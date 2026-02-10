# autofocus_main.py
import time
import os
from datetime import datetime
import cv2
import numpy as np
import globals
from motioncontrols import move_relative, get_toolhead_position

from autofocus_back import process_frame, detect_largest_object_square_roi, edge_definition_score
from cameracontrol import converter
import porthandler


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def final_out_of_frame_check(
    motion_platform,
    current_z,
    target_z,
    frame_scale,
    grab_timeout_ms,
    debug,
    debug_buffer,
    margin_px=2,
    min_area_ratio=0.001
):
    """
    Elmegy target_z-re, lő egy frame-et, és ellenőrzi hogy a legnagyobb Otsu objektum
    érinti-e a képszélt. Ha igen -> (False, error_payload_dict). Ha ok -> (True, None)
    """
    # menjünk a végső Z-re, hogy valósan azt ellenőrizzük, amit visszaadunk
    current_z = move_to_virtual_z(motion_platform, current_z, float(target_z))

    frame = acquire_frame(timeout_ms=grab_timeout_ms)

    if frame_scale is not None and float(frame_scale) != 1.0:
        frame = cv2.resize(frame, None, fx=float(frame_scale), fy=float(frame_scale),
                           interpolation=cv2.INTER_AREA)

    if debug and debug_buffer is not None:
        debug_buffer.append({
            "stage": "final_check",
            "z": float(target_z),
            "idx": 0,
            "frame": frame.copy(),
        })

    # 1x Otsu + bbox (ugyanazzal a logikával mint nálatok)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape[:2]

    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fg = np.mean(gray[bw == 255]) if np.any(bw == 255) else 0
    bg = np.mean(gray[bw == 0]) if np.any(bw == 0) else 0
    if fg < bg:
        bw = 255 - bw

    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        # nincs objektum -> nálatok ez inkább hiba lehet, de itt "nem lóg ki"
        return True, None

    c = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    if area < (float(min_area_ratio) * H * W):
        return True, None  # túl kicsi -> zaj, ne dobjunk hibát

    x, y, w, h = cv2.boundingRect(c)

    touches = (
        x <= margin_px or
        y <= margin_px or
        (x + w) >= (W - 1 - margin_px) or
        (y + h) >= (H - 1 - margin_px)
    )

    if not touches:
        return True, None

    error_code = "AF_OBJECT_OUT_OF_FRAME_FINAL"
    error_dir = dump_debug_buffer_to_error(debug_buffer, error_code) if debug else None

    payload = {
        "status": "ERROR",
        "error_code": error_code,
        "error_dir": error_dir,
        "z_rel": float(target_z),
        "score": 0.0,
        "info": {
            "bbox": (int(x), int(y), int(w), int(h)),
            "area": area,
            "H": int(H),
            "W": int(W),
            "margin_px": int(margin_px),
        }
    }
    return False, payload

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

def safe_float_str(x, nd=3) -> str:
    # fájlnévbarát: 12.345 -> 12p345, -1.2 -> m1p200
    return f"{float(x):.{nd}f}".replace(".", "p").replace("-", "m")

def wait_motion_done(motion_platform):
    porthandler.write_and_wait_motion(motion_platform, "M400", timeout=30.0)

def clamp(v, vmin, vmax):
    return max(vmin, min(vmax, v))

def dump_debug_buffer_to_error(debug_buffer, error_code: str) -> str:
    """
    Az összes memóriában tárolt frame-et kimenti ide:
      <this_file_dir>/Error/<error_code>/<timestamp>/
    Visszaadja az output mappát.
    """
    if not debug_buffer:
        return None

    this_dir = os.path.dirname(os.path.abspath(__file__))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ensure_dir(os.path.join(this_dir, "Error", error_code, ts))

    # meta infó
    try:
        with open(os.path.join(out_dir, "error.txt"), "w", encoding="utf-8") as f:
            f.write(f"error_code={error_code}\n")
            f.write(f"frames={len(debug_buffer)}\n")
            f.write(f"archived_at={ts}\n")
    except Exception:
        pass

    # képek mentése
    for item in debug_buffer:
        stage = item.get("stage", "unk")
        z = float(item.get("z", 0.0))
        idx = int(item.get("idx", 0))
        frame = item.get("frame", None)
        if frame is None:
            continue

        z_str = safe_float_str(z, 3)
        fname = f"{stage}_z{z_str}_i{idx:02d}.png"
        cv2.imwrite(os.path.join(out_dir, fname), frame)

    return out_dir


def has_peak_shape(scores, prominence_ratio=0.05, eps=1e-12) -> bool:
    """
    "Peak" alak ellenőrzés a coarse scan ÖSSZES pontjára:

    True, ha:
      - van legalább 3 pont
      - a maximum nem az első/utolsó ponton van
      - a maximumtól BALRA és JOBBRA is van legalább 'prominence' esés

    prominence_ratio=0.05 -> 5% esést várunk a csúcsról mindkét oldalon.
    """
    if not scores or len(scores) < 3:
        return False

    best = max(scores)
    best_i = max(range(len(scores)), key=lambda i: scores[i])

    # csúcs nem lehet a szélen (különben "végig nő/csökken" jelleg)
    if best_i == 0 or best_i == len(scores) - 1:
        return False

    thr = best * (1.0 - float(prominence_ratio))

    left_min = min(scores[:best_i]) if best_i > 0 else best
    right_min = min(scores[best_i + 1:]) if best_i < len(scores) - 1 else best

    left_ok = left_min < thr - eps
    right_ok = right_min < thr - eps

    return left_ok and right_ok


# ---------------------------------------------------------------------
# Camera + motion
# ---------------------------------------------------------------------
def acquire_frame(timeout_ms=2000, retries=2):
    from pypylon import pylon

    cam = globals.camera
    if cam is None or not cam.IsOpen():
        raise RuntimeError("Camera not ready")

    lock = globals.grab_lock
    attempts = max(1, int(retries) + 1)
    last_error = None

    with lock:
        if not cam.IsGrabbing():
            try:
                cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            except Exception as e:
                raise RuntimeError(f"Camera is not grabbing and could not be started: {e}")

        for attempt in range(attempts):
            grab_result = None
            try:
                grab_result = cam.RetrieveResult(int(timeout_ms), pylon.TimeoutHandling_ThrowException)
                if not grab_result.GrabSucceeded():
                    last_error = RuntimeError("Grab failed")
                    continue

                # Convert -> BGR8 for OpenCV
                frame_bgr = converter.Convert(grab_result).GetArray()
                return frame_bgr.copy()
            except Exception as e:
                last_error = e
            finally:
                try:
                    if grab_result is not None:
                        grab_result.Release()
                except Exception:
                    pass

            if attempt < attempts - 1:
                time.sleep(0.05)

    if last_error:
        raise RuntimeError(str(last_error))
    raise RuntimeError("Grab failed")


def move_to_virtual_z(motion_platform, current_z, target_z, settle_s=1):
    dz = float(target_z) - float(current_z)
    print("Menj " + str(target_z) + " pozícióra")
    if abs(dz) > 1e-9:
        move_relative(motion_platform, z=dz)
        time.sleep(float(settle_s))
        wait_motion_done(motion_platform)

    return float(target_z)


# ---------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------
def measure_focus_score(frame, roi):
    return process_frame(frame, roi=roi)


def measure_score(
    motion_platform, current_z, target_z, roi,
    n_frames=1, timeout_ms=2000,
    debug=False, debug_buffer=None, stage="unk",
    frame_scale=1.0,
    score_fn=None,   # ha megadod, ezt használja (pl. edge score)
):
    current_z = move_to_virtual_z(motion_platform, current_z, target_z)

    scores = []
    n = max(1, int(n_frames))

    for i in range(n):
        frame = acquire_frame(timeout_ms=timeout_ms)

        # downscale (gyorsítás + stabilabb)
        if frame_scale is not None and float(frame_scale) != 1.0:
            frame = cv2.resize(frame, None, fx=float(frame_scale), fy=float(frame_scale), interpolation=cv2.INTER_AREA)

        # debug buffer (a már kicsinyített képet mentjük)
        if debug and debug_buffer is not None:
            debug_buffer.append({
                "stage": stage,
                "z": float(target_z),
                "idx": i,
                "frame": frame.copy(),
            })

        if score_fn is None:
            s = measure_focus_score(frame, roi)
        else:
            s = score_fn(frame)

        scores.append(s)

    scores.sort()
    score = scores[len(scores) // 2]
    print("Scores: " + str(scores))
    return current_z, score

# ---------------------------------------------------------------------
# Autofocus main
# ---------------------------------------------------------------------
def autofocus_coarse(
    motion_platform,
    z_min=0.0,
    z_max=30.0,
    frame_scale=0.35,   # még kisebb kép (pl. 0.5 helyett 0.35 vagy 0.3)
    edge_ring_width=5,  # perem-sáv vastagság
    # Coarse scan
    coarse_step=2.0,
    drop_ratio=0.92,
    bad_needed=2,
    min_points=4,

    # Fine 1 around coarse best
    fine1_offsets=(-1.0, -0.5, 0.0, +0.5, +1.0),

    # Fine 2 around fine1 best
    fine2_step=0.1,
    fine2_halfspan=0.5,

    # Measurement
    n_frames=1,
    roi_square_scale=0.8,
    grab_timeout_ms=2000,

    # Return / behavior
    move_to_best=True,

    # Debug
    debug=True,

    #  ÚJ: "peak" érzékenység
    coarse_peak_prominence_ratio=0.05,
):
    """
    Coarse görbe validálás:
      - nem elég, hogy "nem monoton"
      - a teljes coarse sorozatnak domb alakúnak kell lennie:
        * max nem lehet az első/utolsó ponton
        * a max-tól balra és jobbra is legyen érdemi visszaesés
    """

    debug_buffer = [] if debug else None

    current_z = 0.0
    # induljunk a legmagasabb pozícióról
    current_z = move_to_virtual_z(motion_platform, current_z, float(z_max))

    # --- ROI detektálás 1x ---
    first_frame = acquire_frame(timeout_ms=grab_timeout_ms)

    # ugyanaz a downscale az ROI detekthez is
    if frame_scale is not None and float(frame_scale) != 1.0:
        first_frame = cv2.resize(first_frame, None, fx=float(frame_scale), fy=float(frame_scale),
                                 interpolation=cv2.INTER_AREA)

    if debug and debug_buffer is not None:
        debug_buffer.append({
            "stage": "roi_detect",
            "z": float(current_z),
            "idx": 0,
            "frame": first_frame.copy(),
        })

    roi = detect_largest_object_square_roi(
        first_frame,
        square_scale=roi_square_scale,
        debug_scale=0.3,
        show_debug=False
    )
    if roi is None:
        roi = None

    # -----------------------------------------------------------------
    # 1) COARSE SCAN (✅ z_max -> z_min)
    # -----------------------------------------------------------------
    coarse_points = []  # (z, score)
    best_z = None
    best_score = float("-inf")
    bad_count = 0

    z = float(z_max)
    i = 0
    step = abs(float(coarse_step))  # biztos pozitív
    while z >= float(z_min):
        # if globals.autofocus_abort:
        #     return {
        #         "status": "ABORTED",
        #         "z_rel": float(best_z) if best_z else float(z),
        #         "score": float(best_score) if best_score != float("-inf") else 0.0,
        #     }
            # Check for abort signal
        if globals.autofocus_abort:
            return {
                "status": "ABORTED",
                "z_rel": float(best_z) if best_z else float(z),
                "score": float(best_score) if best_score != float("-inf") else 0.0,
            }
        current_z, s = measure_score(
            motion_platform, current_z, z,
            roi=None,  # ✅ coarse-hoz nem kell ROI
            n_frames=n_frames,
            timeout_ms=grab_timeout_ms,
            debug=debug,
            debug_buffer=debug_buffer,
            stage="coarse",
            frame_scale=frame_scale,
            score_fn=lambda fr: edge_definition_score(fr, ring_width=edge_ring_width),
        )
        coarse_points.append((z, s))

        if s > best_score:
            best_score = s
            best_z = z
            bad_count = 0

        # Early-stop lefelé:
        # akkor "best után vagyunk", ha z < best_z (mert z csökken)
        if (i + 1) >= int(min_points) and (best_z is not None) and (z < best_z):
            if s < best_score * float(drop_ratio):
                bad_count += 1
            else:
                bad_count = 0
            if bad_count >= int(bad_needed):
                break

        z -= step
        i += 1

    if best_z is None:
        raise RuntimeError("Coarse scan nem adott best_z-t.")
    print("Best z: " + str(best_z))

    # A COARSE összes pontját vizsgáljuk: "peak" van-e?
    coarse_scores_in_order = [s for _, s in coarse_points]
    ok_peak = has_peak_shape(
        coarse_scores_in_order,
        prominence_ratio=float(coarse_peak_prominence_ratio),
        eps=1e-12
    )

    if not ok_peak:
        error_code = "AF_NO_PEAK_COARSE"
        error_dir = dump_debug_buffer_to_error(debug_buffer, error_code) if debug else None

        print(f"[ERROR] Coarse görbe nem domb alakú. Képek mentve ide: {error_dir}")

        return {
            "status": "ERROR",
            "error_code": error_code,
            "error_dir": error_dir,
            "z_rel": float(best_z),
            "score": float(best_score),
        }

    # -----------------------------------------------------------------
    # 2) FINE 1
    # -----------------------------------------------------------------
    fine1_targets = sorted(set(
        clamp(best_z + float(off), float(z_min), float(z_max))
        for off in fine1_offsets
    ))
    print("Fine1_targets_z: " + str(fine1_targets))

    fine1_points = []
    for zf in fine1_targets:
        # Check for abort signal (measurement stopped)
        if globals.autofocus_abort:
            if fine1_points:
                best1_z, best1_s = max(fine1_points, key=lambda p: p[1])
            else:
                best1_z = best_z if best_z else 0.0
                best1_s = 0.0
            return {
                "status": "ABORTED",
                "z_rel": float(best1_z),
                "score": float(best1_s),
            }

        current_z, sf = measure_score(
            motion_platform, current_z, zf,
            roi=roi,
            n_frames=n_frames,
            timeout_ms=grab_timeout_ms,
            debug=debug,
            debug_buffer=debug_buffer,
            stage="fine1",
            frame_scale=frame_scale,
            score_fn=None
        )
        fine1_points.append((zf, sf))

    best1_z, best1_s = max(fine1_points, key=lambda p: p[1])
    print("Fine best z: " + str(best1_z))

    # -----------------------------------------------------------------
    # 3) FINE 2
    # -----------------------------------------------------------------
    halfspan = float(fine2_halfspan)
    step2 = float(fine2_step)
    if step2 <= 0:
        raise ValueError("fine2_step legyen > 0.")

    k = int(round(halfspan / step2))
    offsets2 = [j * step2 for j in range(-k, k + 1)]

    fine2_targets = sorted(set(
        clamp(best1_z + off, float(z_min), float(z_max))
        for off in offsets2
    ))
    print("Fine2 targets: " + str(fine2_targets))

    fine2_points = []
    for zf in fine2_targets:
        # Check for abort signal (measurement stopped)
        if globals.autofocus_abort:
            if fine2_points:
                final_z, final_s = max(fine2_points, key=lambda p: p[1])
            else:
                final_z = best1_z if best1_z else 0.0
                final_s = 0.0
            return {
                "status": "ABORTED",
                "z_rel": float(final_z),
                "score": float(final_s),
            }
        current_z, sf = measure_score(
            motion_platform, current_z, zf,
            roi=roi,
            n_frames=n_frames,
            timeout_ms=grab_timeout_ms,
            debug=debug,
            debug_buffer=debug_buffer,
            stage="fine2",
            frame_scale=frame_scale,
            score_fn=None
        )
        fine2_points.append((zf, sf))

    final_z, final_s = max(fine2_points, key=lambda p: p[1])
    print("Final_z: " + str(final_z))
    globals.last_best_z = float(final_z)

    # ✅ VÉGSO KILÓGÁS CHECK (return előtt)
    ok, err_payload = final_out_of_frame_check(
        motion_platform=motion_platform,
        current_z=current_z,
        target_z=final_z,
        frame_scale=frame_scale,
        grab_timeout_ms=grab_timeout_ms,
        debug=debug,
        debug_buffer=debug_buffer,
        margin_px=2,
        min_area_ratio=0.001
    )
    if not ok:
        return err_payload

    # ha ok, és kell mozogni -> már amúgy is a final_z-n vagyunk, de hagyjuk konzisztensen:
    if move_to_best:
        current_z = move_to_virtual_z(motion_platform, current_z, final_z)

    return {"status": "OK", "z_rel": float(final_z), "score": float(final_s)}

def autofocus_fine_only(
    motion_platform,

    # ha None, akkor globals.last_best_z-t használja
    start_z=None,

    z_min=0.0,
    z_max=30.0,

    frame_scale=0.35,
    roi_square_scale=0.8,

    fine1_offsets=(-1.0, -0.5, 0.0, +0.5, +1.0),
    fine2_step=0.1,
    fine2_halfspan=0.5,

    n_frames=1,
    grab_timeout_ms=2000,

    move_to_best=True,
    debug=True,

    # Fine1 "domb" érzékenység
    fine1_peak_prominence_ratio=0.05,

    # ✅ Fallback: ha nincs peak fine1-ben, fusson a teljes coarse+fine
    fallback_to_coarse=True,

    # (opcionális) coarse paraméterek, ha át akarod adni:
    edge_ring_width=5,
    coarse_step=2.0,
    drop_ratio=0.92,
    bad_needed=2,
    min_points=4,
    coarse_peak_prominence_ratio=0.05,
):
    debug_buffer = [] if debug else None

    # start_z feloldás: paraméter > globals
    if start_z is None:
        start_z = globals.last_best_z

    if start_z is None:
        # No previous best Z available — fall back to full coarse+fine autofocus
        print("[FINE_ONLY] No start_z and globals.last_best_z is None -> falling back to autofocus_coarse")
        return autofocus_coarse(
            motion_platform,
            z_min=float(z_min),
            z_max=float(z_max),
            frame_scale=float(frame_scale),
            edge_ring_width=int(edge_ring_width),

            coarse_step=float(coarse_step),
            drop_ratio=float(drop_ratio),
            bad_needed=int(bad_needed),
            min_points=int(min_points),

            fine1_offsets=fine1_offsets,
            fine2_step=float(fine2_step),
            fine2_halfspan=float(fine2_halfspan),

            n_frames=int(n_frames),
            roi_square_scale=float(roi_square_scale),
            grab_timeout_ms=int(grab_timeout_ms),

            move_to_best=bool(move_to_best),
            debug=bool(debug),

            coarse_peak_prominence_ratio=float(coarse_peak_prominence_ratio),
        )

    start_z = float(start_z)
    start_z = clamp(start_z, float(z_min), float(z_max))

    # induljunk a start_z-re
    # induljunk a JELENLEGI valós Z-ről (M114), és ne mozduljunk az elején
    try:
        pos = get_toolhead_position(motion_platform, timeout=0.4)
        current_z = float(pos["z"])
        print(f"[FINE_ONLY] Start from M114 Z={current_z:.3f} (target start_z={start_z:.3f})")
    except Exception as e:
        # ha busy / nem olvasható, akkor fallback (utolsó ismert / 0)
        current_z = float(getattr(globals, "current_virtual_z", 0.0))
        print(f"[FINE_ONLY] M114 failed ({e}) -> fallback current_z={current_z:.3f}")

    # --- ROI detektálás 1x ---
    first_frame = acquire_frame(timeout_ms=grab_timeout_ms)

    if frame_scale is not None and float(frame_scale) != 1.0:
        first_frame = cv2.resize(
            first_frame, None,
            fx=float(frame_scale), fy=float(frame_scale),
            interpolation=cv2.INTER_AREA
        )

    if debug and debug_buffer is not None:
        debug_buffer.append({
            "stage": "roi_detect",
            "z": float(current_z),
            "idx": 0,
            "frame": first_frame.copy(),
        })

    roi = detect_largest_object_square_roi(
        first_frame,
        square_scale=roi_square_scale,
        debug_scale=0.3,
        show_debug=False
    )
    if roi is None:
        roi = None

    # -----------------------------------------------------------------
    # 1) FINE 1 (start_z körül)
    # -----------------------------------------------------------------
    fine1_targets = sorted(set(
        clamp(start_z + float(off), float(z_min), float(z_max))
        for off in fine1_offsets
    ))
    print("Fine1_targets_z:", fine1_targets)

    fine1_points = []
    for zf in fine1_targets:
        # Check for abort signal (measurement stopped)
        if globals.autofocus_abort:
            if fine1_points:
                best1_z, best1_s = max(fine1_points, key=lambda p: p[1])
            else:
                best1_z = start_z
                best1_s = 0.0
            return {
                "status": "ABORTED",
                "z_rel": float(best1_z),
                "score": float(best1_s),
            }
        current_z, sf = measure_score(
            motion_platform, current_z, zf,
            roi=roi,
            n_frames=n_frames,
            timeout_ms=grab_timeout_ms,
            debug=debug,
            debug_buffer=debug_buffer,
            stage="fine1",
            frame_scale=frame_scale,
            score_fn=None
        )
        fine1_points.append((zf, sf))

    best1_z, best1_s = max(fine1_points, key=lambda p: p[1])
    print("Fine1 best z:", best1_z)

    # FINE1 "Peak" ellenőrzés (z szerint rendezett)
    fine1_points_sorted = sorted(fine1_points, key=lambda p: p[0])
    fine1_scores_in_order = [s for _, s in fine1_points_sorted]

    ok_peak = has_peak_shape(
        fine1_scores_in_order,
        prominence_ratio=float(fine1_peak_prominence_ratio),
        eps=1e-12
    )

    if not ok_peak:
        print("[NO PEAK FIND] Fine1 görbe nem domb alakú -> fallback COARSE+FINE")

        # (opcionális) mentsük ki a fine-only debug képeket is
        if debug:
            dump_debug_buffer_to_error(debug_buffer, "AF_NO_PEAK_FINE1_FALLBACK")

        # mentsük el, ami most a legjobb volt
        globals.last_best_z = float(best1_z)

        if fallback_to_coarse:
            # fusson a teljes coarse+fine
            return autofocus_coarse(
                motion_platform,
                z_min=float(z_min),
                z_max=float(z_max),
                frame_scale=float(frame_scale),
                edge_ring_width=int(edge_ring_width),

                coarse_step=float(coarse_step),
                drop_ratio=float(drop_ratio),
                bad_needed=int(bad_needed),
                min_points=int(min_points),

                fine1_offsets=fine1_offsets,
                fine2_step=float(fine2_step),
                fine2_halfspan=float(fine2_halfspan),

                n_frames=int(n_frames),
                roi_square_scale=float(roi_square_scale),
                grab_timeout_ms=int(grab_timeout_ms),

                move_to_best=bool(move_to_best),
                debug=bool(debug),

                coarse_peak_prominence_ratio=float(coarse_peak_prominence_ratio),
            )

        # ha valamiért nem akarod fallback-ként coarse-t futtatni:
        return {
            "status": "ERROR",
            "error_code": "AF_NO_PEAK_FINE1",
            "z_rel": float(best1_z),
            "score": float(best1_s),
        }

    # -----------------------------------------------------------------
    # 2) FINE 2 (best1 körül)
    # -----------------------------------------------------------------
    halfspan = float(fine2_halfspan)
    step2 = float(fine2_step)
    if step2 <= 0:
        raise ValueError("fine2_step legyen > 0.")

    k = int(round(halfspan / step2))
    offsets2 = [j * step2 for j in range(-k, k + 1)]

    fine2_targets = sorted(set(
        clamp(best1_z + off, float(z_min), float(z_max))
        for off in offsets2
    ))
    print("Fine2 targets:", fine2_targets)

    fine2_points = []
    for zf in fine2_targets:
        # Check for abort signal (measurement stopped)
        if globals.autofocus_abort:
            if fine2_points:
                final_z, final_s = max(fine2_points, key=lambda p: p[1])
            else:
                final_z = best1_z if best1_z else 0.0
                final_s = 0.0
            return {
                "status": "ABORTED",
                "z_rel": float(final_z),
                "score": float(final_s),
            }
        current_z, sf = measure_score(
            motion_platform, current_z, zf,
            roi=roi,
            n_frames=n_frames,
            timeout_ms=grab_timeout_ms,
            debug=debug,
            debug_buffer=debug_buffer,
            stage="fine2",
            frame_scale=frame_scale,
            score_fn=None
        )
        fine2_points.append((zf, sf))

    final_z, final_s = max(fine2_points, key=lambda p: p[1])
    print("Final_z:", final_z)

    globals.last_best_z = float(final_z)

    # ✅ VÉGSO KILÓGÁS CHECK (return előtt)
    ok, err_payload = final_out_of_frame_check(
        motion_platform=motion_platform,
        current_z=current_z,
        target_z=final_z,
        frame_scale=frame_scale,
        grab_timeout_ms=grab_timeout_ms,
        debug=debug,
        debug_buffer=debug_buffer,
        margin_px=2,
        min_area_ratio=0.001
    )
    if not ok:
        return err_payload

    if move_to_best:
        move_to_virtual_z(motion_platform, current_z, final_z)

    return {"status": "OK", "z_rel": float(final_z), "score": float(final_s)}
