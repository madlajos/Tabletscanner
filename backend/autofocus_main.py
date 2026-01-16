# autofocus_main.py
import time
import os
from datetime import datetime
import cv2

import globals
from motioncontrols import move_relative
from autofocus_back import process_frame, detect_largest_object_square_roi
from cameracontrol import converter
import porthandler


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

def safe_float_str(x, nd=3) -> str:
    # fájlnévbarát: 12.345 -> 12p345, -1.2 -> m1p200
    return f"{float(x):.{nd}f}".replace(".", "p").replace("-", "m")

def wait_motion_done(motion_platform):
    porthandler.write(motion_platform, "M400")

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
    "Domb" alak ellenőrzés a coarse scan ÖSSZES pontjára:

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
def acquire_frame(timeout_ms=2000):
    from pypylon import pylon

    cam = globals.camera
    if cam is None or not cam.IsOpen():
        raise RuntimeError("Camera not ready")

    lock = globals.grab_lock

    with lock:
        if not cam.IsGrabbing():
            raise RuntimeError("Camera is not grabbing. Stream must be running.")

        grab_result = cam.RetrieveResult(int(timeout_ms), pylon.TimeoutHandling_ThrowException)
        try:
            if not grab_result.GrabSucceeded():
                raise RuntimeError("Grab failed")

            # Convert -> BGR8 for OpenCV
            frame_bgr = converter.Convert(grab_result).GetArray()
            frame_bgr = frame_bgr.copy()

        finally:
            grab_result.Release()

    return frame_bgr


def move_to_virtual_z(motion_platform, current_z, target_z, settle_s=1):
    dz = float(target_z) - float(current_z)
    print("Menj " + str(target_z) + " pozira")
    if abs(dz) > 1e-9:
        move_relative(motion_platform, z=dz)
        wait_motion_done(motion_platform)

        if settle_s and settle_s > 0:
            time.sleep(settle_s)

    return float(target_z)


# ---------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------
def measure_focus_score(frame, roi):
    return process_frame(frame, roi=roi)


def measure_score(
    motion_platform, current_z, target_z, roi,
    n_frames=1, timeout_ms=2000,
    debug=False, debug_buffer=None, stage="unk"
):
    """
    Elmegy target_z-re, készít n_frames képet, score-ol.
    Debug módban NEM ment fájlba, hanem memóriába gyűjti a nyers frame-eket.
    """
    current_z = move_to_virtual_z(motion_platform, current_z, target_z)

    scores = []
    n = max(1, int(n_frames))

    for i in range(n):
        frame = acquire_frame(timeout_ms=timeout_ms)

        # memóriába gyűjtés
        if debug and debug_buffer is not None:
            debug_buffer.append({
                "stage": stage,
                "z": float(target_z),
                "idx": i,
                "frame": frame.copy(),
            })

        scores.append(measure_focus_score(frame, roi))

    scores.sort()
    score = scores[len(scores) // 2]  # median
    print("Scores: " + str(scores))

    return current_z, score


# ---------------------------------------------------------------------
# Autofocus main
# ---------------------------------------------------------------------
def autofocus_coarse(
    motion_platform,
    z_min=0.0,
    z_max=25.0,

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

    # ✅ ÚJ: "domb" érzékenység
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
    current_z = move_to_virtual_z(motion_platform, current_z, float(z_min))

    # --- ROI detektálás 1x ---
    first_frame = acquire_frame(timeout_ms=grab_timeout_ms)

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
    # 1) COARSE SCAN
    # -----------------------------------------------------------------
    coarse_points = []  # (z, score)
    best_z = None
    best_score = float("-inf")
    bad_count = 0

    z = float(z_min)
    i = 0
    while z <= float(z_max):
        current_z, s = measure_score(
            motion_platform, current_z, z,
            roi=roi,
            n_frames=n_frames,
            timeout_ms=grab_timeout_ms,
            debug=debug,
            debug_buffer=debug_buffer,
            stage="coarse"
        )
        coarse_points.append((z, s))

        if s > best_score:
            best_score = s
            best_z = z
            bad_count = 0

        if (i + 1) >= int(min_points) and (best_z is not None) and (z > best_z):
            if s < best_score * float(drop_ratio):
                bad_count += 1
            else:
                bad_count = 0
            if bad_count >= int(bad_needed):
                break

        z += float(coarse_step)
        i += 1

    if best_z is None:
        raise RuntimeError("Coarse scan nem adott best_z-t.")
    print("Best z: " + str(best_z))

    # ✅ A COARSE összes pontját vizsgáljuk: "domb" van-e?
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
        current_z, sf = measure_score(
            motion_platform, current_z, zf,
            roi=roi,
            n_frames=n_frames,
            timeout_ms=grab_timeout_ms,
            debug=debug,
            debug_buffer=debug_buffer,
            stage="fine1"
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
        current_z, sf = measure_score(
            motion_platform, current_z, zf,
            roi=roi,
            n_frames=n_frames,
            timeout_ms=grab_timeout_ms,
            debug=debug,
            debug_buffer=debug_buffer,
            stage="fine2"
        )
        fine2_points.append((zf, sf))

    final_z, final_s = max(fine2_points, key=lambda p: p[1])
    print("Final_z: " + str(final_z))

    if move_to_best:
        current_z = move_to_virtual_z(motion_platform, current_z, final_z)

    return {"status": "OK", "z_rel": float(final_z), "score": float(final_s)}
