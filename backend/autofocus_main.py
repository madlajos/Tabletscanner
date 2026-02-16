import time
import os
from datetime import datetime

import cv2
import numpy as np

import globals
from motioncontrols import move_relative, get_toolhead_position
from cameracontrol import converter
import porthandler

from autofocus_back import (
    grayscale_difference_score,
    sobel_topk_score,
    lap_sq_from_bbox_gray,

    # a korábban main-ben lévő helper-ek most innen jönnek:
    rounded_by_curvature_ignore_border,
    edge_ring_strength_from_roi_gray,
    largest_contour_from_gray_otsu,
)

# ---------------------------------------------------------------------
# Small utils
# ---------------------------------------------------------------------
def contour_to_points_list(c):
    """OpenCV contour -> [[x,y], ...] (JSON/barátságos)."""
    if c is None:
        return None
    try:
        return c.reshape(-1, 2).tolist()
    except Exception:
        return None


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def safe_float_str(x, nd=3) -> str:
    return f"{float(x):.{nd}f}".replace(".", "p").replace("-", "m")


def wait_motion_done(motion_platform):
    porthandler.write_and_wait_motion(motion_platform, "M400", timeout=30.0)


def move_to_virtual_z(motion_platform, current_z, target_z, settle_s=0):
    dz = float(target_z) - float(current_z)
    print("Menj " + str(target_z) + " pozícióra")
    if abs(dz) > 1e-9:
        move_relative(motion_platform, z=dz)
        time.sleep(float(settle_s))
        wait_motion_done(motion_platform)
    return float(target_z)


def near_zero(v, eps=1e-9) -> bool:
    try:
        return abs(float(v)) <= float(eps)
    except Exception:
        return True


def longest_consecutive_near_zero(arr, eps=1e-9) -> int:
    best = 0
    cur = 0
    for v in arr:
        if near_zero(v, eps=eps):
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def _err(code: str, **extra):
    d = {"status": "ERROR", "code": str(code)}
    if extra:
        d.update(extra)
    return d


def _ok(**extra):
    d = {"status": "OK"}
    if extra:
        d.update(extra)
    return d


# ---------------------------------------------------------------------
# Debug-dump gating (csak bizonyos hibáknál)
# ---------------------------------------------------------------------
def should_dump_debug_for_error(error_code: str) -> bool:
    code = str(error_code)

    # Sobel/Lap "lépkedős" logika hibák
    sobel_lap_fail_codes = {"E2008", "E2009", "E2010", "E2011"}

    # Rounded / sok éles sarok
    rounded_fail_codes = {"E2015"}

    return (code in sobel_lap_fail_codes) or (code in rounded_fail_codes)


def maybe_dump_debug(debug: bool, debug_buffer, error_code: str) -> None:
    if debug and debug_buffer is not None and should_dump_debug_for_error(error_code):
        dump_debug_buffer_to_error(debug_buffer, str(error_code))


# ---------------------------------------------------------------------
# Camera
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


# ---------------------------------------------------------------------
# SCRIPT-style FIXED BBOX (first frame)
# ---------------------------------------------------------------------
def init_bbox_state(first_frame_bgr, pad=20):
    if first_frame_bgr is None or first_frame_bgr.size == 0:
        return None
    H, W = first_frame_bgr.shape[:2]

    if first_frame_bgr.ndim == 2:
        gray = first_frame_bgr
    elif first_frame_bgr.ndim == 3 and first_frame_bgr.shape[2] == 3:
        gray = cv2.cvtColor(first_frame_bgr, cv2.COLOR_BGR2GRAY)
    elif first_frame_bgr.ndim == 3 and first_frame_bgr.shape[2] == 4:
        gray = cv2.cvtColor(first_frame_bgr, cv2.COLOR_BGRA2GRAY)
    else:
        return None

    g = cv2.GaussianBlur(gray, (5, 5), 0)
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    fg = np.mean(gray[bw == 255]) if np.any(bw == 255) else 0
    bg = np.mean(gray[bw == 0]) if np.any(bw == 0) else 0
    if fg < bg:
        bw = 255 - bw

    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    c = max(contours, key=cv2.contourArea)
    x, y, ww, hh = cv2.boundingRect(c)

    x1 = max(0, x - int(pad))
    y1 = max(0, y - int(pad))
    x2 = min(W, x + ww + int(pad))
    y2 = min(H, y + hh + int(pad))

    return {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)}


def crop_bbox(frame_bgr, bbox):
    x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
    roi = frame_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    return roi


def bbox_roi_gray(frame_bgr, bbox_state):
    roi = crop_bbox(frame_bgr, bbox_state)
    if roi is None:
        return None
    if roi.ndim == 2:
        return roi
    return cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)


# ---------------------------------------------------------------------
# SCRIPT-style coarse metrics on bbox ROI
# ---------------------------------------------------------------------
def coarse_metrics_on_bbox(frame_bgr, bbox_state, top_k=500):
    roi = crop_bbox(frame_bgr, bbox_state)
    if roi is None:
        return None

    gray = roi if roi.ndim == 2 else cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    sob = float(sobel_topk_score(gray, top_k=int(top_k)))
    lap, _sq = lap_sq_from_bbox_gray(gray)
    return float(sob), float(lap)


# ---------------------------------------------------------------------
# Normalization + crossings (script)
# ---------------------------------------------------------------------
def _normalize_01(arr, eps=1e-12):
    a = np.asarray(arr, dtype=np.float32)
    return (a - a.min()) / (a.max() - a.min() + eps)


def estimate_crossings_linear(x_vals, sob_norm, lap_norm):
    d = np.asarray(sob_norm, dtype=np.float32) - np.asarray(lap_norm, dtype=np.float32)
    out = []
    for i in range(len(d) - 1):
        if float(d[i]) == 0.0:
            out.append((i, i, float(x_vals[i]), float(sob_norm[i]), 0.0))
        elif float(d[i]) * float(d[i + 1]) < 0.0:
            di = float(d[i])
            dj = float(d[i + 1])
            t = di / (di - dj)
            xi, xj = float(x_vals[i]), float(x_vals[i + 1])
            x_star = xi + t * (xj - xi)
            y_star = float(sob_norm[i] + t * (sob_norm[i + 1] - sob_norm[i]))
            out.append((i, i + 1, x_star, y_star, t))
    return out


def pick_best_crossing(crossings):
    if not crossings:
        return None
    return max(crossings, key=lambda c: float(c[3]))


# ---------------------------------------------------------------------
# Peak check (Sobel-en)
# ---------------------------------------------------------------------
def has_peak_shape(scores, prominence_ratio=0.05, eps=1e-12) -> bool:
    if not scores or len(scores) < 3:
        return False

    best = max(scores)
    best_i = max(range(len(scores)), key=lambda i: scores[i])
    if best_i == 0 or best_i == len(scores) - 1:
        return False

    thr = best * (1.0 - float(prominence_ratio))
    left_min = min(scores[:best_i]) if best_i > 0 else best
    right_min = min(scores[best_i + 1:]) if best_i < len(scores) - 1 else best

    left_ok = left_min < thr - eps
    right_ok = right_min < thr - eps
    return left_ok and right_ok


# ---------------------------------------------------------------------
# Debug dump
# ---------------------------------------------------------------------
def dump_debug_buffer_to_error(debug_buffer, error_code: str) -> str:
    if not debug_buffer:
        return None

    this_dir = os.path.dirname(os.path.abspath(__file__))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ensure_dir(os.path.join(this_dir, "Error", str(error_code), ts))

    try:
        with open(os.path.join(out_dir, "error.txt"), "w", encoding="utf-8") as f:
            f.write(f"error_code={error_code}\n")
            f.write(f"frames={len(debug_buffer)}\n")
            f.write(f"archived_at={ts}\n")
    except Exception:
        pass

    for item in debug_buffer:
        if not isinstance(item, dict):
            continue
        stage = item.get("stage", "unk")
        z = float(item.get("z", 0.0))
        idx = int(item.get("idx", 0))
        frame = item.get("frame", None)
        if frame is None:
            continue

        z_str = safe_float_str(z, 3)
        fname = f"{stage}_z{z_str}_i{idx:02d}.png"
        try:
            cv2.imwrite(os.path.join(out_dir, fname), frame)
        except Exception:
            pass

    return out_dir


# ---------------------------------------------------------------------
# Final check
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
    min_area_ratio=0.001,
    bbox_state=None,
    edge_ring_width=5,
    min_edge_strength=None,

    do_rounded_check=False,
    rounded_margin_px=15,
    rounded_step=12,
    rounded_angle_threshold_deg=55.0,
    rounded_max_sharp=20,
    rounded_min_used=20,
    rounded_downsample=4,
    rounded_error_code="E2015",
):
    current_z = move_to_virtual_z(motion_platform, current_z, float(target_z))
    frame = acquire_frame(timeout_ms=grab_timeout_ms)

    if frame_scale is not None and float(frame_scale) != 1.0:
        frame = cv2.resize(frame, None, fx=float(frame_scale), fy=float(frame_scale), interpolation=cv2.INTER_AREA)

    if debug and debug_buffer is not None:
        debug_buffer.append({"stage": "final_check", "z": float(target_z), "idx": 0, "frame": frame.copy()})

    # Edge strength check (opcionális) -> NEM dumpolunk (nem kértél rá)
    if (min_edge_strength is not None) and (bbox_state is not None):
        roi_g = bbox_roi_gray(frame, bbox_state)
        edge_strength = edge_ring_strength_from_roi_gray(roi_g, ring_w=int(edge_ring_width))
        print(f"[FINAL_EDGE] edge_strength={edge_strength:.4f} (min={float(min_edge_strength):.4f})")
        if edge_strength < float(min_edge_strength):
            return False, "E2012", None

    # Gray
    if frame.ndim == 2:
        gray = frame
    elif frame.shape[2] == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif frame.shape[2] == 4:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
    else:
        return False, "E2013", None

    H, W = gray.shape[:2]

    # Largest contour (back)
    c = largest_contour_from_gray_otsu(gray)
    if c is None:
        return False, "E2014", None

    area = float(cv2.contourArea(c))
    if area < (float(min_area_ratio) * H * W):
        return True, None, c

    # Rounded check (back) -> HA BUKIK, dump kell
    if bool(do_rounded_check):
        ok_round, info = rounded_by_curvature_ignore_border(
            c, W, H,
            margin_px=int(rounded_margin_px),
            step=int(rounded_step),
            angle_threshold_deg=float(rounded_angle_threshold_deg),
            max_sharp=int(rounded_max_sharp),
            min_used=int(rounded_min_used),
            downsample=int(rounded_downsample),
        )
        print(f"[ROUNDED_CHECK] ok={ok_round} info={info}")

        if (ok_round is None) or (ok_round is False):
            maybe_dump_debug(debug, debug_buffer, str(rounded_error_code))
            return False, str(rounded_error_code), c

    # Frame touch logic -> NEM dumpolunk (nem kértél rá)
    x, y, w, h = cv2.boundingRect(c)
    left   = (x <= margin_px)
    top    = (y <= margin_px)
    right  = ((x + w) >= (W - 1 - margin_px))
    bottom = ((y + h) >= (H - 1 - margin_px))
    touch_count = int(left) + int(top) + int(right) + int(bottom)

    print(f"[FRAME_TOUCH] L={left} T={top} R={right} B={bottom} count={touch_count}")

    if touch_count == 0:
        return True, None, c

    if touch_count == 1:
        return False, "E2004", c

    if touch_count == 2:
        opposite_ok = (left and right) or (top and bottom)
        if opposite_ok:
            return True, None, c
        return False, "E2004", c

    if touch_count == 3:
        return False, "E2004", c

    if touch_count == 4:
        return True, None, c

    return True, None, c


# ---------------------------------------------------------------------
# Measure at a Z
# ---------------------------------------------------------------------
def measure_score(
    motion_platform,
    current_z,
    target_z,
    bbox_state,
    top_k=500,
    n_frames=1,
    timeout_ms=2000,
    frame_scale=1.0,
    debug=False,
    debug_buffer=None,
    stage="unk",
):
    current_z = move_to_virtual_z(motion_platform, current_z, target_z)

    sobs = []
    laps = []
    n = max(1, int(n_frames))

    for i in range(n):
        frame = acquire_frame(timeout_ms=timeout_ms)

        if frame_scale is not None and float(frame_scale) != 1.0:
            frame = cv2.resize(frame, None, fx=float(frame_scale), fy=float(frame_scale), interpolation=cv2.INTER_AREA)

        if debug and debug_buffer is not None:
            debug_buffer.append({"stage": stage, "z": float(target_z), "idx": i, "frame": frame.copy()})

        m = coarse_metrics_on_bbox(frame, bbox_state, top_k=top_k)
        if m is None:
            sob, lap = 0.0, 0.0
        else:
            sob, lap = float(m[0]), float(m[1])

        sobs.append(float(sob))
        laps.append(float(lap))

    sobs.sort()
    laps.sort()
    sob_med = sobs[len(sobs) // 2] if sobs else 0.0
    lap_med = laps[len(laps) // 2] if laps else 0.0

    print("Sobel scores:", sobs)
    print("Lap scores  :", laps)
    return current_z, float(sob_med), float(lap_med)


# ---------------------------------------------------------------------
# COARSE only (best_z = crossing)
# ---------------------------------------------------------------------
def autofocus_coarse(
    motion_platform,
    z_min=0.0,
    z_max=30.0,
    frame_scale=0.1,
    edge_ring_width=5,
    coarse_step=2.0,
    drop_ratio=0.92,
    bad_needed=7,
    min_points=4,
    fine1_offsets=(-2, -1.5, -1.0, -0.5, 0.0, +0.5, +1.0, +1.5, +2),
    fine2_step=0.1,
    fine2_halfspan=0.5,
    n_frames=1,
    roi_square_scale=1,
    grab_timeout_ms=2000,
    move_to_best=True,
    debug=True,
    coarse_peak_prominence_ratio=0.05,
    inner_margin_px=10,
    lap_zero_eps=1e-6,
    lap_zero_run_needed=6,
    min_edge_strength=None,
):
    debug_buffer = [] if debug else None

    current_z = 0.0
    current_z = move_to_virtual_z(motion_platform, current_z, float(z_max))

    first_frame = acquire_frame(timeout_ms=grab_timeout_ms)
    if frame_scale is not None and float(frame_scale) != 1.0:
        first_frame = cv2.resize(first_frame, None, fx=float(frame_scale), fy=float(frame_scale), interpolation=cv2.INTER_AREA)

    std_gray, mean_abs_diff, min_gray, max_gray = grayscale_difference_score(first_frame)
    if std_gray is None:
        return _err("E2005")
    if std_gray < 10 and mean_abs_diff <= 5:
        return _err("E2000")
    if 5 < mean_abs_diff < 10:
        return _err("E2002")
    if mean_abs_diff > 100:
        return _err("E2003")

    if debug and debug_buffer is not None:
        debug_buffer.append({"stage": "roi_detect", "z": float(current_z), "idx": 0, "frame": first_frame.copy()})

    bbox_state = init_bbox_state(first_frame, pad=20)
    if bbox_state is None:
        return _err("E2006")

    z_list = []
    sobels = []
    laps = []

    best_sobel = float("-inf")
    best_sobel_z = None
    bad_count = 0

    z = float(z_max)
    i = 0
    step = abs(float(coarse_step))

    while z >= float(z_min) - 1e-9:
        if globals.autofocus_abort:
            return _err("E2007")

        current_z, sob, lap = measure_score(
            motion_platform, current_z, z,
            bbox_state=bbox_state,
            top_k=500,
            n_frames=n_frames,
            timeout_ms=grab_timeout_ms,
            frame_scale=frame_scale,
            debug=debug,
            debug_buffer=debug_buffer,
            stage="coarse",
        )

        z_list.append(float(z))
        sobels.append(float(sob))
        laps.append(float(lap))

        if sob > best_sobel:
            best_sobel = float(sob)
            best_sobel_z = float(z)
            bad_count = 0

        if (i + 1) >= int(min_points) and (best_sobel_z is not None) and (z < best_sobel_z):
            if sob < best_sobel * float(drop_ratio):
                bad_count += 1
            else:
                bad_count = 0
            if bad_count >= int(bad_needed):
                break

        z -= step
        i += 1

    if len(z_list) < 3:
        maybe_dump_debug(debug, debug_buffer, "E2008")
        return _err("E2008")

    lz = longest_consecutive_near_zero(laps, eps=float(lap_zero_eps))
    if lz >= int(lap_zero_run_needed):
        maybe_dump_debug(debug, debug_buffer, "E2009")
        return _err("E2009")

    ok_peak = has_peak_shape(sobels, prominence_ratio=float(coarse_peak_prominence_ratio))
    if not ok_peak:
        maybe_dump_debug(debug, debug_buffer, "E2010")
        return _err("E2010")

    sob_norm = _normalize_01(sobels)
    lap_norm = _normalize_01(laps)

    crossings = estimate_crossings_linear(z_list, sob_norm, lap_norm)
    if not crossings:
        maybe_dump_debug(debug, debug_buffer, "E2011")
        return _err("E2011")

    best_cross = pick_best_crossing(crossings)
    _i, _j, best_z, y_star, t = best_cross
    best_z = float(best_z * 0.99)

    print("Crossings:", crossings)
    print(f"BEST crossing -> z*={best_z:.6f}, y*={float(y_star):.6f}, t={float(t):.6f}")

    globals.last_best_z = float(best_z)

    ok, err_code, final_contour = final_out_of_frame_check(
        motion_platform=motion_platform,
        current_z=current_z,
        target_z=best_z,
        frame_scale=frame_scale,
        grab_timeout_ms=grab_timeout_ms,
        debug=debug,
        debug_buffer=debug_buffer,
        margin_px=2,
        min_area_ratio=0.001,
        bbox_state=bbox_state,
        edge_ring_width=edge_ring_width,
        min_edge_strength=min_edge_strength,

        do_rounded_check=True,
        rounded_margin_px=15,
        rounded_step=12,
        rounded_angle_threshold_deg=55,
        rounded_max_sharp=20,
        rounded_min_used=20,
        rounded_downsample=4,
        rounded_error_code="E2015",
    )

    if not ok:
        # itt már csak E2015 dumpolhat (a többihez nem kérsz mentést)
        maybe_dump_debug(debug, debug_buffer, str(err_code))
        return _err(str(err_code))

    final_contour_pts = contour_to_points_list(final_contour)

    return _ok(
        z_rel=float(best_z),
        score=float(max(sobels)) if sobels else 0.0,
        final_contour=final_contour_pts,
    )