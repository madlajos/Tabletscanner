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


def clamp(v, vmin, vmax):
    return max(vmin, min(vmax, v))


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
    H, W = first_frame_bgr.shape[:2]
    gray = cv2.cvtColor(first_frame_bgr, cv2.COLOR_BGR2GRAY)
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
    return cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)


# ---------------------------------------------------------------------
# Edge ring strength (blur / weak edge check)
# ---------------------------------------------------------------------
def edge_ring_strength_from_roi_gray(roi_gray: np.ndarray, ring_w: int = 5) -> float:
    if roi_gray is None or roi_gray.size == 0:
        return 0.0

    H, W = roi_gray.shape[:2]
    rw = int(max(1, ring_w))
    if H < 2 * rw + 2 or W < 2 * rw + 2:
        return 0.0

    gx = cv2.Sobel(roi_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(roi_gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)

    mask = np.zeros((H, W), dtype=np.uint8)
    mask[:, :] = 255
    mask[rw:H - rw, rw:W - rw] = 0  # keep only border ring

    vals = mag[mask == 255]
    if vals.size == 0:
        return 0.0

    return float(np.median(vals))


# ---------------------------------------------------------------------
# SCRIPT-style coarse metrics on bbox ROI
# ---------------------------------------------------------------------
def coarse_metrics_on_bbox(frame_bgr, bbox_state, top_k=500):
    roi = crop_bbox(frame_bgr, bbox_state)
    if roi is None:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    sob = float(sobel_topk_score(gray, top_k=int(top_k)))
    lap, _sq = lap_sq_from_bbox_gray(gray)  # <-- tuple
    lap = float(lap)

    return sob, lap


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
            t = di / (di - dj)  # (0..1)
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
# Debug dump + final check
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
):
    current_z = move_to_virtual_z(motion_platform, current_z, float(target_z))
    frame = acquire_frame(timeout_ms=grab_timeout_ms)

    if frame_scale is not None and float(frame_scale) != 1.0:
        frame = cv2.resize(
            frame, None,
            fx=float(frame_scale),
            fy=float(frame_scale),
            interpolation=cv2.INTER_AREA
        )

    if debug and debug_buffer is not None:
        debug_buffer.append({
            "stage": "final_check",
            "z": float(target_z),
            "idx": 0,
            "frame": frame.copy()
        })

    # -----------------------------
    # Edge strength check (opcionális)
    # -----------------------------
    if (min_edge_strength is not None) and (bbox_state is not None):
        roi_g = bbox_roi_gray(frame, bbox_state)
        edge_strength = edge_ring_strength_from_roi_gray(
            roi_g, ring_w=int(edge_ring_width)
        )
        print(f"[FINAL_EDGE] edge_strength={edge_strength:.4f} "
              f"(min={float(min_edge_strength):.4f})")

        if edge_strength < float(min_edge_strength):
            error_code = "AF_EDGE_TOO_WEAK"
            if debug:
                dump_debug_buffer_to_error(debug_buffer, error_code)
            return False, error_code, None

    # -----------------------------
    # Teljes képes OTSU szegmentálás
    # -----------------------------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape[:2]

    _, bw = cv2.threshold(gray, 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    fg = np.mean(gray[bw == 255]) if np.any(bw == 255) else 0
    bg = np.mean(gray[bw == 0]) if np.any(bw == 0) else 0
    if fg < bg:
        bw = 255 - bw

    contours, _ = cv2.findContours(
        bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return True, None, None

    c = max(contours, key=cv2.contourArea)

    area = float(cv2.contourArea(c))
    if area < (float(min_area_ratio) * H * W):
        return True, None, c

    x, y, w, h = cv2.boundingRect(c)

    # -----------------------------
    # ÚJ SZÉLERINTÉSI LOGIKA
    # -----------------------------
    left   = (x <= margin_px)
    top    = (y <= margin_px)
    right  = ((x + w) >= (W - 1 - margin_px))
    bottom = ((y + h) >= (H - 1 - margin_px))

    touch_count = int(left) + int(top) + int(right) + int(bottom)

    print(f"[FRAME_TOUCH] "
          f"L={left} T={top} R={right} B={bottom} "
          f"count={touch_count}")

    # ---- 0 oldal -> OK ----
    if touch_count == 0:
        return True, None, c

    # ---- 1 oldal -> ERROR ----
    if touch_count == 1:
        error_code = "E2004"
        if debug:
            dump_debug_buffer_to_error(debug_buffer, error_code)
        return False, error_code, c

    # ---- 2 oldal ----
    if touch_count == 2:
        opposite_ok = (left and right) or (top and bottom)
        if opposite_ok:
            return True, None, c
        else:
            error_code = "E2004"
            if debug:
                dump_debug_buffer_to_error(debug_buffer, error_code)
            return False, error_code, c

    # ---- 3 oldal -> ERROR ----
    if touch_count == 3:
        error_code = "E2004"
        if debug:
            dump_debug_buffer_to_error(debug_buffer, error_code)
        return False, error_code, c

    # ---- 4 oldal -> OK ----
    if touch_count == 4:
        return True, None, c

    # fallback (elvileg nem kell)
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

        sobs.append(sob)
        laps.append(lap)

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
        first_frame = cv2.resize(
            first_frame, None,
            fx=float(frame_scale), fy=float(frame_scale),
            interpolation=cv2.INTER_AREA
        )

    std_gray, mean_abs_diff, min_gray, max_gray = grayscale_difference_score(first_frame)
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
        return _err("AF_NO_BBOX")

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
            return _err("ABORTED")

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
        return _err("AF_TOO_FEW_POINTS")

    lz = longest_consecutive_near_zero(laps, eps=float(lap_zero_eps))
    if lz >= int(lap_zero_run_needed):
        if debug:
            dump_debug_buffer_to_error(debug_buffer, "AF_LAP_TOO_LONG_ZERO")
        return _err("AF_LAP_TOO_LONG_ZERO")

    ok_peak = has_peak_shape(sobels, prominence_ratio=float(coarse_peak_prominence_ratio))
    if not ok_peak:
        if debug:
            dump_debug_buffer_to_error(debug_buffer, "AF_NO_PEAK_COARSE_SOBEL")
        return _err("AF_NO_PEAK_COARSE_SOBEL")

    sob_norm = _normalize_01(sobels)
    lap_norm = _normalize_01(laps)

    crossings = estimate_crossings_linear(z_list, sob_norm, lap_norm)
    if not crossings:
        if debug:
            dump_debug_buffer_to_error(debug_buffer, "AF_NO_CROSSING")
        return _err("AF_NO_CROSSING")

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
    )

    if not ok:
        return _err(str(err_code))

    final_contour_pts = contour_to_points_list(final_contour)

    return _ok(
        z_rel=float(best_z),
        score=float(max(sobels)) if sobels else 0.0,
        final_contour=final_contour_pts,
    )