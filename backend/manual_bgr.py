import time
import os
from datetime import datetime

import cv2
import numpy as np

import globals
from cameracontrol import converter
from autofocus_back import grayscale_difference_score


# ==========================================================
# Small utils
# ==========================================================
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


def safe_float_str(x, nd=3) -> str:
    return f"{float(x):.{nd}f}".replace(".", "p").replace("-", "m")


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


# ==========================================================
# Debug dump
# ==========================================================
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


# ==========================================================
# Camera grab
# ==========================================================
def acquire_frame_manual(timeout_ms=2000, retries=2):
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


# ==========================================================
# Manual out-of-frame check (same logic as AF final check)
# ==========================================================
def final_out_of_frame_check_manual(
        frame_scale,
        grab_timeout_ms,
        debug,
        debug_buffer,
        margin_px=2,
        min_area_ratio=0.001,
):
    # --- grab ---
    try:
        frame = acquire_frame_manual(timeout_ms=grab_timeout_ms)
    except Exception as e:
        # képelemzés szempontból ez "nincs frame"
        return False, "E2200", None  # (új) grab fail

    # --- resize (logika marad, csak try) ---
    if frame_scale is not None and float(frame_scale) != 1.0:
        try:
            frame = cv2.resize(
                frame, None,
                fx=float(frame_scale),
                fy=float(frame_scale),
                interpolation=cv2.INTER_AREA
            )
        except Exception:
            return False, "E2201", None  # (új) resize fail

    # --- stats check (AF stílus) ---
    std_gray, mean_abs_diff, min_gray, max_gray = grayscale_difference_score(frame)
    if std_gray is None:
        return False, "E2205", None

    if std_gray < 10 and mean_abs_diff <= 5:
        return False, "E2000", None
    if 5 < mean_abs_diff < 10:
        return False, "E2002", None
    if mean_abs_diff > 100:
        return False, "E2003", None

    # --- Debug buffer (mint AF-ben) ---
    if debug and debug_buffer is not None:
        debug_buffer.append({
            "stage": "final_check_manual",
            "z": 0.0,
            "idx": 0,
            "frame": frame.copy()
        })

    # --- grayscale conversion (ugyanaz mint AF-ben) ---
    if frame.ndim == 2:
        gray = frame
    elif frame.ndim == 3 and frame.shape[2] == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif frame.ndim == 3 and frame.shape[2] == 4:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
    else:
        return False, "E2206", None

    H, W = gray.shape[:2]

    # --- OTSU (logika marad) ---
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    fg = np.mean(gray[bw == 255]) if np.any(bw == 255) else 0
    bg = np.mean(gray[bw == 0]) if np.any(bw == 0) else 0
    if fg < bg:
        bw = 255 - bw

    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # itt nálad eddig: return True, None, None
    if not contours:
        return True, None, None

    c = max(contours, key=cv2.contourArea)

    area = float(cv2.contourArea(c))
    if area < (float(min_area_ratio) * H * W):
        return True, None, c

    x, y, w, h = cv2.boundingRect(c)

    left = (x <= margin_px)
    top = (y <= margin_px)
    right = ((x + w) >= (W - 1 - margin_px))
    bottom = ((y + h) >= (H - 1 - margin_px))

    touch_count = int(left) + int(top) + int(right) + int(bottom)

    print(f"[FRAME_TOUCH] L={left} T={top} R={right} B={bottom} count={touch_count}")

    if touch_count == 0:
        return True, None, c

    if touch_count == 1:
        error_code = "E2004"
        if debug:
            dump_debug_buffer_to_error(debug_buffer, error_code)
        return False, error_code, c

    if touch_count == 2:
        opposite_ok = (left and right) or (top and bottom)
        if opposite_ok:
            return True, None, c
        else:
            error_code = "E2004"
            if debug:
                dump_debug_buffer_to_error(debug_buffer, error_code)
            return False, error_code, c

    if touch_count == 3:
        error_code = "E2004"
        if debug:
            dump_debug_buffer_to_error(debug_buffer, error_code)
        return False, error_code, c

    if touch_count == 4:
        return True, None, c

    return True, None, c




# ==========================================================
# Wrapper: run check and return AF-style dict
# ==========================================================
def manual_return(
        frame_scale=0.1,
        grab_timeout_ms=2000,
        margin_px=2,
        min_area_ratio=0.001,
        debug=True,
):
    debug_buffer = [] if debug else None

    ok, err_code, final_contour = final_out_of_frame_check_manual(
        frame_scale=frame_scale,
        grab_timeout_ms=grab_timeout_ms,
        debug=debug,
        debug_buffer=debug_buffer,
        margin_px=margin_px,
        min_area_ratio=min_area_ratio,
    )

    if not ok:
        return _err(str(err_code))

    final_contour_pts = contour_to_points_list(final_contour)

    return _ok(
        z_rel=0.0,
        score=0.0,
        final_contour=final_contour_pts,
    )
