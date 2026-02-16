import time
import os
from datetime import datetime

import cv2
import numpy as np

import globals
from cameracontrol import converter

from autofocus_back import (
    grayscale_difference_score,
    edge_ring_strength_from_roi_gray,
    rounded_by_curvature_ignore_border,
    largest_contour_from_gray_otsu,
)
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

        # --- kötelező edge check paramok ---
        min_edge_strength=10.0,     # <- állítsd be amit használsz
        edge_ring_width=5,
        bbox_pad=20,

        # --- kötelező rounded check paramok (AF defaultok) ---
        rounded_margin_px=15,
        rounded_step=12,
        rounded_angle_threshold_deg=55.0,
        rounded_max_sharp=20,
        rounded_min_used=20,
        rounded_downsample=4,
):
    # --- grab ---
    try:
        frame = acquire_frame_manual(timeout_ms=grab_timeout_ms)
    except Exception:
        return False, "E2200", None  # grab fail

    # --- resize ---
    if frame_scale is not None and float(frame_scale) != 1.0:
        try:
            frame = cv2.resize(
                frame, None,
                fx=float(frame_scale),
                fy=float(frame_scale),
                interpolation=cv2.INTER_AREA
            )
        except Exception:
            return False, "E2201", None  # resize fail
    # --- stats gate (AF stílus) ---
    std_gray, mean_abs_diff, min_gray, max_gray = grayscale_difference_score(frame)
    if std_gray is None:
        return False, "E2205", None

    if std_gray < 10 and mean_abs_diff <= 5:
        return False, "E2000", None
    if 5 < mean_abs_diff < 10:
        return False, "E2002", None
    if mean_abs_diff > 100:
        return False, "E2003", None

    # --- Debug buffer ---
    if debug and debug_buffer is not None:
        debug_buffer.append({
            "stage": "final_check_manual",
            "z": 0.0,
            "idx": 0,
            "frame": frame.copy()
        })

    # --- Debug buffer ---
    if debug and debug_buffer is not None:
        debug_buffer.append({
            "stage": "final_check_manual",
            "z": 0.0,
            "idx": 0,
            "frame": frame.copy()
        })

    # -----------------------------
    # Kötelező EDGE RING check
    # -----------------------------
    # bbox detektálás (ha nem tudsz külső bbox_state-et adni manualnál)
    bbox_state = init_bbox_state(frame, pad=int(bbox_pad))
    if bbox_state is None:
        return False, "E2006", None  # ROI/bbox fail (AF-ben is ez)

    roi_g = bbox_roi_gray(frame, bbox_state)
    edge_strength = edge_ring_strength_from_roi_gray(roi_g, ring_w=int(edge_ring_width))
    print(f"[FINAL_EDGE] edge_strength={edge_strength:.4f} (min={float(min_edge_strength):.4f})")
    if edge_strength < float(min_edge_strength):
        return False, "E2012", None  # AF kód

    # -----------------------------
    # Grayscale + OTSU largest contour
    # -----------------------------
    if frame.ndim == 2:
        gray = frame
    elif frame.ndim == 3 and frame.shape[2] == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif frame.ndim == 3 and frame.shape[2] == 4:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
    else:
        return False, "E2206", None

    H, W = gray.shape[:2]

    c = largest_contour_from_gray_otsu(gray)
    if c is None:
        return False, "E2207", None

    # -----------------------------
    # Area check + frame-touch (AF logika)
    # -----------------------------
    area = float(cv2.contourArea(c))
    if area < (float(min_area_ratio) * H * W):
        return True, None, c

    x, y, w, h = cv2.boundingRect(c)




    # -----------------------------
    # Kötelező ROUNDED check
    # -----------------------------
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
        return False, "E2015", c  # AF kód

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

        # kötelező edge + rounded paramok továbbadva
        min_edge_strength=10.0,
        edge_ring_width=5,

        rounded_margin_px=15,
        rounded_step=12,
        rounded_angle_threshold_deg=55.0,
        rounded_max_sharp=20,
        rounded_min_used=20,
        rounded_downsample=4,
):
    debug_buffer = [] if debug else None

    ok, err_code, final_contour = final_out_of_frame_check_manual(
        frame_scale=frame_scale,
        grab_timeout_ms=grab_timeout_ms,
        debug=debug,
        debug_buffer=debug_buffer,
        margin_px=margin_px,
        min_area_ratio=min_area_ratio,

        min_edge_strength=min_edge_strength,
        edge_ring_width=edge_ring_width,

        rounded_margin_px=rounded_margin_px,
        rounded_step=rounded_step,
        rounded_angle_threshold_deg=rounded_angle_threshold_deg,
        rounded_max_sharp=rounded_max_sharp,
        rounded_min_used=rounded_min_used,
        rounded_downsample=rounded_downsample,
    )

    if not ok:
        # ha akarsz: csak ezeknél dump
        if debug and debug_buffer is not None and str(err_code) in {"E2012", "E2015"}:
            dump_debug_buffer_to_error(debug_buffer, str(err_code))
        return _err(str(err_code))

    final_contour_pts = contour_to_points_list(final_contour)
    return _ok(
        z_rel=0.0,
        score=0.0,
        final_contour=final_contour_pts,
    )