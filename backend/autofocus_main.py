

import time
import os
from datetime import datetime

import cv2
import numpy as np

import globals
from motioncontrols import move_relative, get_toolhead_position  # (get_toolhead_position lehet unused, hagyom)
from cameracontrol import converter
import porthandler

from autofocus_back import (
    sobel_topk_score,
    lap_sq_from_bbox_gray,
    rounded_by_curvature_ignore_border,
    edge_ring_strength_from_roi_gray,
    largest_contour_from_gray_otsu,
)
def _mask_from_largest_otsu(gray_u8: np.ndarray):
    """
    largest_contour_from_gray_otsu -> filled mask (0/255) + area_ratio
    """
    if gray_u8 is None or getattr(gray_u8, "size", 0) == 0:
        return None, 0.0

    H, W = gray_u8.shape[:2]
    c = largest_contour_from_gray_otsu(gray_u8)
    if c is None:
        return None, 0.0

    area = float(cv2.contourArea(c))
    area_ratio = area / float(H * W + 1e-9)

    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.drawContours(mask, [c], -1, 255, thickness=-1)
    return mask, float(area_ratio)


def _color_signature_from_bgr(img_bgr: np.ndarray, sample_stride: int = 4, mask_u8: np.ndarray | None = None) -> dict:
    """
    Ugyanaz a signature, mint check_only.py-ben: median + span B/G/R + gray.
    Maszk esetén csak a maszk pixelekből számol (downsample után).
    """
    if img_bgr is None or getattr(img_bgr, "size", 0) == 0:
        return {}

    s = max(1, int(sample_stride))

    m = None
    if mask_u8 is not None and getattr(mask_u8, "size", 0) != 0:
        m = (mask_u8[::s, ::s] > 0)
        if not np.any(m):
            m = None

    def pick2d(x2d_u8: np.ndarray) -> np.ndarray:
        xs = x2d_u8[::s, ::s]
        if m is not None:
            px = xs[m]
            return px if px.size else xs.reshape(-1)
        return xs.reshape(-1)

    # gray
    if img_bgr.ndim == 2:
        gray = img_bgr.astype(np.uint8)
        pgray = pick2d(gray)
        return {
            "gray_median": float(np.median(pgray)),
            "gray_span": float(np.max(pgray) - np.min(pgray)),
        }

    if img_bgr.ndim == 3 and img_bgr.shape[2] >= 3:
        b2 = img_bgr[..., 0].astype(np.uint8)
        g2 = img_bgr[..., 1].astype(np.uint8)
        r2 = img_bgr[..., 2].astype(np.uint8)

        gray = cv2.cvtColor(img_bgr[..., :3], cv2.COLOR_BGR2GRAY).astype(np.uint8)

        pb = pick2d(b2)
        pg = pick2d(g2)
        pr = pick2d(r2)
        pgray = pick2d(gray)

        return {
            "b_median": float(np.median(pb)),
            "b_span": float(np.max(pb) - np.min(pb)),
            "g_median": float(np.median(pg)),
            "g_span": float(np.max(pg) - np.min(pg)),
            "r_median": float(np.median(pr)),
            "r_span": float(np.max(pr) - np.min(pr)),
            "gray_median": float(np.median(pgray)),
            "gray_span": float(np.max(pgray) - np.min(pgray)),
        }

    return {}


def _signature_diff(cur: dict, ref: dict) -> dict:
    dif = {}
    keys = set(cur.keys()) & set(ref.keys())
    for k in keys:
        try:
            dif[f"d_{k}"] = float(abs(float(cur[k]) - float(ref[k])))
        except Exception:
            pass
    return dif


def compute_color_values_from_frame(
    frame_bgr: np.ndarray,
    use_otsu_mask: bool = True,
    min_obj_area_ratio: float = 0.10,
    sig_stride: int = 4,
) -> dict:
    """
    check_only.py-szerű signature készítés:
      - OTSU mask (largest contour) ha area_ratio >= 0.30, különben full-frame
      - cur_sig = _color_signature_from_bgr(..., mask_u8=mask)
    Visszaadja:
      {
        "cur_sig": {...},
        "mask_area_ratio": ...,
        "mask_used": ...,
      }
    """
    if frame_bgr is None or getattr(frame_bgr, "size", 0) == 0:
        return {"cur_sig": {}, "mask_area_ratio": 0.0, "mask_used": False}

    gray_u8 = _to_gray_u8(frame_bgr)
    if gray_u8 is None or getattr(gray_u8, "size", 0) == 0:
        return {"cur_sig": {}, "mask_area_ratio": 0.0, "mask_used": False}

    mask_u8 = None
    area_ratio = 0.0
    if bool(use_otsu_mask):
        mask_u8, area_ratio = _mask_from_largest_otsu(gray_u8)
        if area_ratio < float(min_obj_area_ratio):
            mask_u8 = None  # fallback full-frame

    cur_sig = _color_signature_from_bgr(frame_bgr, sample_stride=int(sig_stride), mask_u8=mask_u8)

    return {
        "cur_sig": cur_sig,
        "mask_area_ratio": float(area_ratio),
        "mask_used": bool(mask_u8 is not None),
    }


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
    # !!! MOZGÁS: NEM NYÚLTUNK !!!
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
# Gates
# ---------------------------------------------------------------------
def exposure_gate(
    gray_u8,
    white_thr=250,
    frac_white_thr=0.20,
    under_p95_thr=30,
    under_dr_thr=30,
    # extra "nagyon sötét" kapu (a.jpg-hez), de d-t ne vigye el
    under_p10_thr=10,
    under_p95_hi=55,
    under_p99_thr=90,
):
    """
    Returns:
      exp_code: "EXP_OK" | "E_OVER" | "E_UNDER"
      metrics:  dict(p95, dr, white)
    """
    if gray_u8 is None or gray_u8.size == 0:
        return "E_UNDER", {"p95": 0.0, "dr": 0.0, "white": 0.0}

    g = gray_u8.astype(np.float32)
    p1, p10, p95, p99 = np.percentile(g, [1, 10, 95, 99])
    dr = float(p99 - p1)

    frac_white = float((gray_u8 >= int(white_thr)).mean())
    metrics = {"p95": float(p95), "dr": dr, "white": frac_white}

    if frac_white > float(frac_white_thr) or p95 > 240:
        return "E_OVER", metrics

    # eredeti under (nagyon sötét + lapos)
    if (p95 < float(under_p95_thr)) and (dr < float(under_dr_thr)):
        return "E_UNDER", metrics

    # extra under (nagyon sötét összkép), de csak ha p10 is extrém alacsony
    if (p10 < float(under_p10_thr)) and (p95 < float(under_p95_hi)) and (p99 < float(under_p99_thr)):
        return "E_UNDER", metrics

    return "EXP_OK", metrics


def center_bbox_state(frame_bgr, w_frac=0.5, h_frac=0.5):
    """Fallback ROI: kép közepe (OTSU nélkül)."""
    if frame_bgr is None or frame_bgr.size == 0:
        return None
    H, W = frame_bgr.shape[:2]

    ww = int(max(20, min(W, int(W * float(w_frac)))))
    hh = int(max(20, min(H, int(H * float(h_frac)))))

    cx, cy = W // 2, H // 2
    x1 = max(0, cx - ww // 2)
    y1 = max(0, cy - hh // 2)
    x2 = min(W, x1 + ww)
    y2 = min(H, y1 + hh)

    return {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)}


def full_frame_bbox_state(frame_bgr):
    """ROI = teljes kép."""
    if frame_bgr is None or frame_bgr.size == 0:
        return None
    H, W = frame_bgr.shape[:2]
    return {"x1": 0, "y1": 0, "x2": int(W), "y2": int(H)}


def full_frame_points_from_frame(frame_bgr):
    """Full-frame 'contour' pontok a returnbe."""
    if frame_bgr is None or getattr(frame_bgr, "size", 0) == 0:
        return None
    H, W = frame_bgr.shape[:2]
    return [
        [0, 0],
        [int(W - 1), 0],
        [int(W - 1), int(H - 1)],
        [0, int(H - 1)],
    ]


# ---------------------------------------------------------------------
# Debug-dump gating (csak bizonyos hibáknál)
# ---------------------------------------------------------------------
def should_dump_debug_for_error(error_code: str) -> bool:
    code = str(error_code)
    sobel_lap_fail_codes = {"E2008", "E2009", "E2010", "E2011"}
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
# ROI helpers
# ---------------------------------------------------------------------
def _to_gray_u8(frame_bgr):
    if frame_bgr is None or getattr(frame_bgr, "size", 0) == 0:
        return None
    if frame_bgr.ndim == 2:
        return frame_bgr
    if frame_bgr.shape[2] == 3:
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    if frame_bgr.shape[2] == 4:
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGRA2GRAY)
    return None


def crop_bbox(frame_bgr, bbox):
    x1, y1, x2, y2 = int(bbox["x1"]), int(bbox["y1"]), int(bbox["x2"]), int(bbox["y2"])
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


def _bbox_clamp(x1, y1, x2, y2, W, H):
    x1 = int(max(0, min(W, x1)))
    y1 = int(max(0, min(H, y1)))
    x2 = int(max(0, min(W, x2)))
    y2 = int(max(0, min(H, y2)))
    if x2 <= x1 + 1:
        x2 = min(W, x1 + 2)
    if y2 <= y1 + 1:
        y2 = min(H, y1 + 2)
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def _safe_padded_bbox_from_rect(x, y, w, h, W, H, pad):
    pad = int(max(0, pad))
    left_free = int(x)
    top_free = int(y)
    right_free = int(W - (x + w))
    bottom_free = int(H - (y + h))
    pad_eff = int(min(pad, left_free, right_free, top_free, bottom_free))

    x1 = x - pad_eff
    y1 = y - pad_eff
    x2 = x + w + pad_eff
    y2 = y + h + pad_eff
    return _bbox_clamp(x1, y1, x2, y2, W, H)


def _largest_contour_otsu(gray_u8):
    g = cv2.GaussianBlur(gray_u8, (5, 5), 0)
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    fg = np.mean(gray_u8[bw == 255]) if np.any(bw == 255) else 0.0
    bg = np.mean(gray_u8[bw == 0]) if np.any(bw == 0) else 0.0
    if fg < bg:
        bw = 255 - bw

    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def _bbox_touches_frame(b, W, H, margin=2):
    if b is None:
        return True
    return (
        (b["x1"] <= margin)
        or (b["y1"] <= margin)
        or (b["x2"] >= (W - 1 - margin))
        or (b["y2"] >= (H - 1 - margin))
    )


def _inner_square_from_bbox(bbox, shrink_scale=0.5, min_side=20, frame_W=None, frame_H=None):
    """
    FIXED: a régi verzió hibásan _bbox_clamp(..., x2, y2)-t hívott W,H helyett.
    Ez most vagy a teljes frame-re clamp-el (ha frame_W/H adott),
    vagy valóban bbox-on belül marad.
    """
    x1, y1, x2, y2 = int(bbox["x1"]), int(bbox["y1"]), int(bbox["x2"]), int(bbox["y2"])
    w = max(2, x2 - x1)
    h = max(2, y2 - y1)

    side = float(min(w, h)) * float(shrink_scale)
    side = max(float(min_side), side)

    cx = x1 + w / 2.0
    cy = y1 + h / 2.0

    nx1 = int(round(cx - side / 2.0))
    ny1 = int(round(cy - side / 2.0))
    nx2 = int(round(cx + side / 2.0))
    ny2 = int(round(cy + side / 2.0))

    if frame_W is not None and frame_H is not None:
        return _bbox_clamp(nx1, ny1, nx2, ny2, int(frame_W), int(frame_H))

    # bbox-on belüli clamp
    nx1 = max(x1, min(x2, nx1))
    ny1 = max(y1, min(y2, ny1))
    nx2 = max(x1, min(x2, nx2))
    ny2 = max(y1, min(y2, ny2))
    if nx2 <= nx1 + 1:
        nx2 = min(x2, nx1 + 2)
    if ny2 <= ny1 + 1:
        ny2 = min(y2, ny1 + 2)
    return {"x1": int(nx1), "y1": int(ny1), "x2": int(nx2), "y2": int(ny2)}


def build_sobel_and_lap_rois(first_frame_bgr, uniform_mode, base_bbox_state, pad=20, lap_shrink=0.5):
    """
    uniform_mode:
      - sobel_bbox = base_bbox_state (fullframe vagy center)
      - lap_bbox   = base_bbox_state belsejébe rakott shrinkelt négyzet

    non-uniform:
      - OTSU kontúr
      - sobel_bbox = kontúr boundingRect + safe pad
      - lap_bbox   = kontúr centroid körüli DT-arányos négyzet (minta közepe)
                    + ha képszélt érint / None -> sobel_bbox belső nagy négyzet fallback
    """
    H, W = first_frame_bgr.shape[:2]
    if uniform_mode:
        sobel_bbox = dict(base_bbox_state)
        lap_bbox = _inner_square_from_bbox(
            base_bbox_state,
            shrink_scale=float(lap_shrink),
            min_side=20,
            frame_W=W,
            frame_H=H,
        )
        return sobel_bbox, lap_bbox, None

    gray = _to_gray_u8(first_frame_bgr)
    if gray is None:
        return None, None, None

    c = _largest_contour_otsu(gray)
    if c is None:
        return None, None, None

    x, y, ww, hh = cv2.boundingRect(c)
    sobel_bbox = _safe_padded_bbox_from_rect(x, y, ww, hh, W, H, pad=int(pad))

    # Lap ROI: centroid körüli négyzet (eredeti logika)
    # (NEM nyúlunk hozzá a non-uniform ághoz)
    # Az itt levő kontúr már a "minta"
    # (ha kellene, itt lehetne contour-mask lap, de a kérés szerint nem bántjuk)
    lap_bbox = _inner_square_from_bbox(sobel_bbox, shrink_scale=float(lap_shrink), min_side=20, frame_W=W, frame_H=H)

    if lap_bbox is None or _bbox_touches_frame(lap_bbox, W, H, margin=2):
        lap_bbox = _inner_square_from_bbox(sobel_bbox, shrink_scale=0.85, min_side=40, frame_W=W, frame_H=H)

    return sobel_bbox, lap_bbox, c


# ---------------------------------------------------------------------
# SPECIAL: near-fullframe detection + contour-mask Lap
# ---------------------------------------------------------------------
def bbox_is_near_fullframe(b, W, H, ratio_thr=0.85, touch_margin=2):
    if b is None:
        return False
    w = max(0, int(b["x2"]) - int(b["x1"]))
    h = max(0, int(b["y2"]) - int(b["y1"]))
    area = float(w * h)
    full = float(W) * float(H) + 1e-9
    ratio = area / full

    touches = (
        (int(b["x1"]) <= int(touch_margin)) and
        (int(b["y1"]) <= int(touch_margin)) and
        (int(b["x2"]) >= (int(W) - 1 - int(touch_margin))) and
        (int(b["y2"]) >= (int(H) - 1 - int(touch_margin)))
    )
    return (ratio >= float(ratio_thr)) or bool(touches)


def lap_sq_from_contour_gray(gray_u8, contour, erode_px=2, min_pixels=800):
    """
    Laplacian^2 átlag CSAK a minta kontúrján belül (maszkon).
    """
    if gray_u8 is None or getattr(gray_u8, "size", 0) == 0 or contour is None:
        return 0.0

    H, W = gray_u8.shape[:2]
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=-1)

    if erode_px and int(erode_px) > 0:
        k = int(erode_px) * 2 + 1
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.erode(mask, ker, iterations=1)

    m = (mask > 0)
    n = int(m.sum())
    if n < int(min_pixels):
        return 0.0

    g = gray_u8.astype(np.float32)
    lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
    v = lap[m]
    return float(np.mean(v * v))


# ---------------------------------------------------------------------
# SCRIPT-style coarse metrics:
#   Sobel -> sobel_bbox
#   Lap   -> lap_bbox (alapeset)
# ---------------------------------------------------------------------
def coarse_metrics_on_rois(frame_bgr, sobel_bbox, lap_bbox, top_k=500):
    sobel_roi = crop_bbox(frame_bgr, sobel_bbox) if sobel_bbox is not None else None
    lap_roi = crop_bbox(frame_bgr, lap_bbox) if lap_bbox is not None else None

    if sobel_roi is None or lap_roi is None:
        return 0.0, 0.0

    if sobel_roi.ndim == 3:
        sobel_roi = cv2.cvtColor(sobel_roi, cv2.COLOR_BGR2GRAY)
    if lap_roi.ndim == 3:
        lap_roi = cv2.cvtColor(lap_roi, cv2.COLOR_BGR2GRAY)

    sob = float(sobel_topk_score(sobel_roi, top_k=int(top_k)))
    lap, _ = lap_sq_from_bbox_gray(lap_roi)
    return float(sob), float(lap)


# ---------------------------------------------------------------------
# Normalization + crossings (UNWEIGHTED)
# ---------------------------------------------------------------------
def _normalize_01(arr, eps=1e-12):
    a = np.asarray(arr, dtype=np.float32)
    return (a - a.min()) / (a.max() - a.min() + eps)


def estimate_crossings_linear(x_vals, sob_norm, lap_norm):
    """
    Unweighted crossing:
      d = sob_norm - lap_norm
    """
    s = np.asarray(sob_norm, dtype=np.float32)
    l = np.asarray(lap_norm, dtype=np.float32)
    d = s - l

    out = []
    for i in range(len(d) - 1):
        if float(d[i]) == 0.0:
            out.append((i, i, float(x_vals[i]), float(s[i]), 0.0))
        elif float(d[i]) * float(d[i + 1]) < 0.0:
            di = float(d[i])
            dj = float(d[i + 1])
            t = di / (di - dj)
            xi, xj = float(x_vals[i]), float(x_vals[i + 1])
            x_star = xi + t * (xj - xi)
            y_star = float(s[i] + t * (s[i + 1] - s[i]))
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


def sobel_is_informative(
    scores,
    peak_prominence_ratio=0.05,
    flat_rel_range_thr=0.08,
    flat_abs_range_thr=None,
    eps=1e-12,
) -> bool:
    """
    True, ha a Sobel görbe elég informatív.
    - Ha túl lapos: (max-min)/max < flat_rel_range_thr -> False
    - Ha nincs peak (has_peak_shape) -> False
    - Opcionális abszolút laposság: (max-min) < flat_abs_range_thr -> False
    """
    if not scores or len(scores) < 3:
        return False

    s = np.asarray(scores, dtype=np.float32)
    smax = float(np.max(s))
    smin = float(np.min(s))
    rng = float(smax - smin)
    rel_rng = rng / (abs(smax) + eps)

    if rel_rng < float(flat_rel_range_thr):
        return False

    if flat_abs_range_thr is not None and rng < float(flat_abs_range_thr):
        return False

    if not has_peak_shape(list(map(float, scores)), prominence_ratio=float(peak_prominence_ratio)):
        return False

    return True


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
# PATTERN / EMPTY decision (uniform + EXP_OK) + colorfulness split
# (változatlan)
# ---------------------------------------------------------------------
def _preprocess_gray(gray_u8, scale=0.10, blur_ksize=11):
    g = gray_u8
    if scale is not None and float(scale) < 1.0:
        g = cv2.resize(g, None, fx=float(scale), fy=float(scale), interpolation=cv2.INTER_AREA)

    if blur_ksize and int(blur_ksize) > 0:
        k = int(blur_ksize)
        if k % 2 == 0:
            k += 1
        if k < 3:
            k = 3
        g = cv2.GaussianBlur(g, (k, k), 0)

    return g


def _percentile_span(gray_u8, p_low=10, p_high=90):
    g = gray_u8.astype(np.float32)
    pl, ph = np.percentile(g, [p_low, p_high])
    return float(pl), float(ph), float(ph - pl)


def _robust_texture_and_edge_activity(gray_u8, hp_sigma=12.0, hp_clip_lo=5, hp_clip_hi=95, act_k=6.0):
    g = gray_u8.astype(np.float32)

    low = cv2.GaussianBlur(g, (0, 0), float(hp_sigma))
    hp = g - low
    lo, hi = np.percentile(hp, [float(hp_clip_lo), float(hp_clip_hi)])
    hp_w = np.clip(hp, lo, hi)

    hp_med = float(np.median(hp_w))
    hp_mad = float(np.median(np.abs(hp_w - hp_med)))
    hp_rstd = float(1.4826 * hp_mad)

    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)

    sobel_med = float(np.median(mag))
    sobel_mad = float(np.median(np.abs(mag - sobel_med)))
    sobel_p90 = float(np.percentile(mag, 90))

    thr = sobel_med + float(act_k) * (1.4826 * sobel_mad + 1e-6)
    edge_act = float((mag > thr).mean())

    return {"hp_rstd": hp_rstd, "sobel_p90": sobel_p90, "edge_act": edge_act}


def _vignette_profile_metrics(gray_u8, blur_ksize=11, bins=24):
    k = int(blur_ksize)
    if k % 2 == 0:
        k += 1
    if k < 31:
        k = 31

    g = cv2.GaussianBlur(gray_u8, (k, k), 0).astype(np.float32)

    H, W = g.shape[:2]
    cy, cx = H / 2.0, W / 2.0
    yy, xx = np.indices((H, W))
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r = r / (r.max() + 1e-9)

    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    means = []
    for i in range(int(bins)):
        m = (r >= edges[i]) & (r < edges[i + 1])
        means.append(float(g[m].mean()) if np.any(m) else float(np.mean(g)))

    center = float(means[0])
    edge = float(np.mean(means[-2:]))
    dEC = float(edge - center)
    dAbs = float(abs(dEC))

    diffs = np.diff(np.array(means, dtype=np.float32))
    mono = float(max((diffs > 0).mean(), (diffs < 0).mean()))

    radii = np.linspace(0.0, 1.0, int(bins)).astype(np.float32)
    corr = float(np.corrcoef(radii, np.array(means, dtype=np.float32))[0, 1])

    return {"center": center, "edge": edge, "dEC": dEC, "dAbs": dAbs, "mono": mono, "corr": corr}


def _colorfulness_metric(frame_bgr: np.ndarray, sample_stride: int = 4) -> float:
    if frame_bgr is None or getattr(frame_bgr, "size", 0) == 0:
        return 0.0
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] < 3:
        return 0.0

    b = frame_bgr[::sample_stride, ::sample_stride, 0].astype(np.float32)
    g = frame_bgr[::sample_stride, ::sample_stride, 1].astype(np.float32)
    r = frame_bgr[::sample_stride, ::sample_stride, 2].astype(np.float32)

    return float(np.mean(np.abs(r - g) + np.abs(r - b) + np.abs(g - b)) / 3.0)


def _decide_pattern(
    gray_u8,
    cf=None,
    cf_thr=5.5,
    cf_apply_span_max=20.0,
    cf_apply_dabs_max=10.0,
    p10_zoom_thr=60.0,
    p10_bright_empty_max=68.5,
    dabs_sus_thr=10.0,
    span_thr=25.0,
    hp_thr=2.5,
    sobel_p90_thr=10.0,
    edge_act_empty_thr=0.002,
    edge_act_tex_thr=0.010,
    edge_act_bright_empty_thr=0.020,
    edge_act_classic_empty_max=0.012,
    edge_act_anti_zero_thr=0.0005,
    v_corr_abs_thr=0.55,
    v_mono_thr=0.60,
):
    p10, p90, span = _percentile_span(gray_u8, 10, 90)
    tm = _robust_texture_and_edge_activity(gray_u8, hp_sigma=12.0, hp_clip_lo=5, hp_clip_hi=95, act_k=6.0)
    vm = _vignette_profile_metrics(gray_u8, blur_ksize=11, bins=24)

    info = {"p10": p10, "p90": p90, "span": span, **tm, **vm}

    if cf is not None:
        cfv = float(cf)
        info["cf"] = cfv
        if (span <= float(cf_apply_span_max)) and (vm["dAbs"] <= float(cf_apply_dabs_max)):
            if cfv >= float(cf_thr):
                return "PATTERN_PRESENT_ZOOM", info
            else:
                return "E_EMPTY", info

    if vm["dAbs"] > float(dabs_sus_thr):
        return "PATTERN_PRESENT", info

    radial_ok = (abs(vm["corr"]) >= float(v_corr_abs_thr)) and (vm["mono"] >= float(v_mono_thr))
    info["radial_ok"] = bool(radial_ok)

    if (p10 >= float(p10_zoom_thr)) and (p10 <= float(p10_bright_empty_max)) \
       and (span < float(span_thr)) and radial_ok \
       and (tm["edge_act"] < float(edge_act_anti_zero_thr)):
        return "PATTERN_PRESENT_ZOOM", info

    if tm["edge_act"] <= float(edge_act_empty_thr):
        return "E_EMPTY", info

    if (span < float(span_thr)) and radial_ok and (tm["edge_act"] <= float(edge_act_classic_empty_max)):
        return "E_EMPTY", info

    if (p10 >= float(p10_zoom_thr)) and (p10 <= float(p10_bright_empty_max)) \
       and (span < float(span_thr)) and radial_ok \
       and (tm["edge_act"] <= float(edge_act_bright_empty_thr)):
        return "E_EMPTY", info

    has_texture = (
        ((tm["hp_rstd"] >= float(hp_thr)) or (tm["sobel_p90"] >= float(sobel_p90_thr)))
        and (tm["edge_act"] >= float(edge_act_tex_thr))
    )
    info["has_texture"] = bool(has_texture)

    if has_texture:
        return "PATTERN_PRESENT_ZOOM", info

    if p10 >= float(p10_zoom_thr):
        return "PATTERN_PRESENT_ZOOM", info

    return "E_EMPTY", info


def pattern_empty_gate_from_frame(
    first_frame_bgr,
    uniform_std_thr=10.0,
    pattern_scale=0.10,
    pattern_pre_blur_ksize=11,
    cf_stride=4,
    cf_thr=5.5,
):
    if first_frame_bgr is None or getattr(first_frame_bgr, "size", 0) == 0:
        return {"run": False, "pattern_code": None, "is_empty": None, "info": {}}

    gray_u8 = _to_gray_u8(first_frame_bgr)
    if gray_u8 is None:
        return {"run": False, "pattern_code": None, "is_empty": None, "info": {}}

    std_gray = float(gray_u8.std())
    is_uniform = bool(std_gray < float(uniform_std_thr))

    exp_code, exp_m = exposure_gate(gray_u8, white_thr=250, frac_white_thr=0.20, under_p95_thr=30, under_dr_thr=30)

    if (not is_uniform) or (exp_code != "EXP_OK"):
        return {"run": False, "pattern_code": None, "is_empty": None,
                "info": {"std_gray": std_gray, "exp_code": exp_code, **exp_m}}

    cf = _colorfulness_metric(first_frame_bgr, sample_stride=int(cf_stride))

    gray_pat = _preprocess_gray(gray_u8, scale=float(pattern_scale), blur_ksize=int(pattern_pre_blur_ksize))
    pattern_code, info = _decide_pattern(gray_pat, cf=cf, cf_thr=float(cf_thr))

    return {
        "run": True,
        "pattern_code": pattern_code,
        "is_empty": bool(pattern_code == "E_EMPTY"),
        "info": {"std_gray": std_gray, "exp_code": exp_code, "cf": float(cf), **exp_m, **info},
    }


# ---------------------------------------------------------------------
# Final check (eredeti) + NEW FLAG: do_frame_touch_check
# (változatlan)
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
    do_frame_touch_check=True
):
    current_z = move_to_virtual_z(motion_platform, current_z, float(target_z))
    frame = acquire_frame(timeout_ms=grab_timeout_ms)

    if frame_scale is not None and float(frame_scale) != 1.0:
        frame = cv2.resize(frame, None, fx=float(frame_scale), fy=float(frame_scale), interpolation=cv2.INTER_AREA)

    if debug and debug_buffer is not None:
        debug_buffer.append({"stage": "final_check", "z": float(target_z), "idx": 0, "frame": frame.copy()})

    if (min_edge_strength is not None) and (bbox_state is not None):
        roi_g = bbox_roi_gray(frame, bbox_state)
        edge_strength = edge_ring_strength_from_roi_gray(roi_g, ring_w=int(edge_ring_width))
        print(f"[FINAL_EDGE] edge_strength={edge_strength:.4f} (min={float(min_edge_strength):.4f})")
        if edge_strength < float(min_edge_strength):
            return False, "E2012", None

    gray = _to_gray_u8(frame)
    if gray is None:
        return False, "E2013", None

    H, W = gray.shape[:2]

    c = largest_contour_from_gray_otsu(gray)
    if c is None:
        return False, "E2014", None

    area = float(cv2.contourArea(c))
    if area < (float(min_area_ratio) * H * W):
        return True, None, c

    pts = c.reshape(-1, 2)
    m = int(margin_px)

    left = bool(np.any(pts[:, 0] <= m))
    right = bool(np.any(pts[:, 0] >= (W - 1 - m)))
    top = bool(np.any(pts[:, 1] <= m))
    bottom = bool(np.any(pts[:, 1] >= (H - 1 - m)))

    touch_count = int(left) + int(top) + int(right) + int(bottom)
    opposite_ok = (left and right) or (top and bottom)
    allow_not_enough_interior = (touch_count >= 2)

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

        decision = None
        if isinstance(info, dict):
            decision = info.get("decision", None)

        if (ok_round is None) and (decision == "NOT_ENOUGH_INTERIOR") and allow_not_enough_interior:
            print(f"[ROUNDED_CHECK] allow NOT_ENOUGH_INTERIOR (touch_count={touch_count})")
            ok_round = True

        if (ok_round is None) or (ok_round is False):
            maybe_dump_debug(debug, debug_buffer, str(rounded_error_code))
            return False, str(rounded_error_code), c

    if not bool(do_frame_touch_check):
        print("[FRAME_TOUCH] skipped (do_frame_touch_check=False)")
        return True, None, c

    print(f"[FRAME_TOUCH] L={left} T={top} R={right} B={bottom} count={touch_count}")

    if touch_count == 0:
        return True, None, c
    if touch_count == 1:
        return False, "E2004", c
    if touch_count == 2:
        return False, "E2004", c
    if touch_count == 3:
        return False, "E2004", c
    if touch_count == 4:
        return False, "E2004", c


# ---------------------------------------------------------------------
# Measure at a Z
#  - normál: sobel_roi + lap_roi (bbox)
#  - use_lap_only: lap = contour-mask lap (minta kontúrján belül)
# ---------------------------------------------------------------------
def measure_score(
    motion_platform,
    current_z,
    target_z,
    sobel_bbox_state,
    lap_bbox_state,
    top_k=500,
    n_frames=1,
    timeout_ms=2000,
    frame_scale=1.0,
    debug=False,
    debug_buffer=None,
    stage="unk",
    # NEW:
    use_lap_only=False,
    lap_contour=None,
    lap_erode_px=2,
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

        if not bool(use_lap_only):
            sob, lap = coarse_metrics_on_rois(frame, sobel_bbox_state, lap_bbox_state, top_k=top_k)
        else:
            # Sobel-t meghagyjuk kiszámolni (nem használjuk döntésre, de log/debug miatt ok)
            sob, _lap_bbox = coarse_metrics_on_rois(frame, sobel_bbox_state, lap_bbox_state, top_k=top_k)

            gray = _to_gray_u8(frame)
            if gray is None:
                lap = 0.0
            else:
                lap = lap_sq_from_contour_gray(gray, lap_contour, erode_px=int(lap_erode_px), min_pixels=800)

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
# COARSE only (best_z = crossing / fallback)
# + SPECIAL: uniform+PATTERN+near-fullframe -> use_lap_only -> Lap max
# ---------------------------------------------------------------------
def autofocus_coarse(
    motion_platform,
    z_min=0.0,
    z_max=30.0,
    frame_scale=0.1,
    edge_ring_width=5,
    coarse_step=2.0,
    drop_ratio=0.92,
    bad_needed=4,
    min_points=4,
    n_frames=1,
    grab_timeout_ms=2000,
    move_to_best=True,
    debug=True,
    coarse_peak_prominence_ratio=0.05,
    lap_zero_eps=1e-6,
    lap_zero_run_needed=6,
    min_edge_strength=None,
    otsu_pad=20,
    lap_inscribed_shrink=0.5,
    do_frame_touch_check=True,
    sobel_flat_rel_range_thr=0.08,
    sobel_flat_abs_range_thr=None,
    before_auto = False,
    after_auto = False,
    # NEW:
    skip_empty_check=True,
):
    debug_buffer = [] if debug else None

    # start
    current_z = 0.0
    current_z = move_to_virtual_z(motion_platform, current_z, float(z_max))
    time.sleep(0.2)  # wait for stabilization

    # first frame (for gates + ROI init)
    first_frame = acquire_frame(timeout_ms=grab_timeout_ms)
    if frame_scale is not None and float(frame_scale) != 1.0:
        first_frame = cv2.resize(
            first_frame, None,
            fx=float(frame_scale), fy=float(frame_scale),
            interpolation=cv2.INTER_AREA
        )
    # -----------------------------------------------------------------
    # NEW: compare first frame "color values" with globals reference (ha nem None)
    if before_auto is True:
        ref_sig = getattr(globals, "color_values", None)
        if ref_sig is not None:
            out = compute_color_values_from_frame(
                first_frame,
                use_otsu_mask=True,
                min_obj_area_ratio=0.10,
                sig_stride=4,
            )
            cur_sig = out.get("cur_sig", {}) or {}
            dif = _signature_diff(cur_sig, ref_sig)

            # --- a te "too_much" feltételed (mint a példában) ---
            too_much = (
                    cur_sig.get("gray_median", 0.0) < float(ref_sig.get("gray_median", 0.0)) * 0.5
            )

            print(
                f"[COLOR_CHECK] ok={not too_much} "
                f"mask_used={out.get('mask_used')} area={out.get('mask_area_ratio'):.3f} "
                f"cur_gray_med={cur_sig.get('gray_median')} ref_gray_med={ref_sig.get('gray_median')}"
            )

            if bool(too_much):
                maybe_dump_debug(debug, debug_buffer, "E2000")
                return _err(
                    "E2000",
                    message="Global color signature mismatch (gray_median too low vs reference)",
                    mask_area_ratio=float(out.get("mask_area_ratio", 0.0)),
                    mask_used=bool(out.get("mask_used", False)),
                    ref_sig=ref_sig,
                    cur_sig=cur_sig,
                    **dif,
                    too_much=True,
                )

    gray_first = _to_gray_u8(first_frame)
    if gray_first is None:
        return _err("E2005")

    # ---- Exposure gate ----
    exp_code, exp_m = exposure_gate(
        gray_first,
        white_thr=250,
        frac_white_thr=0.20,
        under_p95_thr=30,
        under_dr_thr=30
    )
    print(f"[EXPOSURE_GATE] exp={exp_code} p95={exp_m['p95']:.1f} dr={exp_m['dr']:.1f} white={exp_m['white']:.3f}")

    if exp_code == "E_OVER":
        return _err("E2003", **exp_m)
    if exp_code == "E_UNDER":
        return _err("E2002", **exp_m)

    # ---- Uniformity ----
    std_gray = float(gray_first.std())
    uniform_mode = (std_gray < 10.0)
    print(f"[UNIFORM_CHECK] std_gray={std_gray:.2f} uniform_mode={uniform_mode}")

    if debug and debug_buffer is not None:
        debug_buffer.append({"stage": "roi_detect", "z": float(current_z), "idx": 0, "frame": first_frame.copy()})

    # -----------------------------------------------------------------
    # Base ROI selection
    # -----------------------------------------------------------------
    pat_code = ""
    if uniform_mode:
        if not bool(skip_empty_check):
            gate = pattern_empty_gate_from_frame(
                first_frame,
                uniform_std_thr=10.0,
                pattern_scale=0.10,
                pattern_pre_blur_ksize=11
            )

            if gate.get("run", False):
                pat_code = str(gate.get("pattern_code") or "")
                is_empty = bool(gate.get("is_empty"))
                print(f"[PATTERN_GATE] pattern_code={pat_code} is_empty={is_empty}")

                if is_empty:
                    maybe_dump_debug(debug, debug_buffer, "E2000")
                    return _err("E2000", **gate.get("info", {}))

                if pat_code.startswith("PATTERN_"):
                    base_bbox = full_frame_bbox_state(first_frame)
                    if base_bbox is None:
                        return _err("E2006")
                    print("[ROI] uniform + PATTERN -> using FULL-FRAME bbox")
                else:
                    base_bbox = center_bbox_state(first_frame, w_frac=0.5, h_frac=0.5)
                    if base_bbox is None:
                        return _err("E2006")
                    print("[ROI] uniform -> using CENTER bbox (no OTSU)")
            else:
                base_bbox = center_bbox_state(first_frame, w_frac=0.5, h_frac=0.5)
                if base_bbox is None:
                    return _err("E2006")
                print("[ROI] uniform -> using CENTER bbox (no OTSU)")
        else:
            # skip_empty_check=True: ne fussunk rá a PATTERN/EMPTY gate-re
            # pat_code marad "", így nem aktiváljuk a special lap-only módot sem.
            print("[PATTERN_GATE] skipped (skip_empty_check=True)")
            base_bbox = center_bbox_state(first_frame, w_frac=0.5, h_frac=0.5)
            if base_bbox is None:
                return _err("E2006")
            print("[ROI] uniform -> using CENTER bbox (no OTSU)")
    else:
        base_bbox = full_frame_bbox_state(first_frame)

    # -----------------------------------------------------------------
    # Build SOBEL + LAP ROIs
    # -----------------------------------------------------------------
    sobel_bbox_state, lap_bbox_state, _c_first = build_sobel_and_lap_rois(
        first_frame,
        uniform_mode=uniform_mode,
        base_bbox_state=base_bbox,
        pad=int(otsu_pad),
        lap_shrink=float(lap_inscribed_shrink),
    )
    if sobel_bbox_state is None or lap_bbox_state is None:
        return _err("E2006")

    print(f"[ROI_SOBEL] {sobel_bbox_state}")
    print(f"[ROI_LAP  ] {lap_bbox_state}")

    # -----------------------------------------------------------------
    # SPECIAL MODE: uniform + PATTERN + near-fullframe => use_lap_only + Lap on contour mask
    # -----------------------------------------------------------------
    use_lap_only = False
    lap_contour = None

    if uniform_mode and str(pat_code).startswith("PATTERN_"):
        H0, W0 = gray_first.shape[:2]
        if bbox_is_near_fullframe(sobel_bbox_state, W0, H0, ratio_thr=0.85, touch_margin=2):
            use_lap_only = True
            print(use_lap_only)
            print("[MODE] uniform + PATTERN + near-fullframe => use_lap_only=True (Lap on contour mask)")

            # a minta kontúrját OTSU-val keressük (csak ehhez a speciális ághoz)
            lap_contour = _largest_contour_otsu(gray_first)
            if lap_contour is None:
                print("[MODE] lap_contour OTSU failed -> fallback: disable use_lap_only")
                use_lap_only = False

    # -----------------------------------------------------------------
    # -----------------------------------------------------------------
    # Coarse scan (z_max -> z_min) + REFINE: Sobel max körül visszalép 2mm és adaptív sűrítés
    # -----------------------------------------------------------------
    z_list = []
    sobels = []
    laps = []

    best_sobel = float("-inf")
    best_sobel_z = None
    best_lap = float("-inf")
    best_lap_z = None
    bad_count = 0

    # --- coarse scan param ---
    z = float(z_max)
    i = 0
    step = abs(float(coarse_step))

    while z >= float(z_min) - 1e-9:
        if globals.autofocus_abort:
            return _err("E2007")

        current_z, sob, lap = measure_score(
            motion_platform, current_z, z,
            sobel_bbox_state=sobel_bbox_state,
            lap_bbox_state=lap_bbox_state,
            top_k=1000,
            n_frames=n_frames,
            timeout_ms=grab_timeout_ms,
            frame_scale=frame_scale,
            debug=debug,
            debug_buffer=debug_buffer,
            stage="coarse",
            use_lap_only=use_lap_only,
            lap_contour=lap_contour,
            lap_erode_px=2,
        )

        z_list.append(float(z))
        sobels.append(float(sob))
        laps.append(float(lap))

        # --- best tracking + early stop ---
        if not use_lap_only:
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
        else:
            if lap > best_lap:
                best_lap = float(lap)
                best_lap_z = float(z)
                bad_count = 0

            if (i + 1) >= int(min_points) and (best_lap_z is not None) and (z < best_lap_z):
                if lap < best_lap * float(drop_ratio):
                    bad_count += 1
                else:
                    bad_count = 0
                if bad_count >= int(bad_needed):
                    break

        z -= float(step)
        i += 1

    # --- min points check (marad) ---
    if len(z_list) < 3:
        maybe_dump_debug(debug, debug_buffer, "E2008")
        return _err("E2008")

    # -----------------------------------------------------------------
    # REFINE PASS: Sobel maximum körül visszalép + adaptív lépéssel lescan
    # (csak normál ágon, Lap-only-n nem)
    # -----------------------------------------------------------------
    refine_back_mm = 2.0  # innen indulunk: best_sobel_z + 2mm (felfelé, mert lefelé scan van)
    refine_window_down_mm = 4.0  # meddig menjünk le: best_sobel_z - 4mm (peak alatt)
    adaptive_step = True

    # adaptív lépés paramok (refine-hoz)
    step_min = 0.25
    step_max = min(1.5, abs(float(coarse_step)))
    step_down = 0.7
    step_up = 1.25
    near_best_ratio = 0.92
    rise_thr = 0.03
    fall_thr = 0.06

    # "micro backtrack" paramok: szelíden, feltételesen
    backtrack_max = 2
    backtrack_cap_mm = 0.60  # sose lépjen vissza többet ennél
    edge_guard_mm = 0.80  # ha az új max a start közelében van, akkor backtrack (ráfutás kell)
    drop_confirm_needed = 1  # hány egymás utáni esés kell "átmentünk a peaken" jelzéshez
    drop_rel_thr = 0.015  # esés küszöb local_best-hez képest (rel)
    drop_near_best_ratio = 0.98  # csak akkor érdekes az esés, ha előtte nagyon közel voltunk a maxhoz

    if (not use_lap_only) and (best_sobel_z is not None) and (len(z_list) >= 3):
        z_peak = float(best_sobel_z)

        z_ref_start = min(float(z_max), z_peak + float(refine_back_mm))
        z_ref_end = max(float(z_min), z_peak - float(refine_window_down_mm))

        if z_ref_start > z_ref_end + 1e-6:
            print(f"[REFINE] peak_z={z_peak:.3f} -> start={z_ref_start:.3f} end={z_ref_end:.3f}")

            refine_pts = []
            z = float(z_ref_start)

            step = float(max(step_min, min(step_max, 1.0)))  # finom kezdés
            local_best = float("-inf")

            backtrack_used = 0
            last_backtrack_z = None

            consec_drop = 0
            last_sob = None

            while z >= float(z_ref_end) - 1e-9:
                if globals.autofocus_abort:
                    return _err("E2007")

                current_z, sob, lap = measure_score(
                    motion_platform, current_z, z,
                    sobel_bbox_state=sobel_bbox_state,
                    lap_bbox_state=lap_bbox_state,
                    top_k=1000,
                    n_frames=n_frames,
                    timeout_ms=grab_timeout_ms,
                    frame_scale=frame_scale,
                    debug=debug,
                    debug_buffer=debug_buffer,
                    stage="refine",
                    use_lap_only=False,
                    lap_contour=lap_contour,
                    lap_erode_px=2,
                )

                sob = float(sob)
                lap = float(lap)
                refine_pts.append((float(z), sob, lap))

                # ---- drop detector: "átmentünk a peaken" jel
                if last_sob is not None and local_best > 0:
                    ds = sob - float(last_sob)
                    ds_rel = ds / (abs(local_best) + 1e-9)

                    # esés akkor számít, ha:
                    # - tényleg csökken
                    # - és a csökkenés "érdemi"
                    if ds_rel < -float(drop_rel_thr):
                        consec_drop += 1
                    else:
                        consec_drop = 0

                last_sob = float(sob)

                # ---- local best frissítés + feltételes backtrack
                new_best = (sob > float(local_best))
                if new_best:
                    local_best = float(sob)

                    if backtrack_used < int(backtrack_max):
                        # (A) ha az új max túl közel van a refine start-hoz, backtrackeljünk
                        near_start = (float(z_ref_start) - float(z)) <= float(edge_guard_mm)

                        # (B) ha már átbillentünk: volt legalább N egymás utáni esés
                        # és az esés előtt tényleg közel voltunk a csúcshoz
                        passed_peak = (consec_drop >= int(drop_confirm_needed))
                        # (a "közel voltunk" feltételt itt úgy fogjuk, hogy mostani max a local_best,
                        # és a legutóbbi pontok közt volt olyan, ami ~local_best közelében van)
                        # egyszerűsítés: ha passed_peak igaz, és a local_best már stabil volt (>=2 pont)
                        # akkor engedjük
                        allow_passed_peak = passed_peak and (len(refine_pts) >= 3)

                        do_backtrack = bool(near_start or allow_passed_peak)

                        if do_backtrack:
                            back_mm = float(max(step_min, min(float(backtrack_cap_mm), step)))
                            z_back = float(min(z_ref_start, z + back_mm))  # felfelé = z nő

                            if (z_back > z + 1e-9) and (
                                    last_backtrack_z is None or abs(z_back - last_backtrack_z) > 1e-6):
                                reason = "near_start" if near_start else "passed_peak_drop"
                                print(
                                    f"[REFINE] ({reason}) new/local peak sob={local_best:.3f} at z={z:.3f} -> backtrack to z={z_back:.3f}")

                                backtrack_used += 1
                                last_backtrack_z = float(z_back)

                                # visszalépés után finomítsunk: kisebb step, és reseteljük a drop számlálót,
                                # hogy ne egyből újra backtrackeljünk
                                step = float(max(step_min, min(step, 0.5 * step_max)))
                                consec_drop = 0

                                z = float(z_back)
                                continue  # új mérés a visszalépett ponton

                # ---- adaptív step update (a local_best-hez képest)
                if adaptive_step and len(refine_pts) >= 2 and local_best > 0:
                    sob_prev = float(refine_pts[-2][1])
                    sob_now = float(refine_pts[-1][1])
                    ds = sob_now - sob_prev
                    ds_rel = ds / (abs(local_best) + 1e-9)

                    if sob_now >= float(local_best) * float(near_best_ratio):
                        step *= float(step_down)
                    elif ds_rel > float(rise_thr):
                        step *= float(step_down)
                    elif ds_rel < -float(fall_thr):
                        step *= float(step_up)

                    step = float(max(step_min, min(step, step_max)))

                # ---- következő pont lefelé
                z_next = float(z) - float(step)
                if z_next < float(z_ref_end):
                    z_next = float(z_ref_end)

                if z_next >= float(z) - 1e-12:
                    break

                z = z_next

            # merge: coarse + refine, z szerint (csökkenő), duplák kiszűrése
            eps_z = 1e-6
            all_pts = [(float(zz), float(ss), float(ll)) for (zz, ss, ll) in zip(z_list, sobels, laps)]
            all_pts.extend(refine_pts)

            all_pts.sort(key=lambda t: t[0], reverse=True)

            merged = []
            for (zz, ss, ll) in all_pts:
                if not merged:
                    merged.append((zz, ss, ll))
                else:
                    if abs(zz - merged[-1][0]) > eps_z:
                        merged.append((zz, ss, ll))
                    else:
                        if ss > merged[-1][1]:
                            merged[-1] = (zz, ss, ll)

            z_list = [t[0] for t in merged]
            sobels = [t[1] for t in merged]
            laps = [t[2] for t in merged]

            best_i = int(np.argmax(np.asarray(sobels, dtype=np.float32))) if sobels else 0
            best_sobel = float(sobels[best_i]) if sobels else float("-inf")
            best_sobel_z = float(z_list[best_i]) if z_list else None
            print(f"[REFINE] merged points={len(z_list)} new_best_z={best_sobel_z:.6f} best_sobel={best_sobel:.3f}")

    # -----------------------------------------------------------------
    # REFINE PASS (LAP-ONLY): Lap maximum körül visszalép + adaptív lépés
    # (csak use_lap_only ágon)
    # -----------------------------------------------------------------
    lap_refine_back_mm = 2.0
    lap_refine_window_down_mm = 4.0
    lap_adaptive_step = True

    lap_step_min = 0.25
    lap_step_max = min(1.5, abs(float(coarse_step)))
    lap_step_down = 0.7
    lap_step_up = 1.25

    lap_near_best_ratio = 0.92
    lap_rise_thr = 0.03
    lap_fall_thr = 0.06

    # micro-backtrack (ugyanaz a filozófia, csak lap-ra)
    lap_backtrack_max = 2
    lap_backtrack_cap_mm = 0.60
    lap_edge_guard_mm = 0.80
    lap_drop_confirm_needed = 1
    lap_drop_rel_thr = 0.015

    if use_lap_only and (best_lap_z is not None) and (len(z_list) >= 3):
        z_peak = float(best_lap_z)

        z_ref_start = min(float(z_max), z_peak + float(lap_refine_back_mm))
        z_ref_end = max(float(z_min), z_peak - float(lap_refine_window_down_mm))

        if z_ref_start > z_ref_end + 1e-6:
            print(f"[REFINE_LAP] peak_z={z_peak:.3f} -> start={z_ref_start:.3f} end={z_ref_end:.3f}")

            refine_pts = []
            z = float(z_ref_start)

            step = float(max(lap_step_min, min(lap_step_max, 1.0)))
            local_best = float("-inf")

            backtrack_used = 0
            last_backtrack_z = None
            consec_drop = 0
            last_lap = None

            while z >= float(z_ref_end) - 1e-9:
                if globals.autofocus_abort:
                    return _err("E2007")

                current_z, sob, lap = measure_score(
                    motion_platform, current_z, z,
                    sobel_bbox_state=sobel_bbox_state,
                    lap_bbox_state=lap_bbox_state,
                    top_k=1000,
                    n_frames=n_frames,
                    timeout_ms=grab_timeout_ms,
                    frame_scale=frame_scale,
                    debug=debug,
                    debug_buffer=debug_buffer,
                    stage="refine_lap",
                    use_lap_only=True,
                    lap_contour=lap_contour,
                    lap_erode_px=2,
                )

                sob = float(sob)
                lap = float(lap)
                refine_pts.append((float(z), sob, lap))

                # drop detector (lap)
                if last_lap is not None and local_best > 0:
                    ds = lap - float(last_lap)
                    ds_rel = ds / (abs(local_best) + 1e-9)
                    if ds_rel < -float(lap_drop_rel_thr):
                        consec_drop += 1
                    else:
                        consec_drop = 0

                last_lap = float(lap)

                # local best + feltételes backtrack
                new_best = (lap > float(local_best))
                if new_best:
                    local_best = float(lap)

                    if backtrack_used < int(lap_backtrack_max):
                        near_start = (float(z_ref_start) - float(z)) <= float(lap_edge_guard_mm)
                        passed_peak = (consec_drop >= int(lap_drop_confirm_needed))
                        do_backtrack = bool(near_start or (passed_peak and len(refine_pts) >= 3))

                        if do_backtrack:
                            back_mm = float(max(lap_step_min, min(float(lap_backtrack_cap_mm), step)))
                            z_back = float(min(z_ref_start, z + back_mm))

                            if (z_back > z + 1e-9) and (
                                    last_backtrack_z is None or abs(z_back - last_backtrack_z) > 1e-6):
                                reason = "near_start" if near_start else "passed_peak_drop"
                                print(
                                    f"[REFINE_LAP] ({reason}) new/local peak lap={local_best:.3f} at z={z:.3f} -> backtrack to z={z_back:.3f}")

                                backtrack_used += 1
                                last_backtrack_z = float(z_back)
                                step = float(max(lap_step_min, min(step, 0.5 * lap_step_max)))
                                consec_drop = 0
                                z = float(z_back)
                                continue

                # adaptív step (lap)
                if lap_adaptive_step and len(refine_pts) >= 2 and local_best > 0:
                    lap_prev = float(refine_pts[-2][2])
                    lap_now = float(refine_pts[-1][2])
                    ds = lap_now - lap_prev
                    ds_rel = ds / (abs(local_best) + 1e-9)

                    if lap_now >= float(local_best) * float(lap_near_best_ratio):
                        step *= float(lap_step_down)
                    elif ds_rel > float(lap_rise_thr):
                        step *= float(lap_step_down)
                    elif ds_rel < -float(lap_fall_thr):
                        step *= float(lap_step_up)

                    step = float(max(lap_step_min, min(step, lap_step_max)))

                # következő pont lefelé
                z_next = float(z) - float(step)
                if z_next < float(z_ref_end):
                    z_next = float(z_ref_end)

                if z_next >= float(z) - 1e-12:
                    break
                z = z_next

            # merge: coarse + refine_lap
            eps_z = 1e-6
            all_pts = [(float(zz), float(ss), float(ll)) for (zz, ss, ll) in zip(z_list, sobels, laps)]
            all_pts.extend(refine_pts)
            all_pts.sort(key=lambda t: t[0], reverse=True)

            merged = []
            for (zz, ss, ll) in all_pts:
                if not merged:
                    merged.append((zz, ss, ll))
                else:
                    if abs(zz - merged[-1][0]) > eps_z:
                        merged.append((zz, ss, ll))
                    else:
                        # duplánál: nagyobb lap legyen a nyertes
                        if ll > merged[-1][2]:
                            merged[-1] = (zz, ss, ll)

            z_list = [t[0] for t in merged]
            sobels = [t[1] for t in merged]
            laps = [t[2] for t in merged]

            best_i = int(np.argmax(np.asarray(laps, dtype=np.float32))) if laps else 0
            best_lap = float(laps[best_i]) if laps else float("-inf")
            best_lap_z = float(z_list[best_i]) if z_list else None
            print(f"[REFINE_LAP] merged points={len(z_list)} new_best_z={best_lap_z:.6f} best_lap={best_lap:.3f}")
    # -----------------------------------------------------------------
    # SPECIAL MODE: best_z = LAP max  (REFINE UTÁN!)
    # -----------------------------------------------------------------
    if use_lap_only:
        best_i = int(np.argmax(np.asarray(laps, dtype=np.float32))) if laps else 0
        best_z = float(z_list[best_i]) * 0.99 if z_list else float(z_max) * 0.99
        print(f"[BEST_Z] use_lap_only -> LAP max at z={best_z:.6f}")

        globals.last_best_z = float(best_z)

        score = float(max(laps)) if laps else 0.0
        current_z = move_to_virtual_z(motion_platform, current_z, float(best_z))
        time.sleep(0.2)

        return _ok(
            z_rel=float(best_z),
            score=float(score),
            final_contour=full_frame_points_from_frame(first_frame),
        )
    # -----------------------------------------------------------------
    # NORMAL MODE (változatlan döntési logika)
    # -----------------------------------------------------------------
    lz = longest_consecutive_near_zero(laps, eps=float(lap_zero_eps))

    sob_ok = sobel_is_informative(
        sobels,
        peak_prominence_ratio=float(coarse_peak_prominence_ratio),
        flat_rel_range_thr=float(sobel_flat_rel_range_thr),
        flat_abs_range_thr=sobel_flat_abs_range_thr,
    )

    if (lz >= int(lap_zero_run_needed)) and sob_ok:
        print(f"[WARN] Lap near-zero run (len={lz}), but Sobel informative -> continue (no E2009)")

    sob_norm = _normalize_01(sobels)
    lap_norm = _normalize_01(laps)

    if not sob_ok:
        best_i = int(np.argmax(np.asarray(laps, dtype=np.float32))) if laps else 0
        best_z = float(z_list[best_i]) * 0.99 if z_list else float(z_max) * 0.99
        print(f"[BEST_Z] Sobel not informative -> LAP max at z={best_z:.6f}")
    else:
        crossings = estimate_crossings_linear(z_list, sob_norm, lap_norm)
        if crossings:
            best_cross = pick_best_crossing(crossings)
            _i, _j, best_z, y_star, t = best_cross
            best_z = float(best_z * 0.99)
            print("Crossings:", crossings)
            print(f"BEST crossing -> z*={best_z:.6f}, y*={float(y_star):.6f}, t={float(t):.6f}")
        else:
            best_i = int(np.argmax(np.asarray(sobels, dtype=np.float32))) if sobels else 0
            best_z = float(z_list[best_i]) * 0.99 if z_list else float(z_max) * 0.99
            print(f"[FALLBACK] no crossings -> SOBEL max at z={best_z:.6f}")

    globals.last_best_z = float(best_z)

    # UNIFORM MODE: no final check
    if uniform_mode:
        score = float(max(sobels)) if sobels else 0.0
        current_z = move_to_virtual_z(motion_platform, current_z, float(best_z))
        time.sleep(0)
        if after_auto is True:
            ref_sig = getattr(globals, "color_values", None)
            if ref_sig is not None:
                out = compute_color_values_from_frame(
                    first_frame,
                    use_otsu_mask=True,
                    min_obj_area_ratio=0.10,
                    sig_stride=4,
                )
                cur_sig = out.get("cur_sig", {}) or {}
                dif = _signature_diff(cur_sig, ref_sig)

                # --- a te "too_much" feltételed (mint a példában) ---
                too_much = (
                        cur_sig.get("gray_median", 0.0) < float(ref_sig.get("gray_median", 0.0)) * 0.5
                )

                print(
                    f"[COLOR_CHECK] ok={not too_much} "
                    f"mask_used={out.get('mask_used')} area={out.get('mask_area_ratio'):.3f} "
                    f"cur_gray_med={cur_sig.get('gray_median')} ref_gray_med={ref_sig.get('gray_median')}"
                )

                if bool(too_much):
                    maybe_dump_debug(debug, debug_buffer, "E2000")
                    return _err(
                        "E2000",
                        message="Global color signature mismatch (gray_median too low vs reference)",
                        mask_area_ratio=float(out.get("mask_area_ratio", 0.0)),
                        mask_used=bool(out.get("mask_used", False)),
                        ref_sig=ref_sig,
                        cur_sig=cur_sig,
                        **dif,
                        too_much=True,
                    )

        return _ok(
            z_rel=float(best_z),
            score=float(score),
            final_contour=full_frame_points_from_frame(first_frame),
        )

    # NON-UNIFORM MODE: final check
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
        bbox_state=sobel_bbox_state,
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
        do_frame_touch_check=bool(do_frame_touch_check),
    )
    if err_code is not None:
        return _err(err_code)
    if before_auto is True:
        ref_sig = getattr(globals, "color_values", None)
        if ref_sig is not None:
            out = compute_color_values_from_frame(
                first_frame,
                use_otsu_mask=True,
                min_obj_area_ratio=0.10,
                sig_stride=4,
            )
            cur_sig = out.get("cur_sig", {}) or {}
            dif = _signature_diff(cur_sig, ref_sig)

            # --- a te "too_much" feltételed (mint a példában) ---
            too_much = (
                    cur_sig.get("gray_median", 0.0) < float(ref_sig.get("gray_median", 0.0)) * 0.5
            )

            print(
                f"[COLOR_CHECK] ok={not too_much} "
                f"mask_used={out.get('mask_used')} area={out.get('mask_area_ratio'):.3f} "
                f"cur_gray_med={cur_sig.get('gray_median')} ref_gray_med={ref_sig.get('gray_median')}"
            )

            if bool(too_much):
                maybe_dump_debug(debug, debug_buffer, "E2000")
                return _err(
                    "E2000",
                    message="Global color signature mismatch (gray_median too low vs reference)",
                    mask_area_ratio=float(out.get("mask_area_ratio", 0.0)),
                    mask_used=bool(out.get("mask_used", False)),
                    ref_sig=ref_sig,
                    cur_sig=cur_sig,
                    **dif,
                    too_much=True,
                )

    final_contour_pts = contour_to_points_list(final_contour)

    return _ok(
        z_rel=float(best_z),
        score=float(max(sobels)) if sobels else 0.0,
        final_contour=final_contour_pts,
    )
