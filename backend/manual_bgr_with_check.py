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

# ==========================================================
# BBOX helpers
# ==========================================================
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
# Exposure gate
# ==========================================================
def exposure_gate(gray_u8,
                  white_thr=250,
                  frac_white_thr=0.20,
                  under_p95_thr=30,
                  under_dr_thr=30):

    if gray_u8 is None or gray_u8.size == 0:
        return "E_UNDER", {"p95": 0.0, "dr": 0.0, "white": 0.0}

    g = gray_u8.astype(np.float32)

    p1, p95, p99 = np.percentile(g, [1, 95, 99])
    dr = float(p99 - p1)

    frac_white = float((gray_u8 >= int(white_thr)).mean())

    metrics = {"p95": float(p95), "dr": dr, "white": frac_white}

    # OVER
    if frac_white > float(frac_white_thr) or p95 > 250:
        return "E_OVER", metrics

    # UNDER
    if (p95 < float(under_p95_thr)) and (dr < float(under_dr_thr)):
        return "E_UNDER", metrics

    return "EXP_OK", metrics

# ==========================================================
# EMPTY vs PATTERN logic (beemelve)
# ==========================================================
def preprocess_gray(gray_u8, scale=0.5, blur_ksize=3):
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


def percentile_span(gray_u8, p_low=10, p_high=90):
    g = gray_u8.astype(np.float32)
    pl, ph = np.percentile(g, [p_low, p_high])
    return float(pl), float(ph), float(ph - pl)


def robust_texture_and_edge_activity(gray_u8,
                                     hp_sigma=12.0,
                                     hp_clip_lo=5, hp_clip_hi=95,
                                     act_k=6.0):
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

    return {
        "hp_rstd": hp_rstd,
        "sobel_p90": sobel_p90,
        "edge_act": edge_act,
    }


def vignette_profile_metrics(gray_u8, blur_ksize=11, bins=24):
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

    return {
        "center": center,
        "edge": edge,
        "dEC": dEC,
        "dAbs": dAbs,
        "mono": mono,
        "corr": corr,
    }


def decide_pattern(gray_u8,
                   exp_code,
                   # zoom védelem
                   p10_zoom_thr=60.0,
                   # üres-gyanú (közép–szél különbség kicsi)
                   dabs_sus_thr=10.0,
                   # klasszikus üres (dinamika)
                   span_thr=25.0,

                   # textúra erősség (de csak ha kiterjedt!)
                   hp_thr=2.5,
                   sobel_p90_thr=10.0,

                   # kiterjedtség küszöbök
                   edge_act_empty_thr=0.002,
                   edge_act_tex_thr=0.010,

                   # vignetta “rendezettség”
                   v_corr_abs_thr=0.55,
                   v_mono_thr=0.60):

    p10, p90, span = percentile_span(gray_u8, 10, 90)
    tm = robust_texture_and_edge_activity(gray_u8, hp_sigma=12.0, hp_clip_lo=5, hp_clip_hi=95, act_k=6.0)
    vm = vignette_profile_metrics(gray_u8, blur_ksize=11, bins=24)

    info = {"p10": p10, "p90": p90, "span": span, **tm, **vm}

    if exp_code == "E_OVER":
        return "PATTERN_UNKNOWN_OVER", info
    if exp_code == "E_UNDER":
        return "PATTERN_UNKNOWN_UNDER", info

    # zoom: ha p10 magas, nem üres
    if p10 >= float(p10_zoom_thr):
        return "PATTERN_PRESENT_ZOOM", info

    # ha dAbs nagy: mintás
    if vm["dAbs"] > float(dabs_sus_thr):
        return "PATTERN_PRESENT", info

    radial_ok = (abs(vm["corr"]) >= float(v_corr_abs_thr)) and (vm["mono"] >= float(v_mono_thr))
    info["radial_ok"] = bool(radial_ok)

    if tm["edge_act"] <= float(edge_act_empty_thr):
        return "E_EMPTY", info

    if (span < float(span_thr)) and radial_ok:
        return "E_EMPTY", info

    has_texture = (
        ((tm["hp_rstd"] >= float(hp_thr)) or (tm["sobel_p90"] >= float(sobel_p90_thr)))
        and (tm["edge_act"] >= float(edge_act_tex_thr))
    )
    info["has_texture"] = bool(has_texture)

    if has_texture:
        return "PATTERN_PRESENT_ZOOM", info

    return "E_EMPTY", info

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
#   MÓDOSÍTÁS: std_gray < 10 ágban EMPTY/PATTERN döntés
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

    # --- exposure gate (UNDER / OVER) ---
    if frame.ndim == 2:
        gray_gate = frame
    elif frame.ndim == 3 and frame.shape[2] == 3:
        gray_gate = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif frame.ndim == 3 and frame.shape[2] == 4:
        gray_gate = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
    else:
        return False, "E2206", None

    exp_code, exp_m = exposure_gate(
        gray_gate,
        white_thr=250,
        frac_white_thr=0.20,
        under_p95_thr=30,
        under_dr_thr=30
    )

    print(f"[EXPOSURE_GATE] exp={exp_code} "
          f"p95={exp_m['p95']:.1f} dr={exp_m['dr']:.1f} white={exp_m['white']:.3f}")

    if exp_code == "E_OVER":
        return False, "E2003", None  # túl világos

    if exp_code == "E_UNDER":
        return False, "E2002", None  # túl sötét

    # --- uniformity check (egyszínűség) ---
    std_gray = float(gray_gate.std())
    print(f"[UNIFORM_CHECK] std_gray={std_gray:.2f}")

    # ==========================================================
    # MÓDOSÍTOTT RÉSZ: ha std_gray < 10, akkor decide_pattern
    # ==========================================================
    if std_gray < 10.0:
        # gyors preprocess a pattern/empty döntéshez
        gray_pat = preprocess_gray(gray_gate, scale=0.10, blur_ksize=11)

        pat_code, pat_info = decide_pattern(
            gray_pat,
            exp_code="EXP_OK",   # ide már csak UNDER/OVER kizárása után jutunk
            p10_zoom_thr=60.0,
            dabs_sus_thr=10.0,
            span_thr=25.0,
            hp_thr=2.5,
            sobel_p90_thr=10.0,
            edge_act_empty_thr=0.002,
            edge_act_tex_thr=0.010,
            v_corr_abs_thr=0.55,
            v_mono_thr=0.60
        )

        print(f"[PATTERN_GATE@LOW_STD] pat={pat_code} "
              f"p10={pat_info.get('p10', 0):.1f} span={pat_info.get('span', 0):.1f} "
              f"dAbs={pat_info.get('dAbs', 0):.1f} act={pat_info.get('edge_act', 0):.4f} "
              f"mono={pat_info.get('mono', 0):.2f} corr={pat_info.get('corr', 0):.2f}")

        # ha empty -> E2000 és leáll
        if pat_code == "E_EMPTY":
            return False, "E2000", None

        # ha nem empty -> a teljes kép kontúrja, és leáll (nem fut edge/otsu/rounded)
        H, W = gray_gate.shape[:2]
        full_contour = np.array([[[0, 0]],
                                 [[W - 1, 0]],
                                 [[W - 1, H - 1]],
                                 [[0, H - 1]]], dtype=np.int32)
        return True, None, full_contour

    # --- Debug buffer ---
    if debug and debug_buffer is not None:
        debug_buffer.append({
            "stage": "final_check_manual",
            "z": 0.0,
            "idx": 0,
            "frame": frame.copy()
        })

    # --- Debug buffer --- (szándékosan változatlanul hagyva, mert az eredetiben is így volt)
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