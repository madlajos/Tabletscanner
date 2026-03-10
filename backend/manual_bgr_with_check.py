
import time
import os
from datetime import datetime

import cv2
import numpy as np

import globals
from cameracontrol import converter

from autofocus_back import (
    rounded_by_curvature_ignore_border,
    largest_contour_from_gray_otsu,
)

# ---------------------------------------------------------------------
# Small utils (SCRIPT-STYLE)
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


def _to_gray_u8(frame_bgr):
    if frame_bgr is None or getattr(frame_bgr, "size", 0) == 0:
        return None
    if frame_bgr.ndim == 2:
        return frame_bgr
    if frame_bgr.ndim == 3 and frame_bgr.shape[2] == 3:
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    if frame_bgr.ndim == 3 and frame_bgr.shape[2] == 4:
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGRA2GRAY)
    return None


def full_frame_points_from_gray(gray_u8):
    """Full-frame 'contour' pontok a returnbe."""
    if gray_u8 is None or getattr(gray_u8, "size", 0) == 0:
        return None
    H, W = gray_u8.shape[:2]
    return [
        [0, 0],
        [int(W - 1), 0],
        [int(W - 1), int(H - 1)],
        [0, int(H - 1)],
    ]


# ---------------------------------------------------------------------
# Debug dump gating (csak bizonyos hibáknál)
# ---------------------------------------------------------------------
def should_dump_debug_for_error(error_code: str) -> bool:
    code = str(error_code)
    rounded_fail_codes = {"E2015"}
    return code in rounded_fail_codes


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


def maybe_dump_debug(debug: bool, debug_buffer, error_code: str) -> None:
    if debug and debug_buffer is not None and should_dump_debug_for_error(error_code):
        dump_debug_buffer_to_error(debug_buffer, str(error_code))


# ---------------------------------------------------------------------
# Exposure gate (ugyanaz a logika, mint a másik scriptedben)
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


# ---------------------------------------------------------------------
# PATTERN / EMPTY gate (átvéve a másik scripted stílusára)
#   - csak uniform (std < thr) és EXP_OK esetén fut
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
    """
    OpenCV frame is BGR. Gyors:
      cf = átlag(|R-G| + |R-B| + |G-B|)/3
    """
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
    # color-only split
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

    # 1) color-only decision FIRST, only in near-uniform regime
    if cf is not None:
        cfv = float(cf)
        info["cf"] = cfv
        if (span <= float(cf_apply_span_max)) and (vm["dAbs"] <= float(cf_apply_dabs_max)):
            if cfv >= float(cf_thr):
                return "PATTERN_PRESENT_ZOOM", info
            else:
                return "E_EMPTY", info

    # 2) strong vignette diff => PATTERN
    if vm["dAbs"] > float(dabs_sus_thr):
        return "PATTERN_PRESENT", info

    radial_ok = (abs(vm["corr"]) >= float(v_corr_abs_thr)) and (vm["mono"] >= float(v_mono_thr))
    info["radial_ok"] = bool(radial_ok)

    # 3) anti-a1 guard
    if (p10 >= float(p10_zoom_thr)) and (p10 <= float(p10_bright_empty_max)) \
       and (span < float(span_thr)) and radial_ok \
       and (tm["edge_act"] < float(edge_act_anti_zero_thr)):
        return "PATTERN_PRESENT_ZOOM", info

    # 4) empty via edge_act
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
    """
    Csak akkor fut:
      - std_gray < uniform_std_thr
      - exposure EXP_OK
    """
    if first_frame_bgr is None or getattr(first_frame_bgr, "size", 0) == 0:
        return {"run": False, "pattern_code": None, "is_empty": None, "info": {}}

    gray_u8 = _to_gray_u8(first_frame_bgr)
    if gray_u8 is None:
        return {"run": False, "pattern_code": None, "is_empty": None, "info": {}}

    std_gray = float(gray_u8.std())
    is_uniform = bool(std_gray < float(uniform_std_thr))

    exp_code, exp_m = exposure_gate(gray_u8, white_thr=250, frac_white_thr=0.20, under_p95_thr=30, under_dr_thr=30)

    if (not is_uniform) or (exp_code != "EXP_OK"):
        return {
            "run": False,
            "pattern_code": None,
            "is_empty": None,
            "info": {"std_gray": std_gray, "exp_code": exp_code, **exp_m},
        }

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
# Camera (manual) - ugyanaz a stílus, mint a másik scriptedben
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
# Final manual check (EDGE NÉLKÜL) - SCRIPT-STYLE szerkezetben
# ---------------------------------------------------------------------
def final_out_of_frame_check_manual(
    frame_scale,
    grab_timeout_ms,
    debug,
    debug_buffer,
    min_area_ratio=0.001,
    # pattern gate
    uniform_std_thr=10.0,
    # rounded
    do_rounded_check=True,
    rounded_margin_px=15,
    rounded_step=12,
    rounded_angle_threshold_deg=55.0,
    rounded_max_sharp=20,
    rounded_min_used=20,
    rounded_downsample=4,
    rounded_error_code="E2015",
):
    # --- grab ---
    try:
        frame = acquire_frame(timeout_ms=grab_timeout_ms)
    except Exception:
        return False, "E2200", None  # grab fail

    # --- resize ---
    if frame_scale is not None and float(frame_scale) != 1.0:
        try:
            frame = cv2.resize(frame, None, fx=float(frame_scale), fy=float(frame_scale), interpolation=cv2.INTER_AREA)
        except Exception:
            return False, "E2201", None  # resize fail

    if debug and debug_buffer is not None:
        debug_buffer.append({"stage": "final_check_manual", "z": 0.0, "idx": 0, "frame": frame.copy()})

    gray_gate = _to_gray_u8(frame)
    if gray_gate is None:
        return False, "E2206", None

    # --- exposure gate ---
    exp_code, exp_m = exposure_gate(
        gray_gate,
        white_thr=250,
        frac_white_thr=0.20,
        under_p95_thr=30,
        under_dr_thr=30,
    )
    print(f"[EXPOSURE_GATE] exp={exp_code} p95={exp_m['p95']:.1f} dr={exp_m['dr']:.1f} white={exp_m['white']:.3f}")

    if exp_code == "E_OVER":
        return False, "E2003", None
    if exp_code == "E_UNDER":
        return False, "E2002", None

    # --- uniformity ---
    std_gray = float(gray_gate.std())
    print(f"[UNIFORM_CHECK] std_gray={std_gray:.2f} uniform_mode={std_gray < float(uniform_std_thr)}")

    # -----------------------------------------------------------------
    # UNIFORM MODE => PATTERN/EMPTY gate
    # -----------------------------------------------------------------
    if std_gray < float(uniform_std_thr):
        gate = pattern_empty_gate_from_frame(
            frame,
            uniform_std_thr=float(uniform_std_thr),
            pattern_scale=0.10,
            pattern_pre_blur_ksize=11,
            cf_stride=4,
            cf_thr=5.5,
        )

        if gate.get("run", False):
            pat_code = str(gate.get("pattern_code") or "")
            is_empty = bool(gate.get("is_empty"))
            info = gate.get("info", {}) or {}

            print(f"[PATTERN_GATE] pattern_code={pat_code} is_empty={is_empty}")

            if is_empty:
                return False, "E2000", None

            # uniform + pattern => full frame "contour"
            return True, None, np.array(
                [[[0, 0]],
                 [[gray_gate.shape[1] - 1, 0]],
                 [[gray_gate.shape[1] - 1, gray_gate.shape[0] - 1]],
                 [[0, gray_gate.shape[0] - 1]]],
                dtype=np.int32
            )

        # ha gate nem futott (elvileg itt fut), de legyen stabil fallback
        return True, None, np.array(
            [[[0, 0]],
             [[gray_gate.shape[1] - 1, 0]],
             [[gray_gate.shape[1] - 1, gray_gate.shape[0] - 1]],
             [[0, gray_gate.shape[0] - 1]]],
            dtype=np.int32
        )

    # -----------------------------------------------------------------
    # NON-UNIFORM MODE => OTSU contour + (opcionális) rounded check
    # -----------------------------------------------------------------
    gray = gray_gate
    H, W = gray.shape[:2]

    c = largest_contour_from_gray_otsu(gray)
    if c is None:
        return False, "E2207", None

    area = float(cv2.contourArea(c))
    if area < (float(min_area_ratio) * H * W):
        return True, None, c

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
            return False, str(rounded_error_code), c

    return True, None, c


# ---------------------------------------------------------------------
# Wrapper (manual_return) - ugyanúgy, mint a scriptedben
# ---------------------------------------------------------------------
def manual_return(
    frame_scale=0.1,
    grab_timeout_ms=2000,
    min_area_ratio=0.001,
    debug=True,
    # pattern gate
    uniform_std_thr=10.0,
    # rounded
    do_rounded_check=True,
    rounded_margin_px=15,
    rounded_step=12,
    rounded_angle_threshold_deg=55.0,
    rounded_max_sharp=20,
    rounded_min_used=20,
    rounded_downsample=4,
    rounded_error_code="E2015",
):
    debug_buffer = [] if debug else None

    ok, err_code, final_contour = final_out_of_frame_check_manual(
        frame_scale=frame_scale,
        grab_timeout_ms=grab_timeout_ms,
        debug=debug,
        debug_buffer=debug_buffer,
        min_area_ratio=min_area_ratio,
        uniform_std_thr=uniform_std_thr,
        do_rounded_check=bool(do_rounded_check),
        rounded_margin_px=rounded_margin_px,
        rounded_step=rounded_step,
        rounded_angle_threshold_deg=rounded_angle_threshold_deg,
        rounded_max_sharp=rounded_max_sharp,
        rounded_min_used=rounded_min_used,
        rounded_downsample=rounded_downsample,
        rounded_error_code=str(rounded_error_code),
    )

    if not ok:
        maybe_dump_debug(debug, debug_buffer, str(err_code))
        return _err(str(err_code))

    final_contour_pts = contour_to_points_list(final_contour)
    return _ok(
        z_rel=0.0,
        score=0.0,
        final_contour=final_contour_pts,
    )