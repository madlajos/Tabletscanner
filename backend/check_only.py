

from autofocus_back import (
    sobel_topk_score,
    lap_sq_from_bbox_gray,
    rounded_by_curvature_ignore_border,
    edge_ring_strength_from_roi_gray,
    largest_contour_from_gray_otsu,
)
import cv2
import numpy as np
import globals


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


# ------------------------------------------------------------
# Otsu mask helpers (largest contour)
# ------------------------------------------------------------
def _mask_from_largest_otsu(gray_u8: np.ndarray):
    """
    Otsu + largest contour -> filled mask (uint8 0/255), and area_ratio in [0..1]
    Returns (mask_u8_or_None, area_ratio)
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


def _select_pixels_u8(gray_u8: np.ndarray, mask_u8: np.ndarray | None) -> np.ndarray:
    if mask_u8 is None:
        return gray_u8.reshape(-1)
    m = (mask_u8 > 0)
    if not np.any(m):
        return gray_u8.reshape(-1)
    return gray_u8[m]


def _select_pixels_f32(gray_f32: np.ndarray, mask_u8: np.ndarray | None) -> np.ndarray:
    if mask_u8 is None:
        return gray_f32.reshape(-1)
    m = (mask_u8 > 0)
    if not np.any(m):
        return gray_f32.reshape(-1)
    return gray_f32[m]


# ------------------------------------------------------------
# Exposure gate (UNDER / OVER)  -- MASZKOS
# ------------------------------------------------------------
def exposure_gate(
    gray_u8,
    white_thr=250,
    frac_white_thr=0.20,
    under_p95_thr=30,
    under_dr_thr=30,
    mask_u8=None,  # <-- ÚJ
):
    if gray_u8 is None or getattr(gray_u8, "size", 0) == 0:
        return "E_UNDER", {"p95": 0.0, "dr": 0.0, "white": 0.0}

    px = _select_pixels_u8(gray_u8, mask_u8).astype(np.float32)
    if px.size == 0:
        return "E_UNDER", {"p95": 0.0, "dr": 0.0, "white": 0.0}

    p1, p95, p99 = np.percentile(px, [1, 95, 99])
    dr = float(p99 - p1)

    frac_white = float((px >= float(white_thr)).mean())
    metrics = {"p95": float(p95), "dr": dr, "white": frac_white}

    if frac_white > float(frac_white_thr) or p95 > 250:
        return "E_OVER", metrics

    if (p95 < float(under_p95_thr)) and (dr < float(under_dr_thr)):
        return "E_UNDER", metrics

    return "EXP_OK", metrics


# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------
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


# ---------------------------------------------------------------------
# decide_pattern (marad, de E2000-et NEM ebből hozzuk)
# ---------------------------------------------------------------------
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


def decide_pattern(
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


# ------------------------------------------------------------
# globals.color_signature compare helpers (E2000-hez) -- MASZKOS
# ------------------------------------------------------------
def _color_signature_from_bgr(
    img_bgr: np.ndarray,
    sample_stride: int = 4,
    mask_u8: np.ndarray | None = None,
) -> dict:
    if img_bgr is None or getattr(img_bgr, "size", 0) == 0:
        return {}

    # mask downsample ugyanazzal a stride-dal, hogy kompatibilis legyen a mintavételezéssel
    m = None
    if mask_u8 is not None and getattr(mask_u8, "size", 0) != 0:
        m = (mask_u8[::sample_stride, ::sample_stride] > 0)

    if img_bgr.ndim == 2:
        gray = img_bgr.astype(np.uint8)
        gray_s = gray[::sample_stride, ::sample_stride]
        px = gray_s[m] if (m is not None and np.any(m)) else gray_s.reshape(-1)
        if px.size == 0:
            px = gray_s.reshape(-1)
        return {
            "gray_median": float(np.median(px)),
            "gray_span": float(np.max(px) - np.min(px)),
        }

    if img_bgr.ndim == 3 and img_bgr.shape[2] >= 3:
        b = img_bgr[::sample_stride, ::sample_stride, 0].astype(np.uint8)
        g = img_bgr[::sample_stride, ::sample_stride, 1].astype(np.uint8)
        r = img_bgr[::sample_stride, ::sample_stride, 2].astype(np.uint8)

        gray = cv2.cvtColor(img_bgr[..., :3], cv2.COLOR_BGR2GRAY)
        gray_s = gray[::sample_stride, ::sample_stride].astype(np.uint8)

        def pick(x):
            if m is not None and np.any(m):
                px = x[m]
                if px.size == 0:
                    return x.reshape(-1)
                return px
            return x.reshape(-1)

        pb, pg, pr, pgray = pick(b), pick(g), pick(r), pick(gray_s)

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
        dif[f"d_{k}"] = float(abs(float(cur[k]) - float(ref[k])))
    return dif


# ------------------------------------------------------------
# grayscale_difference_score
#  - OTSU MASZK: számolás + összevetés maszkon
#  - fallback: ha objektum area_ratio < 0.30 => teljes képre számol
# ------------------------------------------------------------
def grayscale_difference_score(
    img_bgr,
    blur_ksize=5,
    # exposure params
    white_thr=250,
    frac_white_thr=0.20,
    under_p95_thr=30,
    under_dr_thr=30,
    # uniformity (csak info)
    uniform_std_thr=10.0,

    # --- OTSU mask params ---
    use_otsu_mask=True,
    min_obj_area_ratio=0.10,

    # --- E2000 globals compare thresholds ---
    use_global_color_for_e2000=True,
    sig_stride=4,
    thr_med=15.0,
    thr_span=15.0,
    thr_gray_med=35.0,
    thr_gray_span=15.0,
    need_ref_signature=True,
):
    """
    Logika:
      1) (opcionális) Otsu largest-contour maszk készítés
         - ha area_ratio < min_obj_area_ratio -> mask=None (full-frame)
      2) Exposure gate -> E2002 / E2003 error (maszkon vagy full-frame)
      3) is_uniform csak INFO (nem kapuz)
      4) E2000 döntés EXP_OK esetén fut (uniformtól függetlenül!):
         globals.color_values vs aktuális signature (maszkon vagy full-frame)
         eltérés > küszöb => E2000
      5) Egyébként OK
    """
    if img_bgr is None or getattr(img_bgr, "size", 0) == 0:
        return _err("E2005")

    # --- grayscale konverzió ---
    if img_bgr.ndim == 2:
        gray_u8 = img_bgr.astype(np.uint8)
    elif img_bgr.ndim == 3 and img_bgr.shape[2] == 3:
        gray_u8 = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    elif img_bgr.ndim == 3 and img_bgr.shape[2] == 4:
        gray_u8 = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2GRAY)
    else:
        return _err("E2005")

    # --- Otsu mask (largest contour) ---
    mask_u8 = None
    area_ratio = 0.0
    if bool(use_otsu_mask):
        mask_u8, area_ratio = _mask_from_largest_otsu(gray_u8)
        if area_ratio < float(min_obj_area_ratio):
            mask_u8 = None  # fallback full-frame

    # --- blur (std/meanAbsDiff metrikákhoz) ---
    gray_blur = gray_u8
    if blur_ksize and int(blur_ksize) > 0:
        k = int(blur_ksize)
        if k % 2 == 0:
            k += 1
        if k < 3:
            k = 3
        gray_blur = cv2.GaussianBlur(gray_u8, (k, k), 0)

    g = gray_blur.astype(np.float32)
    px_f = _select_pixels_f32(g, mask_u8)
    if px_f.size == 0:
        px_f = g.reshape(-1)

    mean_gray = float(px_f.mean())
    std_gray = float(px_f.std())
    mean_abs_diff = float(np.mean(np.abs(px_f - mean_gray)))
    min_gray = float(px_f.min())
    max_gray = float(px_f.max())

    # --- exposure gate (u8-ból, maszkosan) ---
    exp_code, exp_m = exposure_gate(
        gray_u8,
        white_thr=white_thr,
        frac_white_thr=frac_white_thr,
        under_p95_thr=under_p95_thr,
        under_dr_thr=under_dr_thr,
        mask_u8=mask_u8,
    )

    is_uniform = bool(std_gray < float(uniform_std_thr))

    # default mezők
    pattern_code = None
    is_empty = None
    pat_info = {}

    # UNDER / OVER mapping
    if exp_code == "E_UNDER":
        return _err(
            "E2002",
            std_gray=std_gray,
            mean_abs_diff=mean_abs_diff,
            min_gray=min_gray,
            max_gray=max_gray,
            is_uniform=is_uniform,
            pattern_code=pattern_code,
            is_empty=is_empty,
            mask_area_ratio=float(area_ratio),
            **exp_m,
        )

    if exp_code == "E_OVER":
        return _err(
            "E2003",
            std_gray=std_gray,
            mean_abs_diff=mean_abs_diff,
            min_gray=min_gray,
            max_gray=max_gray,
            is_uniform=is_uniform,
            pattern_code=pattern_code,
            is_empty=is_empty,
            mask_area_ratio=float(area_ratio),
            **exp_m,
        )

    # E2000 compare (EXP_OK + uniformtól függetlenül)
    if exp_code == "EXP_OK" and bool(use_global_color_for_e2000):
        ref_sig = getattr(globals, "color_values", None)

        if ref_sig is None:
            if bool(need_ref_signature):
                return _err(
                    "E2007",
                    message="Missing globals.color_values (run calculate_median_and_span first)",
                    std_gray=std_gray,
                    mean_abs_diff=mean_abs_diff,
                    min_gray=min_gray,
                    max_gray=max_gray,
                    is_uniform=is_uniform,
                    mask_area_ratio=float(area_ratio),
                    **exp_m,
                )
            pattern_code = "GLOBAL_COLOR_SKIPPED"
            is_empty = None
            pat_info = {}
        else:
            cur_sig = _color_signature_from_bgr(
                img_bgr,
                sample_stride=int(sig_stride),
                mask_u8=mask_u8,
            )
            dif = _signature_diff(cur_sig, ref_sig)

            too_much = (
               ( cur_sig.get("gray_median", 0.0)<ref_sig.get("gray_median", 0.0) *0.5) 
            )

            pattern_code = "GLOBAL_COLOR_COMPARE"
            is_empty = bool(too_much)  # itt: "eltér-e"
            pat_info = {"ref_sig": ref_sig, "cur_sig": cur_sig, **dif, "too_much": bool(too_much)}

            if is_empty:
                return _err(
                    "E2000",
                    std_gray=std_gray,
                    mean_abs_diff=mean_abs_diff,
                    min_gray=min_gray,
                    max_gray=max_gray,
                    is_uniform=is_uniform,
                    pattern_code=pattern_code,
                    is_empty=True,
                    mask_area_ratio=float(area_ratio),
                    **exp_m,
                    **pat_info,
                )

    print(
        f"[GATE] exp={exp_code} p95={exp_m['p95']:.1f} dr={exp_m['dr']:.1f} white={exp_m['white']:.3f} "
        f"std={std_gray:.2f} uniform={is_uniform} madMean={mean_abs_diff:.2f} "
        f"min={min_gray:.0f} max={max_gray:.0f} "
        f"maskArea={area_ratio:.3f} "
        f"pat={pattern_code} empty={is_empty}"
    )

    return _ok(
        std_gray=std_gray,
        mean_abs_diff=mean_abs_diff,
        min_gray=min_gray,
        max_gray=max_gray,
        is_uniform=is_uniform,
        pattern_code=pattern_code,
        is_empty=is_empty,
        mask_area_ratio=float(area_ratio),
        **exp_m,
        **pat_info,
    )


def final_out_of_frame_check(
    frame,
    frame_scale=0.1,
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
    do_frame_touch_check=True,
    return_contour=False,
):
    if frame is None or getattr(frame, "size", 0) == 0:
        return _err("E2110")

    if frame_scale is not None and float(frame_scale) != 1.0:
        frame = cv2.resize(
            frame, None,
            fx=float(frame_scale), fy=float(frame_scale),
            interpolation=cv2.INTER_AREA
        )

    gray = _to_gray_u8(frame)
    if gray is None:
        return _err("E2113")

    H, W = gray.shape[:2]

    c = largest_contour_from_gray_otsu(gray)
    if c is None:
        return _err("E2114")

    area = float(cv2.contourArea(c))
    if area < (float(min_area_ratio) * H * W):
        if return_contour:
            return _ok(area=area, note="small_area_ok", contour=c)
        return _ok(area=area, note="small_area_ok")

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

        decision = info.get("decision") if isinstance(info, dict) else None

        if (ok_round is None) and (decision == "NOT_ENOUGH_INTERIOR") and allow_not_enough_interior:
            ok_round = True

        if (ok_round is None) or (ok_round is False):
            extra = {
                "touch_count": touch_count,
                "L": left, "T": top, "R": right, "B": bottom,
                "rounded_info": info,
            }
            if return_contour:
                extra["contour"] = c
            return _err(str(rounded_error_code), **extra)

    if not bool(do_frame_touch_check):
        extra = {"touch_count": touch_count, "L": left, "T": top, "R": right, "B": bottom}
        if return_contour:
            extra["contour"] = c
        return _ok(**extra)

    extra = {"touch_count": touch_count, "L": left, "T": top, "R": right, "B": bottom}
    if return_contour:
        extra["contour"] = c

    if touch_count == 0:
        return _ok(**extra)
    if touch_count == 1:
        return _err("E2004", **extra)
    if touch_count == 2:
        return _err("E2004", **extra)
    if touch_count == 3:
        return _err("E2004", **extra)
    if touch_count == 4:
        return _ok(**extra)

    return _ok(**extra)