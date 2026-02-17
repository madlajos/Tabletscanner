
import cv2
import numpy as np


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


# ------------------------------------------------------------
# Exposure gate (UNDER / OVER)
# ------------------------------------------------------------
def exposure_gate(
    gray_u8,
    white_thr=250,
    frac_white_thr=0.20,
    under_p95_thr=30,
    under_dr_thr=30,
):
    if gray_u8 is None or gray_u8.size == 0:
        return "E_UNDER", {"p95": 0.0, "dr": 0.0, "white": 0.0}

    g = gray_u8.astype(np.float32)
    p1, p95, p99 = np.percentile(g, [1, 95, 99])
    dr = float(p99 - p1)

    frac_white = float((gray_u8 >= int(white_thr)).mean())
    metrics = {"p95": float(p95), "dr": dr, "white": frac_white}

    if frac_white > float(frac_white_thr) or p95 > 250:
        return "E_OVER", metrics

    if (p95 < float(under_p95_thr)) and (dr < float(under_dr_thr)):
        return "E_UNDER", metrics

    return "EXP_OK", metrics


# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------
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
    mask[rw:H - rw, rw:W - rw] = 0

    vals = mag[mask == 255]
    if vals.size == 0:
        return 0.0

    return float(np.median(vals))


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


# ------------------------------------------------------------
# EMPTY vs PATTERN logic
# ------------------------------------------------------------
def preprocess_gray(gray_u8, scale=0.5, blur_ksize=3):
    g = gray_u8
    if scale is not None and float(scale) < 1.0:
        g = cv2.resize(
            g, None, fx=float(scale), fy=float(scale), interpolation=cv2.INTER_AREA
        )

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


def robust_texture_and_edge_activity(
    gray_u8,
    hp_sigma=12.0,
    hp_clip_lo=5,
    hp_clip_hi=95,
    act_k=6.0,
):
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


def decide_pattern(
    gray_u8,
    exp_code,
    p10_zoom_thr=60.0,
    dabs_sus_thr=10.0,
    span_thr=25.0,
    hp_thr=2.5,
    sobel_p90_thr=10.0,
    edge_act_empty_thr=0.002,
    edge_act_tex_thr=0.010,
    v_corr_abs_thr=0.55,
    v_mono_thr=0.60,
):
    p10, p90, span = percentile_span(gray_u8, 10, 90)
    tm = robust_texture_and_edge_activity(
        gray_u8, hp_sigma=12.0, hp_clip_lo=5, hp_clip_hi=95, act_k=6.0
    )
    vm = vignette_profile_metrics(gray_u8, blur_ksize=11, bins=24)

    info = {"p10": p10, "p90": p90, "span": span, **tm, **vm}

    if exp_code == "E_OVER":
        return "PATTERN_UNKNOWN_OVER", info
    if exp_code == "E_UNDER":
        return "PATTERN_UNKNOWN_UNDER", info

    if p10 >= float(p10_zoom_thr):
        return "PATTERN_PRESENT_ZOOM", info

    if vm["dAbs"] > float(dabs_sus_thr):
        return "PATTERN_PRESENT", info

    radial_ok = (abs(vm["corr"]) >= float(v_corr_abs_thr)) and (
        vm["mono"] >= float(v_mono_thr)
    )
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


# ------------------------------------------------------------
# grayscale_difference_score
#  - exposure + uniformity flag
#  - pattern/empty ONLY IF (uniform AND exp OK)
#  - NEW: if pattern says EMPTY -> return ERROR E2000
# ------------------------------------------------------------
def grayscale_difference_score(
    img_bgr,
    blur_ksize=5,
    # exposure params
    white_thr=250,
    frac_white_thr=0.20,
    under_p95_thr=30,
    under_dr_thr=30,
    # uniformity
    uniform_std_thr=10.0,
    # pattern preprocess (csak uniform ágba)
    pattern_scale=0.10,
    pattern_pre_blur_ksize=11,
    # decide_pattern thresholds
    p10_zoom_thr=60.0,
    dabs_sus_thr=10.0,
    span_thr=25.0,
    hp_thr=2.5,
    sobel_p90_thr=10.0,
    edge_act_empty_thr=0.002,
    edge_act_tex_thr=0.010,
    v_corr_abs_thr=0.55,
    v_mono_thr=0.60,
):
    """
    Logika:
      1) Exposure gate -> E2002 / E2003 error
      2) Uniform flag: std_gray < uniform_std_thr -> is_uniform=True (NEM error)
      3) Pattern/empty döntés CSAK akkor fut, ha is_uniform=True és exp OK
      4) HA pattern E_EMPTY -> ERROR E2000
      5) Egyébként OK visszaad metrikákat + (ha futott) pattern mezőket
    """
    if img_bgr is None or getattr(img_bgr, "size", 0) == 0:
        return _err("E2005")

    # --- grayscale konverzió ---
    if img_bgr.ndim == 2:
        gray_u8 = img_bgr
    elif img_bgr.ndim == 3 and img_bgr.shape[2] == 3:
        gray_u8 = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    elif img_bgr.ndim == 3 and img_bgr.shape[2] == 4:
        gray_u8 = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2GRAY)
    else:
        return _err("E2005")

    # --- blur (csak a std/meanAbsDiff metrikákhoz) ---
    gray_blur = gray_u8
    if blur_ksize and int(blur_ksize) > 0:
        k = int(blur_ksize)
        if k % 2 == 0:
            k += 1
        if k < 3:
            k = 3
        gray_blur = cv2.GaussianBlur(gray_u8, (k, k), 0)

    g = gray_blur.astype(np.float32)

    mean_gray = float(g.mean())
    std_gray = float(g.std())
    mean_abs_diff = float(np.mean(np.abs(g - mean_gray)))
    min_gray = float(g.min())
    max_gray = float(g.max())

    # --- exposure gate (u8-ból) ---
    exp_code, exp_m = exposure_gate(
        gray_u8,
        white_thr=white_thr,
        frac_white_thr=frac_white_thr,
        under_p95_thr=under_p95_thr,
        under_dr_thr=under_dr_thr,
    )

    is_uniform = bool(std_gray < float(uniform_std_thr))

    # default pattern mezők (ha nem fut)
    pattern_code = None
    is_empty = None
    pat_info = {}

    # CSAK ha uniform és exp OK, akkor fut a pattern/empty döntés
    if is_uniform and exp_code == "EXP_OK":
        try:
            gray_pat = preprocess_gray(
                gray_u8, scale=float(pattern_scale), blur_ksize=int(pattern_pre_blur_ksize)
            )
            pattern_code, pat_info = decide_pattern(
                gray_pat,
                exp_code="EXP_OK",
                p10_zoom_thr=p10_zoom_thr,
                dabs_sus_thr=dabs_sus_thr,
                span_thr=span_thr,
                hp_thr=hp_thr,
                sobel_p90_thr=sobel_p90_thr,
                edge_act_empty_thr=edge_act_empty_thr,
                edge_act_tex_thr=edge_act_tex_thr,
                v_corr_abs_thr=v_corr_abs_thr,
                v_mono_thr=v_mono_thr,
            )
            is_empty = bool(pattern_code == "E_EMPTY")
        except Exception:
            pattern_code = "PATTERN_UNKNOWN"
            is_empty = None
            pat_info = {}

    print(
        f"[GATE] exp={exp_code} p95={exp_m['p95']:.1f} dr={exp_m['dr']:.1f} white={exp_m['white']:.3f} "
        f"std={std_gray:.2f} uniform={is_uniform} madMean={mean_abs_diff:.2f} "
        f"min={min_gray:.0f} max={max_gray:.0f} "
        f"pat={pattern_code} empty={is_empty}"
    )

    # UNDER / OVER -> marad a régi kód mapping
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
            **exp_m,
            **pat_info,
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
            **exp_m,
            **pat_info,
        )

    # NEW: EMPTY -> E2000 (csak ha volt pattern döntés)
    if exp_code == "EXP_OK" and pattern_code == "E_EMPTY":
        return _err(
            "E2000",
            std_gray=std_gray,
            mean_abs_diff=mean_abs_diff,
            min_gray=min_gray,
            max_gray=max_gray,
            is_uniform=is_uniform,
            pattern_code=pattern_code,
            is_empty=True,
            **exp_m,
            **pat_info,
        )

    # OK (uniform sem hiba)
    return _ok(
        std_gray=std_gray,
        mean_abs_diff=mean_abs_diff,
        min_gray=min_gray,
        max_gray=max_gray,
        is_uniform=is_uniform,
        pattern_code=pattern_code,
        is_empty=is_empty,
        **exp_m,
        **pat_info,
    )


def final_out_of_frame_check(
    frame,
    debug=False,
    debug_buffer=None,
    margin_px=2,
    min_area_ratio=0.001,
    bbox_state=None,
    edge_ring_width=5,
    min_edge_strength=None,
    return_contour=False,
):
    # -----------------------------
    # Edge strength check (opcionális)
    # -----------------------------
    if (min_edge_strength is not None) and (bbox_state is not None):
        roi_g = bbox_roi_gray(frame, bbox_state)
        edge_strength = edge_ring_strength_from_roi_gray(
            roi_g, ring_w=int(edge_ring_width)
        )
        print(
            f"[FINAL_EDGE] edge_strength={edge_strength:.4f} "
            f"(min={float(min_edge_strength):.4f})"
        )

        if edge_strength < float(min_edge_strength):
            return _err("E2112", edge_strength=float(edge_strength))

    # -----------------------------
    # Teljes képes OTSU szegmentálás
    # -----------------------------
    if frame is None or getattr(frame, "size", 0) == 0:
        return _err("E2110")

    if frame.ndim == 2:
        gray = frame
    elif frame.ndim == 3 and frame.shape[2] == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif frame.ndim == 3 and frame.shape[2] == 4:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
    else:
        return _err("E2113")

    H, W = gray.shape[:2]

    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    fg = np.mean(gray[bw == 255]) if np.any(bw == 255) else 0
    bg = np.mean(gray[bw == 0]) if np.any(bw == 0) else 0
    if fg < bg:
        bw = 255 - bw

    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return _err("E2114")

    c = max(contours, key=cv2.contourArea)

    area = float(cv2.contourArea(c))
    if area < (float(min_area_ratio) * H * W):
        if return_contour:
            return _ok(area=area, note="small_area_ok", contour=c)
        return _ok(area=area, note="small_area_ok")

    x, y, w, h = cv2.boundingRect(c)

    left = (x <= margin_px)
    top = (y <= margin_px)
    right = ((x + w) >= (W - 1 - margin_px))
    bottom = ((y + h) >= (H - 1 - margin_px))

    touch_count = int(left) + int(top) + int(right) + int(bottom)

    print(f"[FRAME_TOUCH] L={left} T={top} R={right} B={bottom} count={touch_count}")

    if touch_count == 0:
        if return_contour:
            return _ok(touch_count=touch_count, bbox=(x, y, w, h), contour=c)
        return _ok(touch_count=touch_count, bbox=(x, y, w, h))

    if touch_count == 1:
        return _err("E2004", touch_count=touch_count, bbox=(x, y, w, h))

    if touch_count == 2:
        opposite_ok = (left and right) or (top and bottom)
        if opposite_ok:
            if return_contour:
                return _ok(touch_count=touch_count, bbox=(x, y, w, h), contour=c)
            return _ok(touch_count=touch_count, bbox=(x, y, w, h))
        else:
            return _err("E2004", touch_count=touch_count, bbox=(x, y, w, h))

    if touch_count == 3:
        return _err("E2004", touch_count=touch_count, bbox=(x, y, w, h))

    if touch_count == 4:
        if return_contour:
            return _ok(touch_count=touch_count, bbox=(x, y, w, h), contour=c)
        return _ok(touch_count=touch_count, bbox=(x, y, w, h))

    if return_contour:
        return _ok(touch_count=touch_count, bbox=(x, y, w, h), contour=c)
    return _ok(touch_count=touch_count, bbox=(x, y, w, h))