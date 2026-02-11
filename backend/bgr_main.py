
import os
import cv2
import numpy as np
from PIL import Image


# ============================
# EXIF-safe save
# ============================
def save_bgr_image_keep_exif(image_bgr, src_image_path, dst_image_path, quality=95, subsampling=0):
    os.makedirs(os.path.dirname(dst_image_path), exist_ok=True)

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    out_img = Image.fromarray(rgb)

    exif_bytes = None
    try:
        with Image.open(src_image_path) as src:
            exif_bytes = src.info.get("exif", None)
    except Exception:
        pass

    save_kwargs = dict(format="JPEG", quality=quality, subsampling=subsampling)
    if exif_bytes is not None:
        save_kwargs["exif"] = exif_bytes

    out_img.save(dst_image_path, **save_kwargs)


# ============================
# Mask pipeline (your "good" one) -> returns full-res mask
# ============================
def make_object_mask_from_bgr(
    image_bgr,
    scale=0.3,
    # free-shape params (SMALL mask)
    open_model_k=41,
    close_model_k=21,
    guard_k=19,
    small_area_max=12000,
    # upscale AA
    up_sigma=1.2,
    up_thr=127,
    # final contour smoothing
    do_contour_smooth=True,
    eps_frac=0.002,
    # shape thresholds
    circ_thr=0.88,
    fill_thr=0.88,
    radial_thr=0.04,
    ellipse_aspect_thr=1.35,
):
    """
    Returns:
      mask_up_255: (H,W) uint8 {0,255}
      kind: "circle" | "ellipse" | "free"
      metrics: tuple or None
    """
    I = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    H, W = I.shape

    # ---------------- (speed) caches ----------------
    _kernels = {}
    def _K(k):
        k = int(k)
        if k <= 1:
            k = 1
        if k not in _kernels:
            _kernels[k] = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        return _kernels[k]

    # (speed) reuse CLAHE once per call
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # ---------------- utils ----------------
    def keep_largest(mask_255):
        # (speed) contour fill is often faster than connectedComponentsWithStats on large masks
        m = (mask_255 > 0).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return m
        cnt = max(cnts, key=cv2.contourArea)
        out = np.zeros_like(m)
        cv2.drawContours(out, [cnt], -1, 255, -1)
        return out

    def fill_holes(mask_255):
        ff = mask_255.copy()
        h, w = mask_255.shape
        tmp = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(ff, tmp, (0, 0), 255)
        return cv2.bitwise_or(mask_255, cv2.bitwise_not(ff))

    def refine_mask(mask_255, open_k=7, close_k=21, blur_sigma=0.0):
        m = (mask_255 > 0).astype(np.uint8) * 255

        # (speed) do morph first, then one keep_largest+fill_holes (floodFill is expensive)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  _K(open_k),  iterations=1)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, _K(close_k), iterations=1)

        m = keep_largest(m)
        m = fill_holes(m)

        if blur_sigma and blur_sigma > 0:
            sm = cv2.GaussianBlur(m, (0, 0), float(blur_sigma))
            m = (sm > 127).astype(np.uint8) * 255
            # (speed) no second floodFill; keep largest is enough
            m = keep_largest(m)
        return m

    # ---------------- seed ----------------
    def build_seed(gray_small):
        g = clahe.apply(gray_small)
        g = cv2.GaussianBlur(g, (5, 5), 0)

        _, bw1 = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bw2 = cv2.bitwise_not(bw1)

        def post(bw):
            bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, _K(21), iterations=2)
            bw = keep_largest(bw)
            bw = fill_holes(bw)
            return bw

        m1 = post(bw1)
        m2 = post(bw2)

        def score(m):
            frac = np.count_nonzero(m) / (m.size + 1e-9)
            if frac < 0.02 or frac > 0.98:
                return -1e9
            return frac

        best = m1 if score(m1) >= score(m2) else m2
        return refine_mask(best, open_k=3, close_k=21, blur_sigma=0.0)

    # ---------------- grabcut (free only) ----------------
    def grabcut_refine(img_small_bgr, seed_mask_255, iters=3):
        h, w = seed_mask_255.shape

        sure_fg = cv2.erode(seed_mask_255, _K(15), 1)
        prob_fg = cv2.dilate(seed_mask_255, _K(31), 1)

        gc_mask = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
        gc_mask[prob_fg > 0] = cv2.GC_PR_FGD
        gc_mask[sure_fg > 0] = cv2.GC_FGD

        border = int(0.03 * min(h, w))
        gc_mask[:border, :] = cv2.GC_BGD
        gc_mask[-border:, :] = cv2.GC_BGD
        gc_mask[:, :border] = cv2.GC_BGD
        gc_mask[:, -border:] = cv2.GC_BGD

        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)

        # (speed) usually 1 iter is enough
        iters = int(iters)
        if iters > 1:
            iters = 1

        cv2.grabCut(img_small_bgr, gc_mask, None, bgdModel, fgdModel, iters, cv2.GC_INIT_WITH_MASK)

        out = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        return refine_mask(out, open_k=5, close_k=21, blur_sigma=0.0)

    # ---------------- shape classification ----------------
    def _largest_contour(mask_255):
        m = (mask_255 > 0).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        return max(cnts, key=cv2.contourArea)

    def shape_metrics(cnt):
        A = cv2.contourArea(cnt)
        P = cv2.arcLength(cnt, True)
        if P <= 1e-6 or A <= 1e-6:
            return None

        circularity = 4 * np.pi * A / (P * P)

        (cx, cy), R = cv2.minEnclosingCircle(cnt)
        circle_fill = A / (np.pi * (R * R + 1e-9))

        pts = cnt.reshape(-1, 2).astype(np.float32)
        d = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)
        radial_std = float(np.std(d) / (np.mean(d) + 1e-9))

        aspect = None
        if len(cnt) >= 5:
            (ex, ey), (MA, ma), ang = cv2.fitEllipse(cnt)
            major = max(MA, ma)
            minor = min(MA, ma)
            aspect = float(major / (minor + 1e-9))

        return circularity, circle_fill, radial_std, aspect, (cx, cy, R), cnt

    def classify_shape_from_mask(mask_255):
        cnt = _largest_contour(mask_255)
        if cnt is None:
            return "free", None, None

        met = shape_metrics(cnt)
        if met is None:
            return "free", None, cnt

        circularity, circle_fill, radial_std, aspect, circle, _cnt = met

        is_circle = (circularity >= circ_thr) and (circle_fill >= fill_thr) and (radial_std <= radial_thr)

        is_ellipse = False
        if aspect is not None:
            is_ellipse = (circularity >= 0.75) and (aspect <= ellipse_aspect_thr) and (aspect > 1.05)

        if is_circle:
            return "circle", met, cnt
        if is_ellipse:
            return "ellipse", met, cnt
        return "free", met, cnt

    # ---------------- free-shape ear removal ----------------
    def model_guard_cut(mask_255):
        m = (mask_255 > 0).astype(np.uint8) * 255
        m = keep_largest(m)
        m = fill_holes(m)

        k_open  = _K(open_model_k)
        k_close = _K(close_model_k)
        k_guard = _K(guard_k)

        model = cv2.morphologyEx(m, cv2.MORPH_OPEN,  k_open,  iterations=1)
        model = cv2.morphologyEx(model, cv2.MORPH_CLOSE, k_close, iterations=1)
        model = keep_largest(model)
        model = fill_holes(model)

        guard = cv2.dilate(model, k_guard, iterations=1)

        extra = cv2.subtract(m, model)
        extra_outside = cv2.bitwise_and(extra, cv2.bitwise_not(guard))

        # (speed) if nothing to cut, skip CC
        if np.count_nonzero(extra_outside) == 0:
            return m

        num, labels, stats, _ = cv2.connectedComponentsWithStats(extra_outside, 8)
        cut_mask = np.zeros_like(extra_outside)
        for i in range(1, num):
            area = stats[i, cv2.CC_STAT_AREA]
            if area <= small_area_max:
                cut_mask[labels == i] = 255

        out = cv2.bitwise_and(m, cv2.bitwise_not(cut_mask))
        out = keep_largest(out)
        out = fill_holes(out)
        return out

    # ---------------- upscale + contour smoothing ----------------
    def upscale_mask_antialias(mask_small_255):
        mf = cv2.resize(mask_small_255.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
        mf = cv2.GaussianBlur(mf, (0, 0), float(up_sigma))
        mu = (mf > float(up_thr)).astype(np.uint8) * 255
        mu = keep_largest(mu)
        mu = fill_holes(mu)
        mu = cv2.morphologyEx(mu, cv2.MORPH_CLOSE, _K(5), iterations=1)
        return mu

    def smooth_contour_mask(mask_255):
        m = (mask_255 > 0).astype(np.uint8) * 255
        cnt = _largest_contour(m)
        if cnt is None:
            return m
        eps = float(eps_frac) * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, eps, True)
        out = np.zeros_like(m)
        cv2.drawContours(out, [approx], -1, 255, -1)
        out = keep_largest(out)
        out = fill_holes(out)
        return out

    # ================= run =================
    # (speed) allow smaller scale for faster run, but keep user default if passed
    newW, newH = int(W * scale), int(H * scale)
    newW = max(32, newW)
    newH = max(32, newH)

    I_s = cv2.resize(I, (newW, newH), interpolation=cv2.INTER_AREA)
    img_s = cv2.cvtColor(I_s, cv2.COLOR_GRAY2BGR)

    seed = build_seed(I_s)
    kind, met, cnt = classify_shape_from_mask(seed)

    # build final mask on SMALL
    if kind == "circle" and met is not None:
        circularity, circle_fill, radial_std, aspect, (cx, cy, R), _ = met
        final_s = np.zeros_like(seed)
        cv2.circle(final_s, (int(round(cx)), int(round(cy))), int(round(R + 3)), 255, -1)

    elif kind == "ellipse" and met is not None:
        circularity, circle_fill, radial_std, aspect, circle, cnt = met
        final_s = np.zeros_like(seed)
        if cnt is not None and len(cnt) >= 5:
            (ex, ey), (MA, ma), ang = cv2.fitEllipse(cnt)
            axes = (int(round(MA / 2 + 3)), int(round(ma / 2 + 3)))
            center = (int(round(ex)), int(round(ey)))
            cv2.ellipse(final_s, center, axes, ang, 0, 360, 255, -1)
        else:
            final_s = seed.copy()

    else:
        frac = np.count_nonzero(seed) / (seed.size + 1e-9)

        # (speed) GrabCut only when seed looks "messy" (many components)
        def _should_use_grabcut(seed_255):
            if not (0.02 < frac < 0.98):
                return False
            num, _ = cv2.connectedComponents((seed_255 > 0).astype(np.uint8), 8)
            return num > 3

        if _should_use_grabcut(seed):
            base = grabcut_refine(img_s, seed, iters=1)  # faster
        else:
            base = seed.copy()

        base = refine_mask(base, open_k=5, close_k=21, blur_sigma=0.0)
        final_s = model_guard_cut(base)
        final_s = refine_mask(final_s, open_k=5, close_k=21, blur_sigma=0.0)

    # upscale + smoothing
    mask_up = upscale_mask_antialias(final_s)
    if do_contour_smooth:
        mask_up = smooth_contour_mask(mask_up)

    return mask_up, kind, met


def apply_mask_zero_background(image_bgr, mask_255):
    out = image_bgr.copy()
    out[mask_255 == 0] = (0, 0, 0)
    return out


# ============================
# Folder processing
# ============================
def process_folder(input_dir):
    """
    Legfrissebb JPG/JPEG feldolgozása:
      - maszk pipeline
      - kinullázás maszk alapján
      - mentés input_dir/masked-be EXIF-fel
      - opcionális: maszk mentése is

    return: 1 siker, 0 nincs kép / hiba
    """
    output_dir = os.path.join(input_dir, "masked")
    os.makedirs(output_dir, exist_ok=True)

    valid_exts = {".jpg", ".jpeg", ".JPG", ".JPEG"}
    files = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if os.path.splitext(f)[1] in valid_exts and os.path.isfile(os.path.join(input_dir, f))
    ]
    if not files:
        return 0

    latest_file = max(files, key=os.path.getmtime)

    img = cv2.imread(latest_file, cv2.IMREAD_COLOR)
    if img is None:
        return 0

    # (speed) you can pass scale=0.25 here if you want it faster
    mask_up, kind, met = make_object_mask_from_bgr(img, scale=1 if False else 0.3)  # keep default behavior
    if mask_up is None:
        return 0

    masked = apply_mask_zero_background(img, mask_up)

    base = os.path.splitext(os.path.basename(latest_file))[0]
    output_path = os.path.join(output_dir, f"{base}_masked.jpg")

    save_bgr_image_keep_exif(
        image_bgr=masked,
        src_image_path=latest_file,
        dst_image_path=output_path
    )

    return 1