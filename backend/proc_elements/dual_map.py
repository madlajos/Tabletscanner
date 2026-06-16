"""dual_map.py

Processes paired grayscale + RGB images loaded from data["images"] in a
single pipeline step.

Typical pipeline
----------------
    load_image  (folder containing both gray and RGB images)
    → dual_map

data["images"] is expected to contain both grayscale and colour images
(e.g. every even index = gray, every odd index = RGB, or alphabetically
sorted pairs).  The node classifies images automatically (default) or via
explicit index parameters, then runs:

  1. Gray soft/hard membership mapping — identical algorithm to gray_map.py
  2. RGB nearest-neighbour assignment with IQR-based soft membership

Cross-masking (optional)
------------------------
After the first-pass assignment, the hard_index_map of one side can be
applied as an additional ROI mask for the other side:

  cross_mask = "gray_masks_rgb"  →  run gray first, mask RGB with component N
  cross_mask = "rgb_masks_gray"  →  run RGB first, mask gray with component N

Output keys (side outputs)
--------------------------
Gray side (identical to gray_map.py):
    soft_membership, soft_membership_jet, soft_membership_raw
    component_map, component_map_jet
    hard_index_map, hard_composite_rgb, hard_jet

RGB side (same structure, rgb_ prefix):
    rgb_soft_membership, rgb_soft_membership_jet
    rgb_component_map, rgb_component_map_jet
    rgb_hard_index_map, rgb_hard_composite_rgb, rgb_hard_jet

Source images for preview:
    gray_source_images   — list of the original gray arrays
    rgb_source_images    — list of the original RGB (BGR) arrays
"""

import numpy as np


# ── colour palette (BGR, OpenCV convention) ──────────────────────────────────

_PALETTE_BGR = np.array([
    (0,   0,   255),   # comp 1 → red
    (0,   255,   0),   # comp 2 → green
    (255,   0,   0),   # comp 3 → blue
    (0,   255, 255),   # comp 4 → yellow
    (255,   0, 255),   # comp 5 → magenta
    (255, 255,   0),   # comp 6 → cyan
], dtype=np.uint8)


# ── low-level helpers ─────────────────────────────────────────────────────────

def _get_masks(data):
    res = data.get("results") or {}
    for k in ("range_masks", "masks"):
        if k in res:
            return res[k]
    meta = data.get("meta") or {}
    if "active_masks" in meta:
        return meta["active_masks"]
    imgs = data.get("images")
    if imgs is None:
        return None
    return [None if i is None else np.ones(i.shape[:2], dtype=np.uint8) * 255 for i in imgs]


def _scale_mask(mask, scale):
    roi = np.asarray(mask, dtype=bool)
    s = float(scale)
    if s <= 0.0 or s >= 0.999:
        return roi
    ys, xs = np.where(roi)
    if ys.size == 0:
        return roi
    cy, cx = (ys.min() + ys.max()) / 2.0, (xs.min() + xs.max()) / 2.0
    hh = max(1.0, (ys.max() - ys.min() + 1) * s / 2.0)
    hw = max(1.0, (xs.max() - xs.min() + 1) * s / 2.0)
    yy = np.arange(roi.shape[0])[:, None]
    xx = np.arange(roi.shape[1])[None, :]
    return roi & (np.abs(yy - cy) <= hh) & (np.abs(xx - cx) <= hw)


def _jet_bgr(v):
    x = np.clip(v, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    return np.stack([b, g, r], axis=-1)   # BGR


def _u8(img):
    return (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)


def _jet_maps(values, roi_mask):
    """(H,W,k) float → list of k BGR uint8 JET images."""
    out = []
    values = np.clip(values, 0.0, 1.0)
    for c in range(values.shape[2]):
        j = _jet_bgr(values[..., c])
        j[~roi_mask] = 0
        out.append(_u8(j))
    return out


def _colorize(index_map, k):
    """1-indexed (H,W) uint8 → BGR (H,W,3) uint8."""
    h, w = index_map.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(k):
        out[index_map == c + 1] = _PALETTE_BGR[c % len(_PALETTE_BGR)]
    return out


def _to_gray01(img):
    arr = np.asarray(img, dtype=np.float64)
    if arr.ndim == 3:
        # coefficient order matches gray_map.py (BGR channel 0 = Blue)
        arr = 0.2989 * arr[..., 0] + 0.5870 * arr[..., 1] + 0.1140 * arr[..., 2]
    if arr.size > 0 and np.nanmax(arr) > 1.0:
        arr /= 255.0
    return np.clip(arr, 0.0, 1.0)


def _to_bgr01(img):
    arr = np.asarray(img, dtype=np.float64)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.concatenate([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[2] > 3:
        arr = arr[..., :3]
    if arr.size > 0 and np.nanmax(arr) > 1.0:
        arr /= 255.0
    return np.clip(arr, 0.0, 1.0)


def _is_grayscale(img, threshold=5.0):
    if img is None:
        return False
    arr = np.asarray(img)
    if arr.ndim == 2 or (arr.ndim == 3 and arr.shape[2] == 1):
        return True
    if arr.ndim == 3 and arr.shape[2] >= 3:
        b = arr[..., 0].astype(np.float32)
        g = arr[..., 1].astype(np.float32)
        r = arr[..., 2].astype(np.float32)
        # Per-pixel colour range (max − min across channels)
        per_pixel_range = (
            np.maximum(np.maximum(b, g), r) - np.minimum(np.minimum(b, g), r)
        ).ravel()
        p95 = float(np.percentile(per_pixel_range, 95))
        # Scale threshold to image range: threshold is specified in 0-255 units,
        # but float images in [0,1] would always pass as "grayscale" without scaling.
        eff_threshold = (
            threshold / 255.0
            if arr.dtype.kind == 'f' and float(np.nanmax(arr)) <= 1.0
            else threshold
        )
        return p95 < eff_threshold
    return False


# ── per-image processing ──────────────────────────────────────────────────────

def _run_gray(gray01, roi_mask, centers, sort_order, k,
              eps_iqr, hard_weights, hard_threshold, c1_threshold, c2_threshold):
    """Run gray_map algorithm on one image. Returns result dict."""
    H, W = gray01.shape

    Ivec = gray01[roi_mask]
    D = np.abs(Ivec[:, None] - centers[None, :])          # (N_roi, k)
    idx_roi = (np.argmin(D, axis=1) + 1).astype(np.uint8)

    cluster_map = np.zeros((H, W), dtype=np.uint8)
    cluster_map[roi_mask] = idx_roi

    # Remap cluster IDs to sorted component IDs
    comp_map = np.zeros((H, W), dtype=np.uint8)
    for comp in range(k):
        comp_map[cluster_map == sort_order[comp] + 1] = comp + 1

    # Adjacent-cluster soft membership
    # Each pixel blends only between its own sorted cluster and the immediately
    # adjacent neighbour cluster (the one it is closer to across the boundary).
    # Membership is computed by linear distance interpolation between the two
    # nearest cluster centroids — all other clusters receive zero membership.
    sorted_centers = centers[sort_order]             # (k,) ascending

    c0          = comp_map.astype(np.int32) - 1      # 0-indexed; -1 for background
    c0_clamped  = np.clip(c0, 0, k - 1)              # safe for indexing
    self_center = sorted_centers[c0_clamped]          # (H, W)

    goes_up    = gray01 >= self_center                # True → look at higher-index neighbour
    neighbor_c = np.where(goes_up, c0_clamped + 1, c0_clamped - 1)
    no_neighbor = (goes_up & (c0_clamped == k - 1)) | (~goes_up & (c0_clamped == 0))
    neighbor_c  = np.clip(neighbor_c, 0, k - 1)

    neighbor_center = sorted_centers[neighbor_c]      # (H, W)

    d_self     = np.abs(gray01 - self_center)
    d_neighbor = np.abs(gray01 - neighbor_center)
    total      = d_self + d_neighbor

    mem_own = np.where(total > 1e-15, d_neighbor / total, 1.0)
    mem_nbr = np.where(total > 1e-15, d_self     / total, 0.0)
    mem_own = np.where(no_neighbor, 1.0, mem_own)
    mem_nbr = np.where(no_neighbor, 0.0, mem_nbr)

    membership = np.zeros((H, W, k), dtype=np.float64)
    for comp in range(k):
        own_mask = roi_mask & (c0_clamped == comp)
        membership[..., comp][own_mask] = mem_own[own_mask]
        nbr_mask = roi_mask & (neighbor_c == comp) & (c0_clamped != comp)
        membership[..., comp][nbr_mask] = mem_nbr[nbr_mask]

    # Membership already sums to 1 per ROI pixel by construction; normalise
    # defensively to handle any floating-point edge cases.
    s = membership.sum(axis=2)
    mem_norm = np.zeros_like(membership)
    valid = roi_mask & (s > 0)
    for comp in range(k):
        mem_norm[..., comp][valid] = (
            membership[..., comp][valid] / (s[valid] + np.finfo(float).eps)
        )

    # Hard composite
    wts = np.asarray(hard_weights, dtype=np.float64)
    if wts.size < k:
        wts = np.pad(wts, (0, k - wts.size), constant_values=1.0)
    wts = wts[:k]

    C = [(mem_norm[..., 0] > c1_threshold).astype(np.float64)]
    if k > 1:
        C.append((mem_norm[..., 1] > c2_threshold).astype(np.float64))
    for c in range(2, k):
        C.append(mem_norm[..., c].astype(np.float64))

    Cn = []
    for c in range(k):
        cn = np.zeros((H, W), dtype=np.float64)
        v = C[c][roi_mask]
        if v.size > 0:
            cn[roi_mask] = (v - v.min()) / (v.max() - v.min() + np.finfo(float).eps)
        Cn.append(cn * wts[c])

    stack = np.stack(Cn, axis=2)
    max_val = stack.max(axis=2)
    hard_idx = (np.argmax(stack, axis=2) + 1).astype(np.uint8)
    hard_idx[max_val < hard_threshold] = 0
    hard_idx[~roi_mask] = 0

    hard_rgb = _colorize(hard_idx, k)
    hard_rgb[~roi_mask] = 0

    soft_disp = np.clip(mem_norm, 0.0, 1.0)
    comp_bin = (comp_map[..., None] == (np.arange(k) + 1)[None, None, :]).astype(np.float64)
    hard_bin = (hard_idx[..., None] == (np.arange(k) + 1)[None, None, :]).astype(np.float64)

    return {
        "soft_membership":     mem_norm.astype(np.float32),
        "soft_membership_jet": _jet_maps(soft_disp, roi_mask),
        "component_map":       _colorize(comp_map, k),
        "component_map_jet":   _jet_maps(comp_bin, roi_mask),
        "hard_index_map":      hard_idx,
        "hard_composite_rgb":  hard_rgb.astype(np.uint8),
        "hard_jet":            _jet_maps(hard_bin, roi_mask),
    }


def _parse_rgb_references(raw_refs):
    """Parse flat B1,G1,R1,B2,G2,R2,... references to (k, 3) array in [0,1]."""
    arr = np.asarray(raw_refs, dtype=np.float64).ravel()
    if arr.size % 3 != 0:
        return None
    refs = arr.reshape(-1, 3)
    if refs.size > 0 and np.nanmax(refs) > 1.0:
        refs /= 255.0
    return np.clip(refs, 0.0, 1.0)


def _sort_rgb_references(refs):
    """Sort BGR reference rows by luminance for stable component order."""
    if refs is None or len(refs) == 0:
        return refs
    lum = 0.1140 * refs[:, 0] + 0.5870 * refs[:, 1] + 0.2989 * refs[:, 2]
    return refs[np.argsort(lum)]


def _scale_rgb_references_to_roi(refs, bgr01_roi, percentiles=(10, 90)):
    """Scale each BGR channel of refs linearly to match the ROI's channel range.

    Mirrors gray_map._scale_centroids_to_roi but applied per-channel independently.
    refs:      (k, 3) float array in [0, 1]
    bgr01_roi: (N_roi, 3) float array of ROI pixels in [0, 1]
    Returns:   scaled refs (k, 3) clipped to [0, 1]
    """
    refs = np.array(refs, dtype=np.float64)
    scaled = refs.copy()
    for ch in range(refs.shape[1]):
        col = refs[:, ch]
        c_min, c_max = float(col.min()), float(col.max())
        if c_max <= c_min:
            continue
        img_low, img_high = np.percentile(bgr01_roi[:, ch], percentiles)
        if img_high <= img_low:
            continue
        scaled[:, ch] = img_low + (col - c_min) * (
            (img_high - img_low) / (c_max - c_min + np.finfo(float).eps)
        )
    return np.clip(scaled, 0.0, 1.0)


def _run_rgb(bgr01, roi_mask, refs, k, eps_iqr):
    """Run RGB nearest-neighbour + IQR soft membership (full BGR)."""
    H, W = bgr01.shape[:2]
    bgr_vec = bgr01[roi_mask]                                          # (N_roi, 3)

    # Scale references to the ROI's actual per-channel value range
    refs = _scale_rgb_references_to_roi(refs, bgr_vec)
    dist = np.sqrt(np.sum(
        (bgr_vec[:, None, :] - refs[None, :, :]) ** 2, axis=-1
    ))                                                                # (N_roi, k)
    hard_roi = (np.argmin(dist, axis=1) + 1).astype(np.uint8)

    hard_idx = np.zeros((H, W), dtype=np.uint8)
    hard_idx[roi_mask] = hard_roi

    feat_flat = bgr01.reshape(-1, 3)
    membership = np.zeros((H, W, k), dtype=np.float64)
    for comp in range(k):
        d_full = np.sqrt(np.sum(
            (feat_flat - refs[comp]) ** 2, axis=-1
        )).reshape(H, W)
        vals = d_full[hard_idx == comp + 1]
        if vals.size == 0:
            continue
        med = np.median(vals)
        spread = max(float(np.percentile(vals, 99) - np.percentile(vals, 1)), float(eps_iqr))
        z = (d_full-med) / spread
        mu = np.exp(-0.5 * z ** 2)
        mu[~roi_mask] = 0.0
        membership[..., comp] = mu

    s = membership.sum(axis=2)
    mem_norm = np.zeros_like(membership)
    valid = roi_mask & (s > 0)
    for comp in range(k):
        mem_norm[..., comp][valid] = (
            membership[..., comp][valid] / (s[valid] + np.finfo(float).eps)
        )

    soft_disp = np.clip(mem_norm, 0.0, 1.0)
    comp_bgr = _colorize(hard_idx, k)
    comp_bgr[~roi_mask] = 0
    comp_bin = (hard_idx[..., None] == (np.arange(k) + 1)[None, None, :]).astype(np.float64)

    return {
        "rgb_soft_membership":     mem_norm.astype(np.float32),
        "rgb_soft_membership_jet": _jet_maps(soft_disp, roi_mask),
        "rgb_component_map":       comp_bgr.astype(np.uint8),
        "rgb_component_map_jet":   _jet_maps(comp_bin, roi_mask),
        "rgb_hard_index_map":      hard_idx,
        "rgb_hard_composite_rgb":  comp_bgr.copy(),
        "rgb_hard_jet":            _jet_maps(comp_bin, roi_mask),
    }


# ── main node ─────────────────────────────────────────────────────────────────

def dual_map_node(
    data,
    # Gray mapping parameters
    initial_centroids=(140, 155, 0),   # gray centroids 0-255
    hard_weights=(2, 1, 1),
    hard_threshold=0.10,
    c1_threshold=0.29,
    c2_threshold=0.55,
    # RGB mapping parameters
    rgb_references=None,               # flat B1,G1,R1,B2,G2,R2,... 0-255
    # Sub-classification parameters (used together with cross_mask)
    sub_initial_centroids=None,        # if set + cross_mask="rgb_masks_gray": run gray sub-pass within RGB comp N
    sub_rgb_references=None,           # if set + cross_mask="gray_masks_rgb": run RGB sub-pass within gray comp N
    # Common parameters
    sort_centroids=True,
    auto_detect_channels=True,
    color_channel_threshold=5.0,       # mean abs diff threshold for gray detection
    gray_image_indices=None,           # tuple of ints, used when auto_detect=False
    rgb_image_indices=None,            # tuple of ints, used when auto_detect=False
    roiScale=1.0,
    # Cross-masking
    cross_mask="none",                 # "none" | "gray_masks_rgb" | "rgb_masks_gray"
    cross_mask_component=1,            # 1-indexed component used as mask
    # Misc
    eps_iqr=1e-12,
    debug=False,
    **_legacy_kwargs,
):
    if data.get("error") is not None:
        return data

    images = data.get("images")
    if images is None:
        data["error"] = "E3001"
        return data

    results = data.setdefault("results", {})
    history = data.setdefault("history", [])

    # ── parse gray centroids ──────────────────────────────────────────────────
    centers = np.asarray(initial_centroids, dtype=np.float64).ravel()
    if centers.size > 0 and np.nanmax(centers) > 1.0:
        centers /= 255.0
    centers = np.clip(centers, 0.0, 1.0)
    k_gray = len(centers)

    sort_order_gray = np.argsort(centers) if sort_centroids else np.arange(k_gray)
    # NOTE: centers stay UNSORTED; sort_order is used for remapping (matches gray_map.py)

    # ── parse RGB references ──────────────────────────────────────────────────
    bgr_refs = None
    k_rgb = 0
    if rgb_references is not None:
        bgr_refs = _parse_rgb_references(rgb_references)
        if bgr_refs is None:
            data["error"] = "E2602"
            return data
        k_rgb = len(bgr_refs)
        if sort_centroids:
            bgr_refs = _sort_rgb_references(bgr_refs)

    # ── parse sub-classification gray centroids ───────────────────────────────
    sub_centers = None
    sub_sort_order = None
    k_sub_gray = 0
    if sub_initial_centroids is not None:
        arr = np.asarray(sub_initial_centroids, dtype=np.float64).ravel()
        if arr.size > 0 and np.nanmax(arr) > 1.0:
            arr /= 255.0
        sub_centers = np.clip(arr, 0.0, 1.0)
        k_sub_gray = len(sub_centers)
        sub_sort_order = np.argsort(sub_centers) if sort_centroids else np.arange(k_sub_gray)

    # ── parse sub-classification RGB references ───────────────────────────────
    sub_bgr_refs = None
    k_sub_rgb = 0
    if sub_rgb_references is not None:
        sub_bgr_refs = _parse_rgb_references(sub_rgb_references)
        if sub_bgr_refs is None:
            data["error"] = "E2602"
            return data
        k_sub_rgb = len(sub_bgr_refs)
        if sort_centroids:
            sub_bgr_refs = _sort_rgb_references(sub_bgr_refs)

    # ── split images into gray / RGB lists ────────────────────────────────────
    if auto_detect_channels:
        gray_pairs, rgb_pairs = [], []
        for i, img in enumerate(images):
            if img is None:
                continue
            if _is_grayscale(img, color_channel_threshold):
                gray_pairs.append((i, img))
            else:
                rgb_pairs.append((i, img))
    else:
        gi = tuple(gray_image_indices) if gray_image_indices is not None else (0,)
        ri = tuple(rgb_image_indices) if rgb_image_indices is not None else (1,)
        gray_pairs = [(i, images[i]) for i in gi if i < len(images) and images[i] is not None]
        rgb_pairs = [(i, images[i]) for i in ri if i < len(images) and images[i] is not None]

    n_pairs = max(len(gray_pairs), len(rgb_pairs))

    if n_pairs == 0:
        data["error"] = "E3001"
        return data

    # ── masks ─────────────────────────────────────────────────────────────────
    all_masks = _get_masks(data)

    def _roi_for(orig_index, shape):
        if all_masks is None or orig_index is None or orig_index >= len(all_masks):
            return np.ones(shape, dtype=bool)
        m = all_masks[orig_index]
        if m is None:
            return np.ones(shape, dtype=bool)
        m_arr = np.asarray(m)
        if m_arr.ndim == 3:
            m_arr = m_arr[..., 0]
        return _scale_mask(m_arr, roiScale)

    # ── accumulators ──────────────────────────────────────────────────────────
    gray_keys = [
        "soft_membership", "soft_membership_jet", "soft_membership_raw",
        "component_map", "component_map_jet",
        "hard_index_map", "hard_composite_rgb", "hard_jet",
    ]
    rgb_keys = [
        "rgb_soft_membership", "rgb_soft_membership_jet",
        "rgb_component_map", "rgb_component_map_jet",
        "rgb_hard_index_map", "rgb_hard_composite_rgb", "rgb_hard_jet",
    ]
    src_keys = ["gray_source_images", "rgb_source_images"]
    sub_gray_keys = [
        "sub_soft_membership_jet", "sub_component_map_jet",
        "sub_hard_jet", "sub_hard_composite_rgb",
    ]
    sub_rgb_keys = [
        "sub_rgb_soft_membership_jet", "sub_rgb_component_map_jet",
        "sub_rgb_hard_jet", "sub_rgb_hard_composite_rgb",
    ]

    accum = {k: [] for k in gray_keys + rgb_keys + src_keys + sub_gray_keys + sub_rgb_keys}

    def _append_none_all():
        for k in gray_keys + rgb_keys + src_keys + sub_gray_keys + sub_rgb_keys:
            accum[k].append(None)

    for pair_i in range(n_pairs):
        has_gray = pair_i < len(gray_pairs)
        has_rgb  = pair_i < len(rgb_pairs)

        gray_orig_i, gray_img = gray_pairs[pair_i] if has_gray else (None, None)
        rgb_orig_i,  rgb_img  = rgb_pairs[pair_i]  if has_rgb  else (None, None)

        # Determine working size
        if has_gray and gray_img is not None:
            H, W = gray_img.shape[:2]
        elif has_rgb and rgb_img is not None:
            H, W = rgb_img.shape[:2]
        else:
            _append_none_all()
            continue

        roi_gray = _roi_for(gray_orig_i, (H, W))
        roi_rgb  = _roi_for(rgb_orig_i,  (H, W))

        # Store source images (for preview)
        accum["gray_source_images"].append(
            _to_gray01(gray_img)[..., None].repeat(3, axis=2)  # (H,W,3) for uniform encoding
            if has_gray and gray_img is not None else None
        )
        accum["rgb_source_images"].append(
            _to_bgr01(rgb_img) if has_rgb and rgb_img is not None else None
        )

        gray01 = None
        bgr01  = None
        gray_res = None
        rgb_res  = None
        sub_gray_res = None
        sub_rgb_res  = None

        # ── run first pass then apply cross-mask ──────────────────────────────
        if cross_mask == "gray_masks_rgb" and sub_bgr_refs is None:
            # Existing: gray → mask → RGB
            if has_gray and gray_img is not None and np.any(roi_gray):
                gray01 = _to_gray01(gray_img)
                gray_res = _run_gray(
                    gray01, roi_gray, centers, sort_order_gray, k_gray,
                    eps_iqr, hard_weights, hard_threshold, c1_threshold, c2_threshold,
                )
            if gray_res is not None and has_rgb and rgb_img is not None and bgr_refs is not None:
                comp_mask = gray_res["hard_index_map"] == int(cross_mask_component)
                effective_rgb_roi = roi_rgb & comp_mask
                if not np.any(effective_rgb_roi):
                    effective_rgb_roi = roi_rgb   # fallback if mask is empty
                bgr01 = _to_bgr01(rgb_img)
                rgb_res = _run_rgb(bgr01, effective_rgb_roi, bgr_refs, k_rgb, eps_iqr)
            elif has_rgb and rgb_img is not None and bgr_refs is not None and np.any(roi_rgb):
                bgr01 = _to_bgr01(rgb_img)
                rgb_res = _run_rgb(bgr01, roi_rgb, bgr_refs, k_rgb, eps_iqr)

        elif cross_mask == "rgb_masks_gray" and sub_centers is None:
            # Existing: RGB → mask → gray
            if has_rgb and rgb_img is not None and bgr_refs is not None and np.any(roi_rgb):
                bgr01 = _to_bgr01(rgb_img)
                rgb_res = _run_rgb(bgr01, roi_rgb, bgr_refs, k_rgb, eps_iqr)
            if rgb_res is not None and has_gray and gray_img is not None and np.any(roi_gray):
                comp_mask = rgb_res["rgb_hard_index_map"] == int(cross_mask_component)
                effective_gray_roi = roi_gray & comp_mask
                if not np.any(effective_gray_roi):
                    effective_gray_roi = roi_gray
                gray01 = _to_gray01(gray_img)
                gray_res = _run_gray(
                    gray01, effective_gray_roi, centers, sort_order_gray, k_gray,
                    eps_iqr, hard_weights, hard_threshold, c1_threshold, c2_threshold,
                )
            elif has_gray and gray_img is not None and np.any(roi_gray):
                gray01 = _to_gray01(gray_img)
                gray_res = _run_gray(
                    gray01, roi_gray, centers, sort_order_gray, k_gray,
                    eps_iqr, hard_weights, hard_threshold, c1_threshold, c2_threshold,
                )

        else:  # cross_mask == "none", OR sub params active → run both fully
            if has_gray and gray_img is not None and np.any(roi_gray):
                gray01 = _to_gray01(gray_img)
                gray_res = _run_gray(
                    gray01, roi_gray, centers, sort_order_gray, k_gray,
                    eps_iqr, hard_weights, hard_threshold, c1_threshold, c2_threshold,
                )
            if has_rgb and rgb_img is not None and bgr_refs is not None and np.any(roi_rgb):
                bgr01 = _to_bgr01(rgb_img)
                rgb_res = _run_rgb(bgr01, roi_rgb, bgr_refs, k_rgb, eps_iqr)

        # ── sub-classification passes (only when sub params provided) ──────────
        if sub_centers is not None and cross_mask == "rgb_masks_gray" \
                and rgb_res is not None and has_gray and gray_img is not None and np.any(roi_gray):
            comp_mask = rgb_res["rgb_hard_index_map"] == int(cross_mask_component)
            effective_gray_roi = roi_gray & comp_mask
            if not np.any(effective_gray_roi):
                effective_gray_roi = roi_gray
            if gray01 is None:
                gray01 = _to_gray01(gray_img)
            sub_gray_res = _run_gray(
                gray01, effective_gray_roi, sub_centers, sub_sort_order, k_sub_gray,
                eps_iqr, hard_weights, hard_threshold, c1_threshold, c2_threshold,
            )

        if sub_bgr_refs is not None and cross_mask == "gray_masks_rgb" \
                and gray_res is not None and has_rgb and rgb_img is not None and np.any(roi_rgb):
            comp_mask = gray_res["hard_index_map"] == int(cross_mask_component)
            effective_rgb_roi = roi_rgb & comp_mask
            if not np.any(effective_rgb_roi):
                effective_rgb_roi = roi_rgb
            if bgr01 is None:
                bgr01 = _to_bgr01(rgb_img)
            sub_rgb_res = _run_rgb(bgr01, effective_rgb_roi, sub_bgr_refs, k_sub_rgb, eps_iqr)

        if debug:
            print(f"[dual_map] pair {pair_i}: gray_orig_i={gray_orig_i}, rgb_orig_i={rgb_orig_i}")
            if gray_res:
                vals, cnts = np.unique(gray_res["hard_index_map"], return_counts=True)
                print(f"  gray hard: {dict(zip(vals.tolist(), cnts.tolist()))}")
            if rgb_res:
                vals, cnts = np.unique(rgb_res["rgb_hard_index_map"], return_counts=True)
                print(f"  rgb hard:  {dict(zip(vals.tolist(), cnts.tolist()))}")

        # Append results
        for k_out in gray_keys:
            if gray_res:
                accum[k_out].append(
                    gray_res.get("soft_membership")
                    if k_out == "soft_membership_raw"
                    else gray_res.get(k_out)
                )
            else:
                accum[k_out].append(None)

        for k_out in rgb_keys:
            accum[k_out].append(rgb_res.get(k_out) if rgb_res else None)

        # Append sub gray results
        for out_k, in_k in [
            ("sub_soft_membership_jet", "soft_membership_jet"),
            ("sub_component_map_jet",   "component_map_jet"),
            ("sub_hard_jet",            "hard_jet"),
            ("sub_hard_composite_rgb",  "hard_composite_rgb"),
        ]:
            accum[out_k].append(sub_gray_res.get(in_k) if sub_gray_res else None)

        # Append sub RGB results
        for out_k, in_k in [
            ("sub_rgb_soft_membership_jet", "rgb_soft_membership_jet"),
            ("sub_rgb_component_map_jet",   "rgb_component_map_jet"),
            ("sub_rgb_hard_jet",            "rgb_hard_jet"),
            ("sub_rgb_hard_composite_rgb",  "rgb_hard_composite_rgb"),
        ]:
            accum[out_k].append(sub_rgb_res.get(in_k) if sub_rgb_res else None)

    # ── write to results ──────────────────────────────────────────────────────
    for k_out in gray_keys + rgb_keys + src_keys + sub_gray_keys + sub_rgb_keys:
        results[k_out] = accum[k_out]

    history.append("dual_map_node")
    return data


dual_map = dual_map_node
