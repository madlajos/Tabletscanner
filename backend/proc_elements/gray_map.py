import numpy as np


_COMPONENT_COLORS = np.array(
    [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
    ],
    dtype=np.uint8,
)

def _get_component_percentiles(comp, k):
    """
    Percentilis tartomány a komponens szórás/spread becsléséhez.

    Rendezett komponensekre értendő:
    comp 0 = sötét
    comp 1 = középső
    comp 2 = világos
    """
    if k == 3:
        if comp == 0:
            return 0, 70
        if comp == 1:
            return 25, 75
        if comp == 2:
            return 15, 100

    return 5, 95

def _get_masks_from_pipeline(data):
    results = data.get("results") or {}

    if "range_masks" in results:
        return results["range_masks"]
    if "masks" in results:
        return results["masks"]

    meta = data.get("meta") or {}
    if "active_masks" in meta:
        return meta["active_masks"]

    images = data.get("images")
    if images is None:
        return None

    return [
        None if image is None else np.ones(image.shape[:2], dtype=np.uint8) * 255
        for image in images
    ]


def _to_gray_float01(image):
    arr = np.asarray(image)

    if arr.ndim == 3:
        arr = (
            0.2989 * arr[..., 0]
            + 0.5870 * arr[..., 1]
            + 0.1140 * arr[..., 2]
        )

    arr = arr.astype(np.float64, copy=False)

    if arr.size == 0:
        return arr

    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr, dtype=np.float64)

    if np.nanmax(finite) > 1.0:
        arr = arr / 255.0

    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(arr, 0.0, 1.0)


def _scale_roi_mask(mask, roi_scale):
    roi_mask = np.asarray(mask, dtype=bool)

    if roi_scale is None:
        return roi_mask

    roi_scale = float(roi_scale)

    if roi_scale <= 0 or roi_scale >= 0.999:
        return roi_mask

    ys, xs = np.where(roi_mask)

    if ys.size == 0:
        return roi_mask

    cy = (ys.min() + ys.max()) / 2.0
    cx = (xs.min() + xs.max()) / 2.0

    half_h = max(1.0, (ys.max() - ys.min() + 1) * roi_scale / 2.0)
    half_w = max(1.0, (xs.max() - xs.min() + 1) * roi_scale / 2.0)

    yy = np.arange(roi_mask.shape[0])[:, None]
    xx = np.arange(roi_mask.shape[1])[None, :]

    scaled = (
        (np.abs(yy - cy) <= half_h)
        & (np.abs(xx - cx) <= half_w)
    )

    return roi_mask & scaled


def _scale_centroids_to_roi(centers, roi_values, percentiles=(5, 95)):
    """
    Keeps the old behavior: maps the relative spacing of the input references
    onto the selected image percentile range.

    If your references are already absolute gray values, call the node with
    scale_centroids=False.
    """
    centers = np.asarray(centers, dtype=np.float64).reshape(-1)
    roi_values = np.asarray(roi_values, dtype=np.float64).reshape(-1)

    if centers.size == 0 or roi_values.size == 0:
        return centers.copy()

    c_min = float(np.min(centers))
    c_max = float(np.max(centers))

    if c_max <= c_min:
        return np.clip(centers.copy(), 0.0, 1.0)

    img_low, img_high = np.percentile(roi_values, percentiles)

    if img_high <= img_low:
        return np.clip(centers.copy(), 0.0, 1.0)

    scaled = img_low + (centers - c_min) * (
        (img_high - img_low) / (c_max - c_min + np.finfo(float).eps)
    )

    return np.clip(scaled, 0.0, 1.0)


def _jet(values):
    x = np.clip(values, 0.0, 1.0)

    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)

    return np.stack([b, g, r], axis=-1)  # BGR for cv2.imencode


def _to_uint8(image):
    return (np.clip(image, 0, 1) * 255).astype(np.uint8)


def _make_component_jet_maps(values, roi_mask):
    maps = []
    values = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)

    for c in range(values.shape[2]):
        img = _jet(values[..., c])
        img[~roi_mask] = 0
        maps.append(_to_uint8(img))

    return maps


def _colorize_hard_rgb(index_map):
    h, w = index_map.shape
    bgr = np.zeros((h, w, 3), dtype=np.uint8)

    max_comp = int(np.max(index_map)) if index_map.size else 0

    for comp in range(1, max_comp + 1):
        color_rgb = _COMPONENT_COLORS[(comp - 1) % len(_COMPONENT_COLORS)]
        color_bgr = color_rgb[::-1]
        bgr[index_map == comp] = color_bgr

    return bgr


def _prepare_centroids(initial_centroids, k):
    centers = np.asarray(initial_centroids, dtype=np.float64).reshape(-1)

    if centers.size == 0:
        return None

    finite = centers[np.isfinite(centers)]
    if finite.size == 0:
        return None

    if np.nanmax(finite) > 1.0:
        centers = centers / 255.0

    centers = np.nan_to_num(centers, nan=0.0, posinf=1.0, neginf=0.0)
    centers = np.clip(centers, 0.0, 1.0)

    if k is None:
        k = int(centers.size)
    else:
        k = int(k)

    if centers.size != k:
        return None

    return centers


def _prepare_component_thresholds(k, component_thresholds, c1_threshold, c2_threshold):
    if component_thresholds is None:
        thresholds = np.zeros(k, dtype=np.float64)
        if k >= 1:
            thresholds[0] = float(c1_threshold)
        if k >= 2:
            thresholds[1] = float(c2_threshold)
    else:
        thresholds = np.asarray(component_thresholds, dtype=np.float64).reshape(-1)
        if thresholds.size < k:
            thresholds = np.pad(
                thresholds,
                (0, k - thresholds.size),
                constant_values=0.0,
            )
        thresholds = thresholds[:k]

    thresholds = np.nan_to_num(thresholds, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(thresholds, 0.0, 1.0)


def _prepare_weights(k, hard_weights):
    weights = np.asarray(hard_weights, dtype=np.float64).reshape(-1)

    if weights.size < k:
        weights = np.pad(weights, (0, k - weights.size), constant_values=1.0)

    weights = weights[:k]
    weights = np.nan_to_num(weights, nan=1.0, posinf=1.0, neginf=0.0)
    return np.maximum(weights, 0.0)


def _sigma_to_float01(sigma):
    sigma_norm = float(sigma)

    if sigma_norm > 1.0:
        sigma_norm = sigma_norm / 255.0

    if not np.isfinite(sigma_norm):
        sigma_norm = 18.0 / 255.0

    return max(sigma_norm, np.finfo(float).eps)


def _append_empty_outputs(
    soft_membership_all,
    soft_membership_jet_all,
    component_map_all,
    component_map_jet_all,
    hard_index_map_all,
    hard_composite_rgb_all,
    hard_jet_all,
    scaled_centroids_all,
):
    soft_membership_all.append(None)
    soft_membership_jet_all.append(None)
    component_map_all.append(None)
    component_map_jet_all.append(None)
    hard_index_map_all.append(None)
    hard_composite_rgb_all.append(None)
    hard_jet_all.append(None)
    scaled_centroids_all.append(None)


def fixed_centroid_soft_hard_node(
    data,
    k=None,
    initial_centroids=(0, 128, 255),
    roiScale=1.0,
    eps_iqr=1e-12,
    hard_weights=(2, 1, 1),
    hard_threshold=0.10,
    c1_threshold=0.29,
    c2_threshold=0.55,
    component_thresholds=None,
    sigma=18,
    sort_centroids=True,
    scale_centroids=False,
    image_percentiles=(0, 100),
    debug=False,
    **_legacy_kwargs,
):
    """
    Fixed centroid soft/hard membership node.
    
    Algorithm:
    1. Initial k-means assignment: each pixel to nearest centroid
    2. Component re-estimation: median + IQR for each component
    3. Soft membership: exp(-0.5 * ((I - median) / IQR)^2)
    4. Hard assignment: thresholded + weighted winner-take-all
    """
    if data.get("error") is not None:
        return data

    images = data.get("images")

    if images is None:
        data["error"] = "E3001"
        return data

    results = data.setdefault("results", {})
    history = data.setdefault("history", [])

    centers_fix = _prepare_centroids(initial_centroids, k)

    if centers_fix is None:
        data["error"] = "E2602"
        return data

    k = int(centers_fix.size)
    component_thresholds = _prepare_component_thresholds(
        k,
        component_thresholds,
        c1_threshold,
        c2_threshold,
    )
    weights = _prepare_weights(k, hard_weights)

    masks = _get_masks_from_pipeline(data)

    soft_membership_all = []
    soft_membership_jet_all = []
    component_map_all = []
    component_map_jet_all = []
    hard_index_map_all = []
    hard_composite_rgb_all = []
    hard_jet_all = []
    scaled_centroids_all = []

    for image_index, image in enumerate(images):
        if image is None:
            _append_empty_outputs(
                soft_membership_all,
                soft_membership_jet_all,
                component_map_all,
                component_map_jet_all,
                hard_index_map_all,
                hard_composite_rgb_all,
                hard_jet_all,
                scaled_centroids_all,
            )
            continue

        I = _to_gray_float01(image)

        if I.ndim != 2:
            I = np.squeeze(I)

        if I.ndim != 2:
            data["error"] = "E2605"
            return data

        H, W = I.shape

        mask = None
        if masks is not None and image_index < len(masks):
            mask = masks[image_index]

        if mask is None:
            roi_mask = np.ones((H, W), dtype=bool)
        else:
            if np.asarray(mask).ndim == 3:
                mask = np.asarray(mask)[..., 0]

            roi_mask = _scale_roi_mask(mask, roiScale)

            if roi_mask.shape != (H, W):
                data["error"] = "E2604"
                return data

        if not np.any(roi_mask):
            _append_empty_outputs(
                soft_membership_all,
                soft_membership_jet_all,
                component_map_all,
                component_map_jet_all,
                hard_index_map_all,
                hard_composite_rgb_all,
                hard_jet_all,
                scaled_centroids_all,
            )
            continue

        Ivec_roi = I[roi_mask]

        if scale_centroids:
            centers_work = _scale_centroids_to_roi(
                centers_fix,
                Ivec_roi,
                percentiles=image_percentiles,
            )
        else:
            centers_work = centers_fix.copy()

        if sort_centroids:
            order = np.argsort(centers_work)
        else:
            order = np.arange(k)

        # === STEP 1: Initial k-means assignment ===
        D = np.abs(Ivec_roi[:, None] - centers_work[None, :])
        idx_roi = np.argmin(D, axis=1) + 1

        cluster_map = np.zeros((H, W), dtype=np.uint8)
        cluster_map[roi_mask] = idx_roi.astype(np.uint8)

        component_map_index = np.zeros((H, W), dtype=np.uint8)
        for comp in range(k):
            original_cluster_id = order[comp] + 1
            component_map_index[cluster_map == original_cluster_id] = comp + 1

        # === STEP 2: Median + component-specific percentile spread ===
        membership = np.zeros((H, W, k), dtype=np.float64)

        for comp in range(k):
            comp_mask = component_map_index == (comp + 1)
            vals = I[comp_mask]

            if vals.size == 0:
                continue

            med = np.median(vals)

            p_low, p_high = _get_component_percentiles(comp, k)
            q_low, q_high = np.percentile(vals, [p_low, p_high])

            spread = q_high - q_low
            denom = max(float(spread), float(eps_iqr))

            z = (I - med) / denom
            mu = np.exp(-0.5 * z ** 2)
            mu[~roi_mask] = 0.0

            membership[..., comp] = mu

        # === Soft membership normalization ===
        sum_mu = np.sum(membership, axis=2)
        membership_norm = np.zeros_like(membership)

        valid = roi_mask & (sum_mu > 0)

        for comp in range(k):
            tmp = membership[..., comp]
            membership_norm[..., comp][valid] = (
                tmp[valid] / (sum_mu[valid] + np.finfo(float).eps)
            )

        # === Hard assignment logic ===
        C = []
        for comp in range(k):
            th = component_thresholds[comp]

            if th > 0:
                C.append((membership_norm[..., comp] > th).astype(np.float64))
            else:
                C.append(membership_norm[..., comp].astype(np.float64))

        Cn = []
        for comp in range(k):
            Cn_comp = np.zeros((H, W), dtype=np.float64)
            v = C[comp][roi_mask]

            if v.size > 0:
                v_min = float(np.min(v))
                v_max = float(np.max(v))
                Cn_comp[roi_mask] = (
                    (v - v_min)
                    / (v_max - v_min + np.finfo(float).eps)
                )

            Cn.append(Cn_comp * weights[comp])

        stack = np.stack(Cn, axis=2)

        max_val = np.max(stack, axis=2)
        hard_index_map = np.argmax(stack, axis=2).astype(np.uint8) + 1

        hard_index_map[max_val < hard_threshold] = 0
        hard_index_map[~roi_mask] = 0

        hard_rgb = _colorize_hard_rgb(hard_index_map)
        hard_rgb[~roi_mask] = 0

        soft_display = np.clip(membership_norm, 0.0, 1.0)

        soft_membership_all.append(membership_norm.astype(np.float32))
        soft_membership_jet_all.append(
            _make_component_jet_maps(soft_display, roi_mask)
        )

        component_binary = (
            component_map_index[..., None]
            == (np.arange(k) + 1)[None, None, :]
        ).astype(np.float64)

        component_map_all.append(_colorize_hard_rgb(component_map_index))
        component_map_jet_all.append(
            _make_component_jet_maps(component_binary, roi_mask)
        )

        hard_index_map_all.append(hard_index_map.astype(np.uint8))
        hard_composite_rgb_all.append(hard_rgb.astype(np.uint8))

        hard_binary = (
            hard_index_map[..., None]
            == (np.arange(k) + 1)[None, None, :]
        ).astype(np.float64)

        hard_jet_all.append(_make_component_jet_maps(hard_binary, roi_mask))
        scaled_centroids_all.append(centers_work[order].astype(np.float32))

        if debug:
            print(f"\nImage {image_index}")
            print("centers input 255:", np.round(centers_fix * 255).astype(int))
            print("centers used 255:", np.round(centers_work * 255).astype(int))
            print("centers order 255:", np.round(centers_work[order] * 255).astype(int))
            print("image percentiles 255:", np.percentile(Ivec_roi, image_percentiles) * 255)
            print("order:", order + 1)

            soft_sum = np.sum(membership_norm[roi_mask], axis=1)
            print("soft sum min/max:", float(soft_sum.min()), float(soft_sum.max()))

            vals, counts = np.unique(hard_index_map[roi_mask], return_counts=True)
            print("hard counts:", {int(v): int(c) for v, c in zip(vals, counts)})

            for comp in range(k):
                v = membership_norm[..., comp][roi_mask]
                print(
                    f"comp{comp + 1}: "
                    f"mean={float(v.mean()):.3f}, "
                    f"min={float(v.min()):.3f}, "
                    f"max={float(v.max()):.3f}"
                )

    results["soft_membership_raw"] = soft_membership_all
    results["soft_membership"] = soft_membership_all
    results["soft_membership_jet"] = soft_membership_jet_all

    results["component_map"] = component_map_all
    results["component_map_jet"] = component_map_jet_all

    results["hard_index_map"] = hard_index_map_all
    results["hard_composite_rgb"] = hard_composite_rgb_all
    results["hard_jet"] = hard_jet_all
    results["scaled_centroids"] = scaled_centroids_all

    history.append("fixed_centroid_soft_hard_node_matlab_logic_scaled_centroids")

    return data


gray_reference_mapping_node = fixed_centroid_soft_hard_node
gray_map = fixed_centroid_soft_hard_node
