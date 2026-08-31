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


def _to_rgb_float01(image):
    arr = np.asarray(image)

    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif arr.ndim == 3 and arr.shape[2] >= 3:
        arr = arr[..., :3]
    else:
        return None

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


def _rgb_to_gray_float01(rgb):
    return (
        0.2989 * rgb[..., 0]
        + 0.5870 * rgb[..., 1]
        + 0.1140 * rgb[..., 2]
    )


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


def _prepare_refs_rgb(reference_colors, k=None):
    refs = np.asarray(reference_colors, dtype=np.float64)

    if refs.ndim == 1:
        refs = refs.reshape(-1, 3)

    if refs.ndim != 2 or refs.shape[1] != 3 or refs.shape[0] == 0:
        return None

    finite = refs[np.isfinite(refs)]
    if finite.size == 0:
        return None

    if np.nanmax(finite) > 1.0:
        refs = refs / 255.0

    refs = np.nan_to_num(refs, nan=0.0, posinf=1.0, neginf=0.0)
    refs = np.clip(refs, 0.0, 1.0)

    if k is None:
        k = refs.shape[0]
    else:
        k = int(k)

    if refs.shape[0] != k:
        return None

    return refs


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


def _srgb_to_linear(x, gamma=2.2):
    x = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)
    return x ** float(gamma)


def _normalize_rgb_by_intensity(rgb, eps=1e-12):
    """
    Optional shading compensation.

    Converts RGB to chromaticity-like values by dividing each pixel by R+G+B.
    This removes a lot of brightness/shading information, but also removes
    useful intensity information. Use only when illumination dominates.
    """
    s = np.sum(rgb, axis=-1, keepdims=True)
    return rgb / (s + eps)


def _project_simplex_rows(A, eps=1e-12):
    """
    Fast practical projection for CLS-like abundance maps:
    negative values are clipped and every row is normalized to sum 1.

    This is not a strict Euclidean simplex projection, but is robust and fast.
    """
    A = np.maximum(A, 0.0)
    s = np.sum(A, axis=1, keepdims=True)

    zero = s[:, 0] <= eps
    A = A / (s + eps)

    if np.any(zero):
        A[zero, :] = 1.0 / A.shape[1]

    return A


def _cls_unconstrained_weights(X, refs, sum_to_one=True, nonnegative=True):
    """
    X:    n_pixels x 3
    refs: k x 3

    Solves X ~= A @ refs, returns A as n_pixels x k.
    """
    R = refs.T  # 3 x k

    # Solves R @ A.T ~= X.T
    A, *_ = np.linalg.lstsq(R, X.T, rcond=None)
    A = A.T

    if nonnegative or sum_to_one:
        A = _project_simplex_rows(A)
    elif sum_to_one:
        s = np.sum(A, axis=1, keepdims=True)
        A = A / (s + np.finfo(float).eps)

    return A


def _cls_three_component_barycentric(X, refs):
    """
    Exact sum-to-one CLS for k=3 in RGB:
        x ~= a1*r1 + a2*r2 + a3*r3
        sum(a)=1

    Implemented as affine/barycentric least squares.
    Then clipped and normalized to nonnegative abundances.
    """
    r1 = refs[0]
    r2 = refs[1]
    r3 = refs[2]

    B = np.stack([r1 - r3, r2 - r3], axis=1)  # 3 x 2
    Y = (X - r3).T                            # 3 x n

    UV, *_ = np.linalg.lstsq(B, Y, rcond=None)
    UV = UV.T                                 # n x 2

    A = np.zeros((X.shape[0], 3), dtype=np.float64)
    A[:, 0] = UV[:, 0]
    A[:, 1] = UV[:, 1]
    A[:, 2] = 1.0 - A[:, 0] - A[:, 1]

    return _project_simplex_rows(A)


def _residual_norm(X, A, refs):
    X_hat = A @ refs
    return np.linalg.norm(X - X_hat, axis=1)


def _append_empty_outputs(
    soft_membership_all,
    soft_membership_jet_all,
    component_map_all,
    component_map_jet_all,
    hard_index_map_all,
    hard_composite_rgb_all,
    hard_jet_all,
    cls_residual_all,
    reference_colors_all,
):
    soft_membership_all.append(None)
    soft_membership_jet_all.append(None)
    component_map_all.append(None)
    component_map_jet_all.append(None)
    hard_index_map_all.append(None)
    hard_composite_rgb_all.append(None)
    hard_jet_all.append(None)
    cls_residual_all.append(None)
    reference_colors_all.append(None)


def rgb_cls_reference_mapping_node(
    data,
    k=None,
    reference_colors=((255, 0, 0), (0, 255, 0), (0, 0, 255)),
    roiScale=1.0,
    hard_weights=(1, 1, 1),
    hard_threshold=0.0,
    c1_threshold=0.0,
    c2_threshold=0.0,
    component_thresholds=None,
    use_gamma_linearization=True,
    gamma=2.2,
    normalize_rgb=False,
    method="auto",
    sort_references_by_gray=False,
    debug=False,
    **_legacy_kwargs,
):
    """
    RGB CLS-like reference mapping node.

    Input
    -----
    data["images"]:
        List of RGB or grayscale images. RGB is preferred.

    reference_colors:
        k x 3 RGB references, either 0..255 or 0..1.
        Example:
            reference_colors=((180, 40, 30), (60, 150, 70), (40, 70, 190))

    Output keys kept pipeline-friendly
    ----------------------------------
    results["soft_membership_raw"]
    results["soft_membership"]
    results["soft_membership_jet"]
    results["component_map"]
    results["component_map_jet"]
    results["hard_index_map"]
    results["hard_composite_rgb"]
    results["hard_jet"]

    Extra outputs
    -------------
    results["cls_residual"]:
        Per-pixel reconstruction error. Lower is better.

    results["reference_colors"]:
        Reference colors actually used, after optional sorting and transforms.

    Model
    -----
    For each pixel x:
        x ~= a1*r1 + a2*r2 + ... + ak*rk

    with practical constraints:
        ai >= 0
        sum(ai) = 1

    Notes
    -----
    - RGB has only 3 channels. For stable results, use k <= 3 if possible.
    - k > 3 is mathematically underdetermined; this function still returns a
      practical least-squares solution, but it may be unstable.
    - use_gamma_linearization=True is recommended for normal sRGB images.
    - normalize_rgb=True can help with uneven illumination, but it removes
      brightness information.
    """
    if data.get("error") is not None:
        return data

    images = data.get("images")

    if images is None:
        data["error"] = "E3001"
        return data

    results = data.setdefault("results", {})
    history = data.setdefault("history", [])

    refs = _prepare_refs_rgb(reference_colors, k=k)

    if refs is None:
        data["error"] = "E2702"
        return data

    k = int(refs.shape[0])

    if sort_references_by_gray:
        ref_gray = 0.2989 * refs[:, 0] + 0.5870 * refs[:, 1] + 0.1140 * refs[:, 2]
        order = np.argsort(ref_gray)
        refs = refs[order]
    else:
        order = np.arange(k)

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
    cls_residual_all = []
    reference_colors_all = []

    refs_work_global = refs.copy()

    if use_gamma_linearization:
        refs_work_global = _srgb_to_linear(refs_work_global, gamma=gamma)

    if normalize_rgb:
        refs_work_global = _normalize_rgb_by_intensity(refs_work_global[None, :, :])[0]

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
                cls_residual_all,
                reference_colors_all,
            )
            continue

        img = _to_rgb_float01(image)

        if img is None or img.ndim != 3 or img.shape[2] != 3:
            data["error"] = "E2705"
            return data

        H, W, _ = img.shape

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
                data["error"] = "E2704"
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
                cls_residual_all,
                reference_colors_all,
            )
            continue

        img_work = img.copy()

        if use_gamma_linearization:
            img_work = _srgb_to_linear(img_work, gamma=gamma)

        if normalize_rgb:
            img_work = _normalize_rgb_by_intensity(img_work)

        X = img_work[roi_mask].reshape(-1, 3)
        refs_work = refs_work_global.copy()

        # Main CLS-like abundance calculation.
        if method == "auto":
            if k == 3:
                A = _cls_three_component_barycentric(X, refs_work)
            else:
                A = _cls_unconstrained_weights(
                    X,
                    refs_work,
                    sum_to_one=True,
                    nonnegative=True,
                )
        elif method == "barycentric":
            if k != 3:
                data["error"] = "E2706"
                return data
            A = _cls_three_component_barycentric(X, refs_work)
        elif method == "lstsq":
            A = _cls_unconstrained_weights(
                X,
                refs_work,
                sum_to_one=True,
                nonnegative=True,
            )
        else:
            data["error"] = "E2707"
            return data

        membership_norm = np.zeros((H, W, k), dtype=np.float64)
        membership_norm[roi_mask, :] = A

        residual = np.zeros((H, W), dtype=np.float64)
        residual[roi_mask] = _residual_norm(X, A, refs_work)

        # Nearest reference map. This is only a diagnostic / preview map.
        D = np.linalg.norm(X[:, None, :] - refs_work[None, :, :], axis=2)
        nearest = np.argmin(D, axis=1) + 1
        component_map_index = np.zeros((H, W), dtype=np.uint8)
        component_map_index[roi_mask] = nearest.astype(np.uint8)

        # Threshold + weight hard logic kept similar to the previous nodes.
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
        cls_residual_all.append(residual.astype(np.float32))
        reference_colors_all.append(refs.astype(np.float32))

        if debug:
            print(f"\nImage {image_index}")
            print("reference colors input/order RGB 255:")
            print(np.round(refs * 255).astype(int))
            print("reference colors work:")
            print(refs_work)
            print("order:", order + 1)
            print("method:", method)
            print("use_gamma_linearization:", use_gamma_linearization, "gamma:", gamma)
            print("normalize_rgb:", normalize_rgb)

            soft_sum = np.sum(membership_norm[roi_mask], axis=1)
            print("soft sum min/max:", float(soft_sum.min()), float(soft_sum.max()))

            vals, counts = np.unique(hard_index_map[roi_mask], return_counts=True)
            print("hard counts:", {int(v): int(c) for v, c in zip(vals, counts)})

            vals_c, counts_c = np.unique(component_map_index[roi_mask], return_counts=True)
            print("nearest-reference counts:", {int(v): int(c) for v, c in zip(vals_c, counts_c)})

            rv = residual[roi_mask]
            print(
                "residual min/mean/max:",
                float(rv.min()),
                float(rv.mean()),
                float(rv.max()),
            )

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

    results["cls_residual"] = cls_residual_all
    results["reference_colors"] = reference_colors_all

    history.append("rgb_cls_reference_mapping_node")

    return data


# Pipeline-friendly aliases
rgb_reference_mapping_node = rgb_cls_reference_mapping_node
rgb_cls_map = rgb_cls_reference_mapping_node
cls_reference_mapping_node = rgb_cls_reference_mapping_node

# Optional old-style alias if you want to replace the gray node in a pipeline.
# Only use this alias when the input images are actually RGB or when grayscale
# fallback to RGB-stacked gray is acceptable.
gray_reference_mapping_node = rgb_cls_reference_mapping_node
gray_map = rgb_cls_reference_mapping_node
