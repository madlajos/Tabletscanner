"""rgb_gray_map.py

Component-mapping node that supports three modes:

  rgb_cluster_gray_measure
      Assignment via BGR reference distances (RGB-space nearest neighbour).
      Soft membership computed from the per-component gray-value distribution
      of the assigned pixels – useful when RGB separates components but the
      grayscale intensity within each component carries quantitative meaning.

  gray_cluster
      Assignment and soft membership purely from grayscale reference values,
      identical in spirit to gray_map.py but using an explicit reference list.

  rgb_gray_combined
      Assignment via a weighted sum of normalised BGR distance and normalised
      grayscale distance (rgb_weight / gray_weight).

Output keys are identical to gray_map.py so the existing pipeline encoder
and frontend inspector handle them without modification.

Images are assumed to be in BGR channel order (OpenCV convention).
Reference colours should therefore be specified as (B, G, R) triplets.

data["gray_images"] – optional list of per-image grayscale arrays parallel
                      to data["images"].  If absent the luminance of the
                      main image is used.
"""

import numpy as np


# ── component colour palette (BGR, OpenCV) ──────────────────────────────────

_COLORS_BGR = np.array(
    [
        (0,   0,   255),   # comp 1 → red
        (0,   255,   0),   # comp 2 → green
        (255,   0,   0),   # comp 3 → blue
        (0,   255, 255),   # comp 4 → yellow
        (255,   0, 255),   # comp 5 → magenta
        (255, 255,   0),   # comp 6 → cyan
    ],
    dtype=np.uint8,
)


# ── internal helpers (self-contained, mirrors gray_map.py) ──────────────────

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
        None if img is None else np.ones(img.shape[:2], dtype=np.uint8) * 255
        for img in images
    ]


def _scale_roi_mask(mask, roi_scale):
    roi_mask = np.asarray(mask, dtype=bool)
    roi_scale = float(roi_scale)
    if roi_scale <= 0.0 or roi_scale >= 0.999:
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
    return roi_mask & (np.abs(yy - cy) <= half_h) & (np.abs(xx - cx) <= half_w)


def _jet_bgr(values):
    """Scalar field → BGR JET image (same convention as gray_map.py)."""
    x = np.clip(values, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    return np.stack([b, g, r], axis=-1)   # BGR


def _to_uint8(img):
    return (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)


def _make_component_jet_maps(values, roi_mask):
    """values: (H,W,k) float [0,1] → list of k BGR uint8 images."""
    maps = []
    values = np.clip(values, 0.0, 1.0)
    for c in range(values.shape[2]):
        jet = _jet_bgr(values[..., c])
        jet[~roi_mask] = 0
        maps.append(_to_uint8(jet))
    return maps


def _colorize_bgr(index_map, k):
    """1-indexed hard assignment (H,W) uint8 → BGR (H,W,3) uint8."""
    h, w = index_map.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(k):
        out[index_map == (c + 1)] = _COLORS_BGR[c % len(_COLORS_BGR)]
    return out


def _to_gray_float01(image):
    """Any ndarray → float64 (H,W) [0,1].  Matches gray_map.py convention."""
    arr = np.asarray(image, dtype=np.float64)
    if arr.ndim == 3:
        # channel order matches the pipeline (BGR), but the coefficient
        # assignment matches gray_map.py for consistency
        arr = 0.2989 * arr[..., 0] + 0.5870 * arr[..., 1] + 0.1140 * arr[..., 2]
    if arr.size > 0 and np.nanmax(arr) > 1.0:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def _to_bgr_float01(image):
    """Any ndarray → float64 (H,W,3) [0,1]."""
    arr = np.asarray(image, dtype=np.float64)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.concatenate([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[2] > 3:
        arr = arr[..., :3]
    if arr.size > 0 and np.nanmax(arr) > 1.0:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


# ── main node ────────────────────────────────────────────────────────────────

def rgb_gray_map_node(
    data,
    mode="rgb_cluster_gray_measure",
    rgb_references=None,       # flat sequence (k*3,) or (k,3), BGR 0-255 or 0-1
    gray_references=None,      # flat sequence (k,), 0-255 or 0-1
    k=None,                    # derived from references when None
    reference_indices=None,    # e.g. (0, 1, 3) to select a subset
    rgb_weight=1.0,
    gray_weight=1.0,
    sort_centroids=True,
    roiScale=1.0,
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

    # gray_images can live at top level or inside results
    gray_images = (
        data.get("gray_images")
        or data.get("results", {}).get("gray_images")
    )

    results = data.setdefault("results", {})
    history = data.setdefault("history", [])

    # ── parse references ─────────────────────────────────────────────────────

    def _parse_ref_sequence(raw, n_cols):
        """raw → float64 ndarray (m, n_cols) in [0,1], or (None, error_code)."""
        arr = np.asarray(raw, dtype=np.float64).ravel()
        if n_cols > 1:
            if arr.size % n_cols != 0:
                return None, "E2602"
            arr = arr.reshape(-1, n_cols)
        if arr.size > 0 and np.nanmax(arr) > 1.0:
            arr = arr / 255.0
        return np.clip(arr, 0.0, 1.0), None

    bgr_refs = None
    gr_refs = None

    if rgb_references is not None:
        bgr_refs, err = _parse_ref_sequence(rgb_references, 3)
        if err:
            data["error"] = err
            return data

    if gray_references is not None:
        gr_refs, err = _parse_ref_sequence(gray_references, 1)
        if err:
            data["error"] = err
            return data
        gr_refs = gr_refs.ravel()

    # Validate mode requirements
    if mode in ("rgb_cluster_gray_measure", "rgb_gray_combined") and bgr_refs is None:
        data["error"] = "E2602"
        return data
    if mode == "gray_cluster" and gr_refs is None:
        data["error"] = "E2602"
        return data

    # Determine k
    if k is None:
        primary = bgr_refs if bgr_refs is not None else gr_refs
        if primary is None:
            data["error"] = "E2602"
            return data
        k = len(primary)

    # Subset by reference_indices
    if reference_indices is not None:
        idx = np.asarray(reference_indices, dtype=int)
        if bgr_refs is not None:
            bgr_refs = bgr_refs[idx]
        if gr_refs is not None:
            gr_refs = gr_refs[idx]
        k = len(idx)

    # Sort references
    if sort_centroids:
        if mode == "gray_cluster" and gr_refs is not None:
            order = np.argsort(gr_refs)
        elif bgr_refs is not None:
            # perceived luminance for BGR: B*0.1140 + G*0.5870 + R*0.2989
            lum = (
                0.1140 * bgr_refs[:, 0]
                + 0.5870 * bgr_refs[:, 1]
                + 0.2989 * bgr_refs[:, 2]
            )
            order = np.argsort(lum)
        elif gr_refs is not None:
            order = np.argsort(gr_refs)
        else:
            order = np.arange(k)

        if bgr_refs is not None:
            bgr_refs = bgr_refs[order]
        if gr_refs is not None:
            gr_refs = gr_refs[order]

    masks = _get_masks_from_pipeline(data)

    # ── output accumulators ──────────────────────────────────────────────────

    soft_membership_all = []
    soft_membership_jet_all = []
    component_map_all = []
    component_map_jet_all = []
    hard_index_map_all = []
    hard_composite_rgb_all = []
    hard_jet_all = []

    def _append_none():
        for lst in (
            soft_membership_all, soft_membership_jet_all,
            component_map_all, component_map_jet_all,
            hard_index_map_all, hard_composite_rgb_all, hard_jet_all,
        ):
            lst.append(None)

    for img_i, image in enumerate(images):
        if image is None:
            _append_none()
            continue

        bgr_float = _to_bgr_float01(image)
        H, W = bgr_float.shape[:2]

        # Gray image: parallel list > convert from main image
        if (
            gray_images is not None
            and img_i < len(gray_images)
            and gray_images[img_i] is not None
        ):
            gray_float = _to_gray_float01(gray_images[img_i])
        else:
            gray_float = _to_gray_float01(image)

        if gray_float.shape != (H, W):
            data["error"] = "E2604"
            return data

        # ROI mask
        raw_mask = (
            masks[img_i]
            if masks is not None and img_i < len(masks)
            else None
        )
        if raw_mask is None:
            roi_mask = np.ones((H, W), dtype=bool)
        else:
            m = np.asarray(raw_mask)
            if m.ndim == 3:
                m = m[..., 0]
            roi_mask = _scale_roi_mask(m, roiScale)
            if roi_mask.shape != (H, W):
                data["error"] = "E2604"
                return data

        if not np.any(roi_mask):
            _append_none()
            continue

        bgr_vec = bgr_float[roi_mask]    # (N_roi, 3)
        gray_vec = gray_float[roi_mask]  # (N_roi,)

        # ── hard assignment ──────────────────────────────────────────────────

        if mode == "gray_cluster":
            # Absolute distance to each gray reference
            dist = np.abs(gray_vec[:, None] - gr_refs[None, :])  # (N_roi, k)

        elif mode == "rgb_cluster_gray_measure":
            # Euclidean distance in BGR space
            dist = np.sqrt(
                np.sum(
                    (bgr_vec[:, None, :] - bgr_refs[None, :, :]) ** 2,
                    axis=-1,
                )
            )  # (N_roi, k)

        else:  # rgb_gray_combined
            rgb_d = np.sqrt(
                np.sum(
                    (bgr_vec[:, None, :] - bgr_refs[None, :, :]) ** 2,
                    axis=-1,
                )
            )
            rgb_d = rgb_d / (rgb_d.max() + 1e-12)

            if gr_refs is not None and gray_weight > 0:
                gray_d = np.abs(gray_vec[:, None] - gr_refs[None, :])
                gray_d = gray_d / (gray_d.max() + 1e-12)
            else:
                gray_d = np.zeros_like(rgb_d)

            dist = rgb_weight * rgb_d + gray_weight * gray_d

        hard_roi = (np.argmin(dist, axis=1) + 1).astype(np.uint8)
        hard_index_map = np.zeros((H, W), dtype=np.uint8)
        hard_index_map[roi_mask] = hard_roi

        # ── soft membership ──────────────────────────────────────────────────

        membership = np.zeros((H, W, k), dtype=np.float64)

        if mode == "gray_cluster":
            # Gaussian centred at the component's gray-value distribution
            for comp in range(k):
                vals = gray_float[hard_index_map == (comp + 1)]
                if vals.size == 0:
                    continue
                med = np.median(vals)
                iqr = max(
                    float(np.percentile(vals, 75) - np.percentile(vals, 25)),
                    float(eps_iqr),
                )
                z = (gray_float - med) / iqr
                mu = np.exp(-0.5 * z ** 2)
                mu[~roi_mask] = 0.0
                membership[..., comp] = mu

        elif mode == "rgb_cluster_gray_measure":
            # Soft membership: how well does each pixel's gray value match
            # the typical gray range of pixels assigned to that component?
            for comp in range(k):
                vals = gray_float[hard_index_map == (comp + 1)]
                if vals.size == 0:
                    continue
                med = np.median(vals)
                iqr = max(
                    float(np.percentile(vals, 75) - np.percentile(vals, 25)),
                    float(eps_iqr),
                )
                z = (gray_float - med) / iqr
                mu = np.exp(-0.5 * z ** 2)
                mu[~roi_mask] = 0.0
                membership[..., comp] = mu

        else:  # rgb_gray_combined
            # Gaussian on BGR distance to each reference, parameterised by the
            # IQR of distances within the component's assigned pixels
            bgr_flat = bgr_float.reshape(-1, 3)
            for comp in range(k):
                d_full = np.sqrt(
                    np.sum((bgr_flat - bgr_refs[comp]) ** 2, axis=-1)
                ).reshape(H, W)
                vals = d_full[hard_index_map == (comp + 1)]
                if vals.size == 0:
                    continue
                med = np.median(vals)
                iqr = max(
                    float(np.percentile(vals, 75) - np.percentile(vals, 25)),
                    float(eps_iqr),
                )
                z = (d_full - med) / iqr
                mu = np.exp(-0.5 * z ** 2)
                mu[~roi_mask] = 0.0
                membership[..., comp] = mu

        # Normalise memberships to sum to 1 across components
        s = membership.sum(axis=2)
        mem_norm = np.zeros_like(membership)
        valid = roi_mask & (s > 0)
        for comp in range(k):
            mem_norm[..., comp][valid] = (
                membership[..., comp][valid] / (s[valid] + np.finfo(float).eps)
            )

        # ── assemble outputs ─────────────────────────────────────────────────

        soft_disp = np.clip(mem_norm, 0.0, 1.0)
        comp_bin = (
            hard_index_map[..., None] == (np.arange(k) + 1)[None, None, :]
        ).astype(np.float64)

        comp_bgr = _colorize_bgr(hard_index_map, k)
        comp_bgr[~roi_mask] = 0

        if debug:
            vals_u, counts = np.unique(hard_index_map[roi_mask], return_counts=True)
            print(f"[rgb_gray_map] image {img_i}, mode={mode}")
            print("  hard counts:", {int(v): int(c) for v, c in zip(vals_u, counts)})
            for comp in range(k):
                v = mem_norm[..., comp][roi_mask]
                print(
                    f"  comp{comp + 1}: "
                    f"mean={float(v.mean()):.3f}  "
                    f"min={float(v.min()):.3f}  "
                    f"max={float(v.max()):.3f}"
                )

        soft_membership_all.append(mem_norm.astype(np.float32))
        soft_membership_jet_all.append(_make_component_jet_maps(soft_disp, roi_mask))
        component_map_all.append(comp_bgr.astype(np.uint8))
        component_map_jet_all.append(_make_component_jet_maps(comp_bin, roi_mask))
        hard_index_map_all.append(hard_index_map)
        hard_composite_rgb_all.append(comp_bgr.copy())
        hard_jet_all.append(_make_component_jet_maps(comp_bin, roi_mask))

    results["soft_membership"] = soft_membership_all
    results["soft_membership_jet"] = soft_membership_jet_all
    results["component_map"] = component_map_all
    results["component_map_jet"] = component_map_jet_all
    results["hard_index_map"] = hard_index_map_all
    results["hard_composite_rgb"] = hard_composite_rgb_all
    results["hard_jet"] = hard_jet_all

    history.append("rgb_gray_map_node")
    return data


rgb_gray_map = rgb_gray_map_node
