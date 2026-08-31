import cv2
import json
import numpy as np


GRAY_GAUSSIAN_SIGMA = 3.0
GRAY_FEATURE_WEIGHTS = np.array([1.0, 1.5, 1.0], dtype=np.float32)
MINIMUM_REFERENCE_SCALE = 3.0
SPATIAL_CLEANUP_KERNEL_SIZE = 3


_COLORS_BGR = np.array(
    [
        (0, 0, 255),
        (0, 255, 0),
        (255, 0, 0),
        (0, 255, 255),
        (255, 0, 255),
        (255, 255, 0),
        (128, 0, 255),
        (255, 128, 0),
        (0, 128, 255),
        (128, 255, 0),
        (255, 0, 128),
        (0, 255, 128),
    ],
    dtype=np.uint8,
)


def _get_masks_from_pipeline(data):
    results = data.get("results") or {}
    if "range_masks" in results:
        return results["range_masks"]
    if "region_masks" in results:
        return results["region_masks"]
    if "masks" in results:
        return results["masks"]
    meta = data.get("meta") or {}
    if "active_masks" in meta:
        return meta["active_masks"]
    return None


def _to_bgr_uint8(img):
    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        if arr.dtype.kind == "f" and arr.size and float(np.nanmax(arr)) <= 1.0:
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    if arr.ndim == 3 and arr.shape[2] == 1:
        return cv2.cvtColor(arr[..., 0], cv2.COLOR_GRAY2BGR)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        return arr[..., :3]
    return None


def _feature_image(img, color_space):
    bgr = _to_bgr_uint8(img)
    if bgr is None:
        return None

    if color_space == "GRAY":
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        local_mean = cv2.GaussianBlur(
            gray, (0, 0), sigmaX=GRAY_GAUSSIAN_SIGMA, sigmaY=GRAY_GAUSSIAN_SIGMA,
        )
        local_mean_sq = cv2.GaussianBlur(
            gray * gray, (0, 0),
            sigmaX=GRAY_GAUSSIAN_SIGMA, sigmaY=GRAY_GAUSSIAN_SIGMA,
        )
        local_var = np.maximum(local_mean_sq - local_mean * local_mean, 0.0)
        local_std = np.sqrt(local_var)
        return np.stack([gray, local_mean, local_std], axis=-1).astype(np.float32)
    if color_space == "HSV":
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    if color_space == "LAB":
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    return bgr


def _cluster_colors(centers, color_space, sort_by_brightness):
    centers_u8 = np.clip(centers, 0, 255).astype(np.uint8)

    if color_space == "GRAY":
        gray_vals = centers_u8[:, 0]
        bgr = np.stack([gray_vals, gray_vals, gray_vals], axis=1)
        brightness = gray_vals.astype(np.float32)
    else:
        if color_space == "HSV":
            bgr = cv2.cvtColor(centers_u8.reshape(-1, 1, 3), cv2.COLOR_HSV2BGR).reshape(-1, 3)
        elif color_space == "LAB":
            bgr = cv2.cvtColor(centers_u8.reshape(-1, 1, 3), cv2.COLOR_LAB2BGR).reshape(-1, 3)
        else:
            bgr = centers_u8.reshape(-1, 3)
        brightness = 0.114 * bgr[:, 0] + 0.587 * bgr[:, 1] + 0.299 * bgr[:, 2]

    order = np.argsort(brightness) if sort_by_brightness else np.arange(len(bgr))
    return bgr[order].astype(np.uint8), order


def _custom_cluster_colors(raw_colors):
    if isinstance(raw_colors, str):
        try:
            raw_colors = json.loads(raw_colors)
        except (TypeError, ValueError):
            return {}
    if not isinstance(raw_colors, dict):
        return {}

    colors = {}
    for raw_label, raw_color in raw_colors.items():
        try:
            label = int(raw_label)
        except (TypeError, ValueError):
            continue
        if not isinstance(raw_color, str):
            continue
        value = raw_color.strip().lstrip("#")
        if len(value) != 6:
            continue
        try:
            red, green, blue = (int(value[index:index + 2], 16) for index in (0, 2, 4))
        except ValueError:
            continue
        colors[label] = np.array((blue, green, red), dtype=np.uint8)
    return colors


def _reference_data_for_image(data, image_index, color_space, reference_source=""):
    results = data.get("results") or {}
    source_id = str(reference_source or "").strip()
    source = None
    if source_id and source_id.lower() != "auto":
        source = (results.get("reference_sources") or {}).get(source_id)
    elif isinstance(results.get("reference_sources"), dict) and results["reference_sources"]:
        source = list(results["reference_sources"].values())[-1]
    reference_rows = source.get("reference_crops") if isinstance(source, dict) else results.get("reference_crops")
    if not isinstance(reference_rows, list) or not reference_rows:
        return None, None, 0

    row_index = min(image_index, len(reference_rows) - 1)
    crops = reference_rows[row_index]
    if not isinstance(crops, list) or not crops:
        return None, source, 0

    centers = []
    scales = []
    for crop in crops:
        features = _feature_image(crop, color_space)
        if features is None or features.size == 0:
            continue
        flat = features.reshape(-1, features.shape[2]).astype(np.float32)
        if flat.size == 0:
            continue
        center = np.median(flat, axis=0)
        mad = np.median(np.abs(flat - center), axis=0)
        centers.append(center)
        scales.append(np.maximum(mad * 1.4826, MINIMUM_REFERENCE_SCALE))

    if not centers:
        return None, source, len(crops)
    return {
        "centers": np.asarray(centers, dtype=np.float32),
        "scales": np.asarray(scales, dtype=np.float32),
    }, source, len(crops)


def _feature_weights(color_space, feature_dimension):
    if color_space == "GRAY":
        return GRAY_FEATURE_WEIGHTS
    return np.ones(feature_dimension, dtype=np.float32)


def _majority_filter_labels(label_map, valid_mask, num_labels, kernel_size=3):
    """Apply one local majority pass without changing pixels outside the ROI."""
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")

    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)
    votes = np.empty((*label_map.shape, num_labels), dtype=np.float32)
    for label in range(1, num_labels + 1):
        votes[..., label - 1] = cv2.filter2D(
            (label_map == label).astype(np.float32), -1, kernel,
            borderType=cv2.BORDER_CONSTANT,
        )

    filtered = label_map.copy()
    filtered[valid_mask] = np.argmax(votes[valid_mask], axis=1).astype(np.uint8) + 1
    filtered[~valid_mask] = 0
    return filtered


def _initial_labels_from_reference_centers(sample, ref_centers):
    """Assign pixels to references while keeping every seed represented.

    OpenCV's KMEANS_USE_INITIAL_LABELS requires all cluster ids to occur in the
    initial labels.  Pure nearest-centre assignment can omit a reference when
    two crop averages are close to each other, even though there are enough
    pixels to run k-means.  Reserve one distinct nearest pixel for every
    reference before handing the labels to OpenCV.
    """
    dist = np.sum((sample[:, None, :] - ref_centers[None, :, :]) ** 2, axis=2)
    labels = np.argmin(dist, axis=1).astype(np.int32)

    claimed_samples = set()
    # Start with the most constrained references, whose nearest pixel is
    # furthest away, so broad/overlapping references cannot claim it first.
    center_order = np.argsort(np.min(dist, axis=0))[::-1]
    for center_idx in center_order:
        for sample_idx in np.argsort(dist[:, center_idx]):
            sample_idx = int(sample_idx)
            if sample_idx not in claimed_samples:
                labels[sample_idx] = int(center_idx)
                claimed_samples.add(sample_idx)
                break

    return dist, labels.reshape(-1, 1)


def kmeans_cluster(
    data,
    k=3,
    color_space="BGR",
    init_mode="auto",
    reference_source="",
    attempts=3,
    max_iter=30,
    epsilon=1.0,
    sort_by_brightness=True,
    output_mode="palette",
    background="black",
    cluster_colors=None,
    debug=False,
):
    if data.get("error") is not None:
        return data

    images = data.get("images")
    if images is None or data.get("count", 0) == 0:
        data["error"] = "E3701"
        return data

    k = int(k)
    attempts = int(attempts)
    max_iter = int(max_iter)
    epsilon = float(epsilon)

    if k < 2 or k > 32:
        data["error"] = "E3702"
        return data
    if color_space not in ("BGR", "HSV", "LAB", "GRAY"):
        data["error"] = "E3703"
        return data
    if init_mode not in ("auto", "reference_fixed", "reference_seeded"):
        data["error"] = "E3711"
        return data
    if attempts < 1 or max_iter < 1 or epsilon <= 0:
        data["error"] = "E3704"
        return data
    if output_mode not in ("palette", "centroid"):
        data["error"] = "E3705"
        return data
    if background not in ("black", "white", "original"):
        data["error"] = "E3706"
        return data

    masks = _get_masks_from_pipeline(data)
    output_images = []
    overlay_images = []
    source_images = []
    label_maps = []
    centers_all = []
    counts_all = []
    percentages_all = []
    compactness_all = []
    reference_info_all = []
    legends_all = []
    custom_colors = _custom_cluster_colors(cluster_colors)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, epsilon)

    for idx, img in enumerate(images):
        if img is None:
            data["error"] = "E3707"
            return data

        features = _feature_image(img, color_space)
        original_bgr = _to_bgr_uint8(img)
        if features is None or original_bgr is None:
            data["error"] = "E3708"
            return data

        h, w = features.shape[:2]
        raw_mask = masks[idx] if masks is not None and idx < len(masks) else None
        if raw_mask is None:
            roi_mask = np.ones((h, w), dtype=bool)
        else:
            m = np.asarray(raw_mask)
            if m.ndim == 3:
                m = m[..., 0]
            if m.shape != (h, w):
                data["error"] = "E3709"
                return data
            roi_mask = m > 0

        sample = features[roi_mask].reshape(-1, features.shape[2]).astype(np.float32)
        available_ref_data, selected_source, raw_ref_count = _reference_data_for_image(
            data, idx, color_space, reference_source,
        )
        available_ref_count = (
            int(len(available_ref_data["centers"])) if available_ref_data is not None else 0
        )
        ref_centers = None
        ref_scales = None
        effective_k = k
        if init_mode != "auto":
            ref_centers = available_ref_data["centers"] if available_ref_data is not None else None
            ref_scales = available_ref_data["scales"] if available_ref_data is not None else None
            if ref_centers is None or len(ref_centers) < 2:
                data["error"] = "E3712"
                return data
            effective_k = int(len(ref_centers))

        if sample.shape[0] < effective_k:
            data["error"] = "E3710"
            return data

        if init_mode == "reference_fixed":
            delta = sample[:, None, :] - ref_centers[None, :, :]
            normalized = delta / ref_scales[None, :, :]
            normalized *= _feature_weights(color_space, features.shape[2])[None, None, :]
            dist = np.sum(normalized * normalized, axis=2)
            labels = np.argmin(dist, axis=1).astype(np.int32).reshape(-1, 1)
            centers = ref_centers
            compactness = float(np.sum(np.min(dist, axis=1)))
        elif init_mode == "reference_seeded":
            dist, initial_labels = _initial_labels_from_reference_centers(sample, ref_centers)
            cv2.setRNGSeed(12345)
            compactness, labels, centers = cv2.kmeans(
                sample,
                effective_k,
                initial_labels,
                criteria,
                attempts,
                cv2.KMEANS_USE_INITIAL_LABELS,
            )
        else:
            cv2.setRNGSeed(12345)
            compactness, labels, centers = cv2.kmeans(
                sample,
                effective_k,
                None,
                criteria,
                attempts,
                cv2.KMEANS_PP_CENTERS,
            )

        center_colors, order = _cluster_colors(centers, color_space, sort_by_brightness)
        inverse_order = np.empty_like(order)
        inverse_order[order] = np.arange(len(order))
        labels_sorted = inverse_order[labels.reshape(-1)]

        label_map = np.zeros((h, w), dtype=np.uint8)
        label_map[roi_mask] = labels_sorted.astype(np.uint8) + 1
        if init_mode == "reference_fixed":
            label_map = _majority_filter_labels(
                label_map, roi_mask, effective_k, SPATIAL_CLEANUP_KERNEL_SIZE,
            )

        if background == "white":
            cluster_img = np.full((h, w, 3), 255, dtype=np.uint8)
        elif background == "original":
            cluster_img = original_bgr.copy()
        else:
            cluster_img = np.zeros((h, w, 3), dtype=np.uint8)

        if output_mode == "centroid":
            colors = center_colors
        else:
            colors = _COLORS_BGR[:effective_k] if effective_k <= len(_COLORS_BGR) else np.resize(_COLORS_BGR, (effective_k, 3))
        colors = np.asarray(colors, dtype=np.uint8).copy()
        for cluster_idx in range(effective_k):
            custom_color = custom_colors.get(cluster_idx + 1)
            if custom_color is not None:
                colors[cluster_idx] = custom_color

        for cluster_idx in range(effective_k):
            cluster_img[label_map == (cluster_idx + 1)] = colors[cluster_idx]

        overlay = original_bgr.copy()
        overlay[roi_mask] = cv2.addWeighted(
            original_bgr[roi_mask], 0.48, cluster_img[roi_mask], 0.52, 0,
        )
        final_labels = label_map[roi_mask].astype(np.int32) - 1
        counts = np.bincount(final_labels, minlength=effective_k).astype(int)
        total = int(counts.sum())
        percentages = (counts / max(total, 1) * 100.0).astype(float)

        output_images.append(cluster_img)
        overlay_images.append(overlay)
        source_images.append(original_bgr.copy())
        label_maps.append(label_map)
        centers_all.append(np.asarray(centers[order], dtype=np.float32))
        counts_all.append(counts.tolist())
        percentages_all.append([float(x) for x in percentages])
        compactness_all.append(float(compactness))
        legends_all.append([
            {
                "label": cluster_idx + 1,
                "color": [
                    int(colors[cluster_idx][2]),
                    int(colors[cluster_idx][1]),
                    int(colors[cluster_idx][0]),
                ],
            }
            for cluster_idx in range(effective_k)
        ])
        reference_info_all.append({
            "mode": init_mode,
            "reference_crops_available": available_ref_count,
            "reference_crop_items": int(raw_ref_count),
            "uses_reference_crops": init_mode != "auto",
            "reference_classifier": init_mode == "reference_fixed",
            "reference_feature_dimension": int(features.shape[2]),
            "reference_uses_robust_scale": init_mode == "reference_fixed",
            "spatial_cleanup": init_mode == "reference_fixed",
            "effective_k": effective_k,
            "reference_sequence_used": "reference_sequence" in data.get("history", []),
            "reference_source_id": str(reference_source or ""),
            "reference_source_label": selected_source.get("label") if isinstance(selected_source, dict) else "",
            "reference_source_type": selected_source.get("type") if isinstance(selected_source, dict) else "",
            "reference_sequence": selected_source.get("reference_sequence") if isinstance(selected_source, dict) else None,
        })
        if color_space == "GRAY":
            reference_info_all[-1]["gray_features"] = [
                "intensity", "local_mean", "local_std",
            ]

        if debug:
            print("[kmeans_cluster]", idx, "counts:", counts.tolist())

    data["images"] = output_images
    data["count"] = len(output_images)

    results = data.setdefault("results", {})
    # Keep the pre-clustering pixels for downstream label-based mapping.  The
    # primary image stream contains the coloured label preview at this point.
    results["kmeans_source_images"] = source_images
    results["kmeans_labeled_images"] = [img.copy() for img in output_images]
    results["kmeans_overlay_images"] = overlay_images
    results["kmeans_legend"] = legends_all
    results["kmeans_label_maps"] = label_maps
    results["kmeans_centers"] = centers_all
    results["kmeans_counts"] = counts_all
    results["kmeans_percentages"] = percentages_all
    results["kmeans_compactness"] = compactness_all
    results["kmeans_reference_info"] = reference_info_all

    data.setdefault("meta", {})["kmeans_cluster"] = {
        "k": k,
        "init_mode": init_mode,
        "reference_source": str(reference_source or ""),
        "color_space": color_space,
        "attempts": attempts,
        "max_iter": max_iter,
        "epsilon": epsilon,
        "sort_by_brightness": bool(sort_by_brightness),
        "output_mode": output_mode,
        "background": background,
        "cluster_colors": {
            str(label): "#{:02x}{:02x}{:02x}".format(
                int(color[2]), int(color[1]), int(color[0]),
            )
            for label, color in custom_colors.items()
        },
    }
    data.setdefault("history", []).append("kmeans_cluster")
    return data
