"""Create a similarity map from selected k-means label clusters."""

import cv2
import json
import numpy as np
from itertools import permutations


MATLAB_EPS_IQR = 1e-6
SOFT_SPATIAL_SIGMA = 1.0
DISPLAY_SPATIAL_SIGMA = 2.0
CLUSTER_PALETTE_BGR = np.asarray([
    [255.0, 0.0, 0.0],
    [0.0, 255.0, 0.0],
    [0.0, 0.0, 255.0],
    [255.0, 255.0, 0.0],
    [255.0, 0.0, 255.0],
    [0.0, 255.0, 255.0],
], dtype=np.float32)


def _parse_labels(value):
    if isinstance(value, (list, tuple)):
        parts = value
    else:
        parts = str(value or "").replace(";", ",").split(",")
    labels = []
    for part in parts:
        try:
            label = int(str(part).strip())
        except (TypeError, ValueError):
            continue
        if label > 0 and label not in labels:
            labels.append(label)
    return labels


def _center_value(pixels, center_mode):
    if center_mode == "min_max_midpoint":
        return (np.min(pixels, axis=0) + np.max(pixels, axis=0)) / 2.0
    if center_mode == "reference_mean":
        return np.mean(pixels, axis=0)
    if center_mode == "reference_mean_half":
        return np.mean(pixels, axis=0) / 2.0
    return np.median(pixels, axis=0)


def _parse_accepted_components(value):
    if isinstance(value, list):
        raw = value
    else:
        try:
            raw = json.loads(str(value or "[]"))
        except (TypeError, ValueError, json.JSONDecodeError):
            return []
    return [item for item in raw if isinstance(item, dict)]


def _reference_crop_context(results, image_index):
    info_rows = results.get("kmeans_reference_info")
    info = (
        info_rows[min(image_index, len(info_rows) - 1)]
        if isinstance(info_rows, list) and info_rows
        else {}
    )
    if not isinstance(info, dict):
        info = {}
    source_id = str(info.get("reference_source_id") or "").strip()
    sources = results.get("reference_sources")
    source = None
    if isinstance(sources, dict):
        if source_id and source_id.lower() != "auto":
            source = sources.get(source_id)
        elif sources:
            source_id, source = list(sources.items())[-1]
    rows = source.get("reference_crops") if isinstance(source, dict) else None
    if not isinstance(rows, list) or not rows:
        rows = results.get("reference_crops")
    sequence_rows = source.get("reference_sequence") if isinstance(source, dict) else None
    if not isinstance(sequence_rows, list) or not sequence_rows:
        sequence_rows = results.get("reference_sequence")
    square_rows = source.get("reference_crop_squares") if isinstance(source, dict) else None
    if not isinstance(square_rows, list) or not square_rows:
        square_rows = results.get("reference_crop_squares")
    if not isinstance(rows, list) or not rows:
        return [], {
            "reference_crop_count": 0,
            "reference_source_id": source_id,
            "reference_source_found": isinstance(source, dict),
            "reference_info": [],
            "reference_sequence_order": [],
        }
    row = rows[min(image_index, len(rows) - 1)]
    if not isinstance(row, list):
        crops = []
    else:
        crops = [np.asarray(crop) for crop in row if np.asarray(crop).size]
    sequence = (
        sequence_rows[min(image_index, len(sequence_rows) - 1)]
        if isinstance(sequence_rows, list) and sequence_rows
        else {}
    )
    sequence_items = sequence.get("items") if isinstance(sequence, dict) else None
    square_row = (
        square_rows[min(image_index, len(square_rows) - 1)]
        if isinstance(square_rows, list) and square_rows
        else []
    )
    reference_info = []
    for crop_index in range(len(crops)):
        sequence_item = (
            sequence_items[crop_index]
            if isinstance(sequence_items, list) and crop_index < len(sequence_items)
            and isinstance(sequence_items[crop_index], dict)
            else {}
        )
        square = (
            square_row[crop_index]
            if isinstance(square_row, list) and crop_index < len(square_row)
            and isinstance(square_row[crop_index], dict)
            else {}
        )
        original_index = sequence_item.get("original_index", crop_index)
        try:
            original_index = int(original_index)
        except (TypeError, ValueError):
            original_index = crop_index
        name = str(
            sequence_item.get("label")
            or square.get("name")
            or (original_index + 1)
        )
        reference_info.append({
            "original_index": original_index,
            "name": name,
        })
    return crops, {
        "reference_crop_count": len(crops),
        "reference_source_id": source_id,
        "reference_source_found": isinstance(source, dict),
        "reference_info": reference_info,
        "reference_sequence_order": [
            int(item["original_index"]) for item in reference_info
        ],
    }


def _reference_crop_row(results, image_index):
    return _reference_crop_context(results, image_index)[0]


def _to_gray(image):
    image = np.asarray(image)
    if image.ndim == 3:
        return cv2.cvtColor(image[..., :3].astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    return image.astype(np.float32)


def _reference_centroids_from_sequence(results, image_index):
    rows = results.get("reference_sequence")
    if not isinstance(rows, list) or not rows:
        return None
    row = rows[min(image_index, len(rows) - 1)]
    items = row.get("items") if isinstance(row, dict) else None
    if not isinstance(items, list) or not items:
        return None

    parsed = []
    used_gray_scores = True
    for fallback_index, item in enumerate(items):
        if not isinstance(item, dict):
            return None
        try:
            original_index = int(item.get("original_index", fallback_index))
        except (TypeError, ValueError):
            return None
        scores = item.get("scores")
        gray_score = scores.get("GRAY") if isinstance(scores, dict) else None
        if gray_score is None:
            gray_score = item.get("score")
            used_gray_scores = False
        try:
            gray_score = float(gray_score)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(gray_score):
            return None
        parsed.append((original_index, item, gray_score))

    parsed.sort(key=lambda entry: entry[0])
    if len({entry[0] for entry in parsed}) != len(parsed):
        return None
    centroids = np.asarray(
        [entry[2] / 255.0 for entry in parsed], dtype=np.float32,
    )
    reference_info = [
        {
            "original_index": original_index,
            "name": str(item.get("label") or original_index + 1),
            "centroid": float(centroid),
        }
        for (original_index, item, _), centroid in zip(parsed, centroids)
    ]
    source = (
        "reference_sequence_gray_score"
        if used_gray_scores
        else "reference_sequence_score"
    )
    return centroids, reference_info, source


def _reference_centroids_from_crops(results, image_index):
    rows = results.get("reference_crops")
    if not isinstance(rows, list) or not rows:
        return None
    row = rows[min(image_index, len(rows) - 1)]
    if not isinstance(row, list):
        return None
    indexed_crops = [
        (original_index, np.asarray(crop))
        for original_index, crop in enumerate(row)
        if np.asarray(crop).size
    ]
    if not indexed_crops:
        return None
    square_rows = results.get("reference_crop_squares")
    square_row = (
        square_rows[min(image_index, len(square_rows) - 1)]
        if isinstance(square_rows, list) and square_rows
        else []
    )
    centroids = np.asarray([
        float(np.median(_to_gray(crop))) / 255.0
        for _, crop in indexed_crops
    ], dtype=np.float32)
    reference_info = []
    for (original_index, _), centroid in zip(indexed_crops, centroids):
        square = (
            square_row[original_index]
            if isinstance(square_row, list) and original_index < len(square_row)
            and isinstance(square_row[original_index], dict)
            else {}
        )
        reference_info.append({
            "original_index": original_index,
            "name": str(square.get("name") or original_index + 1),
            "centroid": float(centroid),
        })
    return centroids, reference_info, "reference_crops_median"


def _reference_centroids_with_identity(results, image_index):
    sequence_result = _reference_centroids_from_sequence(results, image_index)
    if sequence_result is not None:
        return sequence_result
    return _reference_centroids_from_crops(results, image_index)


def _cluster_center_brightness(centers, color_space):
    centers = np.asarray(centers, dtype=np.float32)
    if centers.ndim == 1:
        centers = centers[:, None]
    if centers.ndim != 2 or centers.shape[0] == 0:
        return None
    color_space = str(color_space or "BGR").upper()
    if color_space == "GRAY" or centers.shape[1] == 1:
        return centers[:, 0] / 255.0
    if centers.shape[1] < 3:
        return None
    values = np.clip(centers[:, :3], 0.0, 255.0).astype(np.uint8)
    if color_space == "HSV":
        bgr = cv2.cvtColor(values.reshape(-1, 1, 3), cv2.COLOR_HSV2BGR).reshape(-1, 3)
    elif color_space == "LAB":
        bgr = cv2.cvtColor(values.reshape(-1, 1, 3), cv2.COLOR_LAB2BGR).reshape(-1, 3)
    else:
        bgr = values
    brightness = 0.114 * bgr[:, 0] + 0.587 * bgr[:, 1] + 0.299 * bgr[:, 2]
    return brightness.astype(np.float32) / 255.0


def _match_clusters_to_references(cluster_brightness, reference_centroids, reference_info):
    cluster_brightness = np.asarray(cluster_brightness, dtype=np.float32).reshape(-1)
    reference_centroids = np.asarray(reference_centroids, dtype=np.float32).reshape(-1)
    if (
        cluster_brightness.size == 0
        or cluster_brightness.size != reference_centroids.size
        or len(reference_info) != reference_centroids.size
    ):
        return None
    best_perm = None
    best_cost = None
    for perm in permutations(range(reference_centroids.size)):
        cost = sum(
            abs(float(cluster_brightness[index] - reference_centroids[perm[index]]))
            for index in range(cluster_brightness.size)
        )
        if best_cost is None or cost < best_cost - 1e-9:
            best_cost = cost
            best_perm = perm
    return [
        {
            "cluster_label": cluster_index + 1,
            "cluster_center_brightness": float(cluster_brightness[cluster_index] * 255.0),
            "reference_original_index": int(reference_info[reference_index]["original_index"]),
            "reference_name": str(reference_info[reference_index]["name"]),
            "reference_centroid": float(reference_centroids[reference_index] * 255.0),
            "reference_centroid_normalized": float(reference_centroids[reference_index]),
            "matching_cost": float(
                abs(cluster_brightness[cluster_index] - reference_centroids[reference_index]) * 255.0
            ),
        }
        for cluster_index, reference_index in enumerate(best_perm)
    ]


def _membership_from_existing_labels(gray, roi_mask, labels_arr, cluster_labels):
    """Build ordered soft memberships from the upstream hard segmentation."""
    intensity = np.asarray(gray, dtype=np.float32) / 255.0
    roi_mask = np.asarray(roi_mask, dtype=bool)
    labels_arr = np.asarray(labels_arr)
    cluster_labels = [int(label) for label in cluster_labels]
    if not cluster_labels or not np.any(roi_mask):
        return None
    raw = np.zeros(intensity.shape + (len(cluster_labels),), dtype=np.float32)
    medians, iqrs, denoms = [], [], []
    for component_index, label in enumerate(cluster_labels):
        values = intensity[(labels_arr == label) & roi_mask]
        if values.size:
            median_value = float(np.median(values))
            q25, q75 = np.percentile(values, [25.0, 75.0])
            iqr = float(q75 - q25)
            denom = max(iqr, MATLAB_EPS_IQR)
        else:
            median_value = 0.0
            iqr = 0.0
            denom = MATLAB_EPS_IQR
        medians.append(median_value)
        iqrs.append(iqr)
        denoms.append(denom)

    brightness_order = np.argsort(np.asarray(medians), kind="stable")
    ordered_centers = np.asarray(medians, dtype=np.float32)[brightness_order]
    membership_debug = [None] * len(cluster_labels)
    for brightness_rank, original_index in enumerate(brightness_order):
        center = float(ordered_centers[brightness_rank])
        left_center = (
            float(ordered_centers[brightness_rank - 1])
            if brightness_rank > 0 else None
        )
        right_center = (
            float(ordered_centers[brightness_rank + 1])
            if brightness_rank + 1 < len(ordered_centers) else None
        )
        if len(ordered_centers) == 1:
            mu = np.ones_like(intensity, dtype=np.float32)
            membership_type = "single_component"
        elif brightness_rank == 0:
            mu = np.ones_like(intensity, dtype=np.float32)
            between = (intensity > center) & (intensity < right_center)
            mu[intensity >= right_center] = 0.0
            mu[between] = (right_center - intensity[between]) / max(
                right_center - center, np.finfo(np.float32).eps,
            )
            membership_type = "left_shoulder"
        elif brightness_rank == len(ordered_centers) - 1:
            mu = np.zeros_like(intensity, dtype=np.float32)
            between = (intensity > left_center) & (intensity < center)
            mu[intensity >= center] = 1.0
            mu[between] = (intensity[between] - left_center) / max(
                center - left_center, np.finfo(np.float32).eps,
            )
            membership_type = "right_shoulder"
        else:
            z = (intensity - center) / denoms[int(original_index)]
            mu = np.exp(-0.5 * z * z).astype(np.float32)
            membership_type = "gaussian_median_iqr"
        raw[..., int(original_index)] = np.clip(mu, 0.0, 1.0)
        membership_debug[int(original_index)] = {
            "brightness_rank": int(brightness_rank + 1),
            "center": center,
            "left_neighbor_center": left_center,
            "right_neighbor_center": right_center,
            "membership_type": membership_type,
        }
    raw[~roi_mask] = 0.0
    sum_mu = np.sum(raw, axis=2, keepdims=True)
    normalized = np.zeros_like(raw, dtype=np.float32)
    sum_mu_scalar = sum_mu[..., 0]
    valid = (
        roi_mask
        & np.isfinite(sum_mu_scalar)
        & (sum_mu_scalar > np.finfo(np.float32).tiny)
    )
    normalized[valid] = raw[valid] / sum_mu[valid]

    component_statistics = []
    roi_pixel_count = max(int(np.count_nonzero(roi_mask)), 1)
    for component_index, label in enumerate(cluster_labels):
        component_pixels = (labels_arr == label) & roi_mask
        assigned_values = intensity[component_pixels]
        pixel_count = int(np.count_nonzero(component_pixels))
        raw_values = raw[..., component_index][roi_mask]
        normalized_values = normalized[..., component_index][roi_mask]
        final_similarity_values = raw_values
        component_statistics.append({
            "cluster_label": label,
            "label": label,
            **membership_debug[component_index],
            "pixel_count": pixel_count,
            "pixel_fraction": pixel_count / roi_pixel_count,
            "assigned_min": float(np.min(assigned_values)) if assigned_values.size else None,
            "assigned_max": float(np.max(assigned_values)) if assigned_values.size else None,
            "median": medians[component_index],
            "iqr": iqrs[component_index],
            "denom_used": denoms[component_index],
            "raw_membership_min": float(np.min(raw_values)) if raw_values.size else 0.0,
            "raw_membership_max": float(np.max(raw_values)) if raw_values.size else 0.0,
            "raw_membership_mean": float(np.mean(raw_values)) if raw_values.size else 0.0,
            "normalized_membership_min": float(np.min(normalized_values)) if normalized_values.size else 0.0,
            "normalized_membership_max": float(np.max(normalized_values)) if normalized_values.size else 0.0,
            "normalized_membership_mean": float(np.mean(normalized_values)) if normalized_values.size else 0.0,
            "membership_mean": float(np.mean(final_similarity_values)) if final_similarity_values.size else 0.0,
            "raw_mean": float(np.mean(raw_values)) if raw_values.size else 0.0,
            "normalized_mean": float(np.mean(normalized_values)) if normalized_values.size else 0.0,
            "final_similarity_mean": float(np.mean(final_similarity_values)) if final_similarity_values.size else 0.0,
            "raw_max": float(np.max(raw_values)) if raw_values.size else 0.0,
            "normalized_max": float(np.max(normalized_values)) if normalized_values.size else 0.0,
            "final_similarity_max": float(np.max(final_similarity_values)) if final_similarity_values.size else 0.0,
        })
    return {
        "cluster_labels": cluster_labels,
        "component_map": labels_arr,
        "component_medians": np.asarray(medians, dtype=np.float32),
        "component_iqrs": np.asarray(iqrs, dtype=np.float32),
        "component_denoms": np.asarray(denoms, dtype=np.float32),
        "component_statistics": component_statistics,
        "membership_raw": raw,
        "membership_normalized": normalized,
        "hard_labels": labels_arr.copy(),
    }


def _membership_context(feature, labels_arr, config):
    roi_mask = labels_arr > 0
    gray = feature[..., 0] if feature.ndim == 3 else feature
    membership_result = config.get("_membership_result")
    if membership_result is None:
        cluster_labels = config.get("cluster_labels")
        if not cluster_labels:
            return None
        membership_result = _membership_from_existing_labels(
            gray, roi_mask, labels_arr, cluster_labels,
        )
    if membership_result is None:
        return None
    cluster_labels = membership_result["cluster_labels"]
    models = {
        label: {
            "center": np.asarray([membership_result["component_medians"][index]], dtype=np.float32),
            "scale": np.asarray([membership_result["component_denoms"][index]], dtype=np.float32),
            "pixel_count": int(np.count_nonzero(labels_arr == label)),
        }
        for index, label in enumerate(cluster_labels)
    }
    return (
        cluster_labels,
        models,
        membership_result["membership_raw"],
        membership_result["membership_normalized"],
        roi_mask,
    )


def _component_similarity(feature, labels_arr, config):
    labels = _parse_labels(config.get("selected_labels", "1"))
    reference_labels = _parse_labels(config.get("reference_label", "1"))
    if not labels or len(reference_labels) != 1:
        return None
    reference_label = reference_labels[0]
    context = config.get("_membership_context")
    if context is None:
        context = _membership_context(feature, labels_arr, config)
    if context is None:
        return None
    cluster_labels, models, raw_memberships, _normalized_memberships, roi_mask = context
    if reference_label not in cluster_labels:
        return None
    reference_index = cluster_labels.index(reference_label)
    similarity = raw_memberships[..., reference_index].copy().astype(np.float32)
    evaluation_mask = (
        np.isin(labels_arr, labels)
        if bool(config.get("hard_label_mask", False))
        else roi_mask
    )
    similarity[~evaluation_mask] = 0.0
    reference = np.asarray([models[reference_label]["center"][0]], dtype=np.float32)
    return similarity.astype(np.float32), reference, labels, evaluation_mask


def _normalize_soft_components(component_maps, roi_mask):
    """Normalize memberships per pixel, matching MATLAB membership_norm."""
    if not component_maps:
        return []
    stack = np.stack(component_maps, axis=2).astype(np.float64)
    sum_mu = np.sum(stack, axis=2)
    valid = roi_mask & (sum_mu > 0.0)
    normalized = np.zeros_like(stack)
    normalized[valid] = stack[valid] / (
        sum_mu[valid, None] + np.finfo(float).eps
    )
    return [
        normalized[..., component].astype(np.float32)
        for component in range(normalized.shape[2])
    ]


def _palette_bgr(component_count):
    repeats = int(np.ceil(component_count / len(CLUSTER_PALETTE_BGR)))
    return np.tile(CLUSTER_PALETTE_BGR, (max(repeats, 1), 1))[:component_count].copy()


def _smooth_memberships_for_visualization(memberships, roi_mask):
    memberships = np.asarray(memberships, dtype=np.float32)
    roi_mask = np.asarray(roi_mask, dtype=bool)
    smoothed = np.zeros_like(memberships, dtype=np.float32)
    for component_index in range(memberships.shape[2]):
        smoothed[..., component_index] = cv2.GaussianBlur(
            memberships[..., component_index],
            (0, 0),
            sigmaX=SOFT_SPATIAL_SIGMA,
            sigmaY=SOFT_SPATIAL_SIGMA,
            borderType=cv2.BORDER_REFLECT,
        )
    smoothed[~roi_mask] = 0.0
    total = np.sum(smoothed, axis=2, keepdims=True)
    valid = roi_mask & np.isfinite(total[..., 0]) & (
        total[..., 0] > np.finfo(np.float32).eps
    )
    normalized = np.zeros_like(smoothed, dtype=np.float32)
    normalized[valid] = smoothed[valid] / total[valid]
    return normalized


def _smooth_absolute_memberships_for_display(memberships, roi_mask):
    memberships = np.asarray(memberships, dtype=np.float32)
    roi_mask = np.asarray(roi_mask, dtype=bool)
    display = np.zeros_like(memberships, dtype=np.float32)
    for component_index in range(memberships.shape[2]):
        display[..., component_index] = cv2.GaussianBlur(
            memberships[..., component_index],
            (0, 0),
            sigmaX=DISPLAY_SPATIAL_SIGMA,
            sigmaY=DISPLAY_SPATIAL_SIGMA,
            borderType=cv2.BORDER_REFLECT,
        )
    display = np.clip(display, 0.0, 1.0)
    display[~roi_mask] = 0.0
    return display


def _soft_palette_visualization(memberships, palette_bgr, roi_mask, source_bgr):
    soft_bgr = np.sum(
        memberships[..., :, None] * palette_bgr[None, None, :, :], axis=2,
    )
    soft_bgr = np.clip(soft_bgr, 0.0, 255.0).astype(np.uint8)
    soft_bgr[~roi_mask] = 0
    overlay = np.asarray(source_bgr, dtype=np.uint8).copy()
    overlay_float = (
        source_bgr.astype(np.float32) * 0.42
        + soft_bgr.astype(np.float32) * 0.58
    )
    overlay[roi_mask] = np.clip(overlay_float[roi_mask], 0.0, 255.0).astype(np.uint8)
    return soft_bgr, overlay


def cluster_reference_map(
    data,
    selected_labels="1",
    reference_label="1",
    center_mode="cluster_median",
    map_multiplier=1.0,
    accepted_components="[]",
    remainder_as_last=False,
    remainder_name="Maradék",
    remainder_display_multiplier=1.0,
    remainder_invert=False,
    colormap="jet",
    invert=False,
):
    if data.get("error") is not None:
        return data

    results = data.get("results") or {}
    sources = results.get("kmeans_source_images")
    labeled = results.get("kmeans_labeled_images")
    label_maps = results.get("kmeans_label_maps")
    centers_all = results.get("kmeans_centers")
    if (
        not isinstance(sources, list)
        or not isinstance(label_maps, list)
        or not isinstance(centers_all, list)
        or not sources
    ):
        data["error"] = "E3721"
        return data

    labels = _parse_labels(selected_labels)
    if not labels:
        data["error"] = "E3722"
        return data
    reference_labels = _parse_labels(reference_label)
    if len(reference_labels) != 1:
        data["error"] = "E3727"
        return data
    reference_label_value = reference_labels[0]

    cv_maps = {"turbo": cv2.COLORMAP_TURBO, "jet": cv2.COLORMAP_JET, "viridis": cv2.COLORMAP_VIRIDIS}
    if colormap not in cv_maps:
        data["error"] = "E3723"
        return data
    if center_mode not in {"min_max_midpoint", "cluster_median", "reference_mean", "reference_mean_half"}:
        data["error"] = "E3726"
        return data
    try:
        multiplier = float(map_multiplier)
    except (TypeError, ValueError):
        data["error"] = "E3728"
        return data
    if not 0.0 <= multiplier <= 1.0:
        data["error"] = "E3728"
        return data

    accepted = _parse_accepted_components(accepted_components)
    try:
        remainder_visual_multiplier = float(remainder_display_multiplier)
    except (TypeError, ValueError):
        remainder_visual_multiplier = 1.0
    remainder_visual_multiplier = float(np.clip(remainder_visual_multiplier, 0.0, 1.0))
    output_images, overlay_images, raw_maps, references, label_values_all = [], [], [], [], []
    component_maps_all, component_images_all, component_info_all, remainder_maps = [], [], [], []
    soft_memberships_all, soft_color_images, soft_overlay_images = [], [], []
    display_memberships_all, display_raw_maps = [], []
    heatmap_images = []
    reference_models_debug = []
    classifier_label_maps = []
    matlab_raw_all, matlab_normalized_all = [], []
    matlab_medians_all, matlab_iqrs_all = [], []
    matlab_initial_all, matlab_sorted_all = [], []
    matlab_denoms_all, matlab_statistics_all = [], []
    centroid_sources = []
    reference_crop_debug = []
    empty_upstream_labels = []
    reference_cluster_mappings = []
    for index, source in enumerate(sources):
        if index >= len(label_maps) or index >= len(centers_all):
            data["error"] = "E3721"
            return data
        image = np.asarray(source)
        labels_arr = np.asarray(label_maps[index])
        if image.shape[:2] != labels_arr.shape[:2]:
            data["error"] = "E3724"
            return data

        if image.ndim == 3:
            gray = cv2.cvtColor(image[..., :3].astype(np.uint8), cv2.COLOR_BGR2GRAY)
        else:
            gray = image.astype(np.float32)
        feature = gray[..., None].astype(np.float32)
        _, crop_debug = _reference_crop_context(results, index)
        reference_result = _reference_centroids_with_identity(results, index)
        if reference_result is None:
            data["error"] = "E3727"
            return data
        initial_centroids, reference_info, centroid_source = reference_result
        centroid_sources.append(centroid_source)
        reference_crop_debug.append({
            **{key: value for key, value in crop_debug.items() if key != "reference_info"},
            "initial_centroid_source": centroid_source,
        })
        roi_mask = labels_arr > 0
        cluster_centers = np.asarray(centers_all[index], dtype=np.float32)
        cluster_labels = list(range(1, len(cluster_centers) + 1))
        color_space = data.get("meta", {}).get(
            "kmeans_cluster", {},
        ).get("color_space", "GRAY")
        cluster_brightness = _cluster_center_brightness(cluster_centers, color_space)
        cluster_reference_mapping = _match_clusters_to_references(
            cluster_brightness, initial_centroids, reference_info,
        )
        if cluster_reference_mapping is None:
            data["error"] = "E3727"
            return data
        reference_cluster_mappings.append(cluster_reference_mapping)
        membership_result = _membership_from_existing_labels(
            gray, roi_mask, labels_arr, cluster_labels,
        )
        if membership_result is None:
            data["error"] = "E3727"
            return data
        soft_memberships = _smooth_memberships_for_visualization(
            membership_result["membership_normalized"], roi_mask,
        )
        display_memberships = _smooth_absolute_memberships_for_display(
            membership_result["membership_raw"], roi_mask,
        )
        for component_index, statistics in enumerate(
            membership_result["component_statistics"]
        ):
            raw_values = membership_result["membership_raw"][..., component_index][roi_mask]
            display_values = display_memberships[..., component_index][roi_mask]
            statistics.update({
                "raw_min": float(np.min(raw_values)) if raw_values.size else 0.0,
                "raw_max": float(np.max(raw_values)) if raw_values.size else 0.0,
                "raw_mean": float(np.mean(raw_values)) if raw_values.size else 0.0,
                "display_min": float(np.min(display_values)) if display_values.size else 0.0,
                "display_max": float(np.max(display_values)) if display_values.size else 0.0,
                "display_mean": float(np.mean(display_values)) if display_values.size else 0.0,
            })
        palette_bgr = _palette_bgr(len(cluster_labels))
        empty_upstream_labels.append([
            int(stats["cluster_label"])
            for stats in membership_result["component_statistics"]
            if stats["pixel_count"] == 0
        ])
        current_config = {
            "selected_labels": ",".join(str(label) for label in labels),
            "reference_label": str(reference_label_value),
            "center_mode": center_mode,
            "map_multiplier": multiplier,
            "invert": bool(invert),
            "cluster_labels": cluster_labels,
            "_membership_result": membership_result,
            "hard_label_mask": False,
        }
        membership_context = _membership_context(feature, labels_arr, current_config)
        current_config["_membership_context"] = membership_context
        current_result = _component_similarity(feature, labels_arr, current_config)
        if current_result is None:
            data["error"] = "E3727"
            return data
        current_similarity, reference, _, reference_mask = current_result
        label_values = []
        for label in labels:
            label_pixels = gray[labels_arr == label]
            if label_pixels.size == 0:
                continue
            label_center = _center_value(label_pixels, center_mode)
            label_values.append({
                "label": int(label),
                "pixel_count": int(label_pixels.size),
                "values": {"GRAY": float(label_center)},
            })
        context_models = membership_context[1] if membership_context is not None else {}
        classifier_label_maps.append(labels_arr.copy())
        matlab_raw_all.append(membership_result["membership_raw"])
        matlab_normalized_all.append(membership_result["membership_normalized"])
        matlab_medians_all.append(membership_result["component_medians"])
        matlab_iqrs_all.append(membership_result["component_iqrs"])
        matlab_denoms_all.append(membership_result["component_denoms"])
        matlab_statistics_all.append(membership_result["component_statistics"])
        matlab_initial_all.append(initial_centroids)
        matlab_sorted_all.append(np.sort(initial_centroids, kind="stable"))
        reference_models_debug.append([
            {
                "label": int(label),
                "center": [float(value) for value in context_models[label]["center"]],
                "scale": [float(value) for value in context_models[label]["scale"]],
                "pixel_count": int(context_models[label]["pixel_count"]),
            }
            for label in membership_context[0]
        ] if membership_context is not None else [])
        component_candidates = []
        component_configs = []
        for component_index, config in enumerate(accepted):
            component_config = dict(config)
            component_config["hard_label_mask"] = False
            component_config["_membership_context"] = membership_context
            component_result = _component_similarity(feature, labels_arr, component_config)
            if component_result is None:
                continue
            candidate, _, component_labels, _ = component_result
            component_candidates.append(candidate)
            component_configs.append((component_index, config, component_labels))

        if bool(remainder_as_last):
            remainder_candidate = np.zeros(labels_arr.shape, dtype=np.float32)
            if component_candidates:
                candidate_sum = np.clip(
                    np.sum(np.stack(component_candidates, axis=2), axis=2),
                    0.0, 1.0,
                )
                remainder_candidate[roi_mask] = np.clip(
                    1.0 - candidate_sum[roi_mask], 0.0, 1.0,
                )
            component_candidates.append(remainder_candidate)

        # Candidates contain raw 0..1 percentages. Display-only controls are
        # applied later and never alter these values or the remainder.
        component_maps = component_candidates
        component_info = []
        for normalized_index, (component_index, config, component_labels) in enumerate(component_configs):
            allocated = component_maps[normalized_index]
            accepted_reference_label = int(
                _parse_labels(config.get("reference_label", "1"))[0]
            )
            accepted_reference = cluster_reference_mapping[
                accepted_reference_label - 1
            ]
            component_info.append({
                "index": component_index,
                "name": str(config.get("name") or f"Komponens {component_index + 1}"),
                "selected_labels": component_labels,
                "reference_label": accepted_reference_label,
                "original_reference_index": accepted_reference["reference_original_index"],
                "reference_name": accepted_reference["reference_name"],
                "reference_centroid": accepted_reference["reference_centroid"],
                "map_multiplier": float(config.get("map_multiplier", 1.0)),
                "coverage_percent": float(np.mean(allocated[labels_arr > 0]) * 100.0) if np.any(labels_arr > 0) else 0.0,
                "is_remainder": False,
            })

        similarity = current_similarity
        if bool(remainder_as_last):
            similarity = component_maps[-1]
            component_info.append({
                "index": len(component_info),
                "name": str(remainder_name or "Maradék"),
                "selected_labels": [],
                "reference_label": None,
                "map_multiplier": 1.0,
                "display_multiplier": remainder_visual_multiplier,
                "invert": bool(remainder_invert),
                "coverage_percent": float(np.mean(similarity[roi_mask]) * 100.0) if np.any(roi_mask) else 0.0,
                "is_remainder": True,
            })
        display_similarity = similarity
        if bool(remainder_as_last):
            display_similarity = similarity.copy()
            if bool(remainder_invert):
                display_similarity[roi_mask] = 1.0 - similarity[roi_mask]
            display_similarity = display_similarity * remainder_visual_multiplier
        else:
            reference_index = cluster_labels.index(reference_label_value)
            display_similarity = display_memberships[..., reference_index].copy()
            if bool(invert):
                display_similarity[reference_mask] = 1.0 - display_similarity[reference_mask]
            display_similarity = display_similarity * multiplier
        raw = np.rint(np.clip(display_similarity, 0.0, 1.0) * 255.0).astype(np.uint8)
        mapped = cv2.applyColorMap(raw, cv_maps[colormap])
        evaluation_mask = (labels_arr > 0) if bool(remainder_as_last) else reference_mask
        source_bgr = image[..., :3].astype(np.uint8) if image.ndim == 3 else cv2.cvtColor(
            image.astype(np.uint8), cv2.COLOR_GRAY2BGR,
        )
        soft_bgr, soft_overlay = _soft_palette_visualization(
            soft_memberships, palette_bgr, roi_mask, source_bgr,
        )
        reference_index = cluster_labels.index(reference_label_value)
        display_alpha = soft_memberships[..., reference_index].copy()
        if bool(invert):
            display_alpha[roi_mask] = 1.0 - display_alpha[roi_mask]
        display_alpha *= multiplier
        display_alpha[~roi_mask] = 0.0
        alpha = display_alpha[..., None]
        cluster_color = palette_bgr[reference_index]
        reference_overlay = (
            source_bgr.astype(np.float32) * (1.0 - alpha)
            + cluster_color[None, None, :] * alpha
        )
        reference_overlay = np.clip(reference_overlay, 0.0, 255.0).astype(np.uint8)
        heatmap_overlay = source_bgr.copy()
        heatmap_overlay[evaluation_mask] = cv2.addWeighted(
            source_bgr[evaluation_mask], 0.42, mapped[evaluation_mask], 0.58, 0,
        )
        primary_image = mapped
        overlay = heatmap_overlay if bool(remainder_as_last) else reference_overlay

        raw_maps.append(similarity.astype(np.float32))
        display_raw_maps.append(display_similarity.astype(np.float32))
        output_images.append(primary_image)
        overlay_images.append(overlay)
        heatmap_images.append(mapped)
        soft_memberships_all.append(soft_memberships)
        display_memberships_all.append(display_memberships)
        soft_color_images.append(soft_bgr)
        soft_overlay_images.append(soft_overlay)
        references.append([float(v) for v in reference.reshape(-1)])
        label_values_all.append(label_values)
        component_maps_all.append(component_maps)
        component_images = []
        for component_index, component_map in enumerate(component_maps):
            component_display = component_map
            if bool(remainder_as_last) and component_index == len(component_maps) - 1:
                component_display = component_map.copy()
                if bool(remainder_invert):
                    component_display[roi_mask] = 1.0 - component_map[roi_mask]
                component_display = component_display * remainder_visual_multiplier
            elif component_index < len(component_configs):
                component_config = component_configs[component_index][1]
                component_display = component_map.copy()
                if bool(component_config.get("invert", False)):
                    component_mask = roi_mask
                    component_display[component_mask] = 1.0 - component_map[component_mask]
                component_display = component_display * np.clip(
                    float(component_config.get("map_multiplier", 1.0)), 0.0, 1.0,
                )
            component_images.append(cv2.applyColorMap(
                np.rint(np.clip(component_display, 0.0, 1.0) * 255.0).astype(np.uint8),
                cv_maps[colormap],
            ))
        component_images_all.append(component_images)
        component_info_all.append(component_info)
        remainder_maps.append(
            component_maps[-1]
            if bool(remainder_as_last) and component_maps
            else np.zeros(labels_arr.shape, dtype=np.float32)
        )

    results["cluster_map_images"] = output_images
    results["cluster_map_overlay_images"] = overlay_images
    results["cluster_map_heatmap_images"] = heatmap_images
    results["cluster_soft_memberships"] = soft_memberships_all
    results["cluster_soft_color_images"] = soft_color_images
    results["cluster_soft_overlay_images"] = soft_overlay_images
    results["cluster_palette_bgr"] = _palette_bgr(
        len(reference_models_debug[0]) if reference_models_debug else 0
    ).astype(np.uint8).tolist()
    results["cluster_palette_rgb"] = [
        color[::-1] for color in results["cluster_palette_bgr"]
    ]
    results["cluster_map_raw"] = raw_maps
    results["cluster_map_display_raw"] = display_raw_maps
    results["cluster_display_memberships"] = display_memberships_all
    results["cluster_map_reference"] = references
    results["cluster_map_label_values"] = label_values_all
    results["cluster_map_selected_labels"] = labels
    results["cluster_map_reference_label"] = reference_label_value
    results["cluster_map_components_raw"] = component_maps_all
    results["cluster_map_component_images"] = component_images_all
    results["cluster_map_component_info"] = component_info_all
    results["cluster_map_remainder_raw"] = remainder_maps
    results["cluster_map_class_label_maps"] = classifier_label_maps
    results["matlab_membership_raw"] = matlab_raw_all
    results["matlab_membership_normalized"] = matlab_normalized_all
    results["matlab_component_medians"] = matlab_medians_all
    results["matlab_component_iqrs"] = matlab_iqrs_all
    results["matlab_component_denoms"] = matlab_denoms_all
    results["matlab_component_statistics"] = matlab_statistics_all
    results["matlab_initial_centroids"] = matlab_initial_all
    results["matlab_sorted_centroids"] = matlab_sorted_all
    results["reference_initial_centroids"] = matlab_initial_all
    results["reference_initial_centroids_uint8"] = [
        [float(value * 255.0) for value in centroids]
        for centroids in matlab_initial_all
    ]
    results["reference_sorted_centroids"] = matlab_sorted_all
    results["reference_sorted_centroids_uint8"] = [
        [float(value * 255.0) for value in centroids]
        for centroids in matlab_sorted_all
    ]
    results["initial_centroid_source"] = centroid_sources
    results["reference_crop_debug"] = reference_crop_debug
    results["reference_crop_count"] = [
        item["reference_crop_count"] for item in reference_crop_debug
    ]
    results["reference_source_id"] = [
        item["reference_source_id"] for item in reference_crop_debug
    ]
    results["reference_source_found"] = [
        item["reference_source_found"] for item in reference_crop_debug
    ]
    results["empty_upstream_labels"] = empty_upstream_labels
    results["cluster_reference_mapping"] = reference_cluster_mappings
    results["cluster_label_to_original_reference"] = [
        {item["cluster_label"]: item["reference_original_index"] for item in mapping}
        for mapping in reference_cluster_mappings
    ]
    results["original_reference_to_cluster_label"] = [
        {item["reference_original_index"]: item["cluster_label"] for item in mapping}
        for mapping in reference_cluster_mappings
    ]
    results["cluster_label_to_reference_name"] = [
        {item["cluster_label"]: item["reference_name"] for item in mapping}
        for mapping in reference_cluster_mappings
    ]
    results["cluster_brightness_order"] = [
        [
            item["cluster_label"]
            for item in sorted(mapping, key=lambda value: value["cluster_center_brightness"])
        ]
        for mapping in reference_cluster_mappings
    ]
    results["membership_component_statistics"] = matlab_statistics_all
    results["reference_sequence_order"] = [
        item["reference_sequence_order"] for item in reference_crop_debug
    ]
    # Preserve the left panel even when loading an older cached k-means result.
    results["kmeans_labeled_images"] = labeled if isinstance(labeled, list) else list(data.get("images") or [])
    data["images"] = output_images
    data["count"] = len(output_images)
    data.setdefault("meta", {})["cluster_reference_map"] = {
        "selected_labels": labels,
        "reference_label": reference_label_value,
        "center_mode": center_mode,
        "map_multiplier": multiplier,
        "remainder_name": str(remainder_name or "Maradék"),
        "remainder_display_multiplier": remainder_visual_multiplier,
        "remainder_invert": bool(remainder_invert),
        "colormap": colormap,
        "invert": bool(invert),
        "algorithm": "ordered_hybrid_membership_from_upstream_kmeans_labels",
        "initial_assignment": "upstream_kmeans_label_maps",
        "membership": "ordered_shoulders_with_gaussian_middle",
        "spread_estimator": "IQR_for_middle_components",
        "initial_centroid_source": "reference_sequence_or_reference_crops",
        "soft_manual_thresholds": False,
        "legacy_hard_thresholds": False,
        "manual_spread_scaling": False,
        "normalization": "per_pixel_sum_to_one",
        "hard_output": "upstream_kmeans_label_maps",
        "membership_method": "hybrid_shoulders_gaussian_middle",
        "extreme_cluster_membership": "monotonic_shoulders",
        "middle_cluster_membership": "gaussian_median_iqr",
        "primary_similarity": "absolute_raw_membership",
        "per_pixel_normalization_used_for_primary": False,
        "display_spatial_smoothing": "gaussian_membership_only",
        "display_spatial_sigma": DISPLAY_SPATIAL_SIGMA,
        "manual_thresholds": False,
        "soft_normalization": "per_pixel_sum_to_one",
        "hard_label_mask": False,
        "soft_membership": True,
        "component_weights": [1.0, 1.0, 1.0],
        "intensity_scale": "0_to_1",
        "eps_iqr": MATLAB_EPS_IQR,
        "reference_models": reference_models_debug,
        "visualization": "soft_membership_blend",
        "primary_visualization": "heatmap",
        "overlay_is_primary": False,
        "cluster_map_contains_source_pixels": False,
        "palette_color_order": "BGR",
        "soft_spatial_smoothing": True,
        "soft_spatial_sigma": SOFT_SPATIAL_SIGMA,
        "hard_boundaries_drawn": False,
        "heatmap_mode": "optional_debug",
    }
    data.setdefault("history", []).append("cluster_reference_map")
    return data
