"""Align image tones (and optionally colours) to reference crops."""

import cv2
import numpy as np


def _as_uint8(image):
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        if arr.dtype.kind == "f" and arr.size and float(np.nanmax(arr)) <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _as_bgr(image):
    arr = _as_uint8(image)
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    if arr.ndim == 3 and arr.shape[2] == 1:
        return cv2.cvtColor(arr[..., 0], cv2.COLOR_GRAY2BGR)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        return arr[..., :3]
    return None


def _median_lightness(image):
    bgr = _as_bgr(image)
    if bgr is None or not bgr.size:
        return None
    return float(np.median(cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[..., 0]))


def _strictly_increasing(values, epsilon=1e-3):
    """Return control points that are strictly increasing inside 0..255."""
    points = np.asarray(values, dtype=np.float32)
    offsets = np.arange(len(points), dtype=np.float32) * epsilon
    lower = epsilon + offsets
    upper = (255.0 - epsilon) - offsets[::-1]
    points = np.clip(points, lower, upper)
    for index in range(1, len(points)):
        points[index] = max(points[index], points[index - 1] + epsilon)
    return points


def _reference_lab_medians(references):
    medians = []
    for reference in references:
        bgr = _as_bgr(reference)
        if bgr is None or not bgr.size:
            continue
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).reshape(-1, 3)
        medians.append(np.median(lab, axis=0).astype(np.float32))
    if not medians:
        return None
    values = np.asarray(medians, dtype=np.float32)
    return values[np.argsort(values[:, 0], kind="stable")]


def _reference_tone_map(image, references, strength=1.0,
                        percentiles=(20.0, 50.0, 80.0), match_color=False):
    """Map robust source tone positions to ordered reference crop medians."""
    original = _as_uint8(image)
    source = _as_bgr(original)
    targets = _reference_lab_medians(references)
    if source is None or targets is None:
        return None
    source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
    lightness = source_lab[..., 0].copy()
    count = len(targets)
    requested = percentiles if count == 3 else np.linspace(10.0, 90.0, count)
    if len(requested) != count:
        requested = np.linspace(10.0, 90.0, count)
    source_points = _strictly_increasing(np.percentile(lightness, requested))
    x = np.concatenate(([0.0], source_points, [255.0])).astype(np.float32)
    target_l = targets[:, 0]
    y = np.concatenate(([target_l[0]], target_l, [target_l[-1]])).astype(np.float32)
    mapped_l = np.interp(lightness.reshape(-1), x, y).reshape(lightness.shape)
    amount = min(max(float(strength), 0.0), 1.0)
    source_lab[..., 0] = lightness + amount * (mapped_l - lightness)

    if match_color and original.ndim == 3 and original.shape[2] >= 3:
        for channel in (1, 2):
            target_channel = np.concatenate(
                ([targets[0, channel]], targets[:, channel], [targets[-1, channel]])
            )
            mapped = np.interp(lightness.reshape(-1), x, target_channel).reshape(lightness.shape)
            source_lab[..., channel] += amount * (mapped - source_lab[..., channel])

    corrected = cv2.cvtColor(np.clip(source_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    if original.ndim == 2:
        return cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)
    if original.ndim == 3 and original.shape[2] == 1:
        return cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)[..., None]
    return corrected


def _lab_histograms(image):
    lab = cv2.cvtColor(_as_bgr(image), cv2.COLOR_BGR2LAB)
    histograms = []
    for channel in range(3):
        hist = np.bincount(lab[..., channel].reshape(-1), minlength=256).astype(np.float64)
        peak = float(hist.max())
        histograms.append((hist / peak if peak > 0 else hist).tolist())
    return histograms


def reference_color_align(data, strength=1.0, output_dark=0.0, output_light=255.0,
                          reference_branch="auto", mode="location_scale",
                          source_id="", source_label=""):
    if data.get("error") is not None:
        return data
    images = data.get("images")
    results = data.get("results") or {}
    strength = min(max(float(strength), 0.0), 1.0)
    if mode not in {"location", "location_scale"}:
        mode = "location_scale"
    if reference_branch != "auto":
        crop_rows = results.get("reference_crops")
        # Per-image reference overrides deliberately expose the same merged
        # reference list for every source image. Consume that canonical list
        # once instead of counting/showing it once per image.
        if (
            isinstance(crop_rows, list) and len(crop_rows) > 1
            and isinstance(crop_rows[0], list)
            and all(row is crop_rows[0] for row in crop_rows[1:])
        ):
            crop_rows = [crop_rows[0]]
        crops = [
            crop
            for row in crop_rows if isinstance(row, list)
            for crop in row
            if _as_bgr(crop) is not None
        ] if isinstance(crop_rows, list) else []
        if not crops:
            data["error"] = "E3545"
            return data
        corrected_images = [
            _reference_tone_map(image, crops, strength=strength,
                                percentiles=(20.0, 50.0, 80.0),
                                match_color=(mode == "location_scale"))
            for image in images
        ]
        if any(image is None for image in corrected_images):
            data["error"] = "E3542"
            return data
        data["images"] = corrected_images
        data["count"] = len(corrected_images)
        results["reference_color_align_source_images"] = list(images)
        results["reference_color_align_aligned_images"] = corrected_images
        results["reference_color_align_histograms"] = {
            "source": _lab_histograms(images[0]),
            "reference": _lab_histograms(np.concatenate(
                [_as_bgr(crop).reshape(-1, 1, 3) for crop in crops], axis=0
            )),
            "aligned": _lab_histograms(corrected_images[0]),
            "channels": ["L", "A", "B"],
        }
        corrected_rows = crop_rows
        results["reference_crops"] = crop_rows
        results["reference_color_aligned_crops"] = crop_rows
        results["reference_color_align_ranges"] = [{
            "reference_branch": reference_branch,
            "reference_image_index": 0,
            "reference_crop_count": len(crops),
            "mode": mode,
        } for _ in corrected_images]
        if source_id and corrected_rows is not None:
            results.setdefault("reference_sources", {})[source_id] = {
                "id": source_id,
                "label": source_label or "Reference color alignment",
                "type": "reference_color_align",
                "reference_crops": corrected_rows,
                "reference_crop_squares": results.get("reference_crop_squares"),
            }
        data["results"] = results
        data.setdefault("meta", {})["reference_color_align"] = {
            "strength": strength, "mode": mode, "reference_branch": reference_branch,
        }
        data.setdefault("history", []).append("reference_color_align")
        return data
    crop_rows = results.get("reference_crops")
    square_rows = results.get("reference_crop_squares")
    if not isinstance(images, list) or not images or not isinstance(crop_rows, list) or not crop_rows:
        data["error"] = "E3541"
        return data

    corrected_images, corrected_rows, range_info = [], [], []
    for target_index, image in enumerate(images):
        row_index = min(target_index, len(crop_rows) - 1)
        crops = crop_rows[row_index] if isinstance(crop_rows[row_index], list) else []
        valid = []
        for crop_index, crop in enumerate(crops):
            lightness = _median_lightness(crop)
            if lightness is not None:
                valid.append((crop_index, crop, lightness))
        if not valid:
            data["error"] = "E3541"
            return data

        darkest = min(valid, key=lambda item: item[2])
        lightest = max(valid, key=lambda item: item[2])
        input_dark, input_light = float(darkest[2]), float(lightest[2])
        corrected_image = _reference_tone_map(
            image, [item[1] for item in valid], strength=strength,
            percentiles=(20.0, 50.0, 80.0),
            match_color=(mode == "location_scale"),
        )
        if corrected_image is None:
            data["error"] = "E3542"
            return data
        corrected_images.append(corrected_image)
        corrected_rows.append(crops)
        range_info.append({
            "darkest_reference_index": int(darkest[0]),
            "lightest_reference_index": int(lightest[0]),
            "input_dark_lab_l": input_dark,
            "input_light_lab_l": input_light,
            "output_dark_lab_l": input_dark,
            "output_light_lab_l": input_light,
            "source_percentiles": ([20.0, 50.0, 80.0] if len(valid) == 3 else
                                   np.linspace(10.0, 90.0, len(valid)).tolist()),
        })

    data["images"] = corrected_images
    data["count"] = len(corrected_images)
    results["reference_crops"] = corrected_rows
    results["reference_color_aligned_crops"] = corrected_rows
    results["reference_color_align_ranges"] = range_info
    if source_id:
        results.setdefault("reference_sources", {})[source_id] = {
            "id": source_id,
            "label": source_label or "Reference range alignment",
            "type": "reference_color_align",
            "reference_crops": corrected_rows,
            "reference_crop_squares": square_rows,
        }
    data["results"] = results
    data.setdefault("meta", {})["reference_color_align"] = {
        "strength": strength, "output_dark": output_dark,
        "output_light": output_light, "mode": mode, "ranges": range_info,
    }
    data.setdefault("history", []).append("reference_color_align")
    return data
