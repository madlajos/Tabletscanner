"""Fuse a focus bracket into one extended-depth-of-field image."""

import cv2
import numpy as np


def _gray_float(image):
    array = np.asarray(image)
    if array.ndim == 2:
        gray = array
    elif array.ndim == 3 and array.shape[2] in (3, 4):
        gray = cv2.cvtColor(array[:, :, :3], cv2.COLOR_BGR2GRAY)
    else:
        return None
    return gray.astype(np.float32) / 255.0


def _as_float_image(image):
    array = np.asarray(image)
    if array.ndim == 2:
        return array.astype(np.float32)[:, :, None]
    return array[:, :, :3].astype(np.float32)


def _align_to_reference(image, reference_gray, iterations):
    moving_gray = _gray_float(image)
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                max(10, int(iterations)), 1e-5)
    score, warp = cv2.findTransformECC(
        reference_gray, moving_gray, warp, cv2.MOTION_AFFINE, criteria, None, 5
    )
    aligned = cv2.warpAffine(
        image, warp, (reference_gray.shape[1], reference_gray.shape[0]),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REFLECT101,
    )
    return aligned, float(score), warp.tolist()


def focus_stack(data, focus_radius=9, blend_radius=7, align_images=True,
                alignment_iterations=80):
    """Combine two or more registered images using local focus measures."""
    if not isinstance(data, dict) or data.get("error"):
        return data
    if data.get("_inline_capture_input"):
        from pipeline_live_inputs import live_focus_groups
        groups = live_focus_groups(data)
        if data.get("error"):
            return data
        output_images, output_paths, index_maps, metrics = [], [], [], []
        for indices, metadata in groups:
            subset = {"images": [data["images"][index] for index in indices],
                      "paths": [data["paths"][index] for index in indices],
                      "results": {}, "meta": {}, "history": [], "error": None}
            result = focus_stack(subset, focus_radius, blend_radius, align_images, alignment_iterations)
            if result.get("error"):
                data["error"] = result["error"]
                data["error_detail"] = f"{metadata['tablet_id']}, {metadata['wavelength']}"
                return data
            output_images.extend(result["images"])
            output_paths.extend(result["paths"])
            index_maps.append(result["results"]["focus_index_map"])
            metrics.append(result["results"]["focus_stack_metrics"])
        data.update(images=output_images, paths=output_paths, count=len(output_images),
                    image_metadata=[metadata for _, metadata in groups])
        data.setdefault("results", {})["focus_index_map"] = index_maps
        data["results"]["focus_stack_metrics"] = metrics
        data.setdefault("meta", {})["focus_stack"] = {"group_count": len(groups), "groups": metrics}
        data.setdefault("history", []).append("focus_stack")
        return data
    images = data.get("images") or []
    if len(images) < 2:
        data["error"] = "E2170"
        return data

    arrays = [np.asarray(image) for image in images]
    grays = [_gray_float(image) for image in arrays]
    if any(gray is None for gray in grays):
        data["error"] = "E2171"
        return data
    if len({image.shape for image in arrays}) != 1:
        data["error"] = "E2172"
        return data

    aligned = list(arrays)
    alignment = []
    reference_index = len(arrays) // 2
    reference_gray = grays[reference_index]
    if align_images:
        for index, image in enumerate(arrays):
            if index == reference_index:
                alignment.append({"index": index, "score": 1.0,
                                  "matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]})
                continue
            try:
                aligned[index], score, matrix = _align_to_reference(
                    image, reference_gray, alignment_iterations
                )
                alignment.append({"index": index, "score": score, "matrix": matrix})
            except cv2.error:
                alignment.append({"index": index, "score": None, "matrix": None})

    focus_kernel = max(1, int(focus_radius)) | 1
    scores = []
    for image in aligned:
        gray = _gray_float(image)
        laplacian = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
        scores.append(cv2.GaussianBlur(laplacian, (focus_kernel, focus_kernel), 0))
    score_stack = np.stack(scores, axis=0)
    winners = np.argmax(score_stack, axis=0)

    blend_kernel = max(1, int(blend_radius)) | 1
    weights = []
    for index in range(len(aligned)):
        mask = np.float32(winners == index)
        if blend_kernel > 1:
            mask = cv2.GaussianBlur(mask, (blend_kernel, blend_kernel), 0)
        weights.append(mask)
    weights = np.stack(weights, axis=0)
    weights /= np.maximum(np.sum(weights, axis=0, keepdims=True), 1e-6)

    float_images = np.stack([_as_float_image(image) for image in aligned], axis=0)
    output = np.sum(float_images * weights[:, :, :, None], axis=0)
    output = np.clip(output, 0, 255).astype(np.uint8)
    if arrays[0].ndim == 2:
        output = output[:, :, 0]

    metrics = {
        "source_count": len(images),
        "reference_index": reference_index,
        "alignment_enabled": bool(align_images),
        "alignment": sorted(alignment, key=lambda row: row["index"]),
    }
    results = data.setdefault("results", {})
    results["focus_index_map"] = np.uint8(
        np.round(winners * (255.0 / max(1, len(images) - 1)))
    )
    results["focus_stack_metrics"] = metrics
    data["images"] = [output]
    data["count"] = 1
    data["paths"] = (data.get("paths") or [])[:1]
    data.setdefault("meta", {})["focus_stack"] = metrics
    data.setdefault("history", []).append("focus_stack")
    return data
