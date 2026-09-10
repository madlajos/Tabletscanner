"""Manual and automatic similarity alignment against a reference image branch."""
from __future__ import annotations

import cv2
import numpy as np


def _gray(image: np.ndarray) -> np.ndarray:
    return image if image.ndim == 2 else cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)


def _object_mask(image: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(_gray(image), (5, 5), 0)
    _, normal = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inverted = cv2.bitwise_not(normal)
    mask = normal if cv2.countNonZero(normal) <= cv2.countNonZero(inverted) else inverted
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("no object")
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 25:
        raise ValueError("object too small")
    result = np.zeros_like(mask)
    cv2.drawContours(result, [largest], -1, 255, cv2.FILLED)
    return result


def _manual_matrix(width: int, height: int, angle: float, scale: float,
                   offset_x: float, offset_y: float) -> np.ndarray:
    matrix = cv2.getRotationMatrix2D(((width - 1) / 2.0, (height - 1) / 2.0), angle, scale)
    matrix[:, 2] += (offset_x, offset_y)
    return matrix.astype(np.float32)


def _contour_matrix(moving: np.ndarray, reference: np.ndarray, allow_scale: bool) -> np.ndarray:
    moving_rect = cv2.minAreaRect(cv2.findNonZero(_object_mask(moving)))
    reference_rect = cv2.minAreaRect(cv2.findNonZero(_object_mask(reference)))

    def long_axis(rect):
        (_, _), (width, height), angle = rect
        return (angle + 90.0, height) if height > width else (angle, width)

    moving_angle, moving_length = long_axis(moving_rect)
    reference_angle, reference_length = long_axis(reference_rect)
    scale = reference_length / max(moving_length, 1e-6) if allow_scale else 1.0
    matrix = cv2.getRotationMatrix2D(moving_rect[0], reference_angle - moving_angle, scale)
    mapped_center = matrix @ np.array([*moving_rect[0], 1.0])
    matrix[:, 2] += np.asarray(reference_rect[0]) - mapped_center
    return matrix.astype(np.float32)


def _automatic_matrix(moving: np.ndarray, reference: np.ndarray, allow_scale: bool,
                      refine: bool) -> tuple[np.ndarray, float | None]:
    matrix = _contour_matrix(moving, reference, allow_scale)
    if not refine:
        return matrix, None
    try:
        inverse_initial = cv2.invertAffineTransform(matrix)
        score, inverse_matrix = cv2.findTransformECC(
            _gray(reference).astype(np.float32) / 255.0,
            _gray(moving).astype(np.float32) / 255.0,
            inverse_initial,
            cv2.MOTION_AFFINE if allow_scale else cv2.MOTION_EUCLIDEAN,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-6),
            None,
            5,
        )
        return cv2.invertAffineTransform(inverse_matrix), float(score)
    except cv2.error:
        return matrix, None


def align_images(data: dict, reference_images: list, *, automatic: bool,
                 angle: float = 0.0, scale: float = 1.0,
                 offset_x: float = 0.0, offset_y: float = 0.0,
                 allow_scale: bool = True, refine: bool = True,
                 border_value: int = 0, reference_opacity: float = 0.5) -> dict:
    images = data.get("images") if isinstance(data, dict) else None
    if not isinstance(images, list) or not images:
        data["error"] = "E4101"
        return data
    if not isinstance(reference_images, list) or not reference_images:
        data["error"] = "E4102"
        return data
    aligned, transforms, manual_previews = [], [], []
    try:
        for index, moving in enumerate(images):
            reference = reference_images[min(index, len(reference_images) - 1)]
            if automatic:
                matrix, score = _automatic_matrix(moving, reference, allow_scale, refine)
            else:
                matrix = _manual_matrix(moving.shape[1], moving.shape[0], angle, scale, offset_x, offset_y)
                score = None
            output = cv2.warpAffine(
                moving, matrix, (reference.shape[1], reference.shape[0]),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                borderValue=(border_value,) * (3 if moving.ndim == 3 else 1),
            )
            aligned.append(output)
            transforms.append({"matrix": matrix.tolist(), "score": score})
            if not automatic:
                preview_reference = reference
                if output.ndim == 2 and reference.ndim == 3:
                    preview_reference = _gray(reference)
                elif output.ndim == 3 and reference.ndim == 2:
                    preview_reference = cv2.cvtColor(reference, cv2.COLOR_GRAY2BGR)
                opacity = float(np.clip(reference_opacity, 0.0, 1.0))
                manual_previews.append(cv2.addWeighted(output, 1.0 - opacity, preview_reference, opacity, 0.0))
    except (ValueError, TypeError, cv2.error):
        data["error"] = "E4103"
        return data
    data["images"] = aligned
    data["count"] = len(aligned)
    data.setdefault("results", {})["image_alignment_transforms"] = transforms
    if manual_previews:
        # Used only by the preview endpoint; the primary image stream remains
        # the cleanly transformed moving image.
        data["results"]["manual_alignment_previews"] = manual_previews
    data.setdefault("meta", {})["image_alignment"] = {"mode": "automatic" if automatic else "manual", "transforms": transforms}
    data.setdefault("history", []).append("automatic_image_alignment" if automatic else "manual_image_alignment")
    return data
