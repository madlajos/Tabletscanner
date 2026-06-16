"""Scale bar annotation helper for image overlays."""

from __future__ import annotations

from typing import Dict, Tuple

import cv2
import numpy as np


_FONT_MAP = {
    "sans": cv2.FONT_HERSHEY_SIMPLEX,
    "serif": cv2.FONT_HERSHEY_TRIPLEX,
    "mono": cv2.FONT_HERSHEY_PLAIN,
    "complex": cv2.FONT_HERSHEY_COMPLEX,
    "script": cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
}


def _clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(upper, value))


def _nice_scale_length_mm(target_mm: float) -> float:
    if target_mm <= 0:
        return 0.0

    nice_values = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
    return min(nice_values, key=lambda value: abs(target_mm - value))


def _parse_rgb_color(value: str, fallback: Tuple[int, int, int]) -> Tuple[int, int, int]:
    colors = {
        "white": (255, 255, 255),
        "black": (0, 0, 0),
        "yellow": (0, 255, 255),
    }
    if not isinstance(value, str):
        return fallback
    return colors.get(value.strip().lower(), fallback)


def _resolve_font_scale(font_size: int) -> float:
    return max(0.25, float(font_size) / 24.0)


def _format_length(value_mm: float, unit: str) -> str:
    unit_normalized = str(unit or "mm").strip().lower()
    if unit_normalized == "cm":
        value = value_mm / 10.0
    elif unit_normalized == "um":
        value = value_mm * 1000.0
    else:
        value = value_mm

    return f"{round(value)} {unit_normalized}"


def _draw_single_scale_bar(image: np.ndarray, params: Dict) -> np.ndarray:
    if image is None or image.size == 0:
        return image

    px_per_mm = float(params.get("pixels_per_mm", params.get("px_per_mm", 0.0)) or 0.0)
    if px_per_mm <= 0:
        return image.copy()

    bar_length_mm = float(params.get("bar_length_mm", 0.0) or 0.0)
    if bar_length_mm <= 0:
        target_px = max(0.0, min(image.shape[1] * 0.25, image.shape[1] - 80.0))
        target_mm = target_px / px_per_mm if px_per_mm > 0 else 0.0
        bar_length_mm = _nice_scale_length_mm(target_mm)
        if bar_length_mm <= 0:
            bar_length_mm = 10.0

    font_name = str(params.get("font_family", "sans"))
    font = _FONT_MAP.get(font_name, cv2.FONT_HERSHEY_SIMPLEX)
    font_size = max(8, int(params.get("font_size", 24) or 24))
    font_scale = _resolve_font_scale(font_size)
    font_thickness = max(1, int(params.get("font_thickness", 1) or 1))
    bar_thickness = max(1, int(params.get("bar_thickness", 3) or 3))
    padding_value = params.get("box_padding", 14)
    padding = max(0, int(14 if padding_value is None else padding_value))
    text_gap_value = params.get("text_gap", 48)
    text_gap = max(0, int(16 if text_gap_value is None else text_gap_value))
    background_alpha_value = params.get("background_opacity", 0.55)
    background_alpha = float(0.55 if background_alpha_value is None else background_alpha_value)
    background_alpha = max(0.0, min(1.0, background_alpha))
    show_background = bool(params.get("show_background", False))

    foreground = _parse_rgb_color(str(params.get("bar_color", "white")), (255, 255, 255))
    text_color = _parse_rgb_color(str(params.get("text_color", "white")), (255, 255, 255))
    background_color = _parse_rgb_color(str(params.get("background_color", "black")), (0, 0, 0))

    label_text = _format_length(bar_length_mm, params.get("label_unit", params.get("bar_length_unit", "mm")))

    (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)
    bar_length_px = max(1, int(round(bar_length_mm * px_per_mm)))
    box_width = max(bar_length_px, text_w) + 2 * padding
    box_height = padding + text_h + text_gap + max(bar_thickness * 2, 12) + padding + baseline

    requested_x_value = params.get("position_x", -1)
    requested_y_value = params.get("position_y", -1)
    requested_x = int(-1 if requested_x_value is None else requested_x_value)
    requested_y = int(-1 if requested_y_value is None else requested_y_value)
    image_h, image_w = image.shape[:2]
    if requested_x < 0 or requested_y < 0:
        box_x = max(0, image_w - box_width - 20)
        box_y = max(0, image_h - box_height - 20)
    else:
        box_x = _clamp(requested_x, 0, max(0, image_w - box_width))
        box_y = _clamp(requested_y, 0, max(0, image_h - box_height))

    center_x = box_x + box_width // 2
    bar_start_x = center_x - bar_length_px // 2
    bar_end_x = bar_start_x + bar_length_px
    bar_y = box_y + padding + bar_thickness
    text_baseline_y = bar_y + text_gap + text_h

    overlay = image.copy()

    if show_background:
        x0 = max(0, min(image_w, box_x))
        y0 = max(0, min(image_h, box_y))
        x1 = max(0, min(image_w, box_x + box_width))
        y1 = max(0, min(image_h, box_y + box_height))
        if x1 > x0 and y1 > y0:
            region = overlay[y0:y1, x0:x1].copy()
            bg = np.zeros_like(region)
            bg[:] = background_color
            overlay[y0:y1, x0:x1] = cv2.addWeighted(bg, background_alpha, region, 1.0 - background_alpha, 0.0)

    cv2.line(overlay, (bar_start_x, bar_y), (bar_end_x, bar_y), foreground, bar_thickness, cv2.LINE_AA)
    cap_radius = max(2, int(round(bar_thickness * 0.6)))
    for cap_x in (bar_start_x, bar_end_x):
        cv2.circle(overlay, (cap_x, bar_y), cap_radius, foreground, -1, cv2.LINE_AA)

    cv2.putText(
        overlay,
        label_text,
        (center_x - text_w // 2, text_baseline_y),
        font,
        font_scale,
        text_color,
        font_thickness,
        cv2.LINE_AA,
    )

    return overlay


def scale_bar_overlay(data: dict, **params) -> dict:
    """Draw a configurable scale bar overlay on every image in the pipeline."""
    images = data.get("images") or []
    if not images:
        return data

    updated_images = []
    for image in images:
        if image is None:
            updated_images.append(None)
            continue
        updated_images.append(_draw_single_scale_bar(image, params))

    data["images"] = updated_images
    data["count"] = len(updated_images)
    data.setdefault("meta", {})["scale_bar_overlay_config"] = {
        "pixels_per_mm": float(params.get("pixels_per_mm", params.get("px_per_mm", 0.0)) or 0.0),
        "bar_length_mm": float(params.get("bar_length_mm", 0.0) or 0.0),
        "position_x": int(params.get("position_x", -1) or -1),
        "position_y": int(params.get("position_y", -1) or -1),
        "font_family": str(params.get("font_family", "sans")),
        "font_size": int(params.get("font_size", 24) or 24),
    }
    return data