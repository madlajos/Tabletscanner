"""Build false-colour BGR images from consecutive three-image groups."""

import cv2
import numpy as np


_CHANNEL_INDEX = {"B": 0, "G": 1, "R": 2}


def _extract_channel(image, channel):
    array = np.asarray(image)
    if array.ndim == 2:
        return array
    if array.ndim != 3 or array.shape[2] not in (1, 3, 4):
        return None
    if array.shape[2] == 1:
        return array[:, :, 0]
    bgr = array[:, :, :3]
    if channel == "GRAY":
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    index = _CHANNEL_INDEX.get(channel)
    return bgr[:, :, index] if index is not None else None


def _shift_channel(channel, offset_x, offset_y):
    """Translate a channel without wrapping; newly exposed pixels are black."""
    height, width = channel.shape[:2]
    shifted = np.zeros_like(channel)
    offset_x = int(offset_x)
    offset_y = int(offset_y)

    if abs(offset_x) >= width or abs(offset_y) >= height:
        return shifted

    source_x0 = max(0, -offset_x)
    source_x1 = min(width, width - offset_x)
    source_y0 = max(0, -offset_y)
    source_y1 = min(height, height - offset_y)
    target_x0 = max(0, offset_x)
    target_y0 = max(0, offset_y)
    target_x1 = target_x0 + (source_x1 - source_x0)
    target_y1 = target_y0 + (source_y1 - source_y0)
    shifted[target_y0:target_y1, target_x0:target_x1] = channel[
        source_y0:source_y1, source_x0:source_x1
    ]
    return shifted


def _scale_channel(channel, scale_percent):
    """Scale a channel around its image centre while preserving canvas size."""
    height, width = channel.shape[:2]
    scale = float(scale_percent) / 100.0
    centre = ((width - 1) / 2.0, (height - 1) / 2.0)
    matrix = cv2.getRotationMatrix2D(centre, 0, scale)
    return cv2.warpAffine(
        channel,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _multiply_channel(channel, multiplier):
    """Apply an intensity multiplier without overflowing uint8 channels."""
    return np.clip(channel.astype(np.float32) * float(multiplier), 0, 255).astype(np.uint8)


def create_pseudo_image(
    data,
    blue_source="1-B",
    green_source="1-G",
    red_source="1-R",
    blue_scale_percent=100.0,
    green_scale_percent=100.0,
    red_scale_percent=100.0,
    blue_multiplier=1.0,
    green_multiplier=1.0,
    red_multiplier=1.0,
    move_blue=False,
    move_green=False,
    move_red=False,
    offset_x=0,
    offset_y=0,
    **_,
):
    """Apply one three-image channel recipe to every consecutive image trio."""
    if not isinstance(data, dict) or data.get("error"):
        return data

    images = data.get("images") or []
    if not images:
        data["error"] = "E2150"
        return data
    if len(images) % 3:
        data["error"] = "E2154"
        return data

    selectors_transforms = (
        (blue_source, blue_scale_percent, blue_multiplier, bool(move_blue)),
        (green_source, green_scale_percent, green_multiplier, bool(move_green)),
        (red_source, red_scale_percent, red_multiplier, bool(move_red)),
    )
    parsed_selectors = []
    for selector, scale_percent, multiplier, should_move in selectors_transforms:
        try:
            image_number, channel = str(selector).upper().split("-", 1)
            relative_index = int(image_number) - 1
            if not 0 <= relative_index < 3 or channel not in (*_CHANNEL_INDEX, "GRAY"):
                raise ValueError
        except (TypeError, ValueError):
            data["error"] = "E2153"
            return data
        parsed_selectors.append((relative_index, channel, scale_percent, multiplier, should_move))

    output_images = []
    for group_start in range(0, len(images), 3):
        output_channels = []
        for relative_index, channel, scale_percent, multiplier, should_move in parsed_selectors:
            extracted = _extract_channel(images[group_start + relative_index], channel)
            if extracted is None:
                data["error"] = "E2151"
                return data
            if output_channels and extracted.shape[:2] != output_channels[0].shape[:2]:
                data["error"] = "E2152"
                return data
            transformed = _multiply_channel(_scale_channel(extracted, scale_percent), multiplier)
            output_channels.append(
                _shift_channel(transformed, offset_x, offset_y) if should_move else transformed
            )
        output_images.append(cv2.merge(output_channels))

    source_paths = data.get("paths") or []
    data["_original_paths"] = list(source_paths)
    data["images"] = output_images
    data["count"] = len(output_images)
    data["paths"] = source_paths[::3]
    data.setdefault("meta", {})["pseudo_image"] = {
        "blue_source": blue_source,
        "green_source": green_source,
        "red_source": red_source,
        "scale_percent": {
            "blue": float(blue_scale_percent),
            "green": float(green_scale_percent),
            "red": float(red_scale_percent),
        },
        "multiplier": {
            "blue": float(blue_multiplier),
            "green": float(green_multiplier),
            "red": float(red_multiplier),
        },
        "moving_layers": {
            "blue": bool(move_blue),
            "green": bool(move_green),
            "red": bool(move_red),
        },
        "offset": {"x": int(offset_x), "y": int(offset_y)},
        "group_size": 3,
        "group_count": len(output_images),
    }
    data.setdefault("history", []).append("create_pseudo_image")
    return data
