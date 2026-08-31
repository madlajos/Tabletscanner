"""Build one false-colour BGR image from channels of two input images."""

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


def create_pseudo_image(
    data,
    blue_source="1-B",
    green_source="1-G",
    red_source="1-R",
    move_blue=False,
    move_green=False,
    move_red=False,
    offset_x=0,
    offset_y=0,
    **_,
):
    """Compose the output B, G and R planes from the first two loaded images."""
    if not isinstance(data, dict) or data.get("error"):
        return data

    images = data.get("images") or []
    if not images:
        data["error"] = "E2150"
        return data

    output_channels = []
    selectors_and_move_flags = (
        (blue_source, bool(move_blue)),
        (green_source, bool(move_green)),
        (red_source, bool(move_red)),
    )
    for selector, should_move in selectors_and_move_flags:
        try:
            image_number, channel = str(selector).upper().split("-", 1)
            image_index = int(image_number) - 1
            if not 0 <= image_index < len(images) or channel not in (*_CHANNEL_INDEX, "GRAY"):
                raise ValueError
        except (TypeError, ValueError):
            data["error"] = "E2153"
            return data

        extracted = _extract_channel(images[image_index], channel)
        if extracted is None:
            data["error"] = "E2151"
            return data
        if output_channels and extracted.shape[:2] != output_channels[0].shape[:2]:
            data["error"] = "E2152"
            return data
        output_channels.append(
            _shift_channel(extracted, offset_x, offset_y) if should_move else extracted
        )

    source_paths = data.get("paths") or []
    data["_original_paths"] = list(source_paths)
    data["images"] = [cv2.merge(output_channels)]
    data["count"] = 1
    data["paths"] = source_paths[:1]
    data.setdefault("meta", {})["pseudo_image"] = {
        "blue_source": blue_source,
        "green_source": green_source,
        "red_source": red_source,
        "moving_layers": {
            "blue": bool(move_blue),
            "green": bool(move_green),
            "red": bool(move_red),
        },
        "offset": {"x": int(offset_x), "y": int(offset_y)},
    }
    data.setdefault("history", []).append("create_pseudo_image")
    return data
