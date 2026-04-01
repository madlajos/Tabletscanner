import cv2
import numpy as np
from proc_elements.cache_utils import cached_cvtColor, cached_calcHist


def _get_space_config():
    return {
        "BGR": {
            "channels": ["B", "G", "R"],
            "convert": None,
            "ranges": {
                "B": (0, 255),
                "G": (0, 255),
                "R": (0, 255),
            },
        },
        "HSV": {
            "channels": ["H", "S", "V"],
            "convert": cv2.COLOR_BGR2HSV,
            "ranges": {
                "H": (0, 179),
                "S": (0, 255),
                "V": (0, 255),
            },
        },
        "LAB": {
            "channels": ["L", "A", "B"],
            "convert": cv2.COLOR_BGR2LAB,
            "ranges": {
                "L": (0, 255),
                "A": (0, 255),
                "B": (0, 255),
            },
        },
        "GRAY": {
            "channels": ["GRAY"],
            "convert": cv2.COLOR_BGR2GRAY,
            "ranges": {
                "GRAY": (0, 255),
            },
        },
    }


def _validate_thresholds(space, thresholds, config):
    required_channels = config[space]["channels"]

    if thresholds is None:
        return False, "E2205"

    for ch in required_channels:
        if ch not in thresholds:
            return False, "E2206"

        value = thresholds[ch]

        if not isinstance(value, (list, tuple)) or len(value) != 2:
            return False, "E2207"

        ch_min, ch_max = value
        if ch_min is None or ch_max is None:
            return False, "E2208"

        if ch_min > ch_max:
            return False, "E2209"

        allowed_min, allowed_max = config[space]["ranges"][ch]
        if ch_min < allowed_min or ch_max > allowed_max:
            return False, "E2210"

    return True, None


def _convert_image(data, img, space, config):
    if space == "GRAY":
        if len(img.shape) == 2:
            return img
        if len(img.shape) == 3 and img.shape[2] == 3:
            return cached_cvtColor(data, img, config[space]["convert"], f"cvtColor_BGR2GRAY")
        return None

    if len(img.shape) != 3 or img.shape[2] != 3:
        return None

    convert_code = config[space]["convert"]
    if convert_code is None:
        return img

    op_name = {
        cv2.COLOR_BGR2HSV: "cvtColor_BGR2HSV",
        cv2.COLOR_BGR2LAB: "cvtColor_BGR2LAB",
    }.get(convert_code, f"cvtColor_{convert_code}")
    
    return cached_cvtColor(data, img, convert_code, op_name)


def _build_mask(converted, space, thresholds, config):
    channels = config[space]["channels"]

    if space == "GRAY":
        gray_min, gray_max = thresholds["GRAY"]
        return cv2.inRange(converted, gray_min, gray_max)

    lower = []
    upper = []

    for ch in channels:
        ch_min, ch_max = thresholds[ch]
        lower.append(ch_min)
        upper.append(ch_max)

    lower_np = np.array(lower, dtype=np.uint8)
    upper_np = np.array(upper, dtype=np.uint8)

    return cv2.inRange(converted, lower_np, upper_np)


def _build_ui_schema(space, config, thresholds=None):
    sliders = []

    for ch in config[space]["channels"]:
        min_allowed, max_allowed = config[space]["ranges"][ch]

        current_min = min_allowed
        current_max = max_allowed

        if thresholds is not None and ch in thresholds:
            current_min, current_max = thresholds[ch]

        sliders.append({
            "name": f"{ch}_min",
            "label": f"{ch} min",
            "type": "slider",
            "min": min_allowed,
            "max": max_allowed,
            "step": 1,
            "value": current_min
        })

        sliders.append({
            "name": f"{ch}_max",
            "label": f"{ch} max",
            "type": "slider",
            "min": min_allowed,
            "max": max_allowed,
            "step": 1,
            "value": current_max
        })

    return {
        "space": space,
        "sliders": sliders
    }


def color_threshold(data, space="HSV", thresholds=None, invert=False, white_background=False, debug=False):
    """
    Szín alapú küszöbölés.

    Input:
        data["images"] -> lista képekkel
    Output:
        data["images"] -> bináris maszkok listája

    Paraméterek:
        space: "BGR" | "HSV" | "LAB" | "GRAY"
        thresholds:
            BGR esetén pl:
                {"B": (0, 255), "G": (0, 255), "R": (120, 255)}
            HSV esetén pl:
                {"H": (15, 40), "S": (40, 255), "V": (40, 255)}
            LAB esetén pl:
                {"L": (0, 255), "A": (120, 170), "B": (140, 255)}
            GRAY esetén pl:
                {"GRAY": (80, 255)}
        invert: maszk invertálása
        white_background: levágott területek fehérrel töltése
    """

    if data["error"] is not None:
        return data

    if data.get("images") is None or data.get("count", 0) == 0:
        data["error"] = "E2200"
        return data

    config = _get_space_config()

    if space not in config:
        data["error"] = "E2201"
        return data

    is_valid, error_code = _validate_thresholds(space, thresholds, config)
    if not is_valid:
        data["error"] = error_code
        return data

    output_images = []
    channel_histograms = []

    for img_idx, img in enumerate(data["images"]):
        if img is None:
            data["error"] = "E2202"
            return data

        converted = _convert_image(data, img, space, config)
        if converted is None:
            data["error"] = "E2203"
            return data

        mask = _build_mask(converted, space, thresholds, config)

        if invert:
            mask = cv2.bitwise_not(mask)

        output_images.append(mask)
        
        # Calculate histograms for each channel (cached)
        ch_histograms = {}
        for ch in config[space]["channels"]:
            if space == "GRAY":
                hist = cached_calcHist(data, converted, [0], bins=256, ranges=(0, 256), op_name=f"calcHist_GRAY")
            else:
                ch_idx = config[space]["channels"].index(ch)
                hist = cached_calcHist(data, converted, [ch_idx], bins=256, ranges=(0, 256), op_name=f"calcHist_{ch}")
            ch_histograms[ch] = hist.flatten().tolist()
        
        channel_histograms.append(ch_histograms)

    data["images"] = output_images
    data["count"] = len(output_images)

    if "meta" not in data:
        data["meta"] = {}

    data["meta"]["color_threshold"] = {
        "space": space,
        "thresholds": thresholds,
        "invert": invert,
        "ui_schema": _build_ui_schema(space, config, thresholds)
    }

    if "results" not in data:
        data["results"] = {}
    
    data["results"]["color_thresh_channel_histograms"] = channel_histograms

    if "history" not in data:
        data["history"] = []

    data["history"].append("color_threshold")

    if debug and len(output_images) > 0:
        print(data["meta"]["color_threshold"])
        print(f"Output shape: {output_images[0].shape}")
        print(f"Output dtype: {output_images[0].dtype}")

    return data