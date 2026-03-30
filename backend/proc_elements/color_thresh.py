import cv2
import numpy as np


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


def _convert_image(img, space, config):
    if space == "GRAY":
        if len(img.shape) == 2:
            return img
        if len(img.shape) == 3 and img.shape[2] == 3:
            return cv2.cvtColor(img, config[space]["convert"])
        return None

    if len(img.shape) != 3 or img.shape[2] != 3:
        return None

    convert_code = config[space]["convert"]
    if convert_code is None:
        return img

    return cv2.cvtColor(img, convert_code)


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
    input_images = []  # Keep copies of input images for preview

    for img in data["images"]:
        if img is None:
            data["error"] = "E2202"
            return data

        # Store input image copy for preview
        input_images.append(img.copy())

        converted = _convert_image(img, space, config)
        if converted is None:
            data["error"] = "E2203"
            return data

        mask = _build_mask(converted, space, thresholds, config)

        if invert:
            mask = cv2.bitwise_not(mask)

        output_images.append(mask)
        
        # Calculate histograms for each channel
        ch_histograms = {}
        for ch in config[space]["channels"]:
            if space == "GRAY":
                hist = cv2.calcHist([converted], [0], None, [256], [0, 256])
            else:
                ch_idx = config[space]["channels"].index(ch)
                hist = cv2.calcHist([converted], [ch_idx], None, [256], [0, 256])
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
    data["results"]["color_thresh_input_images"] = input_images
    
    # Create mask overlays: original image where mask matches, black/white elsewhere
    mask_overlays = []
    for i, mask in enumerate(output_images):
        if i < len(input_images):
            original = input_images[i].copy()
            # Apply mask: keep original where mask==255, fill with black/white where mask==0
            overlay = original.copy()
            if white_background:
                overlay[mask == 0] = 255  # White for non-matched regions
            else:
                overlay[mask == 0] = 0  # Black for non-matched regions
            mask_overlays.append(overlay)
    
    data["results"]["color_thresh_mask_overlays"] = mask_overlays

    if "history" not in data:
        data["history"] = []

    data["history"].append("color_threshold")

    if debug and len(output_images) > 0:
        print(data["meta"]["color_threshold"])
        print(f"Output shape: {output_images[0].shape}")
        print(f"Output dtype: {output_images[0].dtype}")

    return data