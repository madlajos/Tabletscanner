import cv2
import numpy as np
from proc_elements.histogram_summary import build_histogram_summary
from proc_elements.mask_utils import get_pipeline_masks


def calculate_histograms(data, bins=256, hist_range=None, debug=False,
                         display_mode='per_image', group_labels='[]'):

    if data["error"] is not None:
        return data

    if data["images"] is None or data["count"] == 0:
        data["error"] = "E2301"
        return data

    if not isinstance(bins, int) or bins <= 0:
        data["error"] = "E2303"
        return data

    if hist_range is None:
        channel_meta = data.get("meta", {}).get("channel", {})
        space = channel_meta.get("space", "GRAY")
        channel = channel_meta.get("channel", "GRAY")

        if space == "HSV" and channel == "H":
            hist_range = (0, 180)
        else:
            hist_range = (0, 256)

    if not isinstance(hist_range, (list, tuple)) or len(hist_range) != 2:
        data["error"] = "E2304"
        return data

    if hist_range[0] >= hist_range[1]:
        data["error"] = "E2305"
        return data

    histograms = []
    histogram_stats = []
    masks = get_pipeline_masks(data)

    if masks is not None and len(masks) != len(data["images"]):
        data["error"] = "E2306"
        return data

    for index, img in enumerate(data["images"]):

        if img is None:
            data["error"] = "E2306"
            return data

        if len(img.shape) != 2:
            data["error"] = "E2302"
            return data

        mask = masks[index] if masks is not None else None
        if mask is not None and (not isinstance(mask, np.ndarray)
                                 or len(mask.shape) != 2
                                 or mask.shape != img.shape):
            data["error"] = "E2306"
            return data

        pixels = img[mask > 0] if mask is not None else img.reshape(-1)
        hist = cv2.calcHist([img], [0], mask, [bins], hist_range)
        hist = hist.flatten()

        histograms.append(hist.tolist())
        histogram_stats.append(None if pixels.size == 0 else {
            "pixel_min": int(np.min(pixels)),
            "pixel_max": int(np.max(pixels)),
            "pixel_mean": float(np.mean(pixels)),
            "pixel_std": float(np.std(pixels))
        })

    if "results" not in data:
        data["results"] = {}

    data["results"]["histograms"] = histograms
    data["results"]["histogram_stats"] = histogram_stats
    build_histogram_summary(data, display_mode, group_labels)
    if data['error']:
        return data
    data["meta"]["histogram"] = {
        "bins": int(bins),
        "range": tuple(hist_range)
    }
    data["history"].append("calculate_histograms")

    if debug:
        print(f"Histogram calculated for {len(histograms)} images")
        print(f"Bins: {bins}")
        print(f"Histogram range: {hist_range}")
        print(f"First histogram length: {len(histograms[0])}")
        print(f"First image stats: {histogram_stats[0]}")

    return data
