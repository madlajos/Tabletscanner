import numpy as np


def _get_masks_from_pipeline(data):
    """
    Retrieve masks from the pipeline in priority order.
    Returns a list of masks parallel to data["images"], or None if no masks found.
    Priority: range_masks > masks > active_masks > (generate full mask)
    """
    if data["results"] is None:
        return None
    
    # Try to get range_masks (from apply_range_mask)
    if "range_masks" in data["results"]:
        return data["results"]["range_masks"]
    
    # Try to get masks (from detect_circles with apply_mask=True)
    if "masks" in data["results"]:
        return data["results"]["masks"]
    
    # Try to get active_masks from meta
    if "meta" in data and "active_masks" in data["meta"]:
        return data["meta"]["active_masks"]
    
    # Generate full masks (include all pixels)
    masks = []
    for img in data["images"]:
        full_mask = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8) * 255
        masks.append(full_mask)
    return masks


def calculate_intensity_stats(data, percentiles=(5, 25, 50, 75, 95), debug=False):

    if data["error"] is not None:
        return data

    if data["images"] is None or data["count"] == 0:
        data["error"] = "E2501"
        return data

    if "results" not in data:
        data["results"] = {}

    if not isinstance(percentiles, (list, tuple)) or len(percentiles) == 0:
        data["error"] = "E2503"
        return data

    for p in percentiles:
        if not isinstance(p, (int, float)) or p < 0 or p > 100:
            data["error"] = "E2504"
            return data

    masks = _get_masks_from_pipeline(data)
    if masks is None or len(masks) == 0:
        data["error"] = "E2502"
        return data

    stats = []

    for img, mask in zip(data["images"], masks):

        if img is None or mask is None:
            data["error"] = "E2505"
            return data

        is_rgb = len(img.shape) == 3
        if not is_rgb and len(img.shape) != 2:
            data["error"] = "E2506"
            return data

        if len(mask.shape) != 2:
            data["error"] = "E2506"
            return data

        if img.shape[:2] != mask.shape:
            data["error"] = "E2507"
            return data

        if is_rgb:
            channel_stats = []
            for ch in range(img.shape[2]):
                ch_pixels = img[:, :, ch][mask > 0]
                if ch_pixels.size == 0:
                    channel_stats.append(None)
                    continue
                st = {
                    "min": float(np.min(ch_pixels)),
                    "max": float(np.max(ch_pixels)),
                    "median": float(np.median(ch_pixels)),
                    "mean": float(np.mean(ch_pixels)),
                    "std": float(np.std(ch_pixels)),
                    "pixel_count": int(ch_pixels.size)
                }
                for p in percentiles:
                    key = f"p{int(p)}" if float(p).is_integer() else f"p{p}"
                    st[key] = float(np.percentile(ch_pixels, p))
                if "p5" in st and "p95" in st:
                    st["dynamic_range"] = st["p95"] - st["p5"]
                else:
                    st["dynamic_range"] = None
                channel_stats.append(st)
            stat = {"channels": channel_stats, "channel_count": img.shape[2]}
            stats.append(stat)
        else:
            pixels = img[mask > 0]

            if pixels.size == 0:
                stats.append(None)
                continue

            stat = {
                "min": float(np.min(pixels)),
                "max": float(np.max(pixels)),
                "median": float(np.median(pixels)),
                "mean": float(np.mean(pixels)),
                "std": float(np.std(pixels)),
                "pixel_count": int(pixels.size)
            }

            for p in percentiles:
                key = f"p{int(p)}" if float(p).is_integer() else f"p{p}"
                stat[key] = float(np.percentile(pixels, p))

            if "p5" in stat and "p95" in stat:
                stat["dynamic_range"] = stat["p95"] - stat["p5"]
            else:
                stat["dynamic_range"] = None

            stats.append(stat)

    data["results"]["intensity_stats"] = stats
    data["meta"]["intensity_stats"] = {
        "percentiles": tuple(percentiles)
    }
    data["history"].append("calculate_intensity_stats")

    if debug:
        print("Intensity stats calculated")
        print(f"Image count: {len(stats)}")
        print(f"First stats: {stats[0]}")

    return data
