import numpy as np


def calculate_intensity_stats(data, percentiles=(5, 25, 50, 75, 95), debug=False):

    if data["error"] is not None:
        return data

    if data["images"] is None or data["count"] == 0:
        data["error"] = "E2501"
        return data

    if "results" not in data or "range_masks" not in data["results"]:
        data["error"] = "E2502"
        return data

    if not isinstance(percentiles, (list, tuple)) or len(percentiles) == 0:
        data["error"] = "E2503"
        return data

    for p in percentiles:
        if not isinstance(p, (int, float)) or p < 0 or p > 100:
            data["error"] = "E2504"
            return data

    stats = []

    for img, mask in zip(data["images"], data["results"]["range_masks"]):

        if img is None or mask is None:
            data["error"] = "E2505"
            return data

        if len(img.shape) != 2 or len(mask.shape) != 2:
            data["error"] = "E2506"
            return data

        if img.shape != mask.shape:
            data["error"] = "E2507"
            return data

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
