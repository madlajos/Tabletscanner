import cv2


def calculate_histograms(data, bins=256, hist_range=(0, 256), debug=False):

    if data["error"] is not None:
        return data

    if data["images"] is None or data["count"] == 0:
        data["error"] = "E2301"
        return data

    if not isinstance(bins, int) or bins <= 0:
        data["error"] = "E2303"
        return data

    if not isinstance(hist_range, (list, tuple)) or len(hist_range) != 2:
        data["error"] = "E2304"
        return data

    if hist_range[0] >= hist_range[1]:
        data["error"] = "E2305"
        return data

    histograms = []

    for img in data["images"]:

        if img is None:
            data["error"] = "E2306"
            return data

        if len(img.shape) != 2:
            data["error"] = "E2302"
            return data

        hist = cv2.calcHist([img], [0], None, [bins], hist_range)
        hist = hist.flatten()

        histograms.append(hist)

    if "results" not in data:
        data["results"] = {}

    data["results"]["histograms"] = histograms
    data["meta"]["histogram"] = {
        "bins": bins,
        "range": tuple(hist_range)
    }
    data["history"].append("calculate_histograms")

    if debug:
        print(f"Histogram calculated for {len(histograms)} images")
        print(f"Bins: {bins}")
        print(f"First histogram shape: {histograms[0].shape}")

    return data
