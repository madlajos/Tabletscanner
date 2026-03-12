import cv2


def histogram_equalization(data, debug=False):

    if data["error"] is not None:
        return data

    if data["images"] is None or data["count"] == 0:
        data["error"] = "E3111"
        return data

    output_images = []
    input_histograms = []
    output_histograms = []

    for img in data["images"]:
        if img is None:
            data["error"] = "E3112"
            return data

        if len(img.shape) != 2:
            data["error"] = "E3113"
            return data

        input_hist = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
        input_histograms.append(input_hist)

        result = cv2.equalizeHist(img)

        output_hist = cv2.calcHist([result], [0], None, [256], [0, 256]).flatten()
        output_histograms.append(output_hist)

        output_images.append(result)

    data["images"] = output_images
    data["count"] = len(output_images)
    data["results"]["histeq_input_histograms"] = input_histograms
    data["results"]["histeq_output_histograms"] = output_histograms
    data["meta"]["histogram_equalization"] = {
        "enabled": True,
        "bins": 256,
        "range": [0, 256],
    }
    data["history"].append("histogram_equalization")

    if debug:
        print(data["meta"]["histogram_equalization"])

    return data
