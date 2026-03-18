import cv2
import numpy as np


def apply_threshold(data, thresh=127, maxval=255, mode="binary", debug=False):

    if data["error"] is not None:
        return data

    threshold_modes = {
        "binary": cv2.THRESH_BINARY,
        "binary_inv": cv2.THRESH_BINARY_INV,
        "trunc": cv2.THRESH_TRUNC,
        "tozero": cv2.THRESH_TOZERO,
        "tozero_inv": cv2.THRESH_TOZERO_INV
    }

    if mode not in threshold_modes:
        data["error"] = "E2201"
        return data

    output_images = []
    input_histograms = []
    output_histograms = []

    for img in data["images"]:
        if len(img.shape) != 2:
            data["error"] = "E2202"
            return data

        input_hist = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()

        _, th = cv2.threshold(img, thresh, maxval, threshold_modes[mode])
        output_images.append(th)
        output_hist = cv2.calcHist([th], [0], None, [256], [0, 256]).flatten()

        input_histograms.append([int(v) for v in np.rint(input_hist).tolist()])
        output_histograms.append([int(v) for v in np.rint(output_hist).tolist()])

    data["images"] = output_images
    data["count"] = len(output_images)
    data["results"]["threshold_input_histograms"] = input_histograms
    data["results"]["threshold_output_histograms"] = output_histograms
    data["meta"]["threshold"] = {
        "thresh": thresh,
        "maxval": maxval,
        "mode": mode
    }
    data["history"].append("apply_threshold")

    if debug:
        print(data["meta"]["threshold"])

    return data
