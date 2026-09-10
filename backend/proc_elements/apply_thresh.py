import cv2
import numpy as np


def apply_threshold(data, thresh=127, maxval=255, mode="binary", channel="GRAY", debug=False):

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

    channel_indices = {"B": 0, "G": 1, "R": 2}
    if channel not in {"GRAY", *channel_indices}:
        data["error"] = "E2202"
        return data

    output_images = []
    input_histograms = []
    output_histograms = []

    for img in data["images"]:
        is_grayscale = len(img.shape) == 2
        is_bgr = len(img.shape) == 3 and img.shape[2] == 3
        if not is_grayscale and not is_bgr:
            data["error"] = "E2202"
            return data

        if is_grayscale:
            threshold_source = img
        elif channel == "GRAY":
            data["error"] = "E2202"
            return data
        else:
            threshold_source = img[:, :, channel_indices[channel]]

        input_hist = cv2.calcHist([threshold_source], [0], None, [256], [0, 256]).flatten()

        _, thresholded_source = cv2.threshold(
            threshold_source, thresh, maxval, threshold_modes[mode]
        )

        if is_grayscale:
            th = thresholded_source
        elif mode in {"binary", "binary_inv"}:
            th = np.repeat(thresholded_source[:, :, np.newaxis], 3, axis=2)
        elif mode == "trunc":
            th = img.copy()
            th[threshold_source > thresh] = np.clip(thresh, 0, 255)
        else:
            keep_mask = (
                threshold_source > thresh
                if mode == "tozero"
                else threshold_source <= thresh
            )
            th = np.zeros_like(img)
            th[keep_mask] = img[keep_mask]

        output_images.append(th)
        output_hist_source = th if th.ndim == 2 else cv2.cvtColor(th, cv2.COLOR_BGR2GRAY)
        output_hist = cv2.calcHist([output_hist_source], [0], None, [256], [0, 256]).flatten()

        input_histograms.append([int(v) for v in np.rint(input_hist).tolist()])
        output_histograms.append([int(v) for v in np.rint(output_hist).tolist()])

    data["images"] = output_images
    data["count"] = len(output_images)
    data["results"]["threshold_input_histograms"] = input_histograms
    data["results"]["threshold_output_histograms"] = output_histograms
    data["meta"]["threshold"] = {
        "thresh": thresh,
        "maxval": maxval,
        "mode": mode,
        "channel": channel
    }
    data["history"].append("apply_threshold")

    if debug:
        print(data["meta"]["threshold"])

    return data
