import cv2


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

    for img in data["images"]:
        if len(img.shape) != 2:
            data["error"] = "E2202"
            return data

        _, th = cv2.threshold(img, thresh, maxval, threshold_modes[mode])
        output_images.append(th)

    data["images"] = output_images
    data["count"] = len(output_images)
    data["meta"]["threshold"] = {
        "thresh": thresh,
        "maxval": maxval,
        "mode": mode
    }
    data["history"].append("apply_threshold")

    if debug:
        print(data["meta"]["threshold"])

    return data
