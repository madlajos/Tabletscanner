import cv2
import numpy as np


def gamma_correction(data, gamma=1.0, debug=False):

    if data["error"] is not None:
        return data

    if data["images"] is None or data["count"] == 0:
        data["error"] = "E3151"
        return data

    if not isinstance(gamma, (int, float)) or gamma <= 0:
        data["error"] = "E3152"
        return data

    inv_gamma = 1.0 / float(gamma)
    table = np.array([
        ((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)
    ]).astype("uint8")

    output_images = []

    for img in data["images"]:
        if img is None:
            data["error"] = "E3153"
            return data

        result = cv2.LUT(img, table)
        output_images.append(result)

    data["images"] = output_images
    data["count"] = len(output_images)
    data["meta"]["gamma_correction"] = {
        "gamma": float(gamma)
    }
    data["history"].append("gamma_correction")

    if debug:
        print(data["meta"]["gamma_correction"])

    return data
