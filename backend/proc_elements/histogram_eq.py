import cv2


def histogram_equalization(data, debug=False):

    if data["error"] is not None:
        return data

    if data["images"] is None or data["count"] == 0:
        data["error"] = "E3111"
        return data

    output_images = []

    for img in data["images"]:
        if img is None:
            data["error"] = "E3112"
            return data

        if len(img.shape) != 2:
            data["error"] = "E3113"
            return data

        result = cv2.equalizeHist(img)
        output_images.append(result)

    data["images"] = output_images
    data["count"] = len(output_images)
    data["meta"]["histogram_equalization"] = {
        "enabled": True
    }
    data["history"].append("histogram_equalization")

    if debug:
        print(data["meta"]["histogram_equalization"])

    return data
