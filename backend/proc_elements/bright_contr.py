import cv2


def adjust_brightness_contrast(data, brightness=0, contrast=1.0, debug=False):

    if data["error"] is not None:
        return data

    if data["images"] is None or data["count"] == 0:
        data["error"] = "E3141"
        return data

    if not isinstance(brightness, (int, float)):
        data["error"] = "E3142"
        return data

    if not isinstance(contrast, (int, float)) or contrast <= 0:
        data["error"] = "E3143"
        return data

    output_images = []

    for img in data["images"]:
        if img is None:
            data["error"] = "E3144"
            return data

        result = cv2.convertScaleAbs(
            img,
            alpha=float(contrast),
            beta=float(brightness)
        )
        output_images.append(result)

    data["images"] = output_images
    data["count"] = len(output_images)
    data["meta"]["brightness_contrast"] = {
        "brightness": float(brightness),
        "contrast": float(contrast)
    }
    data["history"].append("adjust_brightness_contrast")

    if debug:
        print(data["meta"]["brightness_contrast"])

    return data
