import cv2


def normalize_images(data, alpha=0, beta=255, norm_type="minmax", debug=False):

    if data["error"] is not None:
        return data

    if data["images"] is None or data["count"] == 0:
        data["error"] = "E3131"
        return data

    norm_map = {
        "minmax": cv2.NORM_MINMAX,
        "l1": cv2.NORM_L1,
        "l2": cv2.NORM_L2
    }

    if norm_type not in norm_map:
        data["error"] = "E3132"
        return data

    if not isinstance(alpha, (int, float)) or not isinstance(beta, (int, float)):
        data["error"] = "E3133"
        return data

    output_images = []

    for img in data["images"]:
        if img is None:
            data["error"] = "E3134"
            return data

        result = cv2.normalize(
            img,
            None,
            alpha=float(alpha),
            beta=float(beta),
            norm_type=norm_map[norm_type]
        )
        output_images.append(result)

    data["images"] = output_images
    data["count"] = len(output_images)
    data["meta"]["normalize"] = {
        "alpha": float(alpha),
        "beta": float(beta),
        "norm_type": norm_type
    }
    data["history"].append("normalize_images")

    if debug:
        print(data["meta"]["normalize"])

    return data
