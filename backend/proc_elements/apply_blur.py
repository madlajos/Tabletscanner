import cv2
from proc_elements.cache_utils import cached_GaussianBlur, cached_medianBlur


def apply_blur(
    data,
    method="gaussian",
    ksize=5,
    sigma=0,
    sigma_color=75,
    sigma_space=75,
    debug=False
):

    if data["error"] is not None:
        return data

    if data["images"] is None or data["count"] == 0:
        data["error"] = "E3041"
        return data

    if method not in ["gaussian", "median", "bilateral", "average"]:
        data["error"] = "E3042"
        return data

    if not isinstance(ksize, int) or ksize <= 0:
        data["error"] = "E3043"
        return data

    if method in ["gaussian", "median"] and ksize % 2 == 0:
        data["error"] = "E3044"
        return data

    if not isinstance(sigma, (int, float)) or sigma < 0:
        data["error"] = "E3045"
        return data

    if not isinstance(sigma_color, (int, float)) or sigma_color < 0:
        data["error"] = "E3046"
        return data

    if not isinstance(sigma_space, (int, float)) or sigma_space < 0:
        data["error"] = "E3047"
        return data

    output_images = []

    for img in data["images"]:
        if img is None:
            data["error"] = "E3048"
            return data

        if method == "gaussian":
            result = cached_GaussianBlur(data, img, (ksize, ksize), sigma)

        elif method == "median":
            result = cached_medianBlur(data, img, ksize)

        elif method == "average":
            result = cv2.blur(img, (ksize, ksize))

        elif method == "bilateral":
            result = cv2.bilateralFilter(
                img,
                d=ksize,
                sigmaColor=sigma_color,
                sigmaSpace=sigma_space
            )

        output_images.append(result)

    data["images"] = output_images
    data["count"] = len(output_images)
    data["meta"]["blur"] = {
        "method": method,
        "ksize": ksize,
        "sigma": float(sigma),
        "sigma_color": float(sigma_color),
        "sigma_space": float(sigma_space)
    }
    data["history"].append("apply_blur")

    if debug:
        print(data["meta"]["blur"])

    return data
