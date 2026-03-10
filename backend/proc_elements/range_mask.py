import cv2
import numpy as np


def apply_range_mask(data, low=0, high=255, keep_mode="inside", debug=False):

    if data["error"] is not None:
        return data

    if data["images"] is None or data["count"] == 0:
        data["error"] = "E2401"
        return data

    if low < 0 or high > 255 or low > high:
        data["error"] = "E2402"
        return data

    if keep_mode not in ["inside", "outside"]:
        data["error"] = "E2403"
        return data

    output_images = []
    masks = []

    for img in data["images"]:

        if len(img.shape) != 2:
            data["error"] = "E2404"
            return data

        mask = cv2.inRange(img, low, high)

        if keep_mode == "outside":
            mask = cv2.bitwise_not(mask)

        result = cv2.bitwise_and(img, img, mask=mask)

        output_images.append(result)
        masks.append(mask)

    data["images"] = output_images
    data["count"] = len(output_images)

    if "results" not in data:
        data["results"] = {}

    data["results"]["range_masks"] = masks

    data["meta"]["range_mask"] = {
        "low": low,
        "high": high,
        "keep_mode": keep_mode
    }

    data["history"].append("apply_range_mask")

    if debug:
        print(data["meta"]["range_mask"])
        cv2.imshow("mask", masks[0])
        cv2.imshow("result", output_images[0])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return data
