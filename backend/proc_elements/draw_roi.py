import cv2
import numpy as np


def mask_rect_roi(
    data,
    roi,
    background_color=0,
    keep_outside=False,
    debug=False
):

    if data["error"] is not None:
        return data

    if data["images"] is None or data["count"] == 0:
        data["error"] = "E3521"
        return data

    if roi.get("type") != "rect":
        data["error"] = "E3522"
        return data

    x = int(roi["x"])
    y = int(roi["y"])
    w = int(roi["width"])
    h = int(roi["height"])

    output_images = []
    masks = []

    for img in data["images"]:

        if img is None:
            data["error"] = "E3523"
            return data

        H, W = img.shape[:2]

        mask = np.zeros((H, W), dtype=np.uint8)
        mask[y:y+h, x:x+w] = 255

        if keep_outside:
            mask = cv2.bitwise_not(mask)

        # háttér generálása
        if len(img.shape) == 2:
            background = np.full((H, W), background_color, dtype=img.dtype)
        else:
            if isinstance(background_color, (list, tuple)):
                background = np.full((H, W, 3), background_color, dtype=img.dtype)
            else:
                background = np.full((H, W, 3), (background_color,)*3, dtype=img.dtype)

        roi_part = cv2.bitwise_and(img, img, mask=mask)
        inv_mask = cv2.bitwise_not(mask)
        bg_part = cv2.bitwise_and(background, background, mask=inv_mask)

        result = cv2.add(roi_part, bg_part)

        output_images.append(result)
        masks.append(mask)

    data["images"] = output_images
    data["count"] = len(output_images)

    data["results"]["roi_masks"] = masks
    data["meta"]["roi_mask"] = {
        "roi": roi,
        "background_color": background_color,
        "keep_outside": keep_outside
    }

    data["history"].append("mask_rect_roi")

    if debug:
        cv2.imshow("roi_mask", masks[0])
        cv2.imshow("roi_result", output_images[0])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return data
