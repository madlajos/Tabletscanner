import cv2
import numpy as np
import json


def _build_mask(roi, H, W):
    """Build a binary mask (255 inside ROI) for a given roi dict."""
    roi_type = roi.get("type", "rect")
    mask = np.zeros((H, W), dtype=np.uint8)

    if roi_type == "rect":
        x = int(roi["x"])
        y = int(roi["y"])
        w = int(roi["width"])
        h = int(roi["height"])
        mask[y:y+h, x:x+w] = 255

    elif roi_type == "ellipse":
        cx = int(roi["cx"])
        cy = int(roi["cy"])
        rx = int(roi["rx"])
        ry = int(roi["ry"])
        cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)

    elif roi_type == "polygon":
        points = roi.get("points", [])
        if len(points) >= 3:
            pts = np.array([[int(p["x"]), int(p["y"])] for p in points], dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)

    return mask


def mask_roi(
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

    roi_type = roi.get("type", "rect")
    if roi_type not in ("rect", "ellipse", "polygon"):
        data["error"] = "E3522"
        return data

    # Polygon with fewer than 3 points: skip masking, pass through
    if roi_type == "polygon":
        points = roi.get("points", [])
        if len(points) < 3:
            data["history"].append("mask_roi (skipped, polygon < 3 points)")
            return data

    output_images = []
    masks = []

    for img in data["images"]:

        if img is None:
            data["error"] = "E3523"
            return data

        H, W = img.shape[:2]

        mask = _build_mask(roi, H, W)

        # Apply mask: keep only the ROI region
        result = cv2.bitwise_and(img, img, mask=mask)

        output_images.append(result)
        masks.append(mask)

    data["images"] = output_images
    data["count"] = len(output_images)

    data["results"]["roi_masks"] = masks
    data["meta"]["roi_mask"] = {
        "roi": roi,
    }

    data["history"].append("mask_roi")

    return data


# Backward-compatible alias
mask_rect_roi = mask_roi
