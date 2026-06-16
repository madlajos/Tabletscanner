import cv2
import numpy as np
import json


def _build_mask(roi, H, W):
    """Build a binary mask (255 inside ROI) for a given roi dict."""
    roi_type = roi.get("type", "rect")
    angle = float(roi.get("angle", 0.0))
    mask = np.zeros((H, W), dtype=np.uint8)

    if roi_type == "rect":
        x = int(roi["x"])
        y = int(roi["y"])
        w = int(roi["width"])
        h = int(roi["height"])
        if abs(angle) < 0.01:
            mask[y:y+h, x:x+w] = 255
        else:
            # Rotated rectangle: compute 4 corner points around center
            cx = x + w / 2.0
            cy = y + h / 2.0
            rect = ((cx, cy), (w, h), angle)
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            cv2.fillPoly(mask, [box], 255)

    elif roi_type in ("ellipse", "circle"):
        cx = int(roi.get("cx", roi.get("x", 0)))
        cy = int(roi.get("cy", roi.get("y", 0)))
        if roi_type == "circle":
            r = int(roi.get("r", roi.get("radius", roi.get("rx", roi.get("ry", 0)))))
            rx = max(0, r)
            ry = max(0, r)
        else:
            rx = int(roi.get("rx", 0))
            ry = int(roi.get("ry", 0))
        cv2.ellipse(mask, (cx, cy), (rx, ry), angle, 0, 360, 255, -1)

    elif roi_type == "polygon":
        points = roi.get("points", [])
        if len(points) >= 3:
            pts = np.array([[int(p["x"]), int(p["y"])] for p in points], dtype=np.float32)
            if abs(angle) > 0.01:
                # Rotate points around centroid
                cx = pts[:, 0].mean()
                cy = pts[:, 1].mean()
                rad = np.radians(angle)
                cos_a, sin_a = np.cos(rad), np.sin(rad)
                dx = pts[:, 0] - cx
                dy = pts[:, 1] - cy
                pts[:, 0] = cx + dx * cos_a - dy * sin_a
                pts[:, 1] = cy + dx * sin_a + dy * cos_a
            cv2.fillPoly(mask, [np.intp(pts)], 255)

    return mask


def _crop_to_mask_bbox(img, mask):
    """Crop an image to the bounding box of the non-zero mask area."""
    points = cv2.findNonZero(mask)
    if points is None:
        return img
    x, y, w, h = cv2.boundingRect(points)
    if w <= 0 or h <= 0:
        return img
    return img[y:y + h, x:x + w]


def _resolve_outline_color_value(img, shape_outline_color):
    color_value = 255 if str(shape_outline_color).lower() in ("fehér", "feher", "white") else 0
    if img.ndim == 2:
        return color_value
    channels = img.shape[2] if len(img.shape) >= 3 else 3
    return tuple([color_value] * channels)


def _draw_roi_outline(img, current_roi, shape_outline_color, shape_outline_thickness):
    line_color = _resolve_outline_color_value(img, shape_outline_color)
    thickness = max(1, int(shape_outline_thickness))

    result = img.copy()
    roi_type = current_roi.get("type", "rect")
    angle = float(current_roi.get("angle", 0.0))

    if roi_type == "rect":
        x = int(current_roi["x"])
        y = int(current_roi["y"])
        w = int(current_roi["width"])
        h = int(current_roi["height"])
        if abs(angle) < 0.01:
            cv2.rectangle(result, (x, y), (x + w, y + h), line_color, thickness)
        else:
            cx = x + w / 2.0
            cy = y + h / 2.0
            rect = ((cx, cy), (w, h), angle)
            box = np.intp(cv2.boxPoints(rect))
            cv2.polylines(result, [box], True, line_color, thickness)
    elif roi_type in ("ellipse", "circle"):
        cx = int(current_roi.get("cx", current_roi.get("x", 0)))
        cy = int(current_roi.get("cy", current_roi.get("y", 0)))
        if roi_type == "circle":
            r = int(current_roi.get("r", current_roi.get("radius", current_roi.get("rx", current_roi.get("ry", 0)))))
            rx = max(0, r)
            ry = max(0, r)
        else:
            rx = int(current_roi.get("rx", 0))
            ry = int(current_roi.get("ry", 0))
        cv2.ellipse(result, (cx, cy), (rx, ry), angle, 0, 360, line_color, thickness)
    elif roi_type == "polygon":
        points = current_roi.get("points", [])
        if len(points) >= 3:
            pts = np.array([[int(p["x"]), int(p["y"])] for p in points], dtype=np.float32)
            if abs(angle) > 0.01:
                cx = pts[:, 0].mean()
                cy = pts[:, 1].mean()
                rad = np.radians(angle)
                cos_a, sin_a = np.cos(rad), np.sin(rad)
                dx = pts[:, 0] - cx
                dy = pts[:, 1] - cy
                pts[:, 0] = cx + dx * cos_a - dy * sin_a
                pts[:, 1] = cy + dx * sin_a + dy * cos_a
            cv2.polylines(result, [np.intp(pts)], True, line_color, thickness)

    return result


def mask_roi(
    data,
    roi=None,
    roi_list=None,
    background_color=0,
    keep_outside=False,
    apply_mask=True,
    invert_mask=False,
    shape_only=False,
    shape_outline_color="fehér",
    shape_outline_thickness=2,
    output_mode="mask",
    debug=False
):

    if data["error"] is not None:
        return data

    if data["images"] is None or data["count"] == 0:
        data["error"] = "E3521"
        return data

    # Build per-image ROI list
    num_images = len(data["images"])
    if roi_list is not None:
        # Per-image ROI list provided
        per_image_rois = roi_list
        # Pad with None if shorter than image count
        while len(per_image_rois) < num_images:
            per_image_rois.append(roi)
    elif roi is not None:
        # Single ROI broadcast to all images
        per_image_rois = [roi] * num_images
    else:
        # No ROI at all
        return data

    output_images = []
    masks = []
    crop_mode = str(output_mode).lower() == "crop"

    for i, img in enumerate(data["images"]):

        if img is None:
            data["error"] = "E3523"
            return data

        current_roi = per_image_rois[i] if i < len(per_image_rois) else None

        H, W = img.shape[:2]

        if current_roi is None:
            # No ROI for this image — pass through unchanged
            output_images.append(img)
            masks.append(np.zeros((H, W), dtype=np.uint8))
            continue

        roi_type = current_roi.get("type", "rect")
        if roi_type not in ("rect", "ellipse", "circle", "polygon"):
            data["error"] = "E3522"
            return data

        # Polygon with fewer than 3 points: skip masking for this image
        if roi_type == "polygon":
            points = current_roi.get("points", [])
            if len(points) < 3:
                output_images.append(img)
                masks.append(np.zeros((H, W), dtype=np.uint8))
                continue

        mask = _build_mask(current_roi, H, W)

        if invert_mask:
            mask = cv2.bitwise_not(mask)

        if shape_only:
            output_images.append(_draw_roi_outline(img, current_roi, shape_outline_color, shape_outline_thickness))
        elif crop_mode:
            output_images.append(_crop_to_mask_bbox(img, mask))
        elif apply_mask:
            # Apply mask: keep ROI, fill outside with background_color
            if background_color == 0:
                result = cv2.bitwise_and(img, img, mask=mask)
            else:
                bg = np.full_like(img, background_color)
                inv_mask = cv2.bitwise_not(mask)
                fg = cv2.bitwise_and(img, img, mask=mask)
                bg_part = cv2.bitwise_and(bg, bg, mask=inv_mask)
                result = cv2.add(fg, bg_part)
            output_images.append(result)
        else:
            output_images.append(img)

        masks.append(mask)

    data["images"] = output_images
    data["count"] = len(output_images)

    data["results"]["roi_masks"] = masks
    data["meta"]["roi_mask"] = {
        "roi": roi if roi is not None else (per_image_rois[0] if per_image_rois else None),
        "output_mode": "crop" if crop_mode else "mask",
    }

    if apply_mask:
        data["results"]["masks"] = masks
        data["meta"]["active_masks"] = masks

    data["history"].append("mask_roi")

    return data


# Backward-compatible alias
mask_rect_roi = mask_roi
