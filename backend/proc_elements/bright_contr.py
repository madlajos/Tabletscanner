import cv2
import numpy as np
from proc_elements.mask_utils import get_active_masks


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

    # Get active masks if they exist
    masks = get_active_masks(data)
    
    output_images = []

    for img_idx, img in enumerate(data["images"]):
        if img is None:
            data["error"] = "E3144"
            return data

        # Get mask for this image (if any)
        mask = masks[img_idx] if img_idx < len(masks) else None
        
        # Apply brightness/contrast
        result = cv2.convertScaleAbs(
            img,
            alpha=float(contrast),
            beta=float(brightness)
        )
        
        # If we have a mask, only apply changes within the masked region
        if mask is not None and mask.size > 0:
            # Ensure mask is the right shape
            if mask.shape[:2] != result.shape[:2]:
                # Skip if mask doesn't match image size
                pass
            else:
                # Apply changes only to masked region (in-place modification)
                if len(result.shape) == 2:
                    img[mask > 0] = result[mask > 0]
                else:
                    img[mask > 0] = result[mask > 0]
                result = img
        
        output_images.append(result)

    data["images"] = output_images
    data["count"] = len(output_images)
    data["meta"]["brightness_contrast"] = {
        "brightness": float(brightness),
        "contrast": float(contrast)
    }
    # Propagate active masks
    if "meta" in data and "active_masks" in data["meta"]:
        pass  # Masks are already in meta
    
    data["history"].append("adjust_brightness_contrast")

    if debug:
        print(data["meta"]["brightness_contrast"])

    return data
