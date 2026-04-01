import cv2
import numpy as np
from proc_elements.mask_utils import get_active_masks


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

    # Get active masks if they exist
    masks = get_active_masks(data)
    
    output_images = []

    for img_idx, img in enumerate(data["images"]):
        if img is None:
            data["error"] = "E3153"
            return data

        # Get mask for this image (if any)
        mask = masks[img_idx] if img_idx < len(masks) else None
        
        result = cv2.LUT(img, table)
        
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
    data["meta"]["gamma_correction"] = {
        "gamma": float(gamma)
    }
    # Propagate active masks
    if "meta" in data and "active_masks" in data["meta"]:
        pass  # Masks are already in meta
    
    data["history"].append("gamma_correction")

    if debug:
        print(data["meta"]["gamma_correction"])

    return data
