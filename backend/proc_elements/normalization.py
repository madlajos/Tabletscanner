import cv2
import numpy as np
from proc_elements.mask_utils import get_active_masks


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

    # Get active masks if they exist
    masks = get_active_masks(data)
    
    output_images = []

    for img_idx, img in enumerate(data["images"]):
        if img is None:
            data["error"] = "E3134"
            return data

        # Get mask for this image (if any)
        mask = masks[img_idx] if img_idx < len(masks) else None
        
        result = cv2.normalize(
            img,
            None,
            alpha=float(alpha),
            beta=float(beta),
            norm_type=norm_map[norm_type]
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
    data["meta"]["normalize"] = {
        "alpha": float(alpha),
        "beta": float(beta),
        "norm_type": norm_type
    }
    # Propagate active masks
    if "meta" in data and "active_masks" in data["meta"]:
        pass  # Masks are already in meta
    
    data["history"].append("normalize_images")

    if debug:
        print(data["meta"]["normalize"])

    return data
