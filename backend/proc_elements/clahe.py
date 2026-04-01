import cv2
import numpy as np
from proc_elements.mask_utils import get_active_masks


def apply_clahe(data, clip_limit=2.0, tile_grid_size=(8, 8), debug=False):

    if data["error"] is not None:
        return data

    if data["images"] is None or data["count"] == 0:
        data["error"] = "E3121"
        return data

    if not isinstance(clip_limit, (int, float)) or clip_limit <= 0:
        data["error"] = "E3122"
        return data

    if not isinstance(tile_grid_size, (tuple, list)) or len(tile_grid_size) != 2:
        data["error"] = "E3123"
        return data

    tx, ty = tile_grid_size

    if not isinstance(tx, int) or not isinstance(ty, int) or tx <= 0 or ty <= 0:
        data["error"] = "E3124"
        return data

    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=(tx, ty)
    )

    # Get active masks if they exist
    masks = get_active_masks(data)
    
    output_images = []

    for img_idx, img in enumerate(data["images"]):
        if img is None:
            data["error"] = "E3125"
            return data

        if len(img.shape) != 2:
            data["error"] = "E3126"
            return data

        # Get mask for this image (if any)
        mask = masks[img_idx] if img_idx < len(masks) else None
        
        result = clahe.apply(img)
        
        # If we have a mask, only apply changes within the masked region
        if mask is not None and mask.size > 0:
            # Ensure mask is the right shape
            if mask.shape[:2] != result.shape[:2]:
                # Skip if mask doesn't match image size
                pass
            else:
                # Apply changes only to masked region (in-place modification)
                img[mask > 0] = result[mask > 0]
                result = img
        
        output_images.append(result)

    data["images"] = output_images
    data["count"] = len(output_images)
    data["meta"]["clahe"] = {
        "clip_limit": float(clip_limit),
        "tile_grid_size": (tx, ty)
    }
    # Propagate active masks
    if "meta" in data and "active_masks" in data["meta"]:
        pass  # Masks are already in meta
    
    data["history"].append("apply_clahe")

    if debug:
        print(data["meta"]["clahe"])

    return data
