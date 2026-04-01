import cv2
import numpy as np
from proc_elements.mask_utils import get_active_masks, compute_histogram_masked


def histogram_equalization(data, debug=False):

    if data["error"] is not None:
        return data

    if data["images"] is None or data["count"] == 0:
        data["error"] = "E3111"
        return data

    # Get active masks if they exist
    masks = get_active_masks(data)
    
    output_images = []
    input_histograms = []
    output_histograms = []

    for img_idx, img in enumerate(data["images"]):
        if img is None:
            data["error"] = "E3112"
            return data

        if len(img.shape) != 2:
            data["error"] = "E3113"
            return data

        # Get mask for this image (if any)
        mask = masks[img_idx] if img_idx < len(masks) else None
        
        # Calculate input histogram
        if mask is not None and mask.size > 0:
            input_hist = compute_histogram_masked(img, mask)
        else:
            input_hist = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
        input_histograms.append(input_hist)

        result = cv2.equalizeHist(img)
        
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

        # Calculate output histogram
        if mask is not None and mask.size > 0:
            output_hist = compute_histogram_masked(result, mask)
        else:
            output_hist = cv2.calcHist([result], [0], None, [256], [0, 256]).flatten()
        output_histograms.append(output_hist)

        output_images.append(result)

    data["images"] = output_images
    data["count"] = len(output_images)
    data["results"]["histeq_input_histograms"] = input_histograms
    data["results"]["histeq_output_histograms"] = output_histograms
    data["meta"]["histogram_equalization"] = {
        "enabled": True,
        "bins": 256,
        "range": [0, 256],
    }
    # Propagate active masks
    if "meta" in data and "active_masks" in data["meta"]:
        pass  # Masks are already in meta
    
    data["history"].append("histogram_equalization")

    if debug:
        print(data["meta"]["histogram_equalization"])

    return data
