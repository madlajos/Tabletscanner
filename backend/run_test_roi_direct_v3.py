import numpy as np
import cv2
from proc_elements.draw_roi import mask_roi

img = np.zeros((100, 100, 3), dtype=np.uint8)
# Fill the rectangle area with some color to ensure boundingRect finds it
img[40:52, 30:40] = 255 

roi = {"type": "rect", "x": 30, "y": 40, "width": 10, "height": 12}
data = {
    "images": [img], 
    "count": 1, 
    "error": None, 
    "results": {}, 
    "meta": {}, 
    "history": []
}

try:
    res_crop = mask_roi(data, roi=roi, output_mode="crop")
    final_img = res_crop["images"][0]
    print(f"Final image shape: {final_img.shape}")
    
    # Check meta for output_mode
    out_mode = res_crop.get("meta", {}).get("roi_mask", {}).get("output_mode")
    print(f"ROI output_mode: {out_mode}")
    
    # Check if cropped
    is_cropped = final_img.shape == (12, 10, 3)
    print(f"Is cropped: {is_cropped}")
except Exception as e:
    import traceback
    traceback.print_exc()
