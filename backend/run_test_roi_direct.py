import numpy as np
import cv2
import os
from proc_elements.draw_roi import mask_roi

img = np.zeros((100, 100, 3), dtype=np.uint8)
roi = {"type": "rect", "x": 30, "y": 40, "width": 10, "height": 12}
data = {"images": [img], "count": 1, "error": None}

res_crop = mask_roi(data.copy(), roi=roi, output_mode="crop")
final_img = res_crop["images"][0]
print(f"Final image shape: {final_img.shape}")
print(f"ROI output_mode: {res_crop.get('results', {}).get('roi', {}).get('output_mode')}")
