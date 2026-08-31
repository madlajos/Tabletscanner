import numpy as np
from proc_elements.draw_roi import mask_roi
img = np.zeros((100, 100, 3), dtype=np.uint8)
data = {"images": [img], "count": 1, "error": None}
roi = {"type": "rect", "x": 30, "y": 40, "width": 10, "height": 12}
res_mask = mask_roi(data.copy(), roi=roi, output_mode="mask")
res_crop = mask_roi(data.copy(), roi=roi, output_mode="crop")
m_img = res_mask["images"][0]
c_img = res_crop["images"][0]
print(f"Mask shape: {m_img.shape}")
print(f"Crop shape: {c_img.shape}")
print(f"Diff: {not np.array_equal(m_img, c_img)}")
