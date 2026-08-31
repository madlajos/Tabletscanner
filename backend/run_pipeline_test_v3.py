import os
import json
import numpy as np
import cv2
from pipeline_engine import execute_pipeline
from proc_elements.load_img import load_image
from proc_elements.draw_roi import mask_roi

# Setup a temporary image
img_path = "temp_test_input.png"
test_img = np.zeros((200, 200, 3), dtype=np.uint8)
# Draw a white rectangle to see it in crop
cv2.rectangle(test_img, (60, 60), (140, 140), (255, 255, 255), -1)
cv2.imwrite(img_path, test_img)

pipeline_doc = {
    "steps": [
        {
            "id": "step0",
            "type": "load_image",
            "params": {"file_path": img_path}
        },
        {
            "id": "step1",
            "type": "mask_rect_roi",
            "params": {
                "roi": {"type": "rect", "x": 50, "y": 50, "width": 100, "height": 80},
                "output_mode": "crop"
            }
        }
    ]
}

try:
    print("Executing pipeline up to step 1 (crop)...")
    # execute_pipeline signature: def execute_pipeline(pipeline_doc: dict, up_to_step: Optional[int] = None) -> dict: (guessed)
    # The signature from Select-String was def execute_pipeline(
    # Let's check the signature more carefully or just call it.
    
    res = execute_pipeline(pipeline_doc, up_to_step=1)
    
    if res and "images" in res:
        img = res["images"][0]
        print(f"Result Image Shape: {img.shape}")
        meta = res.get("meta", {})
        print(f"Meta: {json.dumps(meta)}")
        
        # Manually save to see result
        cv2.imwrite("test_output_crop.png", img)
        print("Saved crop result to test_output_crop.png")
    else:
        print(f"Result invalid: {res}")
    
except Exception as e:
    import traceback
    traceback.print_exc()
finally:
    if os.path.exists(img_path):
        os.remove(img_path)
