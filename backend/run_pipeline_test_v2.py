import os
import json
import numpy as np
import cv2
from pipeline.engine import PipelineEngine
from pipeline.document import Document

# Setup a temporary image
img_path = "temp_test_input.png"
test_img = np.zeros((200, 200, 3), dtype=np.uint8)
cv2.rectangle(test_img, (50, 50), (150, 150), (255, 255, 255), -1)
cv2.imwrite(img_path, test_img)

pipeline_def = {
    "steps": [
        {
            "id": 0,
            "type": "load_image",
            "params": {"file_path": img_path}
        },
        {
            "id": 1,
            "type": "mask_rect_roi",
            "params": {
                "roi": {"x": 50, "y": 50, "width": 100, "height": 80},
                "output_mode": "crop"
            }
        },
        {
            "id": 2,
            "type": "save_images",
            "params": {"folder": "output_test", "prefix": "test_"}
        }
    ]
}

try:
    engine = PipelineEngine()
    doc = Document(pipeline_def)
    
    print("Executing pipeline up to step 1 (crop)...")
    res = engine.execute_pipeline(doc, up_to_step=1)
    
    if res and "images" in res:
        img = res["images"][0]
        print(f"Result Image Shape: {img.shape}")
        meta = res.get("meta", {})
        print(f"Meta: {json.dumps(meta)}")
    else:
        print(f"Step 1 result invalid: {res}")

    print("\nExecuting full pipeline...")
    full_res = engine.execute_pipeline(doc)
    if full_res and "images" in full_res:
         print(f"Final Count: {len(full_res.get('images', []))}")
    
except Exception as e:
    import traceback
    traceback.print_exc()
finally:
    if os.path.exists(img_path):
        os.remove(img_path)
    if os.path.exists("output_test"):
        import shutil
        shutil.rmtree("output_test", ignore_errors=True)
