import cv2
import numpy as np
import os
import json
from pipeline_types import PipelineDocument, StepInstance
from pipeline_engine import execute_pipeline

# 1. Create temporary test image
img_path = "test_input_v6.png"
test_img = np.zeros((200, 200, 3), dtype=np.uint8)
cv2.rectangle(test_img, (50, 50), (150, 130), (255, 255, 255), -1)
cv2.imwrite(img_path, test_img)

# 2. Define steps
steps = [
    StepInstance(instance_id="0", step_def_id="load_image", param_values={"source": img_path}, order=0),
    StepInstance(instance_id="1", step_def_id="mask_rect_roi", param_values={
        "roi_type": "rect",
        "roi_x": 50,
        "roi_y": 50,
        "roi_width": 100,
        "roi_height": 80,
        "roi_cx": 0,
        "roi_cy": 0,
        "roi_rx": 0,
        "roi_ry": 0,
        "roi_angle": 0.0,
        "roi_points": "[]",
        "output_mode": "crop",
        "apply_mask": True,
        "background_color": "fekete",
        "invert_mask": False,
        "roi_overrides": "{}"
    }, order=1),
    StepInstance(instance_id="2", step_def_id="save_images", param_values={"output_dir": "test_output_v6"}, order=2)
]

doc = PipelineDocument(steps=steps)

try:
    # Execute up to step 1
    result = execute_pipeline(doc, up_to_step=1)

    if result.success and result.data and "images" in result.data:
        img = result.data["images"][0]
        meta = result.data.get("meta", {})
        roi_meta = meta.get("roi_mask", {})
        print(f"Result Image Shape: {img.shape}")
        print(f"ROI Output Mode: {roi_meta.get('output_mode')}")
        print(f"Images to save count: {len(result.data['images'])}")
    else:
        print(f"Pipeline failed. Success: {result.success}")
        for err in result.errors:
            print(f"Error: {getattr(err, 'message', str(err))}")

except Exception as e:
    import traceback
    traceback.print_exc()
finally:
    if os.path.exists(img_path):
        os.remove(img_path)
