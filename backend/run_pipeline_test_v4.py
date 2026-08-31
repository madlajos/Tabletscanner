import os
import json
import numpy as np
import cv2
from pipeline_engine import execute_pipeline
from pipeline_types import PipelineDocument, StepInstance
from proc_elements.load_img import load_image
from proc_elements.draw_roi import mask_roi

# Setup a temporary image
img_path = "temp_test_input.png"
test_img = np.zeros((200, 200, 3), dtype=np.uint8)
cv2.rectangle(test_img, (60, 60), (140, 140), (255, 255, 255), -1)
cv2.imwrite(img_path, test_img)

# Build the document model correctly
steps = [
    StepInstance(instance_id="0", step_def_id="load_image", param_values={"file_path": img_path}, order=0),
    StepInstance(instance_id="1", step_def_id="mask_rect_roi", param_values={
        "roi": {"type": "rect", "x": 50, "y": 50, "width": 100, "height": 80},
        "output_mode": "crop"
    }, order=1)
]
doc = PipelineDocument(steps=steps)

try:
    print("Executing pipeline up to step 1 (crop)...")
    # In many systems, up_to_step is inclusive or exclusive. 
    # Let's try 1 for the second step.
    # The signature in pipeline_engine.py: def execute_pipeline(pipeline_doc: PipelineDocument, up_to_step: Optional[int] = None) -> PipelineResult:
    
    result = execute_pipeline(doc, up_to_step=1)
    
    # result is likely a PipelineResult object
    if result.success and result.data and "images" in result.data:
        img = result.data["images"][0]
        print(f"Result Image Shape: {img.shape}")
        meta = result.data.get("meta", {})
        print(f"Meta: {json.dumps(meta)}")
        
        # Save to confirm
        cv2.imwrite("test_output_crop_v2.png", img)
        print("Saved crop result to test_output_crop_v2.png")
    else:
        print(f"Pipeline failed or no images. Success: {result.success}")
        for err in result.errors:
            print(f"Error: {err.message} (Code: {err.error_code})")
    
except Exception as e:
    import traceback
    traceback.print_exc()
finally:
    if os.path.exists(img_path):
        os.remove(img_path)
