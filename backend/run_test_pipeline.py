import numpy as np
import cv2
import os
from pipeline.engine import PipelineEngine
from pipeline.schema import PipelineDocument

img_path = "temp_test_img.png"
input_img = np.zeros((100, 100, 3), dtype=np.uint8)
cv2.imwrite(img_path, input_img)

try:
    doc_dict = {
        "nodes": [
            {"id": "n1", "type": "load_image", "params": {"path": img_path}, "out": ["data"]},
            {"id": "n2", "type": "mask_rect_roi", "params": {"x": 30, "y": 40, "width": 10, "height": 12, "output_mode": "crop"}, "in": ["data"], "out": ["data"]},
            {"id": "n3", "type": "save_images", "params": {"prefix": "test_output"}, "in": ["data"], "out": ["data"]}
        ]
    }
    doc = PipelineDocument.parse_obj(doc_dict)
    engine = PipelineEngine()
    results = engine.execute_pipeline(doc, stop_node_id="n2")
    data = results.get("n2")
    if data and "images" in data and len(data["images"]) > 0:
        final_img = data["images"][0]
        print(f"Final image shape: {final_img.shape}")
    else: print("No image in results")
    if data and "results" in data and "roi" in data["results"]:
         print(f"ROI output_mode: {data['results']['roi'].get('output_mode')}")
    else: print("ROI metadata not found")

finally:
    if os.path.exists(img_path): os.remove(img_path)
