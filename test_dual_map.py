import numpy as np
import sys
sys.path.insert(0,"backend")
from proc_elements.dual_map import dual_map_node, _is_grayscale
from pipeline_engine import extract_side_outputs

# Create a clear color image: left half blue (B channel high), right half green
H, W = 20, 20
rgb = np.zeros((H, W, 3), dtype=np.uint8)
rgb[:, :W//2, 0] = 180  # left half: high blue channel (BGR: channel 0)
rgb[:, W//2:, 1] = 180  # right half: high green channel (BGR: channel 1)

# Create a grayscale image
gray = np.ones((H, W, 3), dtype=np.uint8) * 128

print("_is_grayscale(rgb, 5.0):", _is_grayscale(rgb, 5.0))
print("_is_grayscale(gray, 5.0):", _is_grayscale(gray, 5.0))

d = {"images": [gray, rgb], "results": {}, "history": []}

# Test 1: references WITHOUT 255
o1 = dual_map_node(d.copy(), initial_centroids=(100,128,160), rgb_references=[0,0,180,0,180,0], auto_detect_channels=True)
side1 = extract_side_outputs(o1)
print("TEST 1 (no 255): rgb_hard_index_map_base64 present?", "rgb_hard_index_map" in o1.get("results", {}))
print("  rgb_soft_membership_jet_base64:", side1.get("rgb_soft_membership_jet_base64") is not None)
print("  Error:", o1.get("error"))

# Test 2: references WITH 255
d2 = {"images": [gray, rgb], "results": {}, "history": []}
o2 = dual_map_node(d2, initial_centroids=(100,128,160), rgb_references=[0,0,255,0,255,0], auto_detect_channels=True)
side2 = extract_side_outputs(o2)
print("TEST 2 (with 255): rgb_hard_index_map_base64 present?", "rgb_hard_index_map" in o2.get("results", {}))
print("  rgb_soft_membership_jet_base64:", side2.get("rgb_soft_membership_jet_base64") is not None)
print("  Error:", o2.get("error"))

# Check _is_grayscale on the float-normalized image (simulate pipeline-processed image)
rgb_float = rgb.astype(np.float32) / 255.0
print("_is_grayscale(rgb_float, 5.0):", _is_grayscale(rgb_float, 5.0))
print("max of rgb_float:", rgb_float.max())
