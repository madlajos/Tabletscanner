#!/usr/bin/env python3
"""Quick test of the enhanced select_channel functionality."""

import cv2
import numpy as np
from proc_elements.select_channel import select_channel

# Create test data structure
def create_test_data(img):
    return {
        "error": None,
        "images": [img],
        "count": 1,
        "meta": {"channel": {}},
        "history": [],
        "paths": ["test.png"],
        "results": {}
    }

# Create a simple test image (BGR format)
test_img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

# Test 1: Single channel extraction (H from HSV)
print("Test 1: Extract H channel from HSV")
data = create_test_data(test_img)
result = select_channel(data, space="HSV", channel="H", debug=True)
print(f"  Output shape: {result['images'][0].shape}")
assert result['images'][0].shape == (100, 100), "Should be grayscale (2D)"
print("  ✓ PASS\n")

# Test 2: Single channel extraction (L from LAB)
print("Test 2: Extract L channel from LAB")
data = create_test_data(test_img)
result = select_channel(data, space="LAB", channel="L", debug=True)
print(f"  Output shape: {result['images'][0].shape}")
assert result['images'][0].shape == (100, 100), "Should be grayscale (2D)"
print("  ✓ PASS\n")

# Test 3: ALL channels from HSV (should return 3-channel image)
print("Test 3: Extract ALL channels from HSV")
data = create_test_data(test_img)
result = select_channel(data, space="HSV", channel="ALL", debug=True)
print(f"  Output shape: {result['images'][0].shape}")
assert result['images'][0].shape == (100, 100, 3), "Should be 3-channel image"
print("  ✓ PASS\n")

# Test 4: ALL channels from LAB (should return 3-channel image)
print("Test 4: Extract ALL channels from LAB")
data = create_test_data(test_img)
result = select_channel(data, space="LAB", channel="ALL", debug=True)
print(f"  Output shape: {result['images'][0].shape}")
assert result['images'][0].shape == (100, 100, 3), "Should be 3-channel image"
print("  ✓ PASS\n")

# Test 5: Grayscale conversion
print("Test 5: Convert to GRAY")
data = create_test_data(test_img)
result = select_channel(data, space="GRAY", channel="GRAY", debug=True)
print(f"  Output shape: {result['images'][0].shape}")
assert result['images'][0].shape == (100, 100), "Should be grayscale (2D)"
print("  ✓ PASS\n")

print("✅ All tests passed!")
