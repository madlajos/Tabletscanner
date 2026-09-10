"""Hardware-free smoke tests for manual and automatic image alignment."""
import cv2
import numpy as np

from proc_elements.image_alignment import align_images


def _data(image):
    return {"images": [image], "results": {}, "meta": {}, "history": [], "error": None}


reference = np.zeros((220, 260, 3), dtype=np.uint8)
cv2.ellipse(reference, (130, 110), (65, 32), 18, 0, 360, (225, 225, 225), -1)
cv2.circle(reference, (145, 103), 8, (80, 80, 80), -1)

manual_input = cv2.warpAffine(reference, np.float32([[1, 0, -12], [0, 1, 7]]), (260, 220))
manual = align_images(_data(manual_input), [reference], automatic=False, offset_x=12, offset_y=-7, reference_opacity=0.35)
assert manual.get("error") is None
assert np.mean(cv2.absdiff(manual["images"][0], reference)) < 1.0
assert len(manual["results"]["manual_alignment_previews"]) == 1

distortion = cv2.getRotationMatrix2D((130, 110), -24, 0.76)
distortion[:, 2] += (17, -10)
moving = cv2.warpAffine(reference, distortion, (260, 220))
automatic = align_images(_data(moving), [reference], automatic=True, allow_scale=True, refine=True)
assert automatic.get("error") is None
before = float(np.mean(cv2.absdiff(moving, reference)))
after = float(np.mean(cv2.absdiff(automatic["images"][0], reference)))
assert after < before * 0.35, (before, after)
assert automatic["results"]["image_alignment_transforms"][0]["matrix"]

missing_reference = align_images(_data(reference), [], automatic=False)
assert missing_reference.get("error") == "E4102"

print("Image alignment tests passed.")
