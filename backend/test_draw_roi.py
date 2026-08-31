import unittest

import cv2
import numpy as np

from proc_elements.draw_roi import mask_roi


def _pipeline_data(image):
    return {
        "images": [image],
        "count": 1,
        "error": None,
        "results": {},
        "meta": {},
        "history": [],
    }


class RotatedRectangleCropTests(unittest.TestCase):
    def test_rotated_rectangle_is_straightened_to_its_requested_size(self):
        image = np.zeros((120, 120, 3), dtype=np.uint8)
        roi = {
            "type": "rect",
            "x": 40,
            "y": 50,
            "width": 40,
            "height": 20,
            "angle": 45.0,
        }

        result = mask_roi(_pipeline_data(image), roi=roi, output_mode="crop")

        self.assertEqual(result["images"][0].shape, (20, 40, 3))

    def test_rotated_crop_keeps_the_rectangle_orientation(self):
        width, height = 40, 20
        patch = np.zeros((height, width, 3), dtype=np.uint8)
        patch[:, : width // 2] = (255, 0, 0)
        patch[:, width // 2 :] = (0, 255, 0)

        x, y, angle = 40, 50, 45.0
        radians = np.radians(angle)
        cos_a, sin_a = np.cos(radians), np.sin(radians)
        cx, cy = x + width / 2.0, y + height / 2.0
        placement = np.array([
            [cos_a, -sin_a, cx - cos_a * width / 2.0 + sin_a * height / 2.0],
            [sin_a, cos_a, cy - sin_a * width / 2.0 - cos_a * height / 2.0],
        ], dtype=np.float32)
        image = cv2.warpAffine(patch, placement, (120, 120), flags=cv2.INTER_NEAREST)
        roi = {
            "type": "rect",
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "angle": angle,
        }

        crop = mask_roi(_pipeline_data(image), roi=roi, output_mode="crop")["images"][0]
        left = crop[2:-2, 2:width // 2 - 2]
        right = crop[2:-2, width // 2 + 2:-2]

        self.assertGreater(float(left[..., 0].mean()), 240.0)
        self.assertLess(float(left[..., 1].mean()), 15.0)
        self.assertGreater(float(right[..., 1].mean()), 240.0)
        self.assertLess(float(right[..., 0].mean()), 15.0)


if __name__ == "__main__":
    unittest.main()
