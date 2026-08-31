import unittest

import numpy as np

from proc_elements.reference_sequence import reference_sequence


def _data(images, results=None):
    return {
        "images": images,
        "count": len(images),
        "results": results or {},
        "meta": {},
        "history": [],
        "error": None,
    }


class ReferenceSequenceTests(unittest.TestCase):
    def test_uses_complete_images_when_there_is_no_crop_step(self):
        dark = np.full((4, 5, 3), 20, dtype=np.uint8)
        light = np.full((6, 3, 3), 180, dtype=np.uint8)

        result = reference_sequence(_data([light, dark]))

        self.assertIsNone(result["error"])
        self.assertEqual(len(result["results"]["reference_crops"]), 1)
        self.assertEqual([item["original_index"] for item in result["results"]["reference_sequence"][0]["items"]], [1, 0])
        self.assertIs(result["results"]["reference_crops"][0][0], dark)
        self.assertIs(result["results"]["reference_crops"][0][1], light)

    def test_combines_crops_from_separate_images_into_one_sequence(self):
        bright = np.full((2, 2, 3), 200, dtype=np.uint8)
        dark = np.full((2, 2, 3), 10, dtype=np.uint8)
        medium = np.full((2, 2, 3), 80, dtype=np.uint8)
        results = {
            "reference_crops": [[bright], [dark, medium]],
            "reference_crop_squares": [
                [{"name": "bright", "source_image_index": 0}],
                [
                    {"name": "dark", "source_image_index": 1},
                    {"name": "medium", "source_image_index": 1},
                ],
            ],
        }

        result = reference_sequence(_data([bright, dark], results))

        rows = result["results"]["reference_sequence"]
        self.assertEqual(len(rows), 1)
        self.assertEqual([item["label"] for item in rows[0]["items"]], ["dark", "medium", "bright"])
        self.assertEqual(len(result["results"]["reference_crops"][0]), 3)


if __name__ == "__main__":
    unittest.main()
