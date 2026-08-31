import unittest

import cv2
import numpy as np

from proc_elements.reference_color_align import _reference_tone_map, reference_color_align
from pipeline_engine import _clamp_params


def _lab_l(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)


class ReferenceColorAlignTests(unittest.TestCase):
    def _data(self, image, crops):
        return {
            "images": [image], "count": 1, "error": None,
            "results": {"reference_crops": [crops]},
            "meta": {}, "history": [],
        }

    def test_maps_source_percentiles_to_reference_tones_without_full_stretch(self):
        dark = np.full((5, 5, 3), 50, dtype=np.uint8)
        middle = np.full((5, 5, 3), 120, dtype=np.uint8)
        light = np.full((5, 5, 3), 200, dtype=np.uint8)
        image = np.hstack([dark, middle, light])

        result = reference_color_align(self._data(image, [middle, light, dark]))

        output_l = _lab_l(result["images"][0])
        self.assertAlmostEqual(float(np.median(output_l[:, :5])), float(np.median(_lab_l(dark))), delta=3.0)
        self.assertAlmostEqual(float(np.median(output_l[:, 5:10])), float(np.median(_lab_l(middle))), delta=3.0)
        self.assertAlmostEqual(float(np.median(output_l[:, -5:])), float(np.median(_lab_l(light))), delta=3.0)
        self.assertGreater(float(output_l.min()), 0.0)
        self.assertLess(float(output_l.max()), 255.0)
        info = result["results"]["reference_color_align_ranges"][0]
        self.assertEqual(info["darkest_reference_index"], 2)
        self.assertEqual(info["lightest_reference_index"], 1)

    def test_keeps_target_reference_crops_unchanged(self):
        dark = np.full((5, 5, 3), 60, dtype=np.uint8)
        light = np.full((5, 5, 3), 180, dtype=np.uint8)
        image = np.full((5, 5, 3), 120, dtype=np.uint8)

        result = reference_color_align(self._data(image, [dark, light]))

        aligned = result["results"]["reference_crops"][0]
        self.assertTrue(np.array_equal(aligned[0], dark))
        self.assertTrue(np.array_equal(aligned[1], light))

    def test_accepts_references_without_brightness_range(self):
        image = np.full((5, 5, 3), 100, dtype=np.uint8)
        result = reference_color_align(self._data(image, [image.copy(), image.copy()]))
        self.assertIsNone(result["error"])

    def test_duplicate_source_percentiles_are_handled(self):
        image = np.full((8, 8, 3), 100, dtype=np.uint8)
        refs = [
            np.full((2, 2, 3), value, dtype=np.uint8)
            for value in (60, 100, 160)
        ]
        output = _reference_tone_map(image, refs)
        self.assertIsNotNone(output)
        self.assertEqual(output.shape, image.shape)

    def test_strength_zero_preserves_grayscale_shape_and_values(self):
        image = np.arange(64, dtype=np.uint8).reshape(8, 8)
        refs = [np.full((2, 2), value, dtype=np.uint8) for value in (20, 80, 140)]
        output = _reference_tone_map(image, refs, strength=0.0)
        self.assertEqual(output.shape, image.shape)
        self.assertTrue(np.allclose(output, image, atol=1))

    def test_uses_current_branch_image_as_alignment_source(self):
        source = np.full((8, 8, 3), (20, 40, 80), dtype=np.uint8)
        reference = np.full((8, 8, 3), (120, 150, 190), dtype=np.uint8)
        data = {"images": [source], "results": {"reference_crops": [[source.copy()]]}, "meta": {
            "branch_reference_sources": [{"id": "ref-branch", "label": "Ref", "image": reference}]
        }}

        result = reference_color_align(data, reference_branch="ref-branch", mode="location")

        self.assertIsNone(result.get("error"))
        self.assertTrue(np.allclose(result["images"][0], source, atol=3))
        self.assertEqual(result["meta"]["reference_color_align"]["reference_branch"], "ref-branch")

    def test_dynamic_branch_id_is_not_reset_to_auto(self):
        params = _clamp_params("reference_color_align", {"reference_branch": "branch-instance-id"})
        self.assertEqual(params["reference_branch"], "branch-instance-id")

    def test_selected_branch_image_is_aligned_to_reference_crops(self):
        source = np.full((8, 8, 3), (20, 40, 80), dtype=np.uint8)
        crop = np.full((4, 4, 3), (30, 50, 90), dtype=np.uint8)
        reference = np.full((8, 8, 3), (120, 150, 190), dtype=np.uint8)
        data = {"images": [source], "results": {"reference_crops": [[crop]]}, "meta": {
            "branch_reference_sources": [{"id": "target", "label": "Ag 2", "image": reference}]
        }}

        original_crop = crop.copy()
        result = reference_color_align(data, reference_branch="target", mode="location")

        self.assertTrue(np.allclose(result["images"][0].mean(axis=(0, 1)), crop.mean(axis=(0, 1)), atol=3))
        self.assertTrue(np.array_equal(result["results"]["reference_crops"][0][0], original_crop))

    def test_reports_missing_reference_crops(self):
        image = np.full((8, 8, 3), 80, dtype=np.uint8)
        missing_crops = reference_color_align({
            "images": [image], "results": {}, "meta": {
                "branch_reference_sources": [{"id": "target", "image": image}]
            }
        }, reference_branch="target")
        self.assertEqual(missing_crops["error"], "E3545")

    def test_repeated_merged_crop_rows_are_consumed_only_once(self):
        source = np.full((8, 8, 3), 100, dtype=np.uint8)
        crops = [
            np.full((3, 3, 3), (20, 40, 60), dtype=np.uint8),
            np.full((3, 3, 3), (80, 100, 120), dtype=np.uint8),
            np.full((3, 3, 3), (140, 160, 180), dtype=np.uint8),
        ]
        data = {"images": [source], "results": {"reference_crops": [crops, crops, crops]}, "meta": {}}

        result = reference_color_align(data, reference_branch="reference-branch")

        self.assertEqual(len(result["results"]["reference_crops"]), 1)
        self.assertEqual(len(result["results"]["reference_crops"][0]), 3)



if __name__ == "__main__":
    unittest.main()
