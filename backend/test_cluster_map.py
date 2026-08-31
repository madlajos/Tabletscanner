import unittest
import numpy as np

from proc_elements.cluster_map import (
    _component_similarity, _match_clusters_to_references,
    _membership_from_existing_labels, _smooth_memberships_for_visualization,
    _smooth_absolute_memberships_for_display, _soft_palette_visualization,
    cluster_reference_map,
)


def _data(image, labels, centers, crops=None, sequence=None):
    results = {
        "kmeans_source_images": [image], "kmeans_labeled_images": [image.copy()],
        "kmeans_label_maps": [labels], "kmeans_centers": [np.asarray(centers, np.float32)],
    }
    if crops is not None:
        results["reference_crops"] = [crops]
    if sequence is not None:
        results["reference_sequence"] = [{"items": sequence}]
    return {"error": None, "images": [image], "count": 1, "results": results,
            "meta": {"kmeans_cluster": {"color_space": "GRAY"}}}


class ClusterReferenceMapTests(unittest.TestCase):
    def setUp(self):
        self.gray = np.array([[10, 20, 30, 100, 110, 120, 200, 210]], np.uint8)
        self.labels = np.array([[1, 1, 1, 2, 2, 2, 3, 3]], np.uint8)

    def test_membership_uses_existing_labels(self):
        result = _membership_from_existing_labels(
            self.gray, self.labels > 0, self.labels, [1, 2, 3])
        np.testing.assert_array_equal(result["component_map"], self.labels)
        np.testing.assert_array_equal(result["hard_labels"], self.labels)
        self.assertEqual([s["pixel_count"] for s in result["component_statistics"]], [3, 3, 2])
        self.assertEqual([s["cluster_label"] for s in result["component_statistics"]], [1, 2, 3])
        self.assertAlmostEqual(result["component_medians"][1], 110.0 / 255.0)
        self.assertTrue(all(s["pixel_count"] > 0 for s in result["component_statistics"]))

    def test_reference_label_is_upstream_channel(self):
        membership = _membership_from_existing_labels(
            self.gray, self.labels > 0, self.labels, [1, 2, 3])
        base = {"cluster_labels": [1, 2, 3], "_membership_result": membership}
        for label in (1, 2, 3):
            similarity, _, _, _ = _component_similarity(
                self.gray[..., None].astype(np.float32), self.labels,
                dict(base, selected_labels="1,2,3", reference_label=str(label)))
            expected = membership["membership_raw"][..., label - 1]
            np.testing.assert_array_equal(similarity, expected)

    def test_one_to_one_matching(self):
        refs = np.asarray([89.18, 130.58, 34.41], np.float32) / 255.0
        info = [{"original_index": 0, "name": "MCC"},
                {"original_index": 1, "name": "2"},
                {"original_index": 2, "name": "3"}]
        mapping = _match_clusters_to_references(
            np.asarray([43.7, 56.5, 79.6], np.float32) / 255.0, refs, info)
        self.assertEqual([m["cluster_label"] for m in mapping], [1, 2, 3])
        self.assertEqual(len({m["reference_original_index"] for m in mapping}), 3)

    def test_node_keeps_upstream_segmentation_and_all_components(self):
        image = np.tile(self.gray, (2, 1))
        labels = np.tile(self.labels, (2, 1))
        crops = [np.full((2, 2), value, np.uint8) for value in (34, 89, 131)]
        result = cluster_reference_map(_data(image, labels, [[20], [110], [205]], crops))
        self.assertIsNone(result["error"])
        np.testing.assert_array_equal(result["results"]["cluster_map_class_label_maps"][0], labels)
        self.assertEqual(result["results"]["empty_upstream_labels"][0], [])
        self.assertEqual(
            [s["pixel_count"] for s in result["results"]["membership_component_statistics"][0]],
            [6, 6, 4])

    def test_sequence_centroids_stay_canonical(self):
        sequence = [
            {"label": "2", "original_index": 1, "score": 130.58154296875,
             "scores": {"GRAY": 130.58154296875}},
            {"label": "MCC", "original_index": 0, "score": 89.18115234375,
             "scores": {"GRAY": 89.18115234375}},
            {"label": "3", "original_index": 2, "score": 34.408447265625,
             "scores": {"GRAY": 34.408447265625}},
        ]
        result = cluster_reference_map(_data(
            self.gray, self.labels, [[20], [110], [205]], sequence=sequence))
        np.testing.assert_allclose(
            result["results"]["reference_initial_centroids_uint8"][0],
            [89.18115234375, 130.58154296875, 34.408447265625], atol=1e-5)
        self.assertEqual(result["results"]["initial_centroid_source"][0],
                         "reference_sequence_gray_score")

    def test_accepted_and_remainder_use_soft_upstream_channel(self):
        crops = [np.full((2, 2), value, np.uint8) for value in (34, 89, 131)]
        result = cluster_reference_map(
            _data(self.gray, self.labels, [[20], [110], [205]], crops),
            accepted_components=[{"selected_labels": "1,2,3", "reference_label": "3"}],
            remainder_as_last=True)
        expected = result["results"]["matlab_membership_raw"][0][..., 2]
        accepted, remainder = result["results"]["cluster_map_components_raw"][0]
        np.testing.assert_array_equal(accepted, expected)
        np.testing.assert_allclose(remainder, 1.0 - expected, atol=1e-7)

    def test_missing_upstream_centers_errors(self):
        data = _data(self.gray, self.labels, [[20], [110], [205]], [self.gray] * 3)
        del data["results"]["kmeans_centers"]
        self.assertEqual(cluster_reference_map(data)["error"], "E3721")

    def test_soft_palette_blends_explicit_bgr_colors(self):
        memberships = np.array([[[1.0, 0.0, 0.0], [0.5, 0.5, 0.0]]], np.float32)
        palette_bgr = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], np.float32)
        roi = np.ones((1, 2), bool)
        soft_bgr, _ = _soft_palette_visualization(
            memberships, palette_bgr, roi, np.zeros((1, 2, 3), np.uint8))
        np.testing.assert_array_equal(soft_bgr[0, 0], [255, 0, 0])
        np.testing.assert_array_equal(soft_bgr[0, 1], [127, 127, 0])

    def test_visual_smoothing_renormalizes_without_changing_hard_labels(self):
        memberships = np.zeros((5, 7, 3), np.float32)
        memberships[:, :3, 0] = 1.0
        memberships[:, 3:, 1] = 1.0
        roi = np.ones((5, 7), bool)
        hard_before = self.labels.copy()
        smoothed = _smooth_memberships_for_visualization(memberships, roi)
        np.testing.assert_allclose(np.sum(smoothed, axis=2), 1.0, atol=1e-6)
        self.assertGreater(float(smoothed[2, 3, 0]), 0.0)
        np.testing.assert_array_equal(self.labels, hard_before)

    def test_primary_is_clean_heatmap_and_overlay_contains_source(self):
        crops = [np.full((2, 2), value, np.uint8) for value in (34, 89, 131)]
        result = cluster_reference_map(
            _data(self.gray, self.labels, [[20], [110], [205]], crops),
            reference_label="2",
        )
        primary = result["images"][0]
        overlay = result["results"]["cluster_map_overlay_images"][0]
        self.assertFalse(np.array_equal(primary, overlay))
        np.testing.assert_array_equal(
            primary, result["results"]["cluster_map_images"][0],
        )
        np.testing.assert_array_equal(
            primary, result["results"]["cluster_map_heatmap_images"][0],
        )
        self.assertTrue(np.any(primary[..., 0] != primary[..., 1]))
        self.assertTrue(np.any(primary[..., 1] != primary[..., 2]))
        self.assertFalse(
            result["meta"]["cluster_reference_map"]["cluster_map_contains_source_pixels"]
        )

    def test_extreme_memberships_are_monotonic_shoulders(self):
        gray = np.array([[42, 57, 81, 57, 69, 81, 90, 110]], np.uint8)
        labels = np.array([[1, 2, 3, 0, 0, 0, 0, 0]], np.uint8)
        result = _membership_from_existing_labels(
            gray, np.ones(gray.shape, bool), labels, [1, 2, 3],
        )
        brightest = result["membership_raw"][0, 3:, 2]
        darkest = result["membership_raw"][0, :, 0]
        np.testing.assert_allclose(brightest, [0.0, 0.5, 1.0, 1.0, 1.0], atol=0.03)
        self.assertTrue(np.all(np.diff(brightest) >= -1e-7))
        self.assertTrue(np.all(np.diff(darkest) <= 1e-7))
        self.assertEqual(
            result["component_statistics"][0]["membership_type"], "left_shoulder",
        )
        self.assertEqual(
            result["component_statistics"][2]["membership_type"], "right_shoulder",
        )

    def test_middle_membership_is_gaussian_from_its_own_median_iqr(self):
        result = _membership_from_existing_labels(
            self.gray, self.labels > 0, self.labels, [1, 2, 3],
        )
        stats = result["component_statistics"][1]
        self.assertEqual(stats["membership_type"], "gaussian_median_iqr")
        self.assertAlmostEqual(float(result["membership_raw"][0, 4, 1]), 1.0)
        z = ((100.0 / 255.0) - stats["median"]) / stats["denom_used"]
        self.assertAlmostEqual(
            float(result["membership_raw"][0, 3, 1]), np.exp(-0.5 * z * z),
            places=6,
        )

    def test_absolute_display_smoothing_does_not_contrast_stretch(self):
        memberships = np.zeros((9, 9, 1), np.float32)
        memberships[4, 4, 0] = 0.4
        display = _smooth_absolute_memberships_for_display(
            memberships, np.ones((9, 9), bool),
        )
        self.assertGreater(float(display[4, 4, 0]), 0.0)
        self.assertLess(float(display[4, 4, 0]), 0.4)
        self.assertLess(float(np.max(display)), 1.0)

    def test_metadata_declares_gaussian_middle_and_raw_primary(self):
        crops = [np.full((2, 2), value, np.uint8) for value in (34, 89, 131)]
        result = cluster_reference_map(
            _data(self.gray, self.labels, [[20], [110], [205]], crops),
            reference_label="2",
        )
        metadata = result["meta"]["cluster_reference_map"]
        self.assertEqual(metadata["middle_cluster_membership"], "gaussian_median_iqr")
        self.assertEqual(metadata["primary_similarity"], "absolute_raw_membership")
        self.assertFalse(metadata["per_pixel_normalization_used_for_primary"])
        statistics = result["results"]["membership_component_statistics"][0]
        self.assertNotIn("triangle", [item["membership_type"] for item in statistics])
        self.assertIn("display_mean", statistics[1])


if __name__ == "__main__":
    unittest.main()
