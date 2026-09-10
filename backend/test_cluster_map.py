import unittest
import numpy as np

from proc_elements.cluster_map import (
    _component_similarity, _match_clusters_to_references,
    _membership_from_existing_labels, _membership_from_fixed_centroids,
    _smooth_memberships_for_visualization,
    _smooth_absolute_memberships_for_display, _soft_palette_visualization,
    _jet_display_curve,
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

    def test_fixed_centroid_membership_matches_matlab_sequence(self):
        gray = np.array([[10, 20, 30, 100, 110, 120, 200, 210]], np.uint8)
        mask = np.ones(gray.shape, bool)
        result = _membership_from_fixed_centroids(
            gray, mask, np.asarray([131, 34, 89], np.float64) / 255.0,
        )
        np.testing.assert_array_equal(
            result["component_map"],
            np.array([[1, 1, 1, 2, 3, 3, 3, 3]], np.uint8),
        )
        self.assertEqual(float(result["component_denoms"][0]), 10.0 / 255.0)
        np.testing.assert_allclose(
            np.sum(result["membership_normalized"], axis=2), 1.0, atol=1e-7,
        )

    def test_fixed_centroid_zero_iqr_uses_requested_epsilon(self):
        gray = np.full((2, 2), 50, np.uint8)
        result = _membership_from_fixed_centroids(
            gray, np.ones(gray.shape, bool), [50.0 / 255.0],
        )
        self.assertEqual(float(result["component_denoms"][0]), 1e-12)
        np.testing.assert_array_equal(result["membership_raw"], 1.0)

    def test_each_membership_is_zero_at_the_other_fixed_centroids(self):
        gray = np.array([[0, 50, 88, 90, 128, 165, 167, 205, 255]], np.uint8)
        centers_uint8 = np.asarray([50, 128, 205], np.float64)
        result = _membership_from_fixed_centroids(
            gray, np.ones(gray.shape, bool), centers_uint8 / 255.0,
        )
        raw = result["membership_raw"][0]
        for component_index, own_center in enumerate(centers_uint8.astype(int)):
            for other_center in centers_uint8.astype(int):
                if other_center == own_center:
                    continue
                pixel_index = int(np.flatnonzero(gray[0] == other_center)[0])
                self.assertEqual(float(raw[pixel_index, component_index]), 0.0)

    def test_reference_label_uses_absolute_membership_channel(self):
        membership = _membership_from_existing_labels(
            self.gray, self.labels > 0, self.labels, [1, 2, 3])
        base = {"cluster_labels": [1, 2, 3], "_membership_result": membership}
        for label in (1, 2, 3):
            similarity, _, _, _ = _component_similarity(
                self.gray[..., None].astype(np.float32), self.labels,
                dict(base, selected_labels="1,2,3", reference_label=str(label)))
            expected = membership["membership_raw"][..., label - 1]
            np.testing.assert_array_equal(similarity, expected)

    def test_two_reference_labels_use_their_combined_median_and_iqr(self):
        membership = _membership_from_existing_labels(
            self.gray, self.labels > 0, self.labels, [1, 2, 3])
        similarity, reference, _, _ = _component_similarity(
            self.gray[..., None].astype(np.float32), self.labels,
            {"cluster_labels": [1, 2, 3], "_membership_result": membership,
             "selected_labels": "1,2", "reference_label": "1,2"})
        pixels = self.gray[np.isin(self.labels, [1, 2])].astype(np.float32) / 255.0
        expected_center = float(np.median(pixels))
        q25, q75 = np.percentile(pixels, [25.0, 75.0])
        expected = np.exp(-0.5 * ((self.gray / 255.0 - expected_center) / (q75 - q25)) ** 2)
        self.assertAlmostEqual(float(reference[0]), expected_center)
        np.testing.assert_allclose(similarity, expected, atol=1e-6)

    def test_reference_membership_is_evaluated_over_the_complete_roi(self):
        membership = _membership_from_existing_labels(
            self.gray, self.labels > 0, self.labels, [1, 2, 3])
        similarity, _, _, evaluation_mask = _component_similarity(
            self.gray[..., None].astype(np.float32), self.labels,
            {"cluster_labels": [1, 2, 3], "_membership_result": membership,
             "selected_labels": "1,2", "reference_label": "1,2"})
        expected_mask = self.labels > 0
        np.testing.assert_array_equal(evaluation_mask, expected_mask)
        self.assertTrue(np.any(similarity[self.labels == 3] > 0.0))

    def test_two_cluster_reference_round_trips_and_can_be_accepted(self):
        crops = [np.full((2, 2), value, np.uint8) for value in (34, 89, 131)]
        result = cluster_reference_map(
            _data(self.gray, self.labels, [[20], [110], [205]], crops),
            selected_labels="1,2", reference_label="1,2",
            accepted_components=[{
                "selected_labels": "1,2", "reference_label": "1,2",
            }],
        )
        self.assertIsNone(result["error"])
        self.assertEqual(result["results"]["cluster_map_reference_label"], "1,2")
        info = result["results"]["cluster_map_component_info"][0][0]
        self.assertEqual(info["reference_label"], "1,2")
        self.assertEqual(len(info["original_reference_index"]), 2)

    def test_one_to_one_matching(self):
        refs = np.asarray([89.18, 130.58, 34.41], np.float32) / 255.0
        info = [{"original_index": 0, "name": "MCC"},
                {"original_index": 1, "name": "2"},
                {"original_index": 2, "name": "3"}]
        mapping = _match_clusters_to_references(
            np.asarray([43.7, 56.5, 79.6], np.float32) / 255.0, refs, info)
        self.assertEqual([m["cluster_label"] for m in mapping], [1, 2, 3])
        self.assertEqual(len({m["reference_original_index"] for m in mapping}), 3)

    def test_node_preserves_upstream_kmeans_labels(self):
        image = np.tile(self.gray, (2, 1))
        labels = np.tile(self.labels, (2, 1))
        crops = [np.full((2, 2), value, np.uint8) for value in (34, 89, 131)]
        result = cluster_reference_map(_data(image, labels, [[20], [110], [205]], crops))
        self.assertIsNone(result["error"])
        np.testing.assert_array_equal(
            result["results"]["cluster_map_class_label_maps"][0], labels,
        )
        self.assertEqual(result["results"]["empty_upstream_labels"][0], [])
        self.assertEqual(
            [s["pixel_count"] for s in result["results"]["membership_component_statistics"][0]],
            [6, 6, 4])

    def test_reference_centroids_cannot_empty_an_upstream_cluster(self):
        image = np.array([[70, 90, 110, 130, 150, 190]], np.uint8)
        labels = np.array([[1, 1, 2, 2, 3, 3]], np.uint8)
        crops = [np.full((2, 2), value, np.uint8) for value in (28, 104, 158)]
        result = cluster_reference_map(
            _data(image, labels, [[80], [120], [170]], crops),
        )
        np.testing.assert_array_equal(
            result["results"]["cluster_map_class_label_maps"][0], labels,
        )
        self.assertEqual(
            [item["pixel_count"] for item in
             result["results"]["membership_component_statistics"][0]],
            [2, 2, 2],
        )
        self.assertEqual(result["results"]["empty_upstream_labels"][0], [])

    def test_reference_models_can_cross_hard_label_regions(self):
        image = np.array([[90, 104, 110, 115, 117, 119]], np.uint8)
        labels = np.array([[1, 2, 2, 3, 3, 3]], np.uint8)
        crops = [np.full((3, 3), value, np.uint8) for value in (28, 104, 158)]
        result = cluster_reference_map(
            _data(image, labels, [[90], [106], [117]], crops),
            selected_labels="1,2,3", reference_label="2",
            center_mode="reference_mean",
        )
        memberships = result["results"]["matlab_membership_raw"][0]
        models = result["meta"]["cluster_reference_map"]["reference_models"][0]
        np.testing.assert_allclose(
            [model["center"][0] for model in models],
            np.asarray([28, 104, 158]) / 255.0,
            atol=1.0 / 255.0,
        )
        expected_label_1 = (104.0 - 90.0) / (104.0 - 28.0)
        self.assertAlmostEqual(
            float(memberships[0, 0, 0]), float(expected_label_1), places=5,
        )
        self.assertAlmostEqual(float(memberships[0, 1, 1]), 1.0, places=5)
        expected_label_2 = (158.0 - 115.0) / (158.0 - 104.0)
        self.assertAlmostEqual(
            float(memberships[0, 3, 1]), expected_label_2, places=5,
        )
        expected_label_3 = (117.0 - 104.0) / (158.0 - 104.0)
        self.assertAlmostEqual(
            float(memberships[0, 4, 2]), float(expected_label_3), places=5,
        )

    def test_reference_membership_is_one_at_own_center_and_zero_at_neighbors(self):
        gray = np.array([[30, 90, 150]], np.uint8)
        labels = np.array([[1, 2, 3]], np.uint8)
        models = [
            {"center": value / 255.0, "spread": 1.0, "amplitude": 1.0}
            for value in (30, 90, 150)
        ]
        result = _membership_from_existing_labels(
            gray, np.ones_like(gray, dtype=bool), labels, [1, 2, 3], models,
        )
        np.testing.assert_allclose(
            result["membership_raw"][0], np.eye(3, dtype=np.float32), atol=1e-7,
        )

    def test_extreme_references_stay_at_one_beyond_their_outer_centers(self):
        gray = np.array([[0, 30, 90, 150, 210]], np.uint8)
        labels = np.array([[1, 1, 2, 3, 3]], np.uint8)
        models = [
            {"center": value / 255.0, "spread": 1.0, "amplitude": 1.0}
            for value in (30, 90, 150)
        ]
        result = _membership_from_existing_labels(
            gray, np.ones_like(gray, dtype=bool), labels, [1, 2, 3], models,
        )
        raw = result["membership_raw"][0]
        self.assertAlmostEqual(float(raw[0, 0]), 1.0, places=6)
        self.assertAlmostEqual(float(raw[4, 2]), 1.0, places=6)
        self.assertAlmostEqual(float(raw[0, 1]), 0.0, places=6)
        self.assertAlmostEqual(float(raw[4, 1]), 0.0, places=6)

    def test_center_mode_controls_the_actual_membership_centers(self):
        image = np.array([[80, 100, 110, 120, 130, 150]], np.uint8)
        labels = np.array([[1, 1, 2, 2, 3, 3]], np.uint8)
        crops = [np.full((2, 2), value, np.uint8) for value in (30, 90, 150)]
        data_args = (image, labels, [[90], [115], [140]], crops)

        expected = {
            "cluster_median": [90, 115, 140],
            "min_max_midpoint": [90, 115, 140],
            "reference_mean": [30, 90, 150],
            "reference_mean_half": [15, 45, 75],
        }
        for mode, centers in expected.items():
            result = cluster_reference_map(_data(*data_args), center_mode=mode)
            models = result["meta"]["cluster_reference_map"]["reference_models"][0]
            np.testing.assert_allclose(
                [model["center"][0] for model in models],
                np.asarray(centers) / 255.0,
                atol=1e-6,
            )

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

    def test_accepted_and_remainder_use_absolute_membership_channel(self):
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
        self.assertFalse(
            result["meta"]["cluster_reference_map"]["cluster_map_contains_source_pixels"]
        )

    def test_upstream_mask_is_black_in_current_and_component_heatmaps(self):
        labels = self.labels.copy()
        labels[:, :2] = 0
        crops = [np.full((2, 2), value, np.uint8) for value in (34, 89, 131)]
        result = cluster_reference_map(
            _data(self.gray, labels, [[20], [110], [205]], crops),
            reference_label="2",
            accepted_components='[{"name":"Teszt","selected_labels":"1,2,3","reference_label":"2"}]',
        )
        outside = labels == 0
        np.testing.assert_array_equal(result["images"][0][outside], 0)
        np.testing.assert_array_equal(
            result["results"]["cluster_map_component_images"][0][0][outside], 0,
        )

    def test_label_one_jet_uses_the_same_membership_as_other_labels(self):
        crops = [np.full((2, 2), value, np.uint8) for value in (34, 89, 131)]
        result = cluster_reference_map(
            _data(self.gray, self.labels, [[20], [110], [205]], crops),
            reference_label="1",
        )
        membership = result["results"]["cluster_soft_memberships"][0][..., 0]
        displayed = result["results"]["cluster_map_display_raw"][0]
        np.testing.assert_allclose(
            displayed, _jet_display_curve(membership),
            atol=1e-7,
        )

    def test_every_component_uses_its_own_median_and_iqr_gaussian(self):
        gray = np.array([[42, 57, 81, 57, 69, 81, 90, 110]], np.uint8)
        labels = np.array([[1, 2, 3, 0, 0, 0, 0, 0]], np.uint8)
        roi = np.array([[True, True, True, True, True, True, True, False]])
        result = _membership_from_existing_labels(
            gray, roi, labels, [1, 2, 3],
        )
        intensity = gray.astype(np.float32) / 255.0
        for component, label in enumerate((1, 2, 3)):
            vals = intensity[(labels == label) & roi]
            median = float(np.median(vals))
            q25, q75 = np.percentile(vals, [25.0, 75.0])
            denom = max(float(q75 - q25), 1e-12)
            expected = np.exp(-0.5 * ((intensity - median) / denom) ** 2)
            expected[~roi] = 0.0
            np.testing.assert_allclose(
                result["membership_raw"][..., component], expected, atol=1e-7,
            )
            self.assertEqual(
                result["component_statistics"][component]["membership_type"],
                "gaussian_median_iqr",
            )

    def test_jet_curve_preserves_normalized_membership_gradients(self):
        values = np.asarray([0.0, 0.2, 0.5, 0.7, 0.9, 1.0], np.float32)
        displayed = _jet_display_curve(values)
        np.testing.assert_array_equal(displayed, values)

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

    def test_metadata_declares_linear_membership_and_raw_primary(self):
        crops = [np.full((2, 2), value, np.uint8) for value in (34, 89, 131)]
        result = cluster_reference_map(
            _data(self.gray, self.labels, [[20], [110], [205]], crops),
            reference_label="2",
        )
        metadata = result["meta"]["cluster_reference_map"]
        self.assertEqual(metadata["membership_method"], "piecewise_linear_between_reference_centers")
        self.assertEqual(metadata["extreme_cluster_membership"], "piecewise_linear_shoulder")
        self.assertEqual(metadata["middle_cluster_membership"], "piecewise_linear_triangle")
        self.assertEqual(metadata["primary_similarity"], "absolute_reference_membership")
        self.assertFalse(metadata["per_pixel_normalization_used_for_primary"])
        statistics = result["results"]["membership_component_statistics"][0]
        self.assertEqual(
            [item["membership_type"] for item in statistics],
            ["piecewise_linear_between_reference_centers"] * 3,
        )
        self.assertIn("display_mean", statistics[1])


if __name__ == "__main__":
    unittest.main()
