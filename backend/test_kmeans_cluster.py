import unittest

import numpy as np

from proc_elements.kmeans_cluster import (
    _feature_image,
    _initial_labels_from_reference_centers,
    _majority_filter_labels,
    kmeans_cluster,
)


class ReferenceSeededKmeansTests(unittest.TestCase):
    def test_gray_features_include_local_context(self):
        image = np.full((9, 9, 3), 100, dtype=np.uint8)
        image[4, 4] = 200
        features = _feature_image(image, "GRAY")
        self.assertEqual(features.shape, (9, 9, 3))
        self.assertEqual(float(features[4, 4, 0]), 200.0)
        self.assertGreater(float(features[4, 4, 2]), 0.0)

    def test_majority_filter_preserves_background_and_filters_roi(self):
        labels = np.array(
            [[0, 0, 0], [0, 2, 1], [0, 1, 1]], dtype=np.uint8,
        )
        valid = labels > 0
        filtered = _majority_filter_labels(labels, valid, 2)
        self.assertTrue(np.all(filtered[~valid] == 0))
        self.assertEqual(int(filtered[1, 1]), 1)

    def test_initial_labels_represent_close_reference_centers(self):
        sample = np.array([[0.0], [10.0], [20.0], [30.0]], dtype=np.float32)
        centers = np.array([[0.0], [1.0], [30.0]], dtype=np.float32)

        _, labels = _initial_labels_from_reference_centers(sample, centers)

        self.assertEqual(set(labels.reshape(-1).tolist()), {0, 1, 2})

    def test_reference_seeded_mode_does_not_fail_for_overlapping_references(self):
        image = np.array(
            [[[0, 0, 0], [10, 10, 10]], [[20, 20, 20], [30, 30, 30]]],
            dtype=np.uint8,
        )
        reference_crops = [[
            np.full((1, 1, 3), 0, dtype=np.uint8),
            np.full((1, 1, 3), 1, dtype=np.uint8),
            np.full((1, 1, 3), 30, dtype=np.uint8),
        ]]
        data = {
            "images": [image],
            "count": 1,
            "results": {"reference_crops": reference_crops},
        }

        result = kmeans_cluster(data, init_mode="reference_seeded", color_space="GRAY")

        self.assertIsNone(result.get("error"))
        self.assertEqual(len(result["results"]["kmeans_centers"][0]), 3)
        self.assertEqual(result["results"]["kmeans_centers"][0].shape, (3, 3))

    def test_reference_fixed_reports_robust_classifier_details(self):
        image = np.tile(np.arange(12, dtype=np.uint8), (12, 1)) * 20
        image = np.dstack([image, image, image])
        reference_crops = [[image[:, :4], image[:, 4:8], image[:, 8:]]]
        data = {
            "images": [image], "count": 1,
            "results": {"reference_crops": reference_crops},
        }
        result = kmeans_cluster(data, init_mode="reference_fixed", color_space="GRAY")
        info = result["results"]["kmeans_reference_info"][0]
        self.assertIsNone(result.get("error"))
        self.assertTrue(info["reference_classifier"])
        self.assertTrue(info["reference_uses_robust_scale"])
        self.assertTrue(info["spatial_cleanup"])
        self.assertEqual(info["reference_feature_dimension"], 3)
        self.assertEqual(sum(result["results"]["kmeans_counts"][0]), 144)


if __name__ == "__main__":
    unittest.main()
