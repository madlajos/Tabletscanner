import unittest

import numpy as np

from proc_elements.generate_histogram import calculate_histograms


def data_for(images, masks=None, mask_key='masks'):
    results = {mask_key: masks} if masks is not None else {}
    return {'images': images, 'count': len(images), 'error': None, 'results': results, 'meta': {}, 'history': []}


class HistogramSummaryTests(unittest.TestCase):
    def test_histogram_and_stats_use_only_pixels_inside_previous_mask(self):
        image = np.array([[10, 20, 250]], np.uint8)
        mask = np.array([[255, 255, 0]], np.uint8)
        data = calculate_histograms(data_for([image], [mask]), bins=2)

        self.assertIsNone(data['error'])
        self.assertEqual(data['results']['histograms'], [[2.0, 0.0]])
        self.assertEqual(data['results']['histogram_stats'][0], {
            'pixel_min': 10, 'pixel_max': 20, 'pixel_mean': 15.0, 'pixel_std': 5.0,
        })

    def test_range_mask_is_also_used(self):
        image = np.array([[0, 255]], np.uint8)
        mask = np.array([[0, 255]], np.uint8)
        data = calculate_histograms(data_for([image], [mask], 'range_masks'), bins=2)
        self.assertEqual(data['results']['histograms'], [[0.0, 1.0]])

    def test_pooled_histogram_sums_counts_but_keeps_per_image_histograms(self):
        data = calculate_histograms(data_for([
            np.array([[0, 0]], np.uint8), np.array([[0, 255]], np.uint8),
        ]), bins=2, display_mode='pooled')
        self.assertIsNone(data['error'])
        self.assertEqual(data['results']['histograms'], [[2.0, 0.0], [1.0, 1.0]])
        group = data['results']['histogram_summary']['groups'][0]
        self.assertEqual(group['histogram'], [3.0, 1.0])
        self.assertEqual(group['sample_count'], 2)

    def test_csv_groups_follow_image_order(self):
        images = [np.full((1, 1), value, np.uint8) for value in (0, 255, 0)]
        data = calculate_histograms(data_for(images), bins=2, display_mode='grouped', group_labels='["A", "B", "A"]')
        groups = data['results']['histogram_summary']['groups']
        self.assertEqual([(row['label'], row['indices'], row['histogram']) for row in groups], [
            ('A', [0, 2], [2.0, 0.0]), ('B', [1], [0.0, 1.0]),
        ])

    def test_invalid_mode_and_labels_fail(self):
        image = np.zeros((1, 1), np.uint8)
        self.assertEqual(calculate_histograms(data_for([image]), display_mode='bad')['error'], 'E2307')
        for labels in ('[]', '["A", "B"]', '[""]', 'invalid'):
            self.assertEqual(calculate_histograms(data_for([image]), display_mode='grouped', group_labels=labels)['error'], 'E2308')


if __name__ == '__main__':
    unittest.main()
