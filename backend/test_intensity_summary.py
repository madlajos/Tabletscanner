import unittest
import numpy as np
from proc_elements.calc_intensity import calculate_intensity_stats
from proc_elements.curve_fitting import fit_curve


def data_for(images, masks=None):
    return {'images': images, 'count': len(images), 'error': None, 'history': [], 'meta': {},
            'results': {'masks': masks} if masks is not None else {}}


class IntensitySummaryTests(unittest.TestCase):
    def test_pooled_pixels_are_weighted_and_percentiles_recomputed(self):
        images = [np.array([[0]], np.uint8), np.full((1, 3), 100, np.uint8)]
        data = calculate_intensity_stats(data_for(images), display_mode='pooled')
        self.assertIsNone(data['error'])
        group = data['results']['intensity_summary']['groups'][0]
        self.assertEqual(group['sample_count'], 2)
        stat = group['channels'][0]
        self.assertEqual(stat['mean'], 75)
        self.assertEqual(stat['median'], 100)
        self.assertEqual(stat['pixel_count'], 4)
        self.assertEqual([s['mean'] for s in data['results']['intensity_stats']], [0, 100])
        self.assertEqual([sample['label'] for sample in data['results']['intensity_summary']['samples']], ['1', '2'])
        self.assertEqual([sample['channels'][0]['mean'] for sample in data['results']['intensity_summary']['samples']], [0, 100])

    def test_groups_pool_only_masked_pixels(self):
        images = [np.array([[v, 255]], np.uint8) for v in (10, 50, 30)]
        masks = [np.array([[255, 0]], np.uint8)] * 3
        data = calculate_intensity_stats(data_for(images, masks), display_mode='grouped', group_labels='["A", "B", "A"]')
        groups = data['results']['intensity_summary']['groups']
        self.assertEqual([(g['label'], g['image_indices']) for g in groups], [('A', [0, 2]), ('B', [1])])
        self.assertEqual([g['channels'][0]['mean'] for g in groups], [20, 50])

    def test_color_channels_and_empty_masks(self):
        image = np.array([[[10, 30, 90]]], np.uint8)
        data = calculate_intensity_stats(data_for([image, image]), display_mode='pooled')
        self.assertEqual([s['mean'] for s in data['results']['intensity_summary']['groups'][0]['channels']], [10, 30, 90])
        data = calculate_intensity_stats(data_for([image], [np.zeros((1, 1), np.uint8)]), display_mode='pooled')
        self.assertEqual(data['results']['intensity_summary']['groups'][0]['channels'], [None, None, None])

    def test_invalid_parameters_fail(self):
        image = np.ones((2, 2), np.uint8)
        for labels in ('[]', '["A", "B"]', '[""]', '{"a":1}', 'invalid'):
            data = calculate_intensity_stats(data_for([image]), display_mode='grouped', group_labels=labels)
            self.assertEqual(data['error'], 'E2509')
        self.assertEqual(calculate_intensity_stats(data_for([image]), display_mode='bad')['error'], 'E2508')
        self.assertEqual(calculate_intensity_stats(data_for([image]), percentiles=[float('nan')])['error'], 'E2504')
        self.assertEqual(calculate_intensity_stats(data_for([image, image], [image]))['error'], 'E2502')
        self.assertEqual(calculate_intensity_stats(data_for([]))['error'], 'E2501')
        mixed = data_for([image, np.ones((2, 2, 3), np.uint8)])
        self.assertEqual(calculate_intensity_stats(mixed, display_mode='pooled')['error'], 'E2510')

    def test_grouping_does_not_change_curve_fitting_input(self):
        images = [np.full((3, 3), v, np.uint8) for v in (10, 20, 30)]
        data = calculate_intensity_stats(data_for(images), display_mode='grouped', group_labels='["A", "A", "A"]')
        data['results']['sequence_value'] = [1, 2, 3]
        result = fit_curve(data, x_name='sequence_value', y_name='mean', model='linear')
        self.assertIsNone(result['error'])
        self.assertEqual(len(result['results']['intensity_stats']), 3)


if __name__ == '__main__':
    unittest.main()
