"""Hardware-free checks for detection-only channel selection."""
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from proc_elements.detect_circ import detect_circles
from pipeline_steps import STEP_DEFINITIONS, STEP_EXECUTORS
from pipeline_engine import execute_pipeline
from pipeline_types import PipelineDocument, StepInstance


def data_for(images):
    return dict(images=images, count=len(images), error=None, results={}, meta={}, history=[])


class DetectionChannelTests(unittest.TestCase):
    def setUp(self):
        self.image = np.full((100, 100, 3), (30, 80, 150), np.uint8)
        self.circle = np.array([[[50, 50, 20]]], np.float32)

    def test_channels_search_selected_plane_and_overlay_original(self):
        for channel, plane in [('B', 0), ('G', 1), ('R', 2), ('GRAY', None)]:
            with self.subTest(channel=channel), patch('cv2.HoughCircles', return_value=self.circle) as hough:
                original = self.image.copy()
                result = detect_circles(data_for([self.image]), detection_channel=channel)
                expected = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY) if plane is None else original[:, :, plane]
                np.testing.assert_array_equal(hough.call_args.args[0], expected)
                np.testing.assert_array_equal(result['images'][0], original)
                np.testing.assert_array_equal(result['results']['circle_overlay'][0][10, 10], original[10, 10])
                self.assertFalse(np.array_equal(result['results']['circle_overlay'][0], original))

    def test_masks_preserve_original_channels_in_batch(self):
        for invert in [False, True]:
            for background in ['black', 'white']:
                with self.subTest(invert=invert, background=background), patch('cv2.HoughCircles', return_value=self.circle):
                    result = detect_circles(data_for([self.image, self.image.copy()]), detection_channel='R',
                                            apply_mask=True, invert_mask=invert, mask_background=background)
                self.assertEqual(result['count'], 2)
                for output, mask in zip(result['images'], result['results']['masks']):
                    np.testing.assert_array_equal(output[mask > 0], self.image[mask > 0])
                    self.assertTrue(np.all(output[mask == 0] == (255 if background == 'white' else 0)))

    def test_gray_no_match_and_empty_input(self):
        gray = self.image[:, :, 0].copy()
        with patch('cv2.HoughCircles', return_value=None) as hough:
            result = detect_circles(data_for([gray]), detection_channel='R', apply_mask=True)
        np.testing.assert_array_equal(hough.call_args.args[0], gray)
        np.testing.assert_array_equal(result['images'][0], gray)
        self.assertEqual(result['results']['circles'], [[]])
        self.assertEqual(detect_circles(data_for([]))['error'], 'E3601')

    def test_invalid_channel_and_catalog_executor_default(self):
        self.assertEqual(detect_circles(data_for([self.image]), detection_channel=[])['error'], 'E3611')
        schema = next(p for p in STEP_DEFINITIONS['detect_circles'].params if p.name == 'detection_channel')
        self.assertEqual(schema.default, 'GRAY')
        for params, expected in [({}, cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)),
                                 ({'detection_channel': 'R'}, self.image[:, :, 2])]:
            with patch('cv2.HoughCircles', return_value=None) as hough:
                STEP_EXECUTORS['detect_circles'](data_for([self.image]), params)
            np.testing.assert_array_equal(hough.call_args.args[0], expected)

    def test_full_execution_and_partial_preview(self):
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, 'image.png').write_bytes(cv2.imencode('.png', self.image)[1].tobytes())
            steps = [StepInstance.create(key, {p.name: p.default for p in STEP_DEFINITIONS[key].params})
                     for key in ('load_image', 'detect_circles')]
            steps[0].param_values['source'] = folder
            steps[1].param_values.update(detection_channel='R', apply_mask=True)
            doc = PipelineDocument(steps=steps)
            for up_to in [-1, 1]:
                with patch('cv2.HoughCircles', return_value=self.circle) as hough:
                    result = execute_pipeline(doc, up_to_step=up_to)
                self.assertTrue(result.success, result.errors)
                np.testing.assert_array_equal(hough.call_args.args[0], self.image[:, :, 2])
                np.testing.assert_array_equal(result.data['images'][0][50, 50], self.image[50, 50])


if __name__ == '__main__':
    unittest.main()
