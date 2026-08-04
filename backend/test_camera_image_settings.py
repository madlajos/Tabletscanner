import unittest
from unittest.mock import ANY, patch

from cameracontrol import (
    center_camera_axis,
    get_camera_properties,
    validate_camera_integer_param,
    validate_param,
)
from settings_manager import migrate_settings


class CameraImageSettingsTests(unittest.TestCase):
    def setUp(self):
        self.properties = {
            'Width': {'min': 64, 'max': 4096, 'inc': 16},
            'OffsetX': {'min': 0, 'max': 1024, 'inc': 8},
        }

    def test_accepts_live_range_and_increment(self):
        self.assertEqual(
            4000,
            validate_camera_integer_param('Width', 4000, self.properties),
        )

    def test_normalizes_to_nearest_supported_value(self):
        self.assertEqual(2000, validate_camera_integer_param('Width', 2001, self.properties))
        self.assertEqual(64, validate_camera_integer_param('Width', 32, self.properties))
        self.assertEqual(4096, validate_camera_integer_param('Width', 5000, self.properties))

    def test_gain_uses_live_range_and_increment(self):
        properties = {'Gain': {'min': 0.0, 'max': 24.0, 'inc': 0.1}}
        self.assertEqual(1.2, validate_param('Gain', 1.2, properties))
        self.assertEqual(1.2, validate_param('Gain', 1.23, properties))

    def test_continuous_gain_accepts_any_in_range_value(self):
        properties = {'Gain': {'min': 0.0, 'max': 24.0, 'inc': 0.0}}
        self.assertEqual(1.23456789, validate_param('Gain', 1.23456789, properties))
        self.assertEqual(24.0, validate_param('Gain', 30, properties))

    def test_float_nodes_without_increment_remain_available(self):
        class Node:
            def __init__(self, minimum, maximum, increment=1):
                self.minimum = minimum
                self.maximum = maximum
                self.increment = increment

            def GetMin(self):
                return self.minimum

            def GetMax(self):
                return self.maximum

            def GetInc(self):
                if self.increment is None:
                    raise RuntimeError('node does not have an increment')
                return self.increment

        class Camera:
            Width = Node(4, 2848, 4)
            Height = Node(1, 2848, 1)
            OffsetX = Node(0, 8, 4)
            OffsetY = Node(0, 8, 1)
            ExposureTime = Node(10.0, 10000000.0, 1.0)
            Gamma = Node(0.0, 4.0, None)
            Gain = Node(0.0, 24.0, None)
            AcquisitionFrameRate = Node(0.1, 1000000.0, None)

        properties = get_camera_properties(Camera())

        self.assertEqual(0.0, properties['Gamma']['inc'])
        self.assertEqual(0.0, properties['Gain']['inc'])

    def test_schema_five_migrates_legacy_geometry_out_of_camera_params(self):
        migrated, changed = migrate_settings({
            'settings_schema_version': 5,
            'camera_params': {
                'Width': 2048,
                'Height': 1536,
                'OffsetX': 16,
                'OffsetY': 24,
                'ExposureTime': 100000,
                'Gamma': 1,
            },
        })
        self.assertTrue(changed)
        self.assertEqual(8, migrated['settings_schema_version'])
        self.assertEqual({
            'override_enabled': False,
            'width': 2048,
            'height': 1536,
            'offset_x': 16,
            'offset_y': 24,
        }, migrated['camera_image_settings'])
        self.assertNotIn('Width', migrated['camera_params'])
        self.assertEqual(100000, migrated['camera_params']['ExposureTime'])
        self.assertEqual(0.0, migrated['camera_params']['Gain'])
        self.assertEqual(
            0.0,
            migrated['auto_measurement_settings']['capture_plan'][0]['gain'],
        )

    def test_center_falls_back_when_native_center_node_lookup_raises(self):
        class CameraWithoutCenterNode:
            def __getattr__(self, name):
                if name == 'CenterX':
                    raise RuntimeError("Node not existing")
                raise AttributeError(name)

        properties = {
            'OffsetX': {'min': 0, 'max': 1648, 'inc': 4},
        }
        geometry = {
            'values': {'width': 1200, 'height': 2840, 'offset_x': 824, 'offset_y': 4},
            'limits': {},
        }
        with (
            patch('cameracontrol.get_camera_properties', return_value=properties),
            patch('cameracontrol.validate_and_set_camera_param') as setter,
            patch('cameracontrol.get_camera_image_geometry', return_value=geometry),
        ):
            result = center_camera_axis(CameraWithoutCenterNode(), 'x')

        setter.assert_called_once_with(
            ANY, 'OffsetX', 824, properties
        )
        self.assertEqual(824, result['values']['offset_x'])


if __name__ == '__main__':
    unittest.main()
