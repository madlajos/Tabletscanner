import unittest

from settings_manager import (
    DEFAULT_FIRST_TABLET_X_MM,
    DEFAULT_FIRST_TABLET_Y_MM,
    DEFAULT_FIRST_TABLET_Z_MM,
    DEFAULT_TABLET_SPACING_MM,
    TrayGeometryError,
    migrate_settings,
    validate_motion_simulation_settings,
)


def advanced_payload(**overrides):
    payload = {
        'use_virtual_com_port': False,
        'max_height_offset_up_mm': 5,
        'max_height_offset_down_mm': -5,
        'first_tablet_x_mm': DEFAULT_FIRST_TABLET_X_MM,
        'first_tablet_y_mm': DEFAULT_FIRST_TABLET_Y_MM,
        'first_tablet_z_mm': DEFAULT_FIRST_TABLET_Z_MM,
        'tablet_spacing_mm': DEFAULT_TABLET_SPACING_MM,
    }
    payload.update(overrides)
    return payload


class TraySettingsTests(unittest.TestCase):
    def test_default_calibration_is_valid(self):
        normalized = validate_motion_simulation_settings(advanced_payload())
        self.assertEqual(0.0, normalized['first_tablet_y_mm'])
        self.assertEqual(18.3, normalized['tablet_spacing_mm'])

    def test_out_of_range_tray_is_rejected(self):
        with self.assertRaisesRegex(TrayGeometryError, '10 x 10 tray'):
            validate_motion_simulation_settings(
                advanced_payload(first_tablet_x_mm=20, tablet_spacing_mm=18.3)
            )

    def test_y_travel_limit_is_165_mm(self):
        with self.assertRaisesRegex(ValueError, 'between 0 and 165 mm'):
            validate_motion_simulation_settings(
                advanced_payload(first_tablet_y_mm=165.1, tablet_spacing_mm=0.001)
            )

    def test_tray_y_edge_uses_y_travel_limit(self):
        with self.assertRaisesRegex(TrayGeometryError, '10 x 10 tray'):
            validate_motion_simulation_settings(
                advanced_payload(first_tablet_y_mm=1, tablet_spacing_mm=18.3)
            )

    def test_schema_four_coordinates_migrate_to_advanced_settings(self):
        migrated, changed = migrate_settings({
            'settings_schema_version': 4,
            'auto_measurement_settings': {
                'first_tablet_x': 3,
                'first_tablet_y': 4,
                'first_tablet_z': 19,
                'tablet_spacing': 18,
                'capture_plan': [],
            },
            'advanced_settings': {
                'use_virtual_com_port': False,
                'max_height_offset_up_mm': 5,
                'max_height_offset_down_mm': -5,
            },
        })
        self.assertTrue(changed)
        self.assertEqual(9, migrated['settings_schema_version'])
        self.assertEqual(3, migrated['advanced_settings']['first_tablet_x_mm'])
        self.assertEqual(18, migrated['advanced_settings']['tablet_spacing_mm'])
        self.assertNotIn('first_tablet_x', migrated['auto_measurement_settings'])


if __name__ == '__main__':
    unittest.main()
