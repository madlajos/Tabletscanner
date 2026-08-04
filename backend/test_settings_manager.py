#!/usr/bin/env python3
"""Focused hardware-free checks for settings migration and validation."""

import json
import os
import tempfile
import unittest

import settings_manager


def height_offset_matrix(*filter_ids, value=0):
    return {
        key: {channel: value for channel in settings_manager.LIGHT_CHANNELS}
        for key in (settings_manager.EMPTY_FILTER_KEY, *filter_ids)
    }


class SettingsMigrationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.settings_path = os.path.join(self.temp_dir.name, 'settings.json')

    def tearDown(self):
        settings_manager.set_settings({})
        self.temp_dir.cleanup()

    def write_json(self, payload):
        with open(self.settings_path, 'w', encoding='utf-8') as file:
            json.dump(payload, file)

    def read_json(self):
        with open(self.settings_path, 'r', encoding='utf-8') as file:
            return json.load(file)

    def test_v1_migrates_from_dome_and_creates_one_backup(self):
        legacy = {
            'camera_params_dome': {'ExposureTime': 123456, 'Gain': 4, 'Gamma': 1.2},
            'camera_params_bar': {'ExposureTime': 999999, 'Gamma': 2.0},
            'other_settings': {'save_location': 'D:/operator-data'},
            'auto_measurement_settings': {'save_location': 'D:/measurements'},
        }
        self.write_json(legacy)

        settings = settings_manager.load_settings(self.settings_path)

        self.assertEqual(8, settings['settings_schema_version'])
        self.assertEqual({'ExposureTime': 123456.0, 'Gain': 4.0, 'Gamma': 1.2}, settings['camera_params'])
        self.assertNotIn('camera_params_dome', settings)
        self.assertNotIn('camera_params_bar', settings)
        self.assertEqual(
            [{'wavelength': 'vis', 'filter_position': 1, 'exposure_time': 123456.0, 'gain': 4.0, 'gamma': 1.2}],
            settings['auto_measurement_settings']['capture_plan'],
        )
        self.assertEqual({}, settings['lamp_settings']['channels'])
        self.assertEqual(
            settings_manager.OCTOPUS_LIGHT_OUTPUT_SELECTORS,
            settings['lamp_settings']['output_selectors'],
        )
        with open(f'{self.settings_path}.v1.bak', 'r', encoding='utf-8') as backup_file:
            self.assertEqual(legacy, json.load(backup_file))
        self.assertEqual(settings, self.read_json())

        settings_manager.load_settings(self.settings_path)
        self.assertTrue(os.path.exists(f'{self.settings_path}.v1.bak'))

    def test_v1_uses_bar_only_when_dome_is_unavailable(self):
        self.write_json({'camera_params_bar': {'ExposureTime': 456, 'Gamma': 1.5}})

        settings = settings_manager.load_settings(self.settings_path)

        self.assertEqual({'ExposureTime': 456.0, 'Gain': 0.0, 'Gamma': 1.5}, settings['camera_params'])

    def test_schema_v2_migrates_verified_light_selectors_and_creates_backup(self):
        v2 = {
            'settings_schema_version': 2,
            'camera_params': {'ExposureTime': 50000.0, 'Gamma': 1.0},
            'lamp_settings': {
                'channels': {'uv255': {'dim_percent': 50}},
                'output_selectors': {'uv255': 'P1', 'uv310': 'P2', 'uv365': 'P3', 'vis': 'P0'},
            },
            'auto_measurement_settings': {'capture_plan': [{'wavelength': 'vis', 'filter_position': 1}]},
        }
        self.write_json(v2)

        migrated = settings_manager.load_settings(self.settings_path)

        self.assertEqual(8, migrated['settings_schema_version'])
        self.assertEqual(
            settings_manager.OCTOPUS_LIGHT_OUTPUT_SELECTORS,
            migrated['lamp_settings']['output_selectors'],
        )
        self.assertEqual(v2['lamp_settings']['channels'], migrated['lamp_settings']['channels'])
        with open(f'{self.settings_path}.v2.bak', 'r', encoding='utf-8') as backup_file:
            self.assertEqual(v2, json.load(backup_file))

    def test_schema_v3_adds_per_row_camera_settings(self):
        v3 = {
            'settings_schema_version': 3,
            'camera_params': {'ExposureTime': 50000.0, 'Gamma': 1.0},
            'lamp_settings': {
                'channels': {},
                'output_selectors': settings_manager.OCTOPUS_LIGHT_OUTPUT_SELECTORS,
            },
        }
        self.write_json(v3)

        migrated = settings_manager.load_settings(self.settings_path)
        self.assertEqual(8, migrated['settings_schema_version'])
        self.assertEqual(
            [{'wavelength': 'vis', 'filter_position': 1, 'exposure_time': 50000.0, 'gain': 0.0, 'gamma': 1.0}],
            migrated['auto_measurement_settings']['capture_plan'],
        )
        self.assertTrue(os.path.exists(f'{self.settings_path}.v3.bak'))

    def test_schema_v3_repairs_rotated_light_selectors(self):
        v3 = {
            'settings_schema_version': 3,
            'lamp_settings': {
                'channels': {'uv255': {'dim_percent': 50}},
                'output_selectors': {'uv255': 'P3', 'uv310': 'P1', 'uv365': 'P2', 'vis': 'P0'},
            },
        }
        self.write_json(v3)

        repaired = settings_manager.load_settings(self.settings_path)

        self.assertEqual(
            settings_manager.OCTOPUS_LIGHT_OUTPUT_SELECTORS,
            repaired['lamp_settings']['output_selectors'],
        )
        self.assertEqual(v3['lamp_settings']['channels'], repaired['lamp_settings']['channels'])
        self.assertEqual(repaired, self.read_json())

    def test_schema_v3_removes_obsolete_preset_name_without_backup(self):
        v3 = {
            'settings_schema_version': 3,
            'camera_params': {'ExposureTime': 50000.0, 'Gamma': 1.0},
            'other_settings': {'settings_preset_name': 'legacy'},
        }
        self.write_json(v3)

        settings = settings_manager.load_settings(self.settings_path)

        self.assertNotIn('settings_preset_name', settings['other_settings'])
        self.assertFalse(os.path.exists(f'{self.settings_path}.v1.bak'))

    def test_invalid_json_falls_back_to_empty_settings(self):
        with open(self.settings_path, 'w', encoding='utf-8') as file:
            file.write('{not valid json')

        self.assertEqual({}, settings_manager.load_settings(self.settings_path))

    def test_lamp_settings_validation_normalizes_valid_values(self):
        payload = {
            'channels': {
                channel: {
                    'dim_percent': 10,
                    'full_percent': 100,
                    'dim_timeout_seconds': 30,
                    'full_timeout_seconds': 5,
                }
                for channel in settings_manager.UV_LAMP_CHANNELS
            }
        }

        self.assertEqual(payload, settings_manager.validate_lamp_settings(payload))

    def test_lamp_settings_validation_rejects_unsafe_values(self):
        payload = {'channels': {channel: {} for channel in settings_manager.UV_LAMP_CHANNELS}}
        with self.assertRaises(ValueError):
            settings_manager.validate_lamp_settings(payload)

    def test_capture_plan_validation_normalizes_valid_rows(self):
        plan = [
            {'wavelength': 'vis', 'filter_position': 1, 'exposure_time': 50000, 'gain': 0, 'gamma': 1},
            {'wavelength': 'uv365', 'filter_position': 6, 'exposure_time': 75000.5, 'gain': 2.5, 'gamma': 1.2},
        ]
        self.assertEqual(plan, settings_manager.validate_capture_plan(plan))

    def test_capture_plan_validation_rejects_unknown_wavelength_and_filter(self):
        with self.assertRaises(ValueError):
            settings_manager.validate_capture_plan([{'wavelength': 'uv240', 'filter_position': 1, 'exposure_time': 1, 'gain': 0, 'gamma': 1}])
        with self.assertRaises(ValueError):
            settings_manager.validate_capture_plan([{'wavelength': 'vis', 'filter_position': 7, 'exposure_time': 1, 'gain': 0, 'gamma': 1}])
        with self.assertRaises(ValueError):
            settings_manager.validate_capture_plan([{'wavelength': 'vis', 'filter_position': 1, 'exposure_time': 'bad', 'gain': 0, 'gamma': 1}])
        with self.assertRaises(ValueError):
            settings_manager.validate_capture_plan([{'wavelength': 'vis', 'filter_position': 1, 'exposure_time': 1, 'gain': -0.1, 'gamma': 1}])
        self.assertEqual(
            [{'wavelength': 'vis', 'filter_position': 1, 'exposure_time': 1.0, 'gain': 0.5, 'gamma': 1.0}],
            settings_manager.validate_capture_plan([
                {'wavelength': 'uv365', 'filter_position': 2, 'exposure_time': 1, 'gain': 0.5, 'gamma': 1}
            ]),
        )

    def test_filter_settings_validation_normalizes_and_checks_slot_references(self):
        payload = {
            'filters': [{'id': 'uv-310', 'name': 'UV 310', 'wavelength_range': '300–320', 'color': '#12AB34'}],
            'slots': [None, 'uv-310', None, None, None, None],
            'height_offsets_mm': height_offset_matrix('uv-310', value=1.5),
        }
        self.assertEqual({
            'filters': [{'id': 'uv-310', 'name': 'UV 310', 'wavelength_range': '300–320', 'color': '#12ab34'}],
            'slots': [None, 'uv-310', None, None, None, None],
            'height_offsets_mm': {
                **height_offset_matrix('uv-310', value=1.5),
                'empty': {'uv255': 1.5, 'uv310': 1.5, 'uv365': 1.5, 'vis': 0.0},
            },
        }, settings_manager.validate_filter_settings(payload))
        with self.assertRaises(ValueError):
            settings_manager.validate_filter_settings({**payload, 'slots': ['missing', None, None, None, None, None]})
        self.assertEqual(
            [None, None, None, None, None, None],
            settings_manager.validate_filter_settings({
                **payload,
                'slots': ['uv-310', None, None, None, None, None],
            })['slots'],
        )
        duplicate_name = {
            **payload,
            'filters': [payload['filters'][0], {**payload['filters'][0], 'id': 'uv-310-copy', 'name': 'uv 310'}],
        }
        with self.assertRaises(ValueError):
            settings_manager.validate_filter_settings(duplicate_name)

    def test_filter_settings_update_persists_without_losing_other_categories(self):
        settings_manager.set_settings({'camera_params': {'Gamma': 1.0}})
        filter_settings = {
            'filters': [{'id': 'vis', 'name': 'VIS', 'wavelength_range': '400–700', 'color': '#ffffff'}],
            'slots': [None, 'vis', None, None, None, None],
            'height_offsets_mm': height_offset_matrix('vis'),
        }

        self.assertTrue(settings_manager.update_filter_settings(filter_settings, self.settings_path))
        self.assertEqual({'Gamma': 1.0}, settings_manager.get_settings()['camera_params'])
        self.assertEqual(filter_settings, self.read_json()['filter_settings'])

    def test_lamp_output_selector_validation_normalizes_values(self):
        payload = {'output_selectors': {
            'uv255': 'p2', 'uv310': 'P3', 'uv365': 'P1', 'vis': 'P0'
        }}
        self.assertEqual({
            'output_selectors': settings_manager.OCTOPUS_LIGHT_OUTPUT_SELECTORS
        }, settings_manager.validate_lamp_output_selectors(payload))

    def test_lamp_output_selector_validation_rejects_commands_and_missing_channels(self):
        with self.assertRaises(ValueError):
            settings_manager.validate_lamp_output_selectors({'output_selectors': {'uv255': 'M106 P0'}})

    def test_lamp_output_selector_validation_rejects_duplicate_outputs(self):
        with self.assertRaises(ValueError):
            settings_manager.validate_lamp_output_selectors({'output_selectors': {
                'uv255': 'P1', 'uv310': 'P1', 'uv365': 'P2', 'vis': 'P0'
            }})

    def test_lamp_output_selector_validation_rejects_rotated_mapping(self):
        with self.assertRaises(ValueError):
            settings_manager.validate_lamp_output_selectors({'output_selectors': {
                'uv255': 'P3', 'uv310': 'P1', 'uv365': 'P2', 'vis': 'P0'
            }})

    def test_motion_simulation_setting_requires_boolean(self):
        payload = {
            'use_virtual_com_port': True,
            'max_height_offset_up_mm': 5,
            'max_height_offset_down_mm': -4,
            'first_tablet_x_mm': 2.9,
            'first_tablet_y_mm': 10.6,
            'first_tablet_z_mm': 20,
            'tablet_spacing_mm': 18.3,
        }
        self.assertEqual(
            {
                'use_virtual_com_port': True,
                'max_height_offset_up_mm': 5.0,
                'max_height_offset_down_mm': -4.0,
                'first_tablet_x_mm': 2.9,
                'first_tablet_y_mm': 10.6,
                'first_tablet_z_mm': 20.0,
                'tablet_spacing_mm': 18.3,
            },
            settings_manager.validate_motion_simulation_settings(payload),
        )
        with self.assertRaises(ValueError):
            settings_manager.validate_motion_simulation_settings({
                **payload,
                'use_virtual_com_port': 'true',
            })
        with self.assertRaises(ValueError):
            settings_manager.validate_motion_simulation_settings({
                **payload,
                'max_height_offset_down_mm': 4,
            })

    def test_filter_height_offset_uses_configured_limits(self):
        payload = {
            'filters': [{
                'id': 'green',
                'name': 'Zöld',
                'wavelength_range': '500–570',
                'color': '#00ff00',
            }],
            'slots': [None, 'green', None, None, None, None],
            'height_offsets_mm': height_offset_matrix('green', value=5),
        }
        self.assertEqual(
            5.0,
            settings_manager.validate_filter_settings(
                payload,
                max_height_offset_up_mm=5,
                max_height_offset_down_mm=-4,
            )['height_offsets_mm']['green']['uv255'],
        )
        with self.assertRaises(ValueError):
            settings_manager.validate_filter_settings(
                {
                    **payload,
                    'height_offsets_mm': {
                        **payload['height_offsets_mm'],
                        'green': {**payload['height_offsets_mm']['green'], 'uv255': -4.1},
                    },
                },
                max_height_offset_up_mm=5,
                max_height_offset_down_mm=-4,
            )

    def test_schema_v7_expands_filter_offsets_per_wavelength(self):
        migrated, changed = settings_manager.migrate_settings({
            'settings_schema_version': 7,
            'filter_settings': {
                'filters': [{
                    'id': 'green',
                    'name': 'Zöld',
                    'wavelength_range': '500–570',
                    'height_offset_mm': 1.25,
                    'color': '#00ff00',
                }],
                'slots': [None, 'green', None, None, None, None],
            },
        })

        self.assertTrue(changed)
        self.assertEqual(8, migrated['settings_schema_version'])
        self.assertNotIn('height_offset_mm', migrated['filter_settings']['filters'][0])
        self.assertEqual(
            {channel: 1.25 for channel in settings_manager.LIGHT_CHANNELS},
            migrated['filter_settings']['height_offsets_mm']['green'],
        )
        self.assertEqual(0.0, migrated['filter_settings']['height_offsets_mm']['empty']['vis'])


if __name__ == '__main__':
    unittest.main()
