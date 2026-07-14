#!/usr/bin/env python3
"""Focused hardware-free checks for schema-v2 settings migration."""

import json
import os
import tempfile
import unittest

import settings_manager


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

        self.assertEqual(2, settings['settings_schema_version'])
        self.assertEqual({'ExposureTime': 123456.0, 'Gamma': 1.2}, settings['camera_params'])
        self.assertNotIn('camera_params_dome', settings)
        self.assertNotIn('camera_params_bar', settings)
        self.assertEqual(
            [{'wavelength': 'vis', 'filter_position': 1}],
            settings['auto_measurement_settings']['capture_plan'],
        )
        self.assertEqual({}, settings['lamp_settings']['channels'])
        with open(f'{self.settings_path}.v1.bak', 'r', encoding='utf-8') as backup_file:
            self.assertEqual(legacy, json.load(backup_file))
        self.assertEqual(settings, self.read_json())

        settings_manager.load_settings(self.settings_path)
        self.assertTrue(os.path.exists(f'{self.settings_path}.v1.bak'))

    def test_v1_uses_bar_only_when_dome_is_unavailable(self):
        self.write_json({'camera_params_bar': {'ExposureTime': 456, 'Gamma': 1.5}})

        settings = settings_manager.load_settings(self.settings_path)

        self.assertEqual({'ExposureTime': 456.0, 'Gamma': 1.5}, settings['camera_params'])

    def test_schema_v2_round_trips_without_backup(self):
        v2 = {
            'settings_schema_version': 2,
            'camera_params': {'ExposureTime': 50000.0, 'Gamma': 1.0},
            'lamp_settings': {'channels': {}},
            'auto_measurement_settings': {'capture_plan': [{'wavelength': 'vis', 'filter_position': 1}]},
        }
        self.write_json(v2)

        self.assertEqual(v2, settings_manager.load_settings(self.settings_path))
        self.assertFalse(os.path.exists(f'{self.settings_path}.v1.bak'))

    def test_schema_v2_removes_obsolete_preset_name_without_backup(self):
        v2 = {
            'settings_schema_version': 2,
            'camera_params': {'ExposureTime': 50000.0, 'Gamma': 1.0},
            'other_settings': {'settings_preset_name': 'legacy'},
        }
        self.write_json(v2)

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
            {'wavelength': 'vis', 'filter_position': 1},
            {'wavelength': 'uv365', 'filter_position': 6},
        ]
        self.assertEqual(plan, settings_manager.validate_capture_plan(plan))

    def test_capture_plan_validation_rejects_unknown_wavelength_and_filter(self):
        with self.assertRaises(ValueError):
            settings_manager.validate_capture_plan([{'wavelength': 'uv240', 'filter_position': 1}])
        with self.assertRaises(ValueError):
            settings_manager.validate_capture_plan([{'wavelength': 'vis', 'filter_position': 7}])

    def test_lamp_output_selector_validation_normalizes_values(self):
        payload = {'output_selectors': {
            'uv255': 'p0', 'uv310': 'P1', 'uv365': 'P2', 'vis': 'P3'
        }}
        self.assertEqual({
            'output_selectors': {'uv255': 'P0', 'uv310': 'P1', 'uv365': 'P2', 'vis': 'P3'}
        }, settings_manager.validate_lamp_output_selectors(payload))

    def test_lamp_output_selector_validation_rejects_commands_and_missing_channels(self):
        with self.assertRaises(ValueError):
            settings_manager.validate_lamp_output_selectors({'output_selectors': {'uv255': 'M106 P0'}})


if __name__ == '__main__':
    unittest.main()
