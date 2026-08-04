#!/usr/bin/env python3
"""Flask contract checks for guarded filter-revolver control."""

import unittest
from unittest.mock import patch

import app as backend_app
import filter_revolver
import globals
import motioncontrols
import porthandler
import height_offset_control
import settings_manager
from virtual_octopus import VirtualOctopusSerial


class SecondMoveNoResponseSerial(VirtualOctopusSerial):
    """Simulate a controller that stops acknowledging the second A-axis move."""

    def __init__(self):
        super().__init__()
        self.a_move_count = 0

    def _execute(self, command):
        if command.upper().startswith('G1 A'):
            self.a_move_count += 1
            if self.a_move_count == 2:
                return b''
        return super()._execute(command)


class HomingFailedSerial(VirtualOctopusSerial):
    """Return Marlin's explicit endstop-validation failure for A homing."""

    def _execute(self, command):
        if command.upper() == 'G28 A':
            # Recoverable firmware returns to the command loop, so Marlin may
            # append its normal command acknowledgement after the error.
            return b'echo:busy: processing\nError:Homing failed\nok\n'
        return super()._execute(command)


class FilterRevolverApiTests(unittest.TestCase):
    def setUp(self):
        self.device = VirtualOctopusSerial()
        globals.motion_platform = self.device
        porthandler.motion_platform = self.device
        globals.motion_busy = False
        globals.toolhead_homed = False
        globals.homed_axes = set()
        globals.filter_revolver_homed = False
        globals.filter_revolver_position = None
        height_offset_control.invalidate_reference()
        settings_manager.set_settings({})
        self.client = backend_app.app.test_client()

    def tearDown(self):
        self.device.close()
        globals.motion_platform = None
        porthandler.motion_platform = None
        globals.motion_busy = False
        globals.toolhead_homed = False
        globals.homed_axes = set()
        globals.filter_revolver_homed = False
        globals.filter_revolver_position = None
        height_offset_control.invalidate_reference()
        settings_manager.set_settings({})

    def test_status_and_rotation_require_homing(self):
        status = self.client.get('/api/filter-revolver/status')
        blocked = self.client.post('/api/filter-revolver/rotate', json={'direction': 'down'})

        self.assertEqual(200, status.status_code)
        self.assertEqual({
            'position': None,
            'homed': False,
            'motion_platform_homed': False,
            'busy': False,
        }, status.get_json())
        self.assertEqual(409, blocked.status_code)
        self.assertEqual([], self.device.command_history)

    def test_acknowledged_rotation_updates_status(self):
        globals.toolhead_homed = True
        globals.homed_axes = {'x', 'y', 'z', 'a'}
        globals.filter_revolver_homed = True
        globals.filter_revolver_position = 1

        response = self.client.post('/api/filter-revolver/rotate', json={'direction': 'up'})

        self.assertEqual(200, response.status_code)
        self.assertEqual(2, response.get_json()['position'])
        self.assertFalse(response.get_json()['busy'])
        self.assertEqual(['G91', 'G1 A-60 F5400', 'M400', 'G90'], self.device.command_history)

    def test_two_consecutive_rotations_both_succeed(self):
        globals.filter_revolver_homed = True
        globals.filter_revolver_position = 1

        first = self.client.post('/api/filter-revolver/rotate', json={'direction': 'up'})
        second = self.client.post('/api/filter-revolver/rotate', json={'direction': 'up'})

        self.assertEqual(200, first.status_code)
        self.assertEqual(2, first.get_json()['position'])
        self.assertEqual(200, second.status_code)
        self.assertEqual(3, second.get_json()['position'])
        self.assertEqual(-120.0, self.device._position['A'])

    def test_second_rotation_timeout_keeps_connection_but_invalidates_reference(self):
        self.device.close()
        self.device = SecondMoveNoResponseSerial()
        globals.motion_platform = self.device
        porthandler.motion_platform = self.device
        globals.filter_revolver_homed = True
        globals.filter_revolver_position = 1

        with patch.object(filter_revolver, 'MOTION_TIMEOUT_SECONDS', 0.02):
            first = self.client.post('/api/filter-revolver/rotate', json={'direction': 'up'})
            second = self.client.post('/api/filter-revolver/rotate', json={'direction': 'up'})

        self.assertEqual(200, first.status_code)
        self.assertEqual(504, second.status_code)
        self.assertIn('movement command', second.get_json()['error'])
        self.assertIn("'G1 A-60 F5400'", second.get_json()['error'])
        self.assertIs(globals.motion_platform, self.device)
        self.assertIs(porthandler.motion_platform, self.device)
        self.assertTrue(self.device.is_open)
        self.assertFalse(globals.filter_revolver_homed)
        self.assertIsNone(globals.filter_revolver_position)
        self.assertEqual(-60.0, self.device._position['A'])
        self.assertEqual('G90', self.device.command_history[-1])

    def test_homing_timeout_does_not_close_serial_connection(self):
        globals.homed_axes = {'a'}
        globals.filter_revolver_homed = True
        globals.filter_revolver_position = 3

        timeout_error = motioncontrols.HomingTimeoutError('A', 60.0, b'echo:busy: processing\n')
        with patch.object(motioncontrols, 'home_axes', side_effect=timeout_error):
            response = self.client.post('/api/home_toolhead', json={'axes': ['a']})

        self.assertEqual(504, response.status_code)
        self.assertEqual('E1203', response.get_json()['code'])
        self.assertIs(globals.motion_platform, self.device)
        self.assertIs(porthandler.motion_platform, self.device)
        self.assertTrue(self.device.is_open)
        self.assertFalse(globals.filter_revolver_homed)
        self.assertIsNone(globals.filter_revolver_position)
        self.assertNotIn('a', globals.homed_axes)

    def test_explicit_homing_failure_returns_popup_without_waiting_or_disconnecting(self):
        self.device.close()
        self.device = HomingFailedSerial()
        globals.motion_platform = self.device
        porthandler.motion_platform = self.device

        response = self.client.post('/api/home_toolhead', json={'axes': ['a']})

        self.assertEqual(422, response.status_code)
        self.assertEqual('E1202', response.get_json()['code'])
        self.assertTrue(response.get_json()['popup'])
        self.assertIs(globals.motion_platform, self.device)
        self.assertIs(porthandler.motion_platform, self.device)
        self.assertTrue(self.device.is_open)
        self.assertFalse(globals.filter_revolver_homed)
        acknowledged, _ = porthandler.write_and_wait(self.device, 'M105')
        self.assertTrue(acknowledged)

    def test_invalid_direction_does_not_touch_hardware(self):
        response = self.client.post('/api/filter-revolver/rotate', json={'direction': 'left'})

        self.assertEqual(400, response.status_code)
        self.assertEqual([], self.device.command_history)

    def test_a_axis_homing_establishes_reference_and_allows_rotation(self):
        response = self.client.post('/api/home_toolhead', json={'axes': ['a']})
        status = self.client.get('/api/filter-revolver/status').get_json()
        rotation = self.client.post('/api/filter-revolver/rotate', json={'direction': 'up'})

        self.assertEqual(200, response.status_code)
        self.assertEqual('G28 A', self.device.command_history[0])
        self.assertTrue(status['homed'])
        self.assertFalse(status['motion_platform_homed'])
        self.assertEqual(1, status['position'])
        self.assertEqual(200, rotation.status_code)
        self.assertEqual(2, rotation.get_json()['position'])

    def test_full_homing_runs_a_axis_last_then_allows_rotation(self):
        homing = self.client.post('/api/home_toolhead', json={})
        rotation = self.client.post('/api/filter-revolver/rotate', json={'direction': 'up'})

        self.assertEqual(200, homing.status_code)
        self.assertEqual(['z', 'y', 'x', 'a'], homing.get_json()['homed_axes'])
        self.assertEqual(
            {'x': 2.0, 'y': 2.0, 'z': 2.0},
            homing.get_json()['position'],
        )
        self.assertEqual(
            ['G28 Z', 'G28 Y', 'G28 X', 'G28 A'],
            [command for command in self.device.command_history if command.startswith('G28')],
        )
        self.assertEqual(173.0, self.device._position['Y'])
        self.assertEqual(200, rotation.status_code)
        self.assertEqual(2, rotation.get_json()['position'])

    def test_select_position_uses_shortest_acknowledged_path(self):
        globals.toolhead_homed = True
        globals.homed_axes = {'x', 'y', 'z', 'a'}
        globals.filter_revolver_homed = True
        globals.filter_revolver_position = 2
        self.device._position['A'] = 60.0

        response = self.client.post('/api/filter-revolver/select', json={'position': 6})

        self.assertEqual(200, response.status_code)
        self.assertEqual(6, response.get_json()['position'])
        self.assertEqual('down', response.get_json()['direction'])
        self.assertEqual(2, response.get_json()['steps'])
        self.assertEqual(60.0, self.device._position['A'])

    def test_filter_change_applies_active_combination_after_manual_autofocus(self):
        globals.toolhead_homed = True
        globals.homed_axes = {'x', 'y', 'z', 'a'}
        globals.filter_revolver_homed = True
        globals.filter_revolver_position = 1
        globals.last_toolhead_pos = {'x': 5.0, 'y': 5.0, 'z': 10.0}
        height_offset_control.record_reference(10.0)
        settings_manager.set_settings({
            'filter_settings': {
                'filters': [{
                    'id': 'green', 'name': 'Green',
                    'wavelength_range': '500-570', 'color': '#00ff00',
                }],
                'slots': [None, 'green', None, None, None, None],
                'height_offsets_mm': {
                    'empty': {'uv255': 0, 'uv310': 0, 'uv365': 0, 'vis': 0},
                    'green': {'uv255': 2.0, 'uv310': 0, 'uv365': 0, 'vis': 0},
                },
            },
        })

        with patch.object(backend_app.light_controller, 'status', return_value={'active_channel': 'uv255'}):
            response = self.client.post('/api/filter-revolver/select', json={'position': 2})

        self.assertEqual(200, response.status_code)
        self.assertEqual(12.0, response.get_json()['height_offset']['target_z'])
        self.assertEqual(
            ['G90', 'G1 Z12.0000', 'M400'],
            self.device.command_history[-3:],
        )

    def test_manual_autofocus_selects_empty_filter_and_vis_before_routine(self):
        globals.toolhead_homed = True
        globals.homed_axes = {'x', 'y', 'z', 'a'}
        globals.filter_revolver_homed = True
        globals.filter_revolver_position = 2
        globals.last_toolhead_pos = {'x': 5.0, 'y': 5.0, 'z': 10.0}
        settings_manager.set_settings({
            'lamp_settings': {
                'channels': {},
                'output_selectors': dict(settings_manager.OCTOPUS_LIGHT_OUTPUT_SELECTORS),
            },
        })

        def autofocus_result(*_args, **_kwargs):
            globals.last_toolhead_pos['z'] = 11.25
            return {'status': 'OK'}

        with patch.object(backend_app.autofocus_main, 'autofocus_coarse', side_effect=autofocus_result):
            response = self.client.post('/api/autofocus_coarse', json={'skip_empty_check': True})

        self.assertEqual(200, response.status_code)
        self.assertEqual(1, globals.filter_revolver_position)
        self.assertEqual('vis', backend_app.light_controller.status()['active_channel'])
        self.assertTrue(response.get_json()['autofocus_reference']['available'])
        self.assertEqual(11.25, response.get_json()['autofocus_reference']['reference_z'])
        filter_move_index = self.device.command_history.index('G1 A60 F5400')
        first_light_index = next(
            index for index, command in enumerate(self.device.command_history)
            if command.startswith('M106')
        )
        self.assertLess(filter_move_index, first_light_index)

    def test_light_change_applies_current_filter_offset_after_autofocus(self):
        globals.toolhead_homed = True
        globals.homed_axes = {'x', 'y', 'z', 'a'}
        globals.filter_revolver_homed = True
        globals.filter_revolver_position = 2
        globals.last_toolhead_pos = {'x': 5.0, 'y': 5.0, 'z': 10.0}
        height_offset_control.record_reference(10.0)
        settings_manager.set_settings({
            'lamp_settings': {
                'output_selectors': dict(settings_manager.OCTOPUS_LIGHT_OUTPUT_SELECTORS),
                'channels': {
                    channel: {
                        'dim_percent': 50,
                        'full_percent': 100,
                        'dim_timeout_seconds': 30,
                        'full_timeout_seconds': 5,
                    }
                    for channel in settings_manager.UV_LAMP_CHANNELS
                },
            },
            'filter_settings': {
                'filters': [{
                    'id': 'green', 'name': 'Green',
                    'wavelength_range': '500-570', 'color': '#00ff00',
                }],
                'slots': [None, 'green', None, None, None, None],
                'height_offsets_mm': {
                    'empty': {'uv255': 0, 'uv310': 0, 'uv365': 0, 'vis': 0},
                    'green': {'uv255': -1.5, 'uv310': 0, 'uv365': 0, 'vis': 0},
                },
            },
        })

        response = self.client.post(
            '/api/lights/activate',
            json={'channel': 'uv255', 'mode': 'dimmed'},
        )

        self.assertEqual(200, response.status_code)
        self.assertEqual(-1.5, response.get_json()['height_offset']['offset_mm'])
        self.assertEqual(8.5, globals.last_toolhead_pos['z'])
        self.assertEqual(
            ['G90', 'G1 Z8.5000', 'M400'],
            self.device.command_history[-3:],
        )


if __name__ == '__main__':
    unittest.main()
