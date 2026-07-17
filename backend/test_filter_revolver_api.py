#!/usr/bin/env python3
"""Flask contract checks for guarded filter-revolver control."""

import unittest

import app as backend_app
import globals
import porthandler
from virtual_octopus import VirtualOctopusSerial


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

        response = self.client.post('/api/filter-revolver/rotate', json={'direction': 'down'})

        self.assertEqual(200, response.status_code)
        self.assertEqual(2, response.get_json()['position'])
        self.assertFalse(response.get_json()['busy'])
        self.assertEqual(['G91', 'G1 A60', 'M400', 'G90'], self.device.command_history)

    def test_invalid_direction_does_not_touch_hardware(self):
        response = self.client.post('/api/filter-revolver/rotate', json={'direction': 'left'})

        self.assertEqual(400, response.status_code)
        self.assertEqual([], self.device.command_history)

    def test_a_axis_homing_establishes_slot_one_reference(self):
        globals.homed_axes = {'x', 'y', 'z'}
        globals.toolhead_homed = True

        response = self.client.post('/api/home_toolhead', json={'axes': ['a']})
        status = self.client.get('/api/filter-revolver/status').get_json()

        self.assertEqual(200, response.status_code)
        self.assertEqual('G28 A', self.device.command_history[0])
        self.assertTrue(status['homed'])
        self.assertTrue(status['motion_platform_homed'])
        self.assertEqual(1, status['position'])

    def test_full_homing_runs_a_axis_last_then_allows_rotation(self):
        homing = self.client.post('/api/home_toolhead', json={})
        rotation = self.client.post('/api/filter-revolver/rotate', json={'direction': 'down'})

        self.assertEqual(200, homing.status_code)
        self.assertEqual(['z', 'y', 'x', 'a'], homing.get_json()['homed_axes'])
        self.assertEqual(
            ['G28 Z', 'G28 Y', 'G28 X', 'G28 A'],
            [command for command in self.device.command_history if command.startswith('G28')],
        )
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
        self.assertEqual('up', response.get_json()['direction'])
        self.assertEqual(2, response.get_json()['steps'])
        self.assertEqual(300.0, self.device._position['A'])


if __name__ == '__main__':
    unittest.main()
