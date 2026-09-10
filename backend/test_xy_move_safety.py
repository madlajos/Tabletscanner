"""Hardware-free checks for configurable collision avoidance before X/Y moves."""

import copy
import unittest

import app as backend_app
import globals
import porthandler
import settings_manager
from virtual_octopus import VirtualOctopusSerial


class XyMoveSafetyTests(unittest.TestCase):
    def setUp(self):
        self.saved_settings = copy.deepcopy(settings_manager.get_settings())
        self.device = VirtualOctopusSerial()
        self.device._position.update({'X': 10.0, 'Y': 10.0, 'Z': 38.0})
        globals.motion_platform = porthandler.motion_platform = self.device
        globals.motion_busy = False
        globals.last_toolhead_pos = {'x': 10.0, 'y': 10.0, 'z': 38.0}
        settings_manager.set_settings({'advanced_settings': {
            'lower_z_before_xy_move': True,
            'xy_move_z_limit_mm': 35.0,
        }})
        self.client = backend_app.app.test_client()

    def tearDown(self):
        self.device.close()
        globals.motion_platform = porthandler.motion_platform = None
        globals.motion_busy = False
        globals.last_toolhead_pos = {'x': None, 'y': None, 'z': None}
        settings_manager.set_settings(self.saved_settings)

    def test_relative_xy_move_lowers_z_and_waits_first(self):
        response = self.client.post(
            '/api/move_toolhead_relative', json={'axis': 'x', 'value': 2}
        )

        self.assertEqual(200, response.status_code)
        self.assertEqual({'from_z': 38.0, 'to_z': 35.0}, response.json['safety_z_move'])
        self.assertEqual(35.0, globals.last_toolhead_pos['z'])
        self.assertEqual(12.0, globals.last_toolhead_pos['x'])
        self.assertEqual(
            ['G90', 'G1 Z35.0', 'M400', 'G91', 'G1 X2.0'],
            self.device.command_history,
        )

    def test_absolute_xy_then_requested_high_z_are_separate_moves(self):
        response = self.client.post(
            '/api/move_toolhead_absolute', json={'y': 20, 'z': 39}
        )

        self.assertEqual(200, response.status_code)
        commands = self.device.command_history
        self.assertLess(commands.index('G1 Z35.0'), commands.index('G1 Y145.0'))
        self.assertLess(commands.index('G1 Y145.0'), commands.index('G1 Z39.0'))
        self.assertEqual(39.0, globals.last_toolhead_pos['z'])

    def test_disabled_setting_leaves_z_unchanged(self):
        settings_manager.get_settings()['advanced_settings']['lower_z_before_xy_move'] = False

        response = self.client.post(
            '/api/move_toolhead_relative', json={'axis': 'y', 'value': 1}
        )

        self.assertEqual(200, response.status_code)
        self.assertNotIn('safety_z_move', response.json)
        self.assertEqual(38.0, globals.last_toolhead_pos['z'])
        self.assertFalse(any(command.startswith('G1 Z') for command in self.device.command_history))

    def test_unknown_z_is_queried_before_xy_move(self):
        globals.last_toolhead_pos['z'] = None

        response = self.client.post(
            '/api/move_toolhead_relative', json={'axis': 'x', 'value': 1}
        )

        self.assertEqual(200, response.status_code)
        self.assertEqual('M114', self.device.command_history[0])
        self.assertEqual(35.0, globals.last_toolhead_pos['z'])


if __name__ == '__main__':
    unittest.main()
