#!/usr/bin/env python3
"""Hardware-free checks for autofocus-referenced height corrections."""

import unittest

import globals
import height_offset_control
from virtual_octopus import VirtualOctopusSerial


def filter_settings(offset=2.5):
    return {
        'filters': [{
            'id': 'green',
            'name': 'Green',
            'wavelength_range': '500-570',
            'color': '#00ff00',
        }],
        'slots': [None, 'green', None, None, None, None],
        'height_offsets_mm': {
            'empty': {'uv255': 0, 'uv310': 0, 'uv365': 0, 'vis': 0},
            'green': {'uv255': offset, 'uv310': 1, 'uv365': -1, 'vis': 0.5},
        },
    }


class HeightOffsetControlTests(unittest.TestCase):
    def setUp(self):
        self.device = VirtualOctopusSerial()
        globals.filter_revolver_position = 2
        globals.last_toolhead_pos = {'x': 5.0, 'y': 5.0, 'z': 10.0}
        height_offset_control.invalidate_reference()

    def tearDown(self):
        self.device.close()
        globals.filter_revolver_position = None
        globals.last_toolhead_pos = {'x': None, 'y': None, 'z': None}
        height_offset_control.invalidate_reference()

    def test_no_move_is_sent_before_autofocus(self):
        result = height_offset_control.apply_active_combination(
            self.device, filter_settings(), 'uv255'
        )

        self.assertEqual({'applied': False, 'reason': 'autofocus_required'}, result)
        self.assertEqual([], self.device.command_history)

    def test_selected_filter_and_light_move_from_autofocus_zero(self):
        height_offset_control.record_reference(10.0)

        result = height_offset_control.apply_active_combination(
            self.device, filter_settings(2.5), 'uv255'
        )

        self.assertEqual(
            ['G90', 'G1 Z12.5000', 'M400'],
            self.device.command_history,
        )
        self.assertEqual(12.5, globals.last_toolhead_pos['z'])
        self.assertEqual({
            'applied': True,
            'offset_mm': 2.5,
            'target_z': 12.5,
            'moved': True,
        }, result)

    def test_target_outside_travel_is_rejected_without_motion(self):
        height_offset_control.record_reference(29.0)

        with self.assertRaises(height_offset_control.HeightOffsetCommandError):
            height_offset_control.apply_active_combination(
                self.device, filter_settings(2.5), 'uv255'
            )

        self.assertEqual([], self.device.command_history)


if __name__ == '__main__':
    unittest.main()
