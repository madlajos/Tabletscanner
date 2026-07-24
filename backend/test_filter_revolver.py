#!/usr/bin/env python3
"""Hardware-free checks for acknowledged six-position revolver movement."""

import unittest

import filter_revolver
from virtual_octopus import VirtualOctopusSerial


class FilterRevolverTests(unittest.TestCase):
    def test_right_up_rotates_a_axis_and_advances_position(self):
        device = VirtualOctopusSerial()

        position = filter_revolver.rotate_one_slot(device, 1, 'up')

        self.assertEqual(2, position)
        self.assertEqual(-60.0, device._position['A'])
        self.assertEqual(
            ['G91', 'G1 A-60 F5400', 'M400', 'G90'],
            device.command_history,
        )

    def test_repeated_right_up_rotations_advance_from_first_to_third_slot(self):
        device = VirtualOctopusSerial()

        position = filter_revolver.rotate_one_slot(device, 1, 'up')
        position = filter_revolver.rotate_one_slot(device, position, 'up')

        self.assertEqual(3, position)
        self.assertEqual(-120.0, device._position['A'])
        self.assertEqual([
            'G91', 'G1 A-60 F5400', 'M400', 'G90',
            'G91', 'G1 A-60 F5400', 'M400', 'G90',
        ], device.command_history)

    def test_left_down_wraps_from_slot_one_to_six(self):
        device = VirtualOctopusSerial()

        position = filter_revolver.rotate_one_slot(device, 1, 'down')

        self.assertEqual(6, position)
        self.assertEqual(60.0, device._position['A'])
        self.assertEqual(
            ['G91', 'G1 A60 F5400', 'M400', 'G90'],
            device.command_history,
        )

    def test_right_up_wrap_normalizes_coordinate_after_physical_step(self):
        device = VirtualOctopusSerial()
        device._position['A'] = -300.0

        position = filter_revolver.rotate_one_slot(device, 6, 'up')

        self.assertEqual(1, position)
        self.assertEqual(0.0, device._position['A'])
        self.assertEqual(
            ['G91', 'G1 A-60 F5400', 'M400', 'G90', 'G92 A0'],
            device.command_history,
        )

    def test_invalid_position_and_direction_are_rejected_without_motion(self):
        device = VirtualOctopusSerial()

        with self.assertRaises(ValueError):
            filter_revolver.rotate_one_slot(device, 0, 'down')
        with self.assertRaises(ValueError):
            filter_revolver.rotate_one_slot(device, 1, 'left')

        self.assertEqual([], device.command_history)

    def test_shortest_path_uses_right_up_for_advancing_and_equal_distance(self):
        self.assertEqual(('up', 1), filter_revolver.shortest_path(6, 1))
        self.assertEqual(('down', 1), filter_revolver.shortest_path(1, 6))
        self.assertEqual(('up', 3), filter_revolver.shortest_path(2, 5))


if __name__ == '__main__':
    unittest.main()
