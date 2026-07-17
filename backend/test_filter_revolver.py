#!/usr/bin/env python3
"""Hardware-free checks for acknowledged six-position revolver movement."""

import unittest

import filter_revolver
from virtual_octopus import VirtualOctopusSerial


class FilterRevolverTests(unittest.TestCase):
    def test_down_rotates_a_axis_and_advances_position(self):
        device = VirtualOctopusSerial()

        position = filter_revolver.rotate_one_slot(device, 1, 'down')

        self.assertEqual(2, position)
        self.assertEqual(60.0, device._position['A'])
        self.assertEqual(['G91', 'G1 A60', 'M400', 'G90'], device.command_history)

    def test_up_wraps_from_slot_one_without_crossing_soft_limit(self):
        device = VirtualOctopusSerial()

        position = filter_revolver.rotate_one_slot(device, 1, 'up')

        self.assertEqual(6, position)
        self.assertEqual(300.0, device._position['A'])
        self.assertEqual(['G92 A360', 'G91', 'G1 A-60', 'M400', 'G90'], device.command_history)

    def test_down_wrap_normalizes_coordinate_after_physical_step(self):
        device = VirtualOctopusSerial()
        device._position['A'] = 300.0

        position = filter_revolver.rotate_one_slot(device, 6, 'down')

        self.assertEqual(1, position)
        self.assertEqual(0.0, device._position['A'])
        self.assertEqual(['G91', 'G1 A60', 'M400', 'G90', 'G92 A0'], device.command_history)

    def test_invalid_position_and_direction_are_rejected_without_motion(self):
        device = VirtualOctopusSerial()

        with self.assertRaises(ValueError):
            filter_revolver.rotate_one_slot(device, 0, 'down')
        with self.assertRaises(ValueError):
            filter_revolver.rotate_one_slot(device, 1, 'left')

        self.assertEqual([], device.command_history)

    def test_shortest_path_wraps_and_uses_down_for_equal_three_step_distance(self):
        self.assertEqual(('down', 1), filter_revolver.shortest_path(6, 1))
        self.assertEqual(('up', 1), filter_revolver.shortest_path(1, 6))
        self.assertEqual(('down', 3), filter_revolver.shortest_path(2, 5))


if __name__ == '__main__':
    unittest.main()
