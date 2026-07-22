#!/usr/bin/env python3
"""Hardware-free checks for the virtual BTT Octopus serial adapter."""

import unittest

import globals
import motioncontrols
import porthandler
from virtual_octopus import VirtualOctopusSerial


class VirtualOctopusTests(unittest.TestCase):
    def tearDown(self):
        device = globals.motion_platform
        if device and getattr(device, 'is_open', False):
            device.close()
        globals.motion_platform = None
        porthandler.motion_platform = None
        globals.motion_busy = False

    def test_virtual_connection_identifies_as_octopus_marlin(self):
        device = porthandler.connect_to_motion_platform(use_virtual=True)

        acknowledged, reply = porthandler.write_and_wait(device, 'M115')

        self.assertTrue(acknowledged)
        self.assertIn(b'FIRMWARE_NAME:Marlin', reply)
        self.assertIn(porthandler.EXPECTED_FIRMWARE_MARKER.encode('ascii'), reply)
        self.assertEqual('VIRTUAL_BTT_OCTOPUS', device.port)
        self.assertTrue(device.is_virtual)

    def test_motion_commands_update_clamped_position(self):
        device = VirtualOctopusSerial()

        motioncontrols.move_to_position(device, x_pos=12.5, y_pos=200, z_pos=4)
        self.assertEqual(
            {'x': 12.5, 'y': 175.0, 'z': 4.0},
            motioncontrols.get_toolhead_position(device),
        )

        motioncontrols.move_relative(device, x=-2.5, z=40)
        self.assertEqual(
            {'x': 10.0, 'y': 175.0, 'z': 30.0},
            motioncontrols.get_toolhead_position(device),
        )

    def test_homing_and_light_commands_are_acknowledged(self):
        device = VirtualOctopusSerial()
        motioncontrols.move_to_position(device, x_pos=10, y_pos=20, z_pos=5)

        porthandler.write(device, 'G28 X Z')
        homing_reply = device.read(device.in_waiting)
        light_ok, _ = porthandler.write_and_wait(device, 'M106 P3 S128')

        self.assertIn(b'ok', homing_reply)
        self.assertTrue(light_ok)
        self.assertEqual(128, device._light_pwm[3])
        self.assertEqual(
            {'x': 0.0, 'y': 20.0, 'z': 0.0},
            motioncontrols.get_toolhead_position(device),
        )

    def test_closed_virtual_port_behaves_like_disconnected_serial(self):
        device = VirtualOctopusSerial()
        device.close()

        with self.assertRaises(OSError):
            porthandler.write_and_wait(device, 'M105')

    def test_firmware_interlock_clears_other_outputs_before_nonzero_m106(self):
        device = VirtualOctopusSerial()
        porthandler.write_and_wait(device, 'M106 P2 S255')
        porthandler.write_and_wait(device, 'M106 P3 S128')

        self.assertEqual({0: 0, 1: 0, 2: 0, 3: 128}, device._light_pwm)

    def test_disconnect_clears_every_light_before_closing(self):
        device = porthandler.connect_to_motion_platform(use_virtual=True)
        for selector in range(4):
            porthandler.write_and_wait(device, f'M106 P{selector} S255')

        porthandler.disconnect_serial_device('motion_platform')

        self.assertEqual({0: 0, 1: 0, 2: 0, 3: 0}, device._light_pwm)
        self.assertFalse(device.is_open)
        self.assertIsNone(globals.motion_platform)
        self.assertIsNone(porthandler.motion_platform)


if __name__ == '__main__':
    unittest.main()
