#!/usr/bin/env python3
"""Hardware-free checks for the virtual BTT Octopus serial adapter."""

import unittest

import globals
import motioncontrols
import porthandler
from virtual_octopus import VirtualOctopusSerial


class InterByteGapSerial(VirtualOctopusSerial):
    """Expose the first response byte, then one temporary zero-byte gap."""

    def __init__(self):
        super().__init__()
        self._fragment_reply = False
        self._first_byte_read = False
        self._gap_pending = False

    def write(self, data):
        result = super().write(data)
        command = bytes(data).decode('ascii', 'ignore').strip().upper()
        if command == 'M105':
            self._fragment_reply = True
            self._first_byte_read = False
            self._gap_pending = False
        return result

    @property
    def in_waiting(self):
        if self._gap_pending:
            self._gap_pending = False
            return 0
        waiting = VirtualOctopusSerial.in_waiting.fget(self)
        if self._fragment_reply and not self._first_byte_read and waiting:
            return 1
        return waiting

    def read(self, size=1):
        result = super().read(size)
        if self._fragment_reply and not self._first_byte_read and result:
            self._first_byte_read = True
            self._gap_pending = True
        return result


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
        self.assertIn('G1 X12.5 Y-25.0 Z4', device.command_history)
        self.assertEqual(
            {'x': 12.5, 'y': 175.0, 'z': 4.0},
            motioncontrols.get_toolhead_position(device),
        )
        self.assertEqual(0.0, device._position['Y'])

        motioncontrols.move_relative(device, x=-2.5, y=-5, z=40)
        self.assertIn('G1 X-2.5 Y5.0 Z40', device.command_history)
        self.assertEqual(
            {'x': 10.0, 'y': 170.0, 'z': 40.0},
            motioncontrols.get_toolhead_position(device),
        )
        self.assertEqual(5.0, device._position['Y'])

    def test_circular_a_axis_is_not_clamped_at_zero_or_360_degrees(self):
        device = VirtualOctopusSerial()

        porthandler.write_and_wait(device, 'G91')
        porthandler.write_and_wait(device, 'G1 A420 F5400')
        self.assertEqual(420.0, device._position['A'])

        porthandler.write_and_wait(device, 'G1 A-840 F5400')
        self.assertEqual(-420.0, device._position['A'])

    def test_homing_and_light_commands_are_acknowledged(self):
        device = VirtualOctopusSerial()
        motioncontrols.move_to_position(device, x_pos=10, y_pos=20, z_pos=5)

        porthandler.write(device, 'G28 X Y Z')
        homing_reply = device.read(device.in_waiting)
        light_ok, _ = porthandler.write_and_wait(device, 'M106 P3 S128')

        self.assertIn(b'ok', homing_reply)
        self.assertTrue(light_ok)
        self.assertEqual(128, device._light_pwm[3])
        self.assertEqual(
            {'x': 2.0, 'y': 2.0, 'z': 2.0},
            motioncontrols.get_toolhead_position(device),
        )
        self.assertEqual(173.0, device._position['Y'])

    def test_closed_virtual_port_behaves_like_disconnected_serial(self):
        device = VirtualOctopusSerial()
        device.close()

        with self.assertRaises(OSError):
            porthandler.write_and_wait(device, 'M105')

    def test_m105_probe_waits_across_inter_byte_gap_and_drains_full_line(self):
        device = InterByteGapSerial()

        acknowledged, reply = porthandler.probe_motion_controller(device, timeout=0.3)

        self.assertTrue(acknowledged)
        self.assertEqual(b'ok T:25.00 /0.00 @:0\n', reply)
        self.assertEqual(0, device.in_waiting)

    def test_failure_marker_ends_acknowledgement_wait(self):
        class RejectedCommandSerial(VirtualOctopusSerial):
            def _execute(self, command):
                return b'echo:Homing Failed\n'

        device = RejectedCommandSerial()

        acknowledged, reply = porthandler.write_and_wait(
            device,
            'G28 A',
            timeout=10.0,
            failure_markers=(b'homing failed',),
        )

        self.assertFalse(acknowledged)
        self.assertIn(b'Homing Failed', reply)

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
