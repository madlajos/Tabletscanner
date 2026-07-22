#!/usr/bin/env python3
import unittest

from light_control import (
    LampSettingsError,
    LightCommandError,
    LightConfigurationError,
    LightController,
    contains_lamp_gcode,
)
from settings_manager import OCTOPUS_LIGHT_OUTPUT_SELECTORS


class FakeSerial:
    is_open = True


class LightControllerTests(unittest.TestCase):
    def setUp(self):
        self.now = 100.0
        self.commands = []
        self.serial = FakeSerial()
        self.settings = {
            'lamp_settings': {
                'output_selectors': dict(OCTOPUS_LIGHT_OUTPUT_SELECTORS),
                'channels': {
                    channel: {'dim_percent': 10, 'full_percent': 100, 'dim_timeout_seconds': 30, 'full_timeout_seconds': 5}
                    for channel in ('uv255', 'uv310', 'uv365')
                },
            }
        }
        self.controller = LightController(lambda: self.settings, lambda: self.serial, self.writer, lambda: self.now)

    def writer(self, _serial, command, timeout):
        self.commands.append((command, timeout))
        return True, b'ok'

    def test_dimmed_pwm_and_mode_timeout(self):
        status = self.controller.activate('uv255', 'dimmed')
        self.assertEqual(
            ['M106 P0 S0', 'M106 P1 S0', 'M106 P2 S0', 'M106 P3 S0', 'M106 P2 S26'],
            [command for command, _ in self.commands],
        )
        self.assertEqual('M106 P2 S26', self.commands[-1][0])
        self.assertEqual('dimmed', status['active_mode'])
        self.now += 31
        self.assertEqual('uv255', self.controller.check_timeouts())
        self.assertEqual(
            ['M106 P0 S0', 'M106 P1 S0', 'M106 P2 S0', 'M106 P3 S0'],
            [command for command, _ in self.commands[-4:]],
        )
        self.assertTrue(self.controller.consume_auto_off_event())
        self.assertFalse(self.controller.consume_auto_off_event())

    def test_mutual_exclusion_and_vis_has_no_timeout(self):
        self.controller.activate('uv310', 'full')
        self.controller.activate('vis')
        self.assertEqual(
            ['M106 P0 S0', 'M106 P1 S0', 'M106 P2 S0', 'M106 P3 S0', 'M106 P3 S255'],
            [command for command, _ in self.commands[:5]],
        )
        self.assertEqual(
            ['M106 P0 S0', 'M106 P1 S0', 'M106 P2 S0', 'M106 P3 S0', 'M106 P0 S255'],
            [command for command, _ in self.commands[5:]],
        )
        self.now += 1000
        self.assertIsNone(self.controller.check_timeouts())
        self.assertEqual('vis', self.controller.status()['active_channel'])

    def test_canonical_physical_output_mapping(self):
        self.settings['lamp_settings']['output_selectors'] = dict(OCTOPUS_LIGHT_OUTPUT_SELECTORS)
        expected_on_commands = {
            'uv255': 'M106 P2 S255',
            'uv310': 'M106 P3 S255',
            'uv365': 'M106 P1 S255',
            'vis': 'M106 P0 S255',
        }

        for channel, expected_command in expected_on_commands.items():
            self.commands.clear()
            mode = None if channel == 'vis' else 'full'
            self.controller.activate(channel, mode)
            self.assertEqual(
                ['M106 P0 S0', 'M106 P1 S0', 'M106 P2 S0', 'M106 P3 S0'],
                [command for command, _ in self.commands[:4]],
            )
            self.assertEqual(expected_command, self.commands[-1][0])

    def test_vis_only_needs_its_output_selector(self):
        self.settings = {
            'lamp_settings': {
                'output_selectors': dict(OCTOPUS_LIGHT_OUTPUT_SELECTORS)
            }
        }
        self.controller.activate('vis')
        self.assertEqual(
            ['M106 P0 S0', 'M106 P1 S0', 'M106 P2 S0', 'M106 P3 S0', 'M106 P0 S255'],
            [command for command, _ in self.commands],
        )
        self.assertEqual('M106 P0 S255', self.commands[-1][0])

    def test_off_sends_command_even_when_cached_state_is_empty(self):
        self.controller.off('uv365')
        self.assertEqual(
            ['M106 P0 S0', 'M106 P1 S0', 'M106 P2 S0', 'M106 P3 S0'],
            [command for command, _ in self.commands],
        )

    def test_status_discards_state_from_a_closed_or_replaced_serial_connection(self):
        self.controller.activate('vis')
        self.assertEqual('vis', self.controller.status()['active_channel'])

        self.serial.is_open = False
        self.assertIsNone(self.controller.status()['active_channel'])

        self.serial = FakeSerial()
        self.assertIsNone(self.controller.status()['active_channel'])

    def test_failed_all_off_attempts_every_output_and_never_sends_on(self):
        def writer(_serial, command, timeout):
            self.commands.append((command, timeout))
            return command != 'M106 P1 S0', b'ok'

        self.controller._command_writer = writer

        with self.assertRaises(LightCommandError):
            self.controller.activate('uv255', 'dimmed')

        self.assertEqual(
            ['M106 P0 S0', 'M106 P1 S0', 'M106 P2 S0', 'M106 P3 S0'],
            [command for command, _ in self.commands],
        )

    def test_all_off_attempts_every_output_after_failure(self):
        def writer(_serial, command, timeout):
            self.commands.append((command, timeout))
            return command != 'M106 P0 S0', b'ok'

        self.controller._command_writer = writer

        with self.assertRaises(LightCommandError):
            self.controller.off()

        self.assertEqual(
            ['M106 P0 S0', 'M106 P1 S0', 'M106 P2 S0', 'M106 P3 S0'],
            [command for command, _ in self.commands],
        )

    def test_all_off_attempts_every_output_after_serial_exception(self):
        def writer(_serial, command, timeout):
            self.commands.append((command, timeout))
            if command == 'M106 P1 S0':
                raise OSError('USB disconnected')
            return True, b'ok'

        self.controller._command_writer = writer

        with self.assertRaises(LightCommandError):
            self.controller.off()

        self.assertEqual(
            ['M106 P0 S0', 'M106 P1 S0', 'M106 P2 S0', 'M106 P3 S0'],
            [command for command, _ in self.commands],
        )

    def test_invalid_or_missing_configuration_is_rejected(self):
        unconfigured = LightController(lambda: {}, lambda: FakeSerial(), self.writer)
        with self.assertRaises(LampSettingsError):
            unconfigured.activate('uv255', 'full')

    def test_raw_lamp_gcode_detection_covers_numbered_and_obfuscated_commands(self):
        for command in ('M106 P0 S255', 'n42 m107', 'N5M0106P1S255', 'G4 P0\nM106 P2 S255'):
            with self.subTest(command=command):
                self.assertTrue(contains_lamp_gcode(command))
        self.assertFalse(contains_lamp_gcode('G1 X10 Y20'))


if __name__ == '__main__':
    unittest.main()
