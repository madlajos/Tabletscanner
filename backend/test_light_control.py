#!/usr/bin/env python3
import unittest

from light_control import LampSettingsError, LightConfigurationError, LightController


class FakeSerial:
    is_open = True


class LightControllerTests(unittest.TestCase):
    def setUp(self):
        self.now = 100.0
        self.commands = []
        self.settings = {
            'lamp_settings': {
                'output_selectors': {'uv255': 'P0', 'uv310': 'P1', 'uv365': 'P2', 'vis': 'P3'},
                'channels': {
                    channel: {'dim_percent': 10, 'full_percent': 100, 'dim_timeout_seconds': 30, 'full_timeout_seconds': 5}
                    for channel in ('uv255', 'uv310', 'uv365')
                },
            }
        }
        self.controller = LightController(lambda: self.settings, lambda: FakeSerial(), self.writer, lambda: self.now)

    def writer(self, _serial, command, timeout):
        self.commands.append((command, timeout))
        return True, b'ok'

    def test_dimmed_pwm_and_mode_timeout(self):
        status = self.controller.activate('uv255', 'dimmed')
        self.assertEqual('M106 P0 S26', self.commands[-1][0])
        self.assertEqual('dimmed', status['active_mode'])
        self.now += 31
        self.assertEqual('uv255', self.controller.check_timeouts())
        self.assertEqual('M106 P0 S0', self.commands[-1][0])

    def test_mutual_exclusion_and_vis_has_no_timeout(self):
        self.controller.activate('uv310', 'full')
        self.controller.activate('vis')
        self.assertEqual(['M106 P1 S255', 'M106 P1 S0', 'M106 P3 S255'], [command for command, _ in self.commands])
        self.now += 1000
        self.assertIsNone(self.controller.check_timeouts())
        self.assertEqual('vis', self.controller.status()['active_channel'])

    def test_vis_only_needs_its_output_selector(self):
        self.settings = {
            'lamp_settings': {
                'output_selectors': {'uv255': 'P0', 'uv310': 'P1', 'uv365': 'P2', 'vis': 'P3'}
            }
        }
        self.controller.activate('vis')
        self.assertEqual('M106 P3 S255', self.commands[-1][0])

    def test_invalid_or_missing_configuration_is_rejected(self):
        unconfigured = LightController(lambda: {}, lambda: FakeSerial(), self.writer)
        with self.assertRaises(LampSettingsError):
            unconfigured.activate('uv255', 'full')


if __name__ == '__main__':
    unittest.main()
