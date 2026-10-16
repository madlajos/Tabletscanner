"""Hardware-free door protocol, lamp cache and capture-boundary tests."""
import unittest
from types import SimpleNamespace
from threading import Event, Thread
from unittest.mock import patch

import globals
import porthandler
from door_interlock import DoorInterlockState
from light_control import LightController, DoorOpenError, CaptureIlluminationError, LightCommandError
from settings_manager import OCTOPUS_LIGHT_OUTPUT_SELECTORS
from virtual_octopus import VirtualOctopusSerial


class DoorInterlockTests(unittest.TestCase):
    def setUp(self):
        self.serial = VirtualOctopusSerial()
        porthandler.motion_platform = self.serial
        globals.motion_platform = self.serial
        globals.motion_busy = False
        self.settings = {'lamp_settings': {
            'output_selectors': dict(OCTOPUS_LIGHT_OUTPUT_SELECTORS),
            'channels': {name: {'dim_percent': 10, 'full_percent': 100,
                                'dim_timeout_seconds': 30, 'full_timeout_seconds': 5}
                         for name in ('uv255', 'uv310', 'uv365')},
        }}
        self.lights = LightController(
            lambda: self.settings, lambda: self.serial, porthandler.write_and_wait,
            operation_lock=porthandler.motion_lock,
            door_status_getter=lambda: porthandler.door_interlock.snapshot(self.serial),
            door_status_refresher=porthandler.refresh_door_status,
        )

    def tearDown(self):
        self.serial.close()
        porthandler.motion_platform = None
        globals.motion_platform = None
        globals.motion_busy = False

    def test_fragmented_reports_and_replaced_connection_fail_closed(self):
        state = DoorInterlockState()
        line = b'TS_DOORS FRONT:0 TOP:1 UV_BLOCKED:1 TRIP:12\r\n'
        for byte in line:
            state.feed(self.serial, bytes([byte]))
        self.assertTrue(state.snapshot(self.serial)['top_open'])
        self.assertEqual(12, state.snapshot(self.serial)['trip_count'])
        self.assertFalse(state.snapshot(VirtualOctopusSerial())['known'])
        self.assertTrue(state.snapshot(None)['uv_blocked'])

    def test_inconsistent_report_cannot_allow_uv(self):
        state = DoorInterlockState()
        state.feed(self.serial, b'TS_DOORS FRONT:1 TOP:0 UV_BLOCKED:0 TRIP:0\n')
        self.assertFalse(state.snapshot(self.serial)['known'])

    def test_either_door_blocks_each_uv_channel_but_vis_remains_usable(self):
        for door in ('front', 'top'):
            self.serial.set_door_open(door, True)
            for channel in ('uv255', 'uv310', 'uv365'):
                with self.subTest(door=door, channel=channel), self.assertRaises(DoorOpenError):
                    self.lights.activate(channel, 'dimmed')
            self.assertEqual('vis', self.lights.activate('vis')['active_channel'])
            self.serial.set_door_open(door, False)

    def test_missing_door_adapters_fail_closed_for_every_uv_channel(self):
        lights = LightController(lambda: self.settings, lambda: self.serial, porthandler.write_and_wait)
        for channel in ('uv255', 'uv310', 'uv365'):
            with self.subTest(channel=channel), self.assertRaises(DoorOpenError):
                lights.activate(channel, 'full')
        self.assertEqual('vis', lights.activate('vis')['active_channel'])
        self.assertFalse(lights.status()['door_interlock']['known'])

    def test_brief_opening_clears_lamp_cache_and_rejects_capture(self):
        self.lights.activate('uv365', 'dimmed')
        self.serial.set_door_open('front', True)
        self.assertEqual(0, self.serial._light_pwm[1])
        self.serial.set_door_open('front', False)
        # No software read or OFF command was needed for the physical cutoff.
        with self.assertRaises(CaptureIlluminationError):
            self.lights.capture_remaining_seconds('uv365', 'dimmed')
        self.assertIsNone(self.lights.status()['active_channel'])
        self.assertEqual('uv365', self.lights.activate('uv365', 'dimmed')['active_channel'])

    def test_unsolicited_events_survive_command_drains(self):
        self.lights.activate('uv255', 'full')
        self.serial.set_door_open('top', True)
        acknowledged, _ = porthandler.write_and_wait(self.serial, 'M105')
        self.assertTrue(acknowledged)
        self.assertTrue(self.lights.status()['door_interlock']['top_open'])
        self.assertIsNone(self.lights.status()['active_channel'])

    def test_raw_uv_rejection_is_failure_even_with_ok(self):
        self.serial.set_door_open('front', True)
        acknowledged, reply = porthandler.write_and_wait(self.serial, 'M106 P2 S255')
        self.assertFalse(acknowledged)
        self.assertIn(b'Error:TS_DOOR_OPEN', reply)
        self.assertIn(b'ok\n', reply)
        self.assertEqual('vis', self.lights.activate('vis')['active_channel'])

    def test_opening_between_preflight_and_on_is_rejected(self):
        original = self.lights._command_writer
        def racing_writer(serial_port, command, timeout):
            if command == 'M106 P3 S26':
                serial_port.set_door_open('top', True)
            return original(serial_port, command, timeout=timeout)
        self.lights._command_writer = racing_writer
        with self.assertRaises(DoorOpenError):
            self.lights.activate('uv310', 'dimmed')
        self.assertIsNone(self.lights.status()['active_channel'])

    def test_brief_open_close_before_on_is_explicitly_switched_off(self):
        original = self.lights._command_writer
        def racing_writer(serial_port, command, timeout):
            if command == 'M106 P3 S26':
                serial_port.set_door_open('top', True)
                serial_port.set_door_open('top', False)
            return original(serial_port, command, timeout=timeout)
        self.lights._command_writer = racing_writer
        with self.assertRaises(DoorOpenError):
            self.lights.activate('uv310', 'dimmed')
        self.assertEqual(0, self.serial._light_pwm[3])
        self.assertIsNone(self.lights.status()['active_channel'])
        self.assertFalse(self.lights.status()['door_interlock']['uv_blocked'])

    def test_post_on_query_detects_trip_even_if_unsolicited_reports_are_delayed(self):
        original = self.lights._command_writer
        def racing_writer(serial_port, command, timeout):
            if command == 'M106 P2 S26':
                serial_port.set_door_open('front', True)
                serial_port.set_door_open('front', False)
                # Model firmware that has not transmitted the async reports yet.
                serial_port.reset_input_buffer()
            return original(serial_port, command, timeout=timeout)
        self.lights._command_writer = racing_writer
        with self.assertRaises(DoorOpenError):
            self.lights.activate('uv255', 'dimmed')
        self.assertEqual(0, self.serial._light_pwm[2])
        self.assertIsNone(self.lights.status()['active_channel'])

    def test_missing_post_on_status_turns_uv_back_off(self):
        original = self.serial._execute
        queries = []
        def execute(command):
            if command == 'M119':
                queries.append(1)
                if len(queries) == 2:
                    return b'ok\n'
            return original(command)
        self.serial._execute = execute
        with self.assertRaises(DoorOpenError):
            self.lights.activate('uv310', 'full')
        self.assertEqual(0, self.serial._light_pwm[3])
        self.assertIsNone(self.lights.status()['active_channel'])

    def test_fragmented_rejection_waits_for_ok_but_still_fails(self):
        original_read = self.serial.read
        self.serial.read = lambda size=1: original_read(min(size, 1))
        self.serial.set_door_open('front', True)
        acknowledged, reply = porthandler.write_and_wait(self.serial, 'M106 P2 S255')
        self.assertFalse(acknowledged)
        self.assertTrue(reply.endswith(b'ok\n'))
        self.assertTrue(porthandler.door_interlock.snapshot(self.serial)['front_open'])

    def test_report_split_between_drains_survives(self):
        self.serial._input.extend(b'TS_DOORS FRONT:0 TOP:1 UV_')
        porthandler.drain_serial_buffer(self.serial)
        self.serial._input.extend(b'BLOCKED:1 TRIP:42\n')
        porthandler.drain_serial_buffer(self.serial)
        self.assertEqual(42, porthandler.door_interlock.snapshot(self.serial)['trip_count'])

    def test_invalid_report_invalidates_previously_closed_state(self):
        state = DoorInterlockState()
        state.feed(self.serial, b'TS_DOORS FRONT:0 TOP:0 UV_BLOCKED:0 TRIP:3\n')
        for report in (b'TS_DOORS FRONT:1 TOP:0 UV_BLOCKED:0 TRIP:4\n',
                       b'TS_DOORS FRONT:0 TOP:0 UV_BLOCKED:0 TRIP:4294967296\n'):
            state.feed(self.serial, report)
            self.assertFalse(state.snapshot(self.serial)['known'])

    def test_trip_counter_wrap_and_reset_invalidate_active_uv(self):
        for before, after in ((0xffffffff, 0), (17, 0)):
            with self.subTest(before=before):
                self.serial._door_trips = before
                self.lights.activate('uv255', 'full')
                self.serial._door_trips = after
                porthandler.refresh_door_status(self.serial)
                with self.assertRaises(CaptureIlluminationError):
                    self.lights.capture_remaining_seconds('uv255', 'full')
                self.assertIsNone(self.lights.status()['active_channel'])

    def test_poll_does_not_wait_for_a_serial_owner(self):
        owned, release = Event(), Event()
        def owner():
            with porthandler.motion_lock:
                owned.set()
                release.wait(2)
        thread = Thread(target=owner)
        thread.start()
        try:
            self.assertTrue(owned.wait(1))
            porthandler.poll_door_interlock()
            self.assertEqual([], self.serial.command_history)
        finally:
            release.set()
            thread.join(2)

    def _connect_real_adapter(self, adapter):
        with patch.object(globals, 'motion_platform', None), patch.object(
            porthandler, 'motion_platform', None
        ), patch('serial.tools.list_ports.comports', return_value=[
            SimpleNamespace(device='TEST_COM', vid=0x0483, pid=0x5740)
        ]), patch('serial.Serial', return_value=adapter), patch.object(porthandler.time, 'sleep'):
            return porthandler.connect_to_motion_platform()

    def test_real_connection_requires_exact_identity_and_initial_report(self):
        for marker in (porthandler.EXPECTED_FIRMWARE_MARKER.replace('-DOORS1', ''),
                       porthandler.EXPECTED_FIRMWARE_MARKER + '0'):
            adapter = VirtualOctopusSerial()
            execute = adapter._execute
            adapter._execute = lambda command: (marker.encode() + b'\nok\n') if command == 'M115' else execute(command)
            self.assertIsNone(self._connect_real_adapter(adapter))
            self.assertFalse(adapter.is_open)
        adapter = VirtualOctopusSerial()
        execute = adapter._execute
        adapter._execute = lambda command: b'ok\n' if command == 'M119' else execute(command)
        self.assertIsNone(self._connect_real_adapter(adapter))
        self.assertFalse(adapter.is_open)

    def test_reconnection_uses_current_doors_instead_of_old_cache(self):
        porthandler.refresh_door_status(self.serial)
        adapter = VirtualOctopusSerial()
        read = adapter.read
        adapter.read = lambda size=1: read(min(size, 2))
        adapter.set_door_open('top', True)
        self.assertIs(adapter, self._connect_real_adapter(adapter))
        self.assertTrue(porthandler.door_interlock.snapshot(adapter)['top_open'])
        self.assertEqual(1, porthandler.door_interlock.snapshot(adapter)['trip_count'])
        adapter.close()

    def test_manual_unknown_and_automatic_preflight_reject_with_e1402(self):
        with patch('logger_config.setup_logger'):
            import app
        for door in ('front', 'top'):
            self.serial.set_door_open(door, True)
            response = app.app.test_client().post('/api/auto_measurement/preflight')
            self.assertEqual(409, response.status_code)
            self.assertEqual('E1402', response.json['code'])
            self.assertTrue(response.json['popup'])
            self.serial.set_door_open(door, False)
        execute = self.serial._execute
        self.serial._execute = lambda command: b'ok\n' if command == 'M119' else execute(command)
        with patch.object(app, 'light_controller', self.lights):
            response = app.app.test_client().post('/api/lights/activate', json={'channel': 'uv365', 'mode': 'full'})
        self.assertEqual(409, response.status_code)
        self.assertEqual('E1402', response.json['code'])

    def test_automatic_preflight_respects_motion_ownership_without_querying(self):
        with patch('logger_config.setup_logger'):
            import app
        globals.motion_busy = True
        before = list(self.serial.command_history)
        response = app.app.test_client().post('/api/auto_measurement/preflight')
        self.assertEqual(409, response.status_code)
        self.assertEqual('E1403', response.json['code'])
        self.assertEqual(before, self.serial.command_history)

    def test_missing_query_report_invalidates_cached_closed_state(self):
        porthandler.refresh_door_status(self.serial)
        original = self.serial._execute
        self.serial._execute = lambda command: b'ok\n' if command == 'M119' else original(command)
        for channel in ('uv255', 'uv310', 'uv365'):
            with self.subTest(channel=channel), self.assertRaises(DoorOpenError):
                self.lights.activate(channel, 'full')
        self.assertFalse(self.lights.status()['door_interlock']['known'])

    def test_unknown_door_status_does_not_cancel_thermal_cutoff(self):
        self.lights.activate('uv365', 'full')
        porthandler.door_interlock.invalidate(self.serial)
        self.assertEqual('uv365', self.lights.status()['active_channel'])
        self.lights._clock = lambda: float('inf')
        self.assertEqual('uv365', self.lights.check_timeouts())
        self.assertEqual(0, self.serial._light_pwm[1])

    def test_background_poll_respects_motion_busy(self):
        globals.motion_busy = True
        porthandler.poll_door_interlock()
        self.assertEqual([], self.serial.command_history)
        globals.motion_busy = False
        porthandler.poll_door_interlock()
        self.assertEqual(['M119'], self.serial.command_history)

    def test_failed_readback_and_failed_off_retain_the_thermal_deadline(self):
        refresh = self.lights._door_status_refresher
        write = self.lights._command_writer
        queries = []
        fail_off = []
        def failing_refresh(serial_port):
            queries.append(1)
            if len(queries) == 2:
                porthandler.door_interlock.invalidate(serial_port)
                fail_off.append(True)
                raise OSError('status readback failed')
            return refresh(serial_port)
        def failing_writer(serial_port, command, timeout):
            if fail_off and command == 'M106 P2 S0':
                fail_off.pop()
                return False, b''
            return write(serial_port, command, timeout=timeout)
        self.lights._door_status_refresher = failing_refresh
        self.lights._command_writer = failing_writer
        with self.assertRaises(LightCommandError):
            self.lights.activate('uv255', 'full')
        self.assertEqual('uv255', self.lights.status()['active_channel'])
        self.assertEqual(255, self.serial._light_pwm[2])
        self.lights._clock = lambda: float('inf')
        self.assertEqual('uv255', self.lights.check_timeouts())
        self.assertEqual(0, self.serial._light_pwm[2])

    def test_manual_activation_returns_door_error_instead_of_disconnect(self):
        with patch('logger_config.setup_logger'):
            import app
        self.serial.set_door_open('front', True)
        with patch.object(app, 'light_controller', self.lights):
            response = app.app.test_client().post('/api/lights/activate', json={
                'channel': 'uv255', 'mode': 'dimmed',
            })
        self.assertEqual(409, response.status_code)
        self.assertEqual('E1402', response.json['code'])
        self.assertFalse(globals.motion_busy)

    def test_light_status_get_returns_cached_doors_without_commands(self):
        with patch('logger_config.setup_logger'):
            import app
        self.lights.activate('uv255', 'dimmed')
        self.serial.set_door_open('top', True)
        porthandler.refresh_door_status(self.serial)
        before = list(self.serial.command_history)
        with patch.object(app, 'light_controller', self.lights), patch.object(
            app.height_offset_control, 'status', return_value={}
        ):
            response = app.app.test_client().get('/api/lights/status')
        self.assertEqual(200, response.status_code)
        self.assertTrue(response.json['door_interlock']['top_open'])
        self.assertIsNone(response.json['active_channel'])
        self.assertEqual(before, self.serial.command_history)

    def test_poll_failure_does_not_skip_thermal_monitor(self):
        with patch('logger_config.setup_logger'):
            import app
        with patch.object(app.light_controller, 'check_timeouts', return_value=None) as check, patch.object(
            porthandler, 'poll_door_interlock', side_effect=OSError('USB read failed')
        ), patch.object(app.app.logger, 'exception'), patch.object(
            app.time, 'sleep', side_effect=StopIteration
        ):
            with self.assertRaises(StopIteration):
                app.four_channel_lamp_timeout_monitor()
        check.assert_called_once()


if __name__ == '__main__':
    unittest.main()
