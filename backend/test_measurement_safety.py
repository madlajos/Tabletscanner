"""Door pauses, fresh retry and cancellation without a camera or controller."""

import unittest
import tempfile
from pathlib import Path
from contextlib import ExitStack
from unittest.mock import Mock, patch

import numpy as np

import globals
import porthandler
from filter_series_camera import capture_series_frame
from light_control import CaptureIlluminationError, LightController
from measurement_safety import MeasurementDoorGuard, MeasurementCancelled, acquisition_checkpoint
from settings_manager import OCTOPUS_LIGHT_OUTPUT_SELECTORS
from virtual_octopus import VirtualOctopusSerial


class MeasurementSafetyTests(unittest.TestCase):
    def setUp(self):
        self.serial = VirtualOctopusSerial()
        settings = {'lamp_settings': {
            'output_selectors': dict(OCTOPUS_LIGHT_OUTPUT_SELECTORS),
            'channels': {channel: {'dim_percent': 10, 'full_percent': 100,
                                  'dim_timeout_seconds': 30, 'full_timeout_seconds': 5}
                         for channel in ('uv255', 'uv310', 'uv365')},
        }}
        self.lights = LightController(
            lambda: settings, lambda: self.serial, porthandler.write_and_wait,
            operation_lock=porthandler.motion_lock,
            door_status_getter=lambda: porthandler.door_interlock.snapshot(self.serial),
            door_status_refresher=porthandler.refresh_door_status,
        )
        self.pauses = []
        self.cancelled = False
        self.guard = MeasurementDoorGuard(
            self.serial, self.lights, porthandler.refresh_door_status,
            self.pauses.append, lambda: self.cancelled, wait=self.close_doors,
        )

    def tearDown(self):
        self.serial.close()

    def close_doors(self, _seconds):
        # Pausing must release camera ownership and serial ownership.
        if self.pauses and self.pauses[-1]:
            self.assertFalse(globals.grab_lock.locked())
            self.assertEqual(0, globals.preview_grab_suppression_count)
        self.serial.set_door_open('front', False)
        self.serial.set_door_open('top', False)

    def test_either_door_pauses_even_vis_phases_then_automatically_continues(self):
        for door in ('front', 'top'):
            self.serial.set_door_open(door, True)
            operation = Mock(return_value='done')
            self.assertEqual('done', self.guard.run(operation))
            operation.assert_called_once()
        self.assertEqual([True, False, True, False], self.pauses)

    def test_brief_opening_retries_only_current_phase_and_rearms_optics(self):
        completed = Mock(return_value='previous image')
        self.guard.run(completed)
        attempts = []
        def phase():
            attempts.append(1)
            if len(attempts) == 1:
                self.serial.set_door_open('front', True)
                self.serial.set_door_open('front', False)
            acquisition_checkpoint()
            return 'valid image'
        prepare = Mock()
        self.assertEqual('valid image', self.guard.run(phase, prepare=prepare))
        self.assertEqual(2, len(attempts))
        self.assertEqual(2, prepare.call_count)
        completed.assert_called_once()
        self.assertEqual([True, False], self.pauses)

    def test_interrupted_guard_or_saved_exposure_is_discarded_and_retried(self):
        for interrupted_frame in (1, 2):
            camera = Mock()
            grabbing = [True]
            camera.IsOpen.return_value = True
            camera.IsGrabbing.side_effect = lambda: grabbing[0]
            camera.StopGrabbing.side_effect = lambda: grabbing.__setitem__(0, False)
            camera.StartGrabbing.side_effect = lambda *_: grabbing.__setitem__(0, True)
            camera.ExposureTime.GetValue.return_value = 100000
            frames = []
            def grab(*_args, **_kwargs):
                frames.append(len(frames) + 1)
                if len(frames) == interrupted_frame:
                    self.serial.set_door_open('front', True)
                    self.serial.set_door_open('front', False)
                return np.full((2, 2, 3), len(frames), dtype=np.uint8)
            with patch('filter_series_camera.grab_and_convert_frame', side_effect=grab):
                result = self.guard.run(lambda: capture_series_frame(
                    camera, self.lights, 'uv255', 'full', self.guard), validate_after=False)
            self.assertEqual(interrupted_frame + 2, len(frames))
            self.assertTrue(np.all(result == len(frames)))
            self.assertIsNone(self.lights.status()['active_channel'])
            self.assertTrue(grabbing[0])

    def test_thermal_failures_are_not_retried_as_door_pauses(self):
        operation = Mock(side_effect=CaptureIlluminationError('thermal deadline'))
        with self.assertRaises(CaptureIlluminationError):
            self.guard.run(operation)
        operation.assert_called_once()
        self.assertEqual([], self.pauses)

    def test_save_errors_do_not_repeat_committed_work_when_a_door_also_trips(self):
        def save():
            self.serial.set_door_open('front', True)
            raise RuntimeError('JPEG write failed after committing an image')
        operation = Mock(side_effect=save)
        with self.assertRaisesRegex(RuntimeError, 'JPEG write failed'):
            self.guard.run(operation, validate_after=False)
        operation.assert_called_once()

    def test_stop_while_open_releases_request_without_resuming(self):
        self.serial.set_door_open('top', True)
        self.guard.wait = lambda _seconds: setattr(self, 'cancelled', True)
        operation = Mock()
        with self.assertRaises(MeasurementCancelled):
            self.guard.run(operation)
        operation.assert_not_called()
        self.assertIsNone(self.lights.status()['active_channel'])
        self.assertFalse(globals.grab_lock.locked())

    def test_no_checkpoint_is_installed_in_preview_or_after_a_phase(self):
        self.guard.run(lambda: 'complete')
        self.serial.set_door_open('top', True)
        acquisition_checkpoint()  # Preview threads continue in their own context.

    def test_flask_capture_retains_completed_rows_and_never_saves_interrupted_frame(self):
        with patch('logger_config.setup_logger'):
            import app
        camera = Mock()
        grabbing = [True]
        camera.IsOpen.return_value = True
        camera.IsGrabbing.side_effect = lambda: grabbing[0]
        camera.StopGrabbing.side_effect = lambda: grabbing.__setitem__(0, False)
        camera.StartGrabbing.side_effect = lambda *_: grabbing.__setitem__(0, True)
        camera.ExposureTime.GetValue.return_value = 100000
        frames = []
        def grab(*_args, **_kwargs):
            frames.append(len(frames) + 1)
            if len(frames) == 4:
                self.serial.set_door_open('top', True)
                self.serial.set_door_open('top', False)
            return np.full((4, 4, 3), len(frames), dtype=np.uint8)
        def guard_factory(*args):
            return MeasurementDoorGuard(*args, wait=lambda _seconds: None)
        saved = []
        original_save = app._save_capture_jpeg
        def save(*args, **kwargs):
            result = original_save(*args, **kwargs)
            saved.append(1)
            if len(saved) == 2:
                # The last row is valid and already saved. Do not recapture it.
                self.serial.set_door_open('top', True)
                self.serial.set_door_open('top', False)
            return result
        rows = [dict(wavelength=channel, brightness='full', filter_position=index + 1,
                     exposure_time=100000, gain=0) for index, channel in enumerate(('vis', 'uv255'))]
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            replacements = {
                'light_controller': self.lights, 'MeasurementDoorGuard': guard_factory,
                '_check_devices_connected': lambda: (self.serial, camera, None, None),
                'validate_capture_plan': lambda rows: rows,
                '_validate_capture_plan_camera_values': lambda rows: None,
                '_apply_capture_plan_camera_settings': lambda row: None,
                '_move_toolhead_absolute_impl': lambda **kwargs: ({}, 200),
                '_wait_for_motion_complete': lambda *args, **kwargs: True,
                '_check_tablet_presence_enabled': lambda settings: False,
                '_measurement_capture_rows': lambda rows, *args: list(enumerate(rows)),
                '_select_measurement_filter_position': lambda *args: None,
                '_current_filter_settings': lambda: {},
                '_current_capture_metadata': lambda *args, **kwargs: {},
                '_save_capture_jpeg': save,
                '_turn_off_all_lights': self.lights.off,
            }
            for key, value in replacements.items():
                stack.enter_context(patch.object(app, key, value))
            stack.enter_context(patch.object(globals, 'camera', camera))
            stack.enter_context(patch.object(app.time, 'sleep'))
            stack.enter_context(patch('filter_series_camera.grab_and_convert_frame', side_effect=grab))
            stack.enter_context(patch.object(app.height_offset_control, 'apply_active_combination',
                                             return_value={'applied': True}))
            response = app.app.test_client().post('/api/auto_measurement/step', json={
                'request_id': 'door-retry-route', 'tablet_index': 2, 'measurement_name': 'test',
                'measurement_folder': folder, 'x': 1, 'y': 2, 'capture_plan': rows,
            })
            self.assertEqual(200, response.status_code, response.json)
            self.assertEqual(2, response.json['door_pause_count'])
            self.assertEqual(6, len(frames))
            self.assertEqual(2, len(response.json['saved_images']))
            self.assertEqual(2, len(list(Path(folder).glob('*.jpg'))))
            self.assertTrue(all(Path(path).is_file() for path in response.json['saved_images']))
            progress = app.measurement_progress.snapshot('door-retry-route')
            self.assertEqual('completed', progress['status'])
            self.assertEqual(2, len(progress['images']))

    def test_flask_first_tablet_with_open_door_has_no_motion_or_folder(self):
        with patch('logger_config.setup_logger'):
            import app
        self.serial.set_door_open('front', True)
        with patch.object(app, '_check_devices_connected', return_value=(self.serial, Mock(), None, None)), \
             patch.object(app, 'validate_capture_plan', side_effect=lambda rows: rows), \
             patch.object(app, '_validate_capture_plan_camera_values'), \
             patch.object(app, '_move_toolhead_absolute_impl') as move, \
             patch.object(app.os, 'makedirs') as mkdir:
            response = app.app.test_client().post('/api/auto_measurement/step', json={
                'request_id': 'door-blocked-first', 'tablet_index': 1, 'is_first_tablet': True,
                'measurement_name': 'test', 'measurement_folder': 'not-created', 'x': 1, 'y': 2,
                'capture_plan': [{'filter_position': 1}],
            })
        self.assertEqual(409, response.status_code)
        self.assertEqual('E1402', response.json['code'])
        move.assert_not_called()
        mkdir.assert_not_called()


    def test_flask_cancel_while_paused_returns_after_cleanup_without_moving(self):
        with patch('logger_config.setup_logger'):
            import app
        self.serial.set_door_open('top', True)
        def guard_factory(*args):
            return MeasurementDoorGuard(*args, wait=lambda _seconds: setattr(globals, 'autofocus_abort', True))
        with patch.object(app, '_check_devices_connected', return_value=(self.serial, Mock(), None, None)), \
             patch.object(app, 'light_controller', self.lights), \
             patch.object(app, 'MeasurementDoorGuard', side_effect=guard_factory), \
             patch.object(app, 'validate_capture_plan', side_effect=lambda rows: rows), \
             patch.object(app, '_validate_capture_plan_camera_values'), \
             patch.object(globals, 'autofocus_abort', False), \
             patch.object(app, '_move_toolhead_absolute_impl') as move, \
             patch.object(app.os, 'makedirs') as mkdir:
            response = app.app.test_client().post('/api/auto_measurement/step', json={
                'request_id': 'cancel-door-pause', 'tablet_index': 2,
                'measurement_name': 'test', 'measurement_folder': 'not-created', 'x': 1, 'y': 2,
                'capture_plan': [{'filter_position': 1}],
            })
        self.assertEqual(200, response.status_code, response.json)
        self.assertTrue(response.json['cancelled'])
        self.assertIsNone(self.lights.status()['active_channel'])
        move.assert_not_called()
        mkdir.assert_not_called()


    def test_stop_arriving_before_tablet_post_does_not_reset_abort_or_touch_devices(self):
        with patch('logger_config.setup_logger'):
            import app
        with patch.object(globals, 'autofocus_abort', False), \
             patch.object(app, '_check_devices_connected') as devices, \
             patch.object(app.os, 'makedirs') as mkdir:
            client = app.app.test_client()
            stopped = client.post('/api/abort-autofocus', json={'request_id': 'stopped-before-post'})
            self.assertEqual(200, stopped.status_code)
            response = client.post('/api/auto_measurement/step', json={
                'request_id': 'stopped-before-post', 'tablet_index': 2,
            })
            self.assertTrue(globals.autofocus_abort)
        self.assertEqual(200, response.status_code)
        self.assertTrue(response.json['cancelled'])
        devices.assert_not_called()
        mkdir.assert_not_called()


if __name__ == '__main__':
    unittest.main()
