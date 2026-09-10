"""Verify acquisition ordering, UV deadlines and cleanup with no physical camera."""

import unittest
from unittest.mock import Mock, patch

import numpy as np
import globals
from cameracontrol import stream_video
from filter_series_camera import capture_series_frame
from filter_capture_series import FilterCaptureSeriesCoordinator
from light_control import CaptureIlluminationError, LightController
from settings_manager import OCTOPUS_LIGHT_OUTPUT_SELECTORS


class FilterSeriesCameraTests(unittest.TestCase):
    def setUp(self):
        self.now = 100.
        self.events = []
        self.grabbing = True
        self.camera = Mock()
        self.camera.IsOpen.return_value = True
        self.camera.IsGrabbing.side_effect = lambda: self.grabbing
        self.camera.StopGrabbing.side_effect = lambda: self.set_grabbing(False)
        self.camera.StartGrabbing.side_effect = lambda *_: self.set_grabbing(True)
        self.camera.ExposureTime.GetValue.return_value = 100_000
        self.device = Mock(is_open=True)
        config = {'lamp_settings': {'output_selectors': dict(OCTOPUS_LIGHT_OUTPUT_SELECTORS), 'channels': {
            channel: {'dim_percent': 50, 'full_percent': 100, 'dim_timeout_seconds': 30, 'full_timeout_seconds': 5}
            for channel in ('uv255', 'uv310', 'uv365')
        }}}
        self.lights = LightController(lambda: config, lambda: self.device, self.write, clock=lambda: self.now)
        self.coordinator = FilterCaptureSeriesCoordinator()
        self.coordinator.begin()

    def write(self, _device, command, **_kwargs):
        self.events.append(command)
        return True, b'ok'

    def set_grabbing(self, grabbing):
        self.grabbing = grabbing
        self.events.append('start' if grabbing else 'stop')

    def test_nearly_expired_uv_gets_fresh_window_and_no_preview_frame_is_reused(self):
        self.lights.activate('uv255', 'full')
        self.now += 4.99
        self.events.clear()
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        grabbed_frames = []
        def grab(camera, timeout_ms, retries):
            self.assertIs(camera, self.camera)
            self.assertEqual(0, retries)
            self.assertLessEqual(timeout_ms, 2100)
            self.assertFalse(globals.grab_lock.acquire(blocking=False))
            self.assertEqual(1, globals.preview_grab_suppression_count)
            self.assertEqual('uv255', self.lights.status()['active_channel'])
            self.assertIsNone(self.lights.check_timeouts())
            grabbed_frames.append(len(grabbed_frames) + 1)
            self.events.append(f'frame-{len(grabbed_frames)}')
            return np.full_like(frame, len(grabbed_frames))
        with patch('filter_series_camera.grab_and_convert_frame', side_effect=grab):
            result = capture_series_frame(self.camera, self.lights, 'uv255', 'full', self.coordinator)
        self.assertEqual([1, 2], grabbed_frames)
        np.testing.assert_array_equal(np.full_like(frame, 2), result)
        self.assertEqual('stop', self.events[0])
        self.assertLess(self.events.index('M106 P2 S255'), self.events.index('start'))
        self.assertLess(self.events.index('start'), self.events.index('frame-1'))
        self.assertLess(self.events.index('frame-1'), self.events.index('frame-2'))
        self.assertIsNone(self.lights.status()['active_channel'])
        self.assertTrue(self.grabbing)
        self.assertTrue(globals.grab_lock.acquire(blocking=False))
        globals.grab_lock.release()
        self.assertEqual(0, globals.preview_grab_suppression_count)

    def test_exposure_longer_than_uv_window_is_rejected_before_acquisition(self):
        self.camera.ExposureTime.GetValue.return_value = 5_000_000
        with patch('filter_series_camera.grab_and_convert_frame') as grab:
            with self.assertRaises(CaptureIlluminationError):
                capture_series_frame(self.camera, self.lights, 'uv365', 'full', self.coordinator)
            grab.assert_not_called()
        self.assertIsNone(self.lights.status()['active_channel'])
        self.assertTrue(self.grabbing)

    def test_long_exposure_uses_two_exposure_scaled_grabs_and_returns_second(self):
        self.camera.ExposureTime.GetValue.return_value = 1_250_000
        frames = [
            np.full((2, 2, 3), 1, dtype=np.uint8),
            np.full((2, 2, 3), 2, dtype=np.uint8),
        ]
        with patch('filter_series_camera.grab_and_convert_frame', side_effect=frames) as grab:
            result = capture_series_frame(self.camera, self.lights, 'vis', None, self.coordinator)
        self.assertEqual(2, grab.call_count)
        self.assertEqual([3250, 3250], [call.kwargs['timeout_ms'] for call in grab.call_args_list])
        np.testing.assert_array_equal(frames[1], result)

    def test_timeout_monitor_remains_enabled_and_expired_frame_is_rejected(self):
        def grab(*_args, **_kwargs):
            self.now += 6
            self.assertEqual('uv310', self.lights.check_timeouts())
            return np.zeros((2, 2, 3), dtype=np.uint8)
        with patch('filter_series_camera.grab_and_convert_frame', side_effect=grab):
            with self.assertRaises(CaptureIlluminationError):
                capture_series_frame(self.camera, self.lights, 'uv310', 'full', self.coordinator)
        self.assertIsNone(self.lights.status()['active_channel'])
        self.assertTrue(self.grabbing)

    def test_cancel_and_camera_failure_both_restore_acquisition_and_turn_off(self):
        self.coordinator.request_cancel()
        with patch('filter_series_camera.grab_and_convert_frame') as grab:
            self.assertIsNone(capture_series_frame(self.camera, self.lights, 'vis', None, self.coordinator))
            grab.assert_not_called()
        self.coordinator.finish()
        self.coordinator.begin()
        with patch('filter_series_camera.grab_and_convert_frame', side_effect=RuntimeError('Disconnected')):
            with self.assertRaises(RuntimeError):
                capture_series_frame(self.camera, self.lights, 'vis', None, self.coordinator)
        self.assertIsNone(self.lights.status()['active_channel'])
        self.assertTrue(self.grabbing)

    def test_busy_camera_displays_owned_frames_without_waiting_for_lock(self):
        original_camera = globals.camera
        original_stream_running = globals.stream_running
        original_owned_image = globals.latest_owned_preview_image
        original_owned_sequence = globals.latest_owned_preview_sequence
        preview_camera = Mock()
        preview_camera.IsOpen.return_value = True
        preview_camera.IsGrabbing.return_value = True
        globals.camera = preview_camera
        globals.stream_running = True
        generator = stream_video(scale_factor=1)
        try:
            image = np.full((2, 2, 3), 127, dtype=np.uint8)
            with patch('cameracontrol.grab_and_convert_frame', return_value=image) as grab:
                live_frame = next(generator)
                globals.grab_lock.acquire()
                self.addCleanup(globals.grab_lock.release)
                from cameracontrol import publish_owned_preview_frame
                publish_owned_preview_frame(np.full((2, 2, 3), 240, dtype=np.uint8))
                owned_frame = next(generator)
                self.assertNotEqual(live_frame, owned_frame)
                grab.assert_called_once()
        finally:
            globals.stream_running = False
            generator.close()
            globals.camera = original_camera
            globals.stream_running = original_stream_running
            globals.latest_owned_preview_image = original_owned_image
            globals.latest_owned_preview_sequence = original_owned_sequence

    def test_measurement_grab_suppression_is_scoped_to_the_acquisition(self):
        from cameracontrol import suppress_preview_grabs

        self.assertEqual(0, globals.preview_grab_suppression_count)
        with suppress_preview_grabs():
            self.assertEqual(1, globals.preview_grab_suppression_count)
            with suppress_preview_grabs():
                self.assertEqual(2, globals.preview_grab_suppression_count)
            self.assertEqual(1, globals.preview_grab_suppression_count)
        self.assertEqual(0, globals.preview_grab_suppression_count)

    def test_live_stream_resumes_immediately_after_measurement_grab(self):
        from cameracontrol import publish_owned_preview_frame, suppress_preview_grabs

        original_camera = globals.camera
        original_stream_running = globals.stream_running
        preview_camera = Mock()
        preview_camera.IsOpen.return_value = True
        preview_camera.IsGrabbing.return_value = True
        globals.camera = preview_camera
        globals.stream_running = True
        generator = stream_video(scale_factor=1)
        try:
            with patch(
                'cameracontrol.grab_and_convert_frame',
                side_effect=[
                    np.full((2, 2, 3), 40, dtype=np.uint8),
                    np.full((2, 2, 3), 80, dtype=np.uint8),
                ],
            ) as grab:
                next(generator)
                with suppress_preview_grabs():
                    publish_owned_preview_frame(np.full((2, 2, 3), 60, dtype=np.uint8))
                    next(generator)
                    self.assertEqual(1, grab.call_count)
                next(generator)
                self.assertEqual(2, grab.call_count)
        finally:
            globals.stream_running = False
            generator.close()
            globals.camera = original_camera
            globals.stream_running = original_stream_running


if __name__ == '__main__':
    unittest.main()
