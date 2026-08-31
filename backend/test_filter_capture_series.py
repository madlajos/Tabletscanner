#!/usr/bin/env python3
"""Focused naming, coordination, and Flask contract checks for BGR capture."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import app as backend_app
from filter_capture_series import (
    FilterCaptureSeriesCoordinator,
    capture_filename,
    capture_folder_is_empty,
    capture_series_stem,
    next_capture_series_index,
    resolve_filter_targets,
)
import globals
import height_offset_control
import porthandler
import settings_manager
from virtual_octopus import VirtualOctopusSerial


def bgr_filter_settings():
    zero_offsets = {'uv255': 0, 'uv310': 0, 'uv365': 0, 'vis': 0}
    return {
        'filters': [
            {'id': 'blue', 'name': 'Kék', 'wavelength_range': '450-500', 'color': '#0000ff'},
            {'id': 'green', 'name': 'Zöld', 'wavelength_range': '500-550', 'color': '#00ff00'},
            {'id': 'red', 'name': 'Piros', 'wavelength_range': '600-650', 'color': '#ff0000'},
        ],
        'slots': [None, 'blue', 'green', 'red', None, None],
        'height_offsets_mm': {
            'empty': dict(zero_offsets),
            'blue': dict(zero_offsets),
            'green': dict(zero_offsets),
            'red': dict(zero_offsets),
        },
    }


class FakeOpenCamera:
    def IsOpen(self):
        return True


class FilterCaptureSeriesHelperTests(unittest.TestCase):
    def test_targets_follow_blue_green_red_order_and_configured_slots(self):
        targets = resolve_filter_targets(bgr_filter_settings())

        self.assertEqual(
            [('Kék', 'b', 2), ('Zöld', 'g', 3), ('Piros', 'r', 4)],
            [(target.name, target.suffix, target.position) for target in targets],
        )

    def test_next_index_is_monotonic_across_all_three_suffixes(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertTrue(capture_folder_is_empty(folder))
            stem = capture_series_stem(folder)
            Path(folder, f'{stem}_1_b.jpg').touch()
            Path(folder, f'{stem}_4_R.JPG').touch()
            Path(folder, f'{stem}_99_x.jpg').touch()

            self.assertEqual(5, next_capture_series_index(folder, stem))
            self.assertEqual(f'{stem}_5_g', capture_filename(stem, 5, 'g'))
            self.assertFalse(capture_folder_is_empty(folder))

    def test_coordinator_cancellation_is_reset_for_the_next_run(self):
        coordinator = FilterCaptureSeriesCoordinator()

        self.assertTrue(coordinator.begin())
        self.assertFalse(coordinator.begin())
        self.assertTrue(coordinator.request_cancel())
        self.assertTrue(coordinator.cancellation_requested())
        coordinator.finish()
        self.assertTrue(coordinator.begin())
        self.assertFalse(coordinator.cancellation_requested())
        coordinator.finish()


class FilterCaptureSeriesApiTests(unittest.TestCase):
    def setUp(self):
        self.device = VirtualOctopusSerial()
        self.original_camera = globals.camera
        globals.camera = FakeOpenCamera()
        globals.motion_platform = self.device
        porthandler.motion_platform = self.device
        globals.motion_busy = False
        globals.toolhead_homed = True
        globals.homed_axes = {'x', 'y', 'z', 'a'}
        globals.filter_revolver_homed = True
        globals.filter_revolver_position = 1
        globals.last_toolhead_pos = {'x': 2.0, 'y': 2.0, 'z': 10.0}
        globals.autofocus_abort = False
        height_offset_control.invalidate_reference()
        settings_manager.set_settings({'filter_settings': bgr_filter_settings()})
        backend_app.bgr_capture_coordinator.finish()
        self.client = backend_app.app.test_client()

    def tearDown(self):
        backend_app.bgr_capture_coordinator.finish()
        self.device.close()
        globals.camera = self.original_camera
        globals.motion_platform = None
        porthandler.motion_platform = None
        globals.motion_busy = False
        globals.toolhead_homed = False
        globals.homed_axes = set()
        globals.filter_revolver_homed = False
        globals.filter_revolver_position = None
        globals.autofocus_abort = False
        height_offset_control.invalidate_reference()
        settings_manager.set_settings({})

    def test_endpoint_saves_one_shared_index_in_bgr_order(self):
        with tempfile.TemporaryDirectory() as folder:
            stem = capture_series_stem(folder)
            settings = bgr_filter_settings()
            settings['height_offsets_mm']['green']['vis'] = 1.0
            settings['height_offsets_mm']['red']['vis'] = 2.0
            settings_manager.set_settings({'filter_settings': settings})
            height_offset_control.record_reference(10.0)

            def fake_capture(target_folder, filename, **_kwargs):
                path = os.path.join(target_folder, f'{filename}.jpg')
                Path(path).touch()
                return [path]

            with (
                patch.object(backend_app, '_capture_and_save_image', side_effect=fake_capture),
                patch.object(
                    backend_app,
                    '_run_configured_manual_autofocus',
                    return_value={'status': 'OK'},
                ) as autofocus,
                patch.object(backend_app.light_controller, 'status', return_value={'active_channel': 'vis'}),
            ):
                response = self.client.post(
                    '/api/bgr-capture-series',
                    json={'target_folder': folder},
                )

            body = response.get_json()
            self.assertEqual(200, response.status_code)
            self.assertEqual('completed', body['status'])
            self.assertEqual(1, body['series_index'])
            self.assertEqual(
                [f'{stem}_1_b.jpg', f'{stem}_1_g.jpg', f'{stem}_1_r.jpg'],
                [os.path.basename(image['path']) for image in body['saved_images']],
            )
            self.assertEqual(['Kék', 'Zöld', 'Piros'], [
                image['filter_name'] for image in body['saved_images']
            ])
            self.assertEqual(
                [0.0, 1.0, 2.0],
                [image['height_offset']['offset_mm'] for image in body['saved_images']],
            )
            self.assertEqual(12.0, globals.last_toolhead_pos['z'])
            self.assertEqual(4, globals.filter_revolver_position)
            self.assertFalse(globals.motion_busy)
            autofocus.assert_called_once_with(self.device, skip_empty_check=True)

    def test_cancellation_after_first_capture_skips_green_and_red(self):
        with tempfile.TemporaryDirectory() as folder:
            def fake_capture(target_folder, filename, **_kwargs):
                path = os.path.join(target_folder, f'{filename}.jpg')
                Path(path).touch()
                backend_app.bgr_capture_coordinator.request_cancel()
                return [path]

            with (
                patch.object(backend_app, '_capture_and_save_image', side_effect=fake_capture),
                patch.object(
                    backend_app,
                    '_run_configured_manual_autofocus',
                    return_value={'status': 'OK'},
                ),
            ):
                response = self.client.post(
                    '/api/bgr-capture-series',
                    json={'target_folder': folder},
                )

            body = response.get_json()
            self.assertEqual(200, response.status_code)
            self.assertEqual('cancelled', body['status'])
            self.assertEqual(['b'], [image['suffix'] for image in body['saved_images']])
            self.assertEqual(2, globals.filter_revolver_position)
            self.assertFalse(globals.motion_busy)

    def test_nonempty_folder_skips_initial_autofocus(self):
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, 'existing-note.txt').touch()

            def fake_capture(target_folder, filename, **_kwargs):
                path = os.path.join(target_folder, f'{filename}.jpg')
                Path(path).touch()
                return [path]

            with (
                patch.object(backend_app, '_capture_and_save_image', side_effect=fake_capture),
                patch.object(backend_app, '_run_configured_manual_autofocus') as autofocus,
            ):
                response = self.client.post(
                    '/api/bgr-capture-series',
                    json={'target_folder': folder},
                )

            self.assertEqual(200, response.status_code)
            self.assertEqual('completed', response.get_json()['status'])
            autofocus.assert_not_called()

    def test_initial_autofocus_failure_prevents_capture(self):
        with tempfile.TemporaryDirectory() as folder:
            with (
                patch.object(
                    backend_app,
                    '_run_configured_manual_autofocus',
                    return_value={'status': 'ERROR', 'code': 'E2007'},
                ),
                patch.object(backend_app, '_capture_and_save_image') as capture,
            ):
                response = self.client.post(
                    '/api/bgr-capture-series',
                    json={'target_folder': folder},
                )

            self.assertEqual(422, response.status_code)
            self.assertEqual('E2007', response.get_json()['code'])
            capture.assert_not_called()

    def test_cancel_endpoint_sets_the_running_series_event(self):
        self.assertTrue(backend_app.bgr_capture_coordinator.begin())
        backend_app.bgr_capture_coordinator.set_autofocus_in_progress(True)

        response = self.client.post('/api/bgr-capture-series/cancel', json={})

        self.assertEqual(200, response.status_code)
        self.assertEqual('cancellation_requested', response.get_json()['status'])
        self.assertTrue(backend_app.bgr_capture_coordinator.cancellation_requested())
        self.assertTrue(globals.autofocus_abort)


if __name__ == '__main__':
    unittest.main()
