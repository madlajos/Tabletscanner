#!/usr/bin/env python3
"""Focused naming, coordination, and Flask contract checks for BGR capture."""

import os
from pathlib import Path
import tempfile
import threading
import copy
import unittest
import numpy as np
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
    class Value:
        def __init__(self, value=0): self.value = value
        def SetValue(self, value): self.value = value
        def GetValue(self): return self.value

    def __init__(self):
        self.ExposureTime = self.Value()
        self.Gain = self.Value()
        self.ReverseX = self.Value()
        self.ReverseY = self.Value()
        self.grabbing = True

    def IsOpen(self):
        return True

    def IsGrabbing(self):
        return self.grabbing

    def StopGrabbing(self):
        self.grabbing = False

    def StartGrabbing(self, *_args):
        self.grabbing = True


def successful_autofocus(*_args, **_kwargs):
    height_offset_control.record_reference(10.)
    return {'status': 'OK'}


class FilterCaptureSeriesHelperTests(unittest.TestCase):
    def test_targets_follow_red_green_blue_order_and_configured_slots(self):
        targets = resolve_filter_targets(bgr_filter_settings())

        self.assertEqual(
            [('Piros', 'r', 4), ('Zöld', 'g', 3), ('Kék', 'b', 2)],
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

    def test_uv_only_partial_set_reserves_its_index(self):
        with tempfile.TemporaryDirectory() as folder:
            stem = capture_series_stem(folder)
            Path(folder, f'{stem}_7_uv.jpg').touch()
            self.assertEqual(8, next_capture_series_index(folder, stem))
            self.assertEqual(f'{stem}_8_uv', capture_filename(stem, 8, 'uv'))

    def test_wavelength_qualified_names_are_unique_and_reserve_the_index(self):
        with tempfile.TemporaryDirectory() as folder:
            stem = capture_series_stem(folder)
            Path(folder, f'{stem}_3_uv255_r.jpg').touch()
            self.assertEqual(4, next_capture_series_index(folder, stem))
            self.assertEqual(
                f'{stem}_4_uv310_uv',
                capture_filename(stem, 4, 'uv', 'uv310'),
            )

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
    def test_scanner_camera_controls_do_not_overwrite_combination_settings(self):
        filters = bgr_filter_settings()
        matrix = settings_manager.default_camera_combination_settings(filters)
        matrix['empty']['vis'] = {'exposure_time': 654321, 'gain': 6.25}
        settings_manager.set_settings({
            'filter_settings': filters,
            'camera_combination_settings': copy.deepcopy(matrix),
            'camera_params': {'ExposureTime': 100000, 'Gain': 0, 'Gamma': 1},
        })

        with patch.object(backend_app, 'save_settings', return_value=True):
            response = self.client.post('/api/update-camera-settings', json={
                'setting_name': 'ExposureTime',
                'setting_value': 123456,
            })

        self.assertEqual(200, response.status_code, response.json)
        self.assertEqual(123456, settings_manager.get_settings()['camera_params']['ExposureTime'])
        self.assertEqual(matrix, settings_manager.get_settings()['camera_combination_settings'])

    def test_camera_combination_api_applies_active_cell(self):
        filters = bgr_filter_settings()
        matrix = settings_manager.default_camera_combination_settings(filters)
        matrix['empty']['vis'] = {'exposure_time': 654321, 'gain': 6.25}
        before = self.client.get('/api/settings/camera/combinations')
        self.assertEqual(200, before.status_code)
        self.assertEqual(set(settings_manager.CAMERA_FILTER_GROUPS), set(before.json['camera_combination_settings']))
        def persist_without_operator_file(settings):
            settings_manager.get_settings()['camera_combination_settings'] = copy.deepcopy(settings)
            return True
        with patch.object(backend_app, 'update_camera_combination_settings', side_effect=persist_without_operator_file):
            response = self.client.put('/api/settings/camera/combinations', json=matrix)
        self.assertEqual(200, response.status_code, response.json)
        self.assertEqual(654321, response.json['camera_params']['ExposureTime'])
        self.assertEqual(6.25, response.json['camera_params']['Gain'])
        self.assertEqual(654321, globals.camera.ExposureTime.GetValue())
        self.assertEqual(6.25, globals.camera.Gain.GetValue())
        invalid = self.client.put('/api/settings/camera/combinations', json={'empty': {}})
        self.assertEqual(400, invalid.status_code)

    def test_clamped_capture_saves_warning_metadata_and_progress(self):
        with tempfile.TemporaryDirectory() as folder:
            settings = bgr_filter_settings()
            settings['height_offsets_mm']['red']['uv255'] = 3
            settings_manager.set_settings({'filter_settings': settings})
            backend_app.light_controller.activate('uv255', 'dimmed')
            height_offset_control.record_reference(39)
            response = self.client.post('/api/bgr-capture-series', json={
                'target_folder': folder, 'mode': 'rgb', 'capture_id': 'clamped-test'})
            self.assertEqual(200, response.status_code)
            body = response.get_json()
            self.assertEqual(3, len(body['saved_images']))
            self.assertEqual(2, body['warnings'][0]['missing_offset_mm'])
            red = body['saved_images'][0]
            self.assertEqual(40, red['metadata']['z'])
            self.assertEqual('uv255', red['metadata']['wavelength'])
            self.assertEqual(['ZOffset difference: 2 mm'], red['metadata']['Errors'])
            read = self.client.get('/api/image-metadata', query_string={'path': red['path']})
            self.assertEqual(red['metadata'], read.get_json()['metadata'])
            self.assertEqual([], body['saved_images'][1]['metadata']['Errors'])
            status = self.client.get('/api/bgr-capture-series/status').get_json()
            self.assertFalse(status['running'])
            self.assertEqual('clamped-test', status['capture_id'])
            self.assertEqual(body['saved_images'], status['saved_images'])

    def test_manual_capture_uses_active_uv_instead_of_stale_client_vis(self):
        with tempfile.TemporaryDirectory() as folder:
            backend_app.light_controller.activate('uv365', 'dimmed')
            def capture_with_monitor_access():
                acquired = []
                def monitor():
                    locked = porthandler.motion_lock.acquire(timeout=.2)
                    acquired.append(locked)
                    if locked:
                        porthandler.motion_lock.release()
                thread = threading.Thread(target=monitor)
                thread.start()
                thread.join(timeout=1)
                self.assertEqual([True], acquired, 'UV timeout monitor must be able to acquire the serial lock during exposure')
                self.assertTrue(globals.motion_busy)
                return np.zeros((2, 2, 3), dtype=np.uint8)
            with patch.object(backend_app, 'grab_camera_image', side_effect=capture_with_monitor_access):
                response = self.client.post('/api/save_raw_image', json={
                    'target_folder': folder, 'light_type': 'dome', 'metadata': {'wavelength': 'vis'}})
            self.assertEqual(200, response.status_code)
            body = response.get_json()
            self.assertEqual('uv365', body['metadata']['wavelength'])
            self.assertEqual(body['metadata'], self.client.get('/api/image-metadata', query_string={'path': body['path']}).get_json()['metadata'])

    def setUp(self):
        self.device = VirtualOctopusSerial()
        self.original_camera = globals.camera
        globals.camera = FakeOpenCamera()
        globals.camera_properties = {
            'ExposureTime': {'min': 1, 'max': 2000000, 'inc': 1},
            'Gain': {'min': 0, 'max': 24, 'inc': 0.000001},
        }
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
        lamp_settings = {
            'output_selectors': dict(settings_manager.OCTOPUS_LIGHT_OUTPUT_SELECTORS),
            'channels': {channel: {'dim_percent': 50, 'full_percent': 100,
                'dim_timeout_seconds': 30, 'full_timeout_seconds': 5}
                for channel in settings_manager.UV_LAMP_CHANNELS},
        }
        lamp_config = patch.object(backend_app.light_controller, '_settings_getter', lambda: {'lamp_settings': lamp_settings})
        lamp_config.start()
        self.addCleanup(lamp_config.stop)
        backend_app.light_controller.activate('vis')
        frame_capture = patch.object(backend_app, 'capture_series_frame', return_value=np.zeros((2, 2, 3), dtype=np.uint8))
        self.frame_capture = frame_capture.start()
        self.addCleanup(frame_capture.stop)
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

    def test_uv_rgb_preview_acquires_during_autofocus_motion_and_saving(self):
        from cameracontrol import stream_video

        settings = bgr_filter_settings()
        settings['filters'].append({
            'id': 'uv', 'name': '255 nm', 'wavelength_range': '255 nm', 'color': '#ffffff'})
        settings['slots'][4] = 'uv'
        settings['height_offsets_mm']['uv'] = dict(settings['height_offsets_mm']['blue'])
        settings_manager.set_settings({'filter_settings': settings})
        backend_app.light_controller.activate('uv255', 'full')
        original_stream_running = globals.stream_running
        self.addCleanup(setattr, globals, 'stream_running', original_stream_running)
        globals.stream_running = True
        preview = stream_video(scale_factor=1)
        self.addCleanup(preview.close)
        phases = []
        with patch('cameracontrol.grab_and_convert_frame',
                   return_value=np.zeros((2, 2, 3), dtype=np.uint8)) as grab:
            def check_preview(phase):
                before = grab.call_count
                next(preview)
                self.assertEqual(before + 1, grab.call_count, phase)
                phases.append(phase)

            def autofocus(*args, **kwargs):
                check_preview('autofocus')
                return successful_autofocus(*args, **kwargs)

            move_filter = backend_app._move_filter_revolver_to_position
            def movement(*args, **kwargs):
                check_preview('filter movement')
                return move_filter(*args, **kwargs)

            def save(folder, filename, **kwargs):
                check_preview('saving')
                return [os.path.join(folder, filename + '.jpg')]

            with (
                tempfile.TemporaryDirectory() as folder,
                patch.object(backend_app, '_run_configured_manual_autofocus', side_effect=autofocus),
                patch.object(backend_app, '_move_filter_revolver_to_position', side_effect=movement),
                patch.object(backend_app, '_capture_and_save_image', side_effect=save),
            ):
                response = self.client.post('/api/bgr-capture-series', json={
                    'target_folder': folder, 'mode': 'uv_rgb'})
            self.assertEqual(200, response.status_code, response.json)
            self.assertIn('autofocus', phases)
            self.assertEqual(5, phases.count('filter movement'))
            self.assertEqual(4, phases.count('saving'))

    def test_available_autofocus_reference_skips_autofocus_and_saves_in_rgb_order(self):
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
                    side_effect=successful_autofocus,
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
                [f'{stem}_1_r.jpg', f'{stem}_1_g.jpg', f'{stem}_1_b.jpg'],
                [os.path.basename(image['path']) for image in body['saved_images']],
            )
            self.assertEqual(['Piros', 'Zöld', 'Kék'], [
                image['filter_name'] for image in body['saved_images']
            ])
            self.assertEqual(
                [2.0, 1.0, 0.0],
                [image['height_offset']['offset_mm'] for image in body['saved_images']],
            )
            self.assertEqual(10.0, globals.last_toolhead_pos['z'])
            self.assertEqual(2, globals.filter_revolver_position)
            self.assertFalse(globals.motion_busy)
            autofocus.assert_not_called()

    def test_initial_autofocus_applies_and_arms_its_vis_blue_camera_cell_first(self):
        filters = bgr_filter_settings()
        matrix = settings_manager.default_camera_combination_settings(filters)
        matrix['rgb']['uv255'] = {'exposure_time': 1_000_000, 'gain': 10}
        matrix['rgb']['vis'] = {'exposure_time': 50_000, 'gain': 2.5}
        settings_manager.set_settings({
            'filter_settings': filters,
            'camera_combination_settings': matrix,
            'autofocus_settings': {
                'channel': 'vis', 'brightness': 'full', 'filter_position': 2,
            },
            'camera_params': {'ExposureTime': 1_000_000, 'Gain': 10, 'Gamma': 1},
        })
        globals.camera.ExposureTime.SetValue(1_000_000)
        globals.camera.Gain.SetValue(10)
        backend_app.light_controller.activate('uv255', 'full')

        def assert_autofocus_is_armed(*_args, **_kwargs):
            self.assertEqual('vis', backend_app.light_controller.status()['active_channel'])
            self.assertEqual(2, globals.filter_revolver_position)
            self.assertEqual(50_000, globals.camera.ExposureTime.GetValue())
            self.assertEqual(2.5, globals.camera.Gain.GetValue())
            self.assertTrue(globals.camera.IsGrabbing())
            globals.last_toolhead_pos['z'] = 10.
            return {'status': 'OK'}

        with (
            patch.object(backend_app.time, 'sleep'),
            patch.object(backend_app.autofocus_main, 'autofocus_coarse', side_effect=assert_autofocus_is_armed),
        ):
            response = backend_app._run_configured_manual_autofocus(
                self.device, skip_empty_check=True,
            )

        self.assertEqual('OK', response['status'])
        self.assertEqual({'ExposureTime': 50_000., 'Gain': 2.5}, response['camera_params'])

    def test_series_applies_camera_values_for_each_filter_and_reports_them_in_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            filters = bgr_filter_settings()
            matrix = settings_manager.default_camera_combination_settings(filters)
            matrix['rgb']['vis'] = {'exposure_time': 120003, 'gain': 3.75}
            settings_manager.set_settings({
                'filter_settings': filters,
                'camera_combination_settings': matrix,
                'camera_params': {'ExposureTime': 100000, 'Gain': 0, 'Gamma': 1},
            })
            height_offset_control.record_reference(10)
            with patch.object(backend_app.light_controller, 'status', return_value={'active_channel': 'vis'}):
                response = self.client.post('/api/bgr-capture-series', json={'target_folder': folder})
            self.assertEqual(200, response.status_code, response.json)
            images = response.get_json()['saved_images']
            self.assertEqual([120003, 120003, 120003], [item['metadata']['exposure_time'] for item in images])
            self.assertEqual([3.75, 3.75, 3.75], [item['metadata']['gain'] for item in images])
            self.assertEqual(120003, globals.camera.ExposureTime.GetValue())
            self.assertEqual(3.75, globals.camera.Gain.GetValue())

    def test_cancellation_after_first_capture_skips_green_and_blue(self):
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
                    side_effect=successful_autofocus,
                ),
            ):
                response = self.client.post(
                    '/api/bgr-capture-series',
                    json={'target_folder': folder},
                )

            body = response.get_json()
            self.assertEqual(200, response.status_code)
            self.assertEqual('cancelled', body['status'])
            self.assertEqual(['r'], [image['suffix'] for image in body['saved_images']])
            self.assertEqual(4, globals.filter_revolver_position)
            self.assertFalse(globals.motion_busy)

    def test_nonempty_folder_runs_autofocus_when_reference_is_unavailable(self):
        with tempfile.TemporaryDirectory() as folder:
            stem = capture_series_stem(folder)
            for suffix in ('b', 'g', 'r'):
                Path(folder, f'{stem}_1_{suffix}.jpg').touch()

            def fake_capture(target_folder, filename, **_kwargs):
                path = os.path.join(target_folder, f'{filename}.jpg')
                Path(path).touch()
                return [path]

            with (
                patch.object(backend_app, '_capture_and_save_image', side_effect=fake_capture),
                patch.object(
                    backend_app,
                    '_run_configured_manual_autofocus',
                    side_effect=successful_autofocus,
                ) as autofocus,
            ):
                response = self.client.post(
                    '/api/bgr-capture-series',
                    json={'target_folder': folder},
                )

            self.assertEqual(200, response.status_code)
            self.assertEqual('completed', response.get_json()['status'])
            self.assertEqual(2, response.get_json()['series_index'])
            autofocus.assert_called_once_with(self.device, skip_empty_check=True)

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

    def test_no_active_lamp_rejects_without_autofocus_capture_or_motion(self):
        backend_app.light_controller.off()
        self.device.command_history.clear()
        with tempfile.TemporaryDirectory() as folder, patch.object(backend_app, '_run_configured_manual_autofocus') as autofocus:
            response = self.client.post('/api/bgr-capture-series', json={'target_folder': folder, 'mode': 'uv_rgb'})
        self.assertEqual(409, response.status_code)
        self.assertEqual('E1505', response.json['code'])
        self.assertFalse(any(command.startswith('G1 ') for command in self.device.command_history))
        self.assertTrue(self.device.command_history)
        self.assertTrue(all(command.endswith(' S0') for command in self.device.command_history))
        autofocus.assert_not_called()
        self.frame_capture.assert_not_called()

    def test_selected_wavelengths_run_in_order_without_an_active_lamp(self):
        settings = bgr_filter_settings()
        settings['filters'].append({
            'id': 'uv255-filter', 'name': '255 nm',
            'wavelength_range': '255 nm', 'color': '#ffffff',
        })
        settings['slots'][4] = 'uv255-filter'
        settings['height_offsets_mm']['uv255-filter'] = {
            'uv255': 0., 'uv310': 0., 'uv365': 0., 'vis': 0.,
        }
        settings_manager.set_settings({'filter_settings': settings})
        height_offset_control.record_reference(10.)
        backend_app.light_controller.off()

        with tempfile.TemporaryDirectory() as folder:
            response = self.client.post('/api/bgr-capture-series', json={
                'target_folder': folder,
                'wavelengths': ['uv255', 'vis'],
            })

        self.assertEqual(200, response.status_code, response.json)
        rows = response.json['saved_images']
        self.assertEqual(
            [('uv255', 'r'), ('uv255', 'g'), ('uv255', 'b'), ('uv255', 'uv'),
             ('vis', 'r'), ('vis', 'g'), ('vis', 'b')],
            [(row['wavelength'], row['suffix']) for row in rows],
        )
        self.assertTrue(all(f"_{row['wavelength']}_{row['suffix']}.jpg" in row['path'] for row in rows))
        self.assertEqual(
            [('uv255', 'dimmed')] * 4 + [('vis', None)] * 3,
            [(call.args[2], call.args[3]) for call in self.frame_capture.call_args_list],
        )

    def test_selected_wavelength_validation_happens_before_motion(self):
        backend_app.light_controller.off()
        self.device.command_history.clear()
        with tempfile.TemporaryDirectory() as folder:
            response = self.client.post('/api/bgr-capture-series', json={
                'target_folder': folder,
                'wavelengths': ['uv255', 'uv255'],
            })
        self.assertEqual(400, response.status_code)
        self.assertEqual('E1507', response.json['code'])
        self.assertFalse(any(command.startswith('G1 ') for command in self.device.command_history))
        self.frame_capture.assert_not_called()

    def test_both_modes_use_start_lamp_and_correct_uv_filter_after_autofocus(self):
        settings = bgr_filter_settings()
        for index, name in ((4, '255 nm'), (5, '365nm')):
            key = f'uv{index}'
            settings['filters'].append({'id': key, 'name': name, 'wavelength_range': name, 'color': '#ffffff'})
            settings['slots'][index] = key
            settings['height_offsets_mm'][key] = {'uv255': 1., 'uv310': 1., 'uv365': 1., 'vis': 0.}
        settings_manager.set_settings({'filter_settings': settings})
        for channel in ('vis', 'uv255', 'uv310', 'uv365'):
            for mode in ('rgb', 'uv_rgb'):
                with self.subTest(channel=channel, mode=mode), tempfile.TemporaryDirectory() as folder:
                    height_offset_control.invalidate_reference()
                    brightness = None if channel == 'vis' else 'full'
                    backend_app.light_controller.activate(channel, brightness)
                    def autofocus_result(*_args, **_kwargs):
                        # Simulate a different configured autofocus illumination.
                        backend_app.light_controller.activate('vis')
                        return successful_autofocus()
                    with patch.object(backend_app, '_run_configured_manual_autofocus', side_effect=autofocus_result) as autofocus:
                        response = self.client.post('/api/bgr-capture-series', json={'target_folder': folder, 'mode': mode})
                    self.assertEqual(200, response.status_code, response.json)
                    has_uv = mode == 'uv_rgb' and channel != 'vis'
                    self.assertEqual(['r', 'g', 'b'] + (['uv'] if has_uv else []), [row['suffix'] for row in response.json['saved_images']])
                    self.assertTrue(all(row['wavelength'] == channel for row in response.json['saved_images']))
                    if has_uv:
                        self.assertEqual(5 if channel in ('uv255', 'uv310') else 6, response.json['saved_images'][-1]['filter_position'])
                        self.assertEqual(11., response.json['saved_images'][-1]['height_offset']['target_z'])
                    self.assertEqual(channel, self.frame_capture.call_args.args[2])
                    self.assertEqual(brightness, self.frame_capture.call_args.args[3])
                    self.assertEqual('vis', backend_app.light_controller.status()['active_channel'])
                    self.assertEqual(2, globals.filter_revolver_position)
                    autofocus.assert_called_once()

    def test_transient_uv_grab_failure_is_retried_and_all_required_images_are_saved(self):
        settings = bgr_filter_settings()
        settings['filters'].append({
            'id': 'uv255-filter', 'name': '255 nm',
            'wavelength_range': '255 nm', 'color': '#ffffff',
        })
        settings['slots'][4] = 'uv255-filter'
        settings['height_offsets_mm']['uv255-filter'] = {
            'uv255': 0., 'uv310': 0., 'uv365': 0., 'vis': 0.,
        }
        settings_manager.set_settings({'filter_settings': settings})
        height_offset_control.record_reference(10.)
        backend_app.light_controller.activate('uv255', 'full')
        failed_once = False

        def capture_with_one_transient_failure(*_args):
            nonlocal failed_once
            if globals.filter_revolver_position == 5 and not failed_once:
                failed_once = True
                raise RuntimeError('temporary camera timeout')
            return np.zeros((2, 2, 3), dtype=np.uint8)

        self.frame_capture.side_effect = capture_with_one_transient_failure
        with tempfile.TemporaryDirectory() as folder:
            response = self.client.post('/api/bgr-capture-series', json={
                'target_folder': folder, 'mode': 'uv_rgb', 'capture_id': 'retry-uv-test',
            })

        self.assertEqual(200, response.status_code, response.json)
        self.assertTrue(failed_once)
        self.assertEqual(['r', 'g', 'b', 'uv'], [row['suffix'] for row in response.json['saved_images']])
        self.assertEqual(5, self.frame_capture.call_count)
        self.assertEqual('vis', backend_app.light_controller.status()['active_channel'])
        self.assertEqual(2, globals.filter_revolver_position)
        self.assertEqual(0, globals.preview_grab_suppression_count)

    def test_invalid_request_switches_active_lamp_off(self):
        backend_app.light_controller.activate('vis')

        response = self.client.post('/api/bgr-capture-series', json={
            'target_folder': 'missing-capture-folder',
        })

        self.assertEqual(400, response.status_code)
        self.assertIsNone(backend_app.light_controller.status()['active_channel'])

    def test_anchor_skips_autofocus_and_missing_uv_is_rejected_before_motion(self):
        height_offset_control.record_combination_reference(10., 0., source='anchor')
        with tempfile.TemporaryDirectory() as folder, patch.object(backend_app, '_run_configured_manual_autofocus') as autofocus:
            response = self.client.post('/api/bgr-capture-series', json={'target_folder': folder, 'mode': 'uv_rgb'})
            self.assertEqual(200, response.status_code)  # VIS ignores the UV filter requirement.
            autofocus.assert_not_called()
            backend_app.light_controller.activate('uv255', 'dimmed')
            self.frame_capture.reset_mock()
            self.device.command_history.clear()
            response = self.client.post('/api/bgr-capture-series', json={'target_folder': folder, 'mode': 'uv_rgb'})
            self.assertEqual(400, response.status_code)
            self.assertEqual('E1502', response.json['code'])
            self.assertFalse(any(command.startswith('G1 ') for command in self.device.command_history))
            self.frame_capture.assert_not_called()
            self.assertIsNone(backend_app.light_controller.status()['active_channel'])

    def test_illumination_failure_does_not_save_dark_frame_and_switches_lamps_off(self):
        from light_control import CaptureIlluminationError
        height_offset_control.record_reference(10.)
        self.frame_capture.side_effect = CaptureIlluminationError('Safety window expired')
        with tempfile.TemporaryDirectory() as folder, patch.object(backend_app, '_capture_and_save_image') as save:
            response = self.client.post('/api/bgr-capture-series', json={'target_folder': folder})
        self.assertEqual(422, response.status_code)
        self.assertEqual('E1506', response.json['code'])
        save.assert_not_called()
        self.assertIsNone(backend_app.light_controller.status()['active_channel'])
        self.assertFalse(globals.motion_busy)


if __name__ == '__main__':
    unittest.main()
