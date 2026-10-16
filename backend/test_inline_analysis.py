"""Hardware-free asynchronous capture/analysis lifecycle and HTTP contracts."""

import json
import io
from contextlib import ExitStack
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import cv2
from flask import Flask
import numpy as np
from PIL import Image

from error_codes import ErrorCode
from inline_analysis import AnalysisCaptureCancelled, InlineAnalysisService
from inline_analysis_api import create_inline_analysis_blueprint
from pipeline_types import PipelineResult, StepError
from pipeline_steps import STEP_DEFINITIONS


def recipe():
    return {'schema_version': 1, 'name': 'Inline test', 'steps': [
        {'instance_id': 'load', 'step_def_id': 'load_image',
         'param_values': {'source': 'this-editor-path-must-never-be-loaded'}, 'order': 0},
    ], 'connections': []}


class InlineAnalysisTests(unittest.TestCase):
    def test_cumulative_gallery_preserves_tablet_identity_and_selected_thumbnail(self):
        service = self.service()
        session_id = self.create(service, input_scope='cumulative', mode='measurement_end')
        self.capture(service, session_id, tablet=31, count=1)
        service.begin_capture(session_id, 'request-32')
        service.capture_finished(session_id, 'request-32', 32, [{'path': self.paths[1]}])
        service.finish(session_id)
        snapshot = self.settled(service, session_id)
        outputs = snapshot['results'][0]['output_images']
        self.assertEqual([(0, '31'), (1, '32')], [(row['image_index'], row['tablet_id']) for row in outputs])
        app = Flask('thumbnail-contract')
        app.register_blueprint(create_inline_analysis_blueprint(service))
        client = app.test_client()
        url = f'/api/inline-analysis/sessions/{session_id}/results/cumulative/thumbnail'
        response = client.get(url + '?image_index=1')
        self.assertEqual(200, response.status_code)
        self.assertEqual('image/jpeg', response.mimetype)
        with Image.open(io.BytesIO(response.data)) as image:
            self.assertAlmostEqual(70, np.asarray(image).mean(), delta=2)
        self.assertEqual(404, client.get(url + '?image_index=5').status_code)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.services, self.gates = [], []
        self.paths = []
        for index, channel in enumerate(('red', 'green', 'blue')):
            path = self.root / f'capture-{index}.jpg'
            image = Image.fromarray(np.full((12, 16, 3), 40 + index * 30, np.uint8))
            exif = image.getexif()
            exif[0x010E] = json.dumps({'channel': channel, 'wavelength': 'uv255', 'tray_position': 'A4'})
            image.save(path, exif=exif)
            self.paths.append(str(path))

    def tearDown(self):
        for gate in self.gates:
            gate.set()
        for service in self.services:
            service.close()
            if service._worker:
                service._worker.join(timeout=5)
        self.directory.cleanup()

    def service(self, **kwargs):
        service = InlineAnalysisService(**kwargs)
        self.services.append(service)
        return service

    def create(self, service, **kwargs):
        payload = dict(pipeline=recipe(), measurement_name='Temporary test',
                       measurement_folder=str(self.root), **kwargs)
        return service.create(payload)['session_id']

    def capture(self, service, session_id, tablet=1, count=1):
        request_id = f'request-{tablet}'
        service.begin_capture(session_id, request_id)
        service.capture_finished(session_id, request_id, tablet,
                                 [{'path': path} for path in self.paths[:count]])

    def settled(self, service, session_id):
        with service._condition:
            ready = service._condition.wait_for(
                lambda: not any(result['status'] in ('queued', 'running')
                                for result in service.snapshot(session_id)['results']), timeout=8)
            self.assertTrue(ready, 'analysis worker did not finish')
        return service.snapshot(session_id)

    @staticmethod
    def fake_result(doc, *, live_inputs):
        return PipelineResult(success=True, executed_up_to=len(doc.steps) - 1, data={
            'images': [cv2.imread(item['path']) for item in live_inputs],
            'paths': [item['path'] for item in live_inputs],
            'image_metadata': [dict(item['metadata'], tablet_id=item['tablet_id']) for item in live_inputs],
            'results': {'summary': {'input_count': len(live_inputs)}},
        })

    def test_complete_tablet_waits_for_next_move_and_last_flushes_at_finish(self):
        service = self.service()
        session_id = self.create(service)
        self.capture(service, session_id, count=3)
        self.assertEqual([], service.snapshot(session_id)['results'])
        service.begin_capture(session_id, 'second-tablet')
        service.release_after_move(session_id)
        snapshot = self.settled(service, session_id)
        self.assertEqual('completed', snapshot['results'][0]['status'])
        preview = service.preview(session_id, 'tablet-1', 2)
        self.assertEqual(3, preview['image_count'])
        self.assertEqual('blue', preview['side_outputs']['image_metadata'][2]['channel'])
        service.capture_finished(session_id, 'second-tablet', 2, [{'path': self.paths[0]}])
        self.assertEqual(1, len(service.snapshot(session_id)['results']))
        service.finish(session_id)
        snapshot = self.settled(service, session_id)
        self.assertEqual('completed', snapshot['state'])
        self.assertEqual(2, len(snapshot['results']))
        self.assertTrue((Path(snapshot['artifact_folder']) / 'recipe.json').is_file())

    def test_real_crop_pipeline_and_parameter_rerun_preserve_source_and_recipe(self):
        service = self.service()
        document = recipe()
        crop_params = {param.name: param.default for param in STEP_DEFINITIONS['mask_rect_roi'].params}
        crop_params.update(roi_type='rect', roi_x=2, roi_y=2, roi_width=6, roi_height=5, output_mode='crop')
        document['steps'].append({'instance_id': 'crop', 'step_def_id': 'mask_rect_roi', 'order': 1,
                                 'param_values': crop_params})
        session_id = service.create(dict(pipeline=document, measurement_name='Crop',
                                         measurement_folder=str(self.root)))['session_id']
        self.capture(service, session_id)
        service.finish(session_id)
        self.settled(service, session_id)
        first = service.preview(session_id, 'tablet-1')
        self.assertTrue(first['success'], first)
        self.assertEqual((6, 5), (first['image_width'], first['image_height']))
        document['steps'][-1]['param_values']['roi_width'] = 4
        service.configure(session_id, {'pipeline': document})
        self.settled(service, session_id)
        self.assertEqual(4, service.preview(session_id, 'tablet-1')['image_width'])
        with Image.open(self.paths[0]) as source:
            self.assertEqual((16, 12), source.size)
        self.assertEqual('this-editor-path-must-never-be-loaded',
                         service.snapshot(session_id)['pipeline']['steps'][0]['param_values']['source'])

    def test_cumulative_end_waits_then_uses_all_complete_tablets(self):
        calls = []
        def execute(doc, *, live_inputs):
            calls.append(live_inputs)
            return self.fake_result(doc, live_inputs=live_inputs)
        service = self.service(executor=execute)
        session_id = self.create(service, input_scope='cumulative', mode='measurement_end')
        self.capture(service, session_id, tablet=1, count=3)
        service.release_after_move(session_id)
        self.capture(service, session_id, tablet=2, count=3)
        service.release_after_move(session_id)
        self.assertEqual([], calls)
        service.finish(session_id)
        self.settled(service, session_id)
        self.assertEqual(1, len(calls))
        self.assertEqual(['1'] * 3 + ['2'] * 3, [item['tablet_id'] for item in calls[0]])
        self.assertEqual(6, service.preview(session_id, 'cumulative', 5)['image_count'])

    def test_cumulative_each_tablet_revision_changes_without_recipe_edit(self):
        service = self.service(executor=self.fake_result)
        session_id = self.create(service, input_scope='cumulative')
        self.capture(service, session_id)
        service.release_after_move(session_id)
        first = self.settled(service, session_id)['results'][0]
        self.capture(service, session_id, tablet=2)
        service.finish(session_id)
        second = self.settled(service, session_id)['results'][0]
        self.assertGreater(second['revision'], first['revision'])
        self.assertEqual(first['pipeline_revision'], second['pipeline_revision'])
        self.assertEqual(2, second['input_count'])

    def test_cancel_during_capture_defers_flush_and_discards_incomplete_tablet(self):
        service = self.service(executor=self.fake_result)
        session_id = self.create(service)
        self.capture(service, session_id)
        service.begin_capture(session_id, 'incomplete')
        result = service.finish(session_id, cancelled=True)
        self.assertEqual('finishing', result['state'])
        self.assertEqual([], result['results'])
        service.capture_finished(session_id, 'incomplete', 2, inputs=None)
        snapshot = self.settled(service, session_id)
        self.assertEqual('cancelled', snapshot['state'])
        self.assertEqual(['1'], [item['tablet_id'] for item in snapshot['tablets']])
        with self.assertRaises(AnalysisCaptureCancelled):
            service.begin_capture(session_id, 'late-request')

    def test_failure_does_not_block_next_tablet(self):
        def execute(doc, *, live_inputs):
            if live_inputs[0]['tablet_id'] == '1':
                return PipelineResult(success=False, errors=[StepError(0, 'load_image', 'E2184', 'test failure')])
            return self.fake_result(doc, live_inputs=live_inputs)
        service = self.service(executor=execute)
        session_id = self.create(service)
        self.capture(service, session_id)
        service.release_after_move(session_id)
        self.settled(service, session_id)
        self.capture(service, session_id, tablet=2)
        service.finish(session_id)
        snapshot = self.settled(service, session_id)
        self.assertEqual(['failed', 'completed'], [result['status'] for result in snapshot['results']])
        self.assertEqual('test failure', snapshot['errors'][0]['error'])
        self.assertEqual('completed', snapshot['state'])

    def test_configuration_supersedes_running_results(self):
        entered, gate = threading.Event(), threading.Event()
        self.gates.append(gate)
        calls = []
        def execute(doc, *, live_inputs):
            calls.append(doc.name)
            if len(calls) == 1:
                entered.set()
                self.assertTrue(gate.wait(5))
            return self.fake_result(doc, live_inputs=live_inputs)
        service = self.service(executor=execute)
        session_id = self.create(service)
        self.capture(service, session_id)
        service.finish(session_id)
        self.assertTrue(entered.wait(5))
        edited = recipe()
        edited['name'] = 'New revision'
        response = service.configure(session_id, {'pipeline': edited})
        self.assertEqual('queued', response['results'][0]['status'])
        gate.set()
        snapshot = self.settled(service, session_id)
        self.assertEqual(['Inline test', 'New revision'], calls)
        self.assertEqual(2, snapshot['results'][0]['pipeline_revision'])
        self.assertEqual([], snapshot['errors'])

    def test_bounded_queue_reports_error_without_losing_capture(self):
        entered, gate = threading.Event(), threading.Event()
        self.gates.append(gate)
        def execute(doc, *, live_inputs):
            entered.set()
            self.assertTrue(gate.wait(5))
            return self.fake_result(doc, live_inputs=live_inputs)
        service = self.service(executor=execute, max_queue=1)
        session_id = self.create(service)
        self.capture(service, session_id)
        service.release_after_move(session_id)
        self.assertTrue(entered.wait(5))
        self.capture(service, session_id, tablet=2)
        service.release_after_move(session_id)
        self.capture(service, session_id, tablet=3)
        service.finish(session_id)
        snapshot = service.snapshot(session_id)
        self.assertEqual(ErrorCode.INLINE_ANALYSIS_CAPACITY, snapshot['errors'][0]['code'])
        self.assertEqual(3, len(snapshot['tablets']))
        gate.set()
        self.settled(service, session_id)

    def test_api_validation_pending_and_paged_preview(self):
        service = self.service(executor=self.fake_result)
        app = Flask(__name__)
        app.register_blueprint(create_inline_analysis_blueprint(service))
        client = app.test_client()
        self.assertEqual(400, client.post('/api/inline-analysis/sessions', json=[]).status_code)
        self.assertEqual(404, client.get('/api/inline-analysis/sessions/missing').status_code)
        response = client.post('/api/inline-analysis/sessions', json=dict(
            pipeline=recipe(), measurement_name='HTTP', measurement_folder=str(self.root)))
        self.assertEqual(201, response.status_code)
        session_id = response.json['session_id']
        self.capture(service, session_id, count=3)
        self.assertEqual(202, client.post(f'/api/inline-analysis/sessions/{session_id}/finish', json={}).status_code)
        self.settled(service, session_id)
        preview = client.get(f'/api/inline-analysis/sessions/{session_id}/results/tablet-1?image_index=2')
        self.assertEqual(200, preview.status_code)
        self.assertTrue(preview.json['success'])
        self.assertEqual('capture-2.jpg', preview.json['source_name'])
        self.assertEqual(400, client.post(f'/api/inline-analysis/sessions/{session_id}/configure',
                                         json={'mode': 'invalid'}).status_code)

    def test_late_cancelled_http_step_never_resets_abort_or_touches_hardware(self):
        import app as backend_app
        import globals
        service = self.service()
        session_id = self.create(service)
        service.finish(session_id, cancelled=True)
        with patch.object(backend_app, 'inline_analysis_service', service), \
                patch.object(backend_app, '_check_devices_connected') as devices, \
                patch.object(globals, 'autofocus_abort', True):
            response = backend_app.app.test_client().post('/api/auto_measurement/step', json={
                'inline_analysis_session_id': session_id, 'tablet_index': 30, 'request_id': 'late-request'})
            self.assertEqual(200, response.status_code)
            self.assertTrue(response.json['cancelled'])
            self.assertTrue(globals.autofocus_abort)
            devices.assert_not_called()

    def test_capture_route_stages_only_complete_original_images_after_confirmed_move(self):
        import app as backend_app
        service = self.service(executor=self.fake_result)
        session_id = self.create(service)
        self.capture(service, session_id, tablet=1)
        row = dict(wavelength='uv255', brightness='dimmed', filter_position=1,
                   exposure_time=1000, gain=0)
        payload = dict(inline_analysis_session_id=session_id, tablet_index=2, request_id='seam-complete',
                       measurement_name='Mocked', measurement_folder=str(self.root),
                       x=1, y=2, autofocus=False, is_first_tablet=False, capture_plan=[row])
        def capture(row, folder, name, tablet, **kwargs):
            self.assertTrue(service.snapshot(session_id)['results'], 'prior tablet not released after motion')
            kwargs['on_image_saved'](self.paths[0], False)
            kwargs['on_image_saved']('generated-mask.jpg', True)
            return [self.paths[0], 'generated-mask.jpg']
        replacements = {
            'inline_analysis_service': service, 'validate_capture_plan': lambda rows: rows,
            '_validate_capture_plan_camera_values': lambda rows: None,
            '_check_devices_connected': lambda: (object(), object(), None, None),
            '_move_toolhead_absolute_impl': lambda **kwargs: ({}, 200),
            '_wait_for_motion_complete': lambda *args, **kwargs: True,
            '_check_tablet_presence_enabled': lambda settings: False,
            '_measurement_capture_rows': lambda rows, *args: list(enumerate(rows)),
            '_select_measurement_filter_position': lambda *args: None,
            '_current_filter_settings': lambda: {}, '_capture_capture_plan_row': capture,
            '_turn_off_all_lights': lambda: None,
        }
        with ExitStack() as stack:
            for key, value in replacements.items():
                stack.enter_context(patch.object(backend_app, key, value))
            stack.enter_context(patch.object(backend_app.time, 'sleep'))
            stack.enter_context(patch.object(backend_app.porthandler, 'refresh_door_status', return_value={
                'known': True, 'front_open': False, 'top_open': False,
                'uv_blocked': False, 'trip_count': 0,
            }))
            stack.enter_context(patch.object(backend_app.light_controller, 'off'))
            stack.enter_context(patch.object(backend_app.height_offset_control, 'apply_active_combination',
                                            return_value={'applied': True}))
            client = backend_app.app.test_client()
            response = client.post('/api/auto_measurement/step', json=payload)
            self.assertEqual(200, response.status_code, response.json)
            snapshot = service.snapshot(session_id)
            self.assertEqual(0, snapshot['active_capture_count'])
            self.assertEqual(1, snapshot['tablets'][1]['image_count'])
            self.assertEqual('waiting', snapshot['tablets'][1]['status'])
            # A failed second row must not stage the first row as a complete tablet.
            payload.update(tablet_index=3, request_id='seam-partial', capture_plan=[row, row])
            count = 0
            def partial(*args, **kwargs):
                nonlocal count
                count += 1
                if count == 2:
                    raise ValueError('test capture failure')
                return capture(*args, **kwargs)
            with patch.object(backend_app, '_capture_capture_plan_row', partial):
                response = client.post('/api/auto_measurement/step', json=payload)
            self.assertEqual(500, response.status_code)
            self.assertEqual(2, len(service.snapshot(session_id)['tablets']))
            self.assertEqual(0, service.snapshot(session_id)['active_capture_count'])
        service.finish(session_id)
        self.settled(service, session_id)


if __name__ == '__main__':
    unittest.main()
