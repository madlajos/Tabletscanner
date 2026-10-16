"""Asynchronous recipe application to complete, released tablet capture sets.

This service owns paths and immutable recipe snapshots, never hardware handles.
The capture route stages a tablet only after its entire image cycle succeeds.
The next confirmed XY move releases it, or finish releases the last tablet once
the active capture request has cleaned up. One bounded worker performs analysis.
"""

from __future__ import annotations

import base64
from collections import OrderedDict
from copy import deepcopy
import json
import logging
from pathlib import Path
import threading
import time
import uuid

import cv2
from PIL import Image

from error_codes import ErrorCode
from inline_analysis_artifacts import load_data, save_data, write_json
import pipeline_engine
from pipeline_types import PipelineDocument

logger = logging.getLogger(__name__)


class AnalysisError(ValueError):
    def __init__(self, message, status=400, code=None):
        super().__init__(message)
        self.status = status
        self.code = code or ErrorCode.INLINE_ANALYSIS_INVALID


class AnalysisCaptureCancelled(AnalysisError):
    """A late measurement request belongs to an explicitly cancelled run."""


class InlineAnalysisService:
    MODES = ('each_tablet', 'measurement_end')
    SCOPES = ('tablet', 'cumulative')

    def __init__(self, *, executor=None, max_sessions=8, max_queue=256,
                 max_input_bytes=768 * 1024 * 1024):
        self._executor = executor or pipeline_engine.execute_pipeline
        self._max_sessions = max_sessions
        self._max_queue = max_queue
        self._max_input_bytes = max_input_bytes
        self._condition = threading.Condition(threading.RLock())
        self._sessions = OrderedDict()
        self._queue = OrderedDict()
        self._worker = None
        self._closed = False

    def _get(self, session_id):
        session = self._sessions.get(session_id)
        if session is None:
            raise AnalysisError('Az elemzési munkamenet nem található.', 404,
                                ErrorCode.INLINE_ANALYSIS_NOT_FOUND)
        return session

    @staticmethod
    def _document(payload):
        if not isinstance(payload, dict) or not isinstance(payload.get('steps'), list):
            raise AnalysisError('Érvénytelen képelemző recept.')
        try:
            doc = PipelineDocument.from_dict(deepcopy(payload))
        except (TypeError, ValueError, KeyError, AttributeError) as error:
            raise AnalysisError('Érvénytelen képelemző recept.') from error
        if doc.schema_version != 1 or not doc.steps or len(doc.steps) > 256:
            raise AnalysisError('A receptnek 1–256 lépést kell tartalmaznia (1. séma).')
        return doc

    def create(self, payload):
        doc = self._document(payload.get('pipeline'))
        mode = payload.get('mode', 'each_tablet')
        infer = getattr(pipeline_engine, 'classify_inline_input_scope', lambda doc: 'tablet')
        scope = payload.get('input_scope') or infer(doc)
        if mode not in self.MODES or scope not in self.SCOPES:
            raise AnalysisError('Érvénytelen elemzési mód.')
        name, folder = payload.get('measurement_name'), payload.get('measurement_folder')
        if not isinstance(name, str) or not name.strip() or len(name) > 200:
            raise AnalysisError('Adja meg a mérés nevét (legfeljebb 200 karakter).')
        if not isinstance(folder, str) or not folder.strip():
            raise AnalysisError('Adja meg a mérés mentési mappáját.')
        with self._condition:
            if self._closed:
                raise AnalysisError('Az elemző szolgáltatás leállt.', 503)
            if len(self._sessions) >= self._max_sessions:
                terminal = next((key for key, value in self._sessions.items()
                                 if value['state'] in ('completed', 'cancelled')
                                 and not value['active']), None)
                if terminal is None:
                    raise AnalysisError('Túl sok aktív elemzési munkamenet.', 409)
                del self._sessions[terminal]
            session_id = uuid.uuid4().hex
            directory = Path(folder).resolve() / 'elemzes' / session_id
            directory.mkdir(parents=True, exist_ok=False)
            session = {
                'session_id': session_id, 'recipe_name': doc.name,
                'measurement_name': name.strip(), 'measurement_folder': str(Path(folder).resolve()),
                'mode': mode, 'input_scope': scope, 'state': 'collecting',
                'pipeline': doc.to_dict(), 'pipeline_revision': 1, 'revision': 1,
                'tablets': OrderedDict(), 'results': OrderedDict(), 'errors': [],
                'active': set(), 'finish_requested': False, 'cancelled': False,
                'directory': directory, 'created_at': time.time(),
            }
            write_json(directory / 'recipe.json', session['pipeline'])
            self._sessions[session_id] = session
            self._persist(session)
            return self._snapshot(session)

    @staticmethod
    def _snapshot(session):
        public = {key: deepcopy(session[key]) for key in (
            'session_id', 'recipe_name', 'measurement_name', 'measurement_folder',
            'mode', 'input_scope', 'state', 'pipeline', 'pipeline_revision', 'revision', 'errors')}
        public['tablets'] = [{key: deepcopy(value) for key, value in tablet.items()
                              if key not in ('inputs', 'released')}
                             for tablet in session['tablets'].values()]
        public['results'] = [{key: deepcopy(value) for key, value in result.items()
                             if key not in ('directory', 'token', 'inputs', 'pipeline')}
                            for result in session['results'].values()]
        public['active_capture_count'] = len(session['active'])
        public['artifact_folder'] = str(session['directory'])
        return public

    def snapshot(self, session_id):
        with self._condition:
            return self._snapshot(self._get(session_id))

    def _persist(self, session):
        try:
            write_json(session['directory'] / 'session.json', self._snapshot(session))
        except OSError:
            logger.exception('Could not persist inline analysis status')

    def begin_capture(self, session_id, request_id):
        with self._condition:
            session = self._get(session_id)
            if session['cancelled']:
                raise AnalysisCaptureCancelled('A mérést a felhasználó leállította.', 409)
            if session['finish_requested']:
                raise AnalysisError('A mérés elemzési bemenete már lezárult.', 409)
            session['active'].add(request_id)
            session['revision'] += 1

    def capture_finished(self, session_id, request_id, tablet_index, inputs=None):
        """Only the caller marking the WHOLE tablet successful supplies inputs."""
        with self._condition:
            session = self._get(session_id)
            session['active'].discard(request_id)
            tablet_id = str(tablet_index)
            if inputs and tablet_id not in session['tablets']:
                if len(session['tablets']) >= 1000:
                    self._error(session, 'Elérte az elemzés 1000 tablettás korlátját.', tablet_id=tablet_id)
                else:
                    owned_inputs = [dict(deepcopy(item), tablet_id=tablet_id) for item in inputs]
                    session['tablets'][tablet_id] = {
                        'tablet_id': tablet_id, 'tablet_index': tablet_index,
                        'label': next((item.get('metadata', {}).get('tray_position') for item in owned_inputs
                                       if item.get('metadata', {}).get('tray_position')), f'Tabletta {tablet_index}'),
                        'image_count': len(owned_inputs), 'status': 'waiting',
                        'result_id': None, 'inputs': owned_inputs, 'released': False,
                    }
            session['revision'] += 1
            if session['finish_requested'] and not session['active']:
                self._release(session)
                self._schedule(session)
            self._settle(session)

    def release_after_move(self, session_id):
        with self._condition:
            session = self._get(session_id)
            self._release(session)
            self._schedule(session)

    @staticmethod
    def _release(session):
        for tablet in session['tablets'].values():
            tablet['released'] = True
        session['revision'] += 1

    def finish(self, session_id, cancelled=False):
        if type(cancelled) is not bool:
            raise AnalysisError('A cancelled értékének logikainak kell lennie.')
        with self._condition:
            session = self._get(session_id)
            session['finish_requested'] = True
            session['cancelled'] = session['cancelled'] or cancelled
            session['state'] = 'finishing'
            if not session['active']:
                self._release(session)
                self._schedule(session)
            self._settle(session)
            self._persist(session)
            return self._snapshot(session)

    def configure(self, session_id, payload):
        if not payload or set(payload) - {'pipeline', 'mode', 'input_scope'}:
            raise AnalysisError('Érvénytelen elemzési beállítás.')
        doc = self._document(payload['pipeline']) if 'pipeline' in payload else None
        with self._condition:
            session = self._get(session_id)
            mode, scope = payload.get('mode', session['mode']), payload.get('input_scope', session['input_scope'])
            if mode not in self.MODES or scope not in self.SCOPES:
                raise AnalysisError('Érvénytelen elemzési mód.')
            if doc:
                session['pipeline'] = doc.to_dict()
                session['recipe_name'] = doc.name
            session['mode'], session['input_scope'] = mode, scope
            session['pipeline_revision'] += 1
            for key in list(self._queue):
                if key[0] == session_id:
                    del self._queue[key]
            session['results'].clear()
            for tablet in session['tablets'].values():
                tablet['status'], tablet['result_id'] = 'waiting', None
            self._schedule(session)
            self._settle(session)
            self._persist(session)
            return self._snapshot(session)

    def _schedule(self, session):
        tablets = [tablet for tablet in session['tablets'].values() if tablet['released']]
        if not tablets:
            return
        if session['input_scope'] == 'cumulative':
            if session['mode'] == 'measurement_end' and not session['finish_requested']:
                return
            if session['finish_requested'] and session['active']:
                return
            inputs = [item for tablet in tablets for item in tablet['inputs']]
            self._enqueue(session, 'cumulative', None, 'Összes tabletta', inputs)
        else:
            for tablet in tablets:
                self._enqueue(session, 'tablet-' + tablet['tablet_id'], tablet['tablet_id'],
                              tablet['label'], tablet['inputs'])

    def _enqueue(self, session, result_id, tablet_id, label, inputs):
        key = (session['session_id'], result_id)
        previous = session['results'].get(result_id)
        signature = (session['pipeline_revision'], len(inputs))
        if previous and (previous['pipeline_revision'], previous['input_count']) == signature:
            return
        result = {
            'result_id': result_id, 'kind': 'tablet' if tablet_id is not None else 'cumulative',
            'tablet_id': tablet_id, 'label': label, 'status': 'queued', 'image_count': 0,
            'input_count': len(inputs), 'pipeline_revision': session['pipeline_revision'],
            'revision': session['revision'] + 1,
            'token': uuid.uuid4().hex, 'inputs': deepcopy(inputs), 'pipeline': deepcopy(session['pipeline']),
        }
        session['results'][result_id] = result
        if len(self._queue) >= self._max_queue and key not in self._queue:
            result['status'] = 'failed'
            self._error(session, 'Az elemzési várólista megtelt. A képek megmaradtak; az elemzés újraindítható.',
                        result_id=result_id, tablet_id=tablet_id, code=ErrorCode.INLINE_ANALYSIS_CAPACITY)
        else:
            self._queue[key] = result
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(target=self._work, name='inline-analysis', daemon=True)
                self._worker.start()
            self._condition.notify_all()
        if tablet_id is not None:
            session['tablets'][tablet_id].update(status=result['status'], result_id=result_id)
        else:
            for tablet in session['tablets'].values():
                if tablet['released']:
                    tablet.update(status=result['status'], result_id=result_id)
        session['revision'] += 1

    def _error(self, session, message, *, result_id=None, tablet_id=None, code=None, details=None):
        event = {'id': uuid.uuid4().hex, 'code': code or ErrorCode.INLINE_ANALYSIS_FAILED,
                 'error': message, 'popup': True, 'result_id': result_id, 'tablet_id': tablet_id}
        if details:
            event['errors'] = details
        session['errors'].append(event)
        session['errors'] = session['errors'][-256:]
        session['revision'] += 1

    def report_error(self, session_id, message):
        with self._condition:
            self._error(self._get(session_id), message)

    def _settle(self, session):
        if session['finish_requested']:
            busy = bool(session['active']) or any(
                result['status'] in ('queued', 'running') for result in session['results'].values())
            session['state'] = 'finishing' if busy else ('cancelled' if session['cancelled'] else 'completed')
        session['revision'] += 1
        self._condition.notify_all()

    def _prepare_inputs(self, inputs):
        prepared, total_bytes = [], 0
        for item in inputs:
            path = item.get('path')
            if not isinstance(path, str):
                raise AnalysisError('Hiányzó elemzési képfájl.')
            with Image.open(path) as image:
                total_bytes += image.width * image.height * 3
                description = image.getexif().get(0x010E)
            if total_bytes > self._max_input_bytes:
                raise AnalysisError('A képcsoport túl nagy a biztonságos elemzéshez. Használjon tablettánkénti módot.',
                                    code=ErrorCode.INLINE_ANALYSIS_CAPACITY)
            metadata = {}
            if description:
                try:
                    metadata = json.loads(description)
                except (ValueError, TypeError):
                    pass
            metadata = metadata if isinstance(metadata, dict) else {}
            metadata.update(item.get('metadata') or {})
            prepared.append(dict(item, metadata=metadata))
        return prepared

    def _work(self):
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._closed or bool(self._queue))
                if self._closed:
                    return
                (session_id, result_id), work = self._queue.popitem(last=False)
                session = self._sessions.get(session_id)
                if session is None:
                    continue
                work['status'] = 'running'
                session['revision'] += 1
                self._update_tablets(session, work)
                directory = session['directory'] / work['token']
            failure, thumbnail, image_count, executed_up_to = None, None, 0, -1
            output_images = []
            try:
                inputs = self._prepare_inputs(work['inputs'])
                with self._condition:
                    for item in inputs:
                        label = item['metadata'].get('tray_position')
                        tablet = session['tablets'].get(item['tablet_id'])
                        if isinstance(label, str) and label and tablet is not None:
                            tablet['label'] = label
                            if work['tablet_id'] == item['tablet_id']:
                                work['label'] = label
                result = self._executor(PipelineDocument.from_dict(work['pipeline']), live_inputs=inputs)
                if not result.success:
                    failure = ([error.to_dict() for error in result.errors], None)
                else:
                    data = result.data or {}
                    save_data(directory, data)
                    write_json(directory / 'recipe.json', work['pipeline'])
                    image_count = len(data.get('images', []))
                    metadata = data.get('image_metadata', [])
                    for index in range(image_count):
                        row = metadata[index] if index < len(metadata) and isinstance(metadata[index], dict) else {}
                        output_images.append({
                            'image_index': index,
                            'tablet_id': row.get('tablet_id', work['tablet_id']),
                            **{key: row[key] for key in ('wavelength', 'channel', 'filter_name') if key in row},
                        })
                    executed_up_to = result.executed_up_to
                    if image_count:
                        image = data['images'][0]
                        if image is not None:
                            height, width = image.shape[:2]
                            scale = min(1., 180. / max(height, width))
                            small = cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))))
                            ok, encoded = cv2.imencode('.jpg', small)
                            if ok:
                                thumbnail = base64.b64encode(encoded).decode('ascii')
                    # Drop full image arrays before acquiring the shared lock.
                    del data, result
            except Exception as error:
                logger.exception('Inline analysis job failed')
                failure = (None, error)
            finally:
                # Failed pipelines may also retain full image stacks. Do not
                # keep the last execution alive while the worker waits idle.
                data = result = image = small = encoded = None
            with self._condition:
                session = self._sessions.get(session_id)
                if session is None or session['results'].get(result_id) is not work:
                    continue  # A configuration edit superseded this result.
                if failure:
                    details, error = failure
                    message = (details[0]['message'] if details else str(error)) or 'A képelemzés sikertelen.'
                    work['status'] = 'failed'
                    work['errors'] = details or []
                    self._error(session, message, result_id=result_id, tablet_id=work['tablet_id'],
                                code=getattr(error, 'code', None), details=details)
                else:
                    work.update(status='completed', image_count=image_count, directory=directory,
                                artifact_folder=str(directory), executed_up_to=executed_up_to,
                                output_images=output_images)
                    if thumbnail:
                        work['thumbnail_base64'] = thumbnail
                self._update_tablets(session, work)
                self._settle(session)
                self._persist(session)

    @staticmethod
    def _update_tablets(session, work):
        for tablet in session['tablets'].values():
            if tablet['result_id'] == work['result_id']:
                tablet['status'] = work['status']

    def preview(self, session_id, result_id, image_index=0):
        with self._condition:
            session = self._get(session_id)
            result = session['results'].get(result_id)
            if result is None:
                raise AnalysisError('Az elemzési eredmény nem található.', 404, ErrorCode.INLINE_ANALYSIS_NOT_FOUND)
            if result['status'] == 'failed':
                return {'success': False, 'errors': deepcopy(result.get('errors', [])), 'executed_up_to': -1}
            if result['status'] != 'completed':
                raise AnalysisError('Az elemzés még folyamatban van.', 409)
            directory, executed_up_to = result['directory'], result['executed_up_to']
            pipeline = deepcopy(result['pipeline'])
        data = load_data(directory)
        count = len(data.get('images', []))
        selected = max(0, min(image_index, max(0, count - 1)))
        side = pipeline_engine.extract_side_outputs(data, preview_image_index=selected)
        response = {'success': True, 'executed_up_to': executed_up_to, 'side_outputs': side, 'image_count': count}
        encoded = pipeline_engine._encode_preview_image_from_data(data, selected)
        final_step = pipeline['steps'][executed_up_to]['step_def_id'] if executed_up_to >= 0 else ''
        comparison_key = {'manual_image_alignment': 'manual_alignment_previews',
                          'resize_to_reference': 'reference_resize_previews'}.get(final_step)
        comparisons = data.get('results', {}).get(comparison_key, []) if comparison_key else []
        if comparisons:
            encoded = pipeline_engine._encode_preview_image_from_data(
                {'images': comparisons, 'paths': data.get('paths', [])}, selected) or encoded
        if final_step == 'reference_sequence':
            encoded = pipeline_engine._encode_reference_sequence_preview_from_data(data, selected) or encoded
        if encoded:
            response.update(encoded)
        if final_step == 'detect_circles':
            overlays = side.get('circle_overlay_base64') or []
            if overlays:
                response['image_base64'] = overlays[min(selected, len(overlays) - 1)]
                response['is_grayscale'] = False
        metadata = data.get('image_metadata', [])
        if metadata:
            side['image_metadata'] = metadata
        return response

    def thumbnail(self, session_id, result_id, image_index=0):
        """Read a selected result image without rerunning analysis or loading its stack."""
        with self._condition:
            result = self._get(session_id)['results'].get(result_id)
            if result is None:
                raise AnalysisError('Az elemzési eredmény nem található.', 404, ErrorCode.INLINE_ANALYSIS_NOT_FOUND)
            if result['status'] != 'completed':
                raise AnalysisError('Az elemzés még nem készült el.', 409)
            directory = result['directory']
        data = load_data(directory)
        images = data.get('images', [])
        if not 0 <= image_index < len(images):
            raise AnalysisError('Érvénytelen képsorszám.', 404)
        image = images[image_index]
        height, width = image.shape[:2]
        scale = min(1., 180. / max(height, width))
        small = cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))))
        ok, encoded = cv2.imencode('.jpg', small)
        if not ok:
            raise AnalysisError('A bélyegkép nem készíthető el.', 500)
        return encoded.tobytes()

    def close(self):
        """Cancel queued CPU work; an active bounded engine call finishes itself."""
        with self._condition:
            self._closed = True
            self._queue.clear()
            self._condition.notify_all()


inline_analysis_service = InlineAnalysisService()
