"""Short-lived HTTP commands and polling for the inline analysis worker."""

from functools import wraps
import logging

from flask import Blueprint, Response, jsonify, request

from error_codes import ErrorCode
from inline_analysis import AnalysisError

logger = logging.getLogger(__name__)


def create_inline_analysis_blueprint(service):
    blueprint = Blueprint('inline_analysis', __name__, url_prefix='/api/inline-analysis')

    def contract(handler):
        @wraps(handler)
        def guarded(*args, **kwargs):
            try:
                return handler(*args, **kwargs)
            except AnalysisError as error:
                return jsonify(error=str(error), code=error.code, popup=True), error.status
            except (OSError, ValueError, TypeError):
                logger.exception('Inline analysis request failed')
                return jsonify(error='Az elemzési adat nem olvasható vagy menthető.',
                               code=ErrorCode.INLINE_ANALYSIS_FAILED, popup=True), 500
        return guarded

    def payload():
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            raise AnalysisError('Hiányzó JSON objektum.')
        return body

    @blueprint.post('/sessions')
    @contract
    def create():
        return jsonify(service.create(payload())), 201

    @blueprint.get('/sessions/<session_id>')
    @contract
    def status(session_id):
        return jsonify(service.snapshot(session_id))

    @blueprint.post('/sessions/<session_id>/finish')
    @contract
    def finish(session_id):
        return jsonify(service.finish(session_id, payload().get('cancelled', False))), 202

    @blueprint.post('/sessions/<session_id>/configure')
    @contract
    def configure(session_id):
        return jsonify(service.configure(session_id, payload())), 202

    @blueprint.get('/sessions/<session_id>/results/<result_id>')
    @contract
    def preview(session_id, result_id):
        try:
            image_index = int(request.args.get('image_index', 0))
        except ValueError as error:
            raise AnalysisError('Érvénytelen képsorszám.') from error
        return jsonify(service.preview(session_id, result_id, image_index))

    @blueprint.get('/sessions/<session_id>/results/<result_id>/thumbnail')
    @contract
    def thumbnail(session_id, result_id):
        try:
            image_index = int(request.args.get('image_index', 0))
        except ValueError as error:
            raise AnalysisError('Érvénytelen képsorszám.') from error
        return Response(service.thumbnail(session_id, result_id, image_index), mimetype='image/jpeg',
                        headers={'Cache-Control': 'no-cache'})

    return blueprint
