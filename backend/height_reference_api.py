"""Explicit operator anchoring of the current physical filter/light Z position."""

from functools import wraps

from flask import Blueprint, jsonify, request

import globals
import height_offset_control
import motioncontrols
import porthandler
from error_codes import ErrorCode


def guard_manual_motion(handler):
    """Keep manual motion and anchor/selection changes in one serial operation."""
    @wraps(handler)
    def guarded(*args, **kwargs):
        with porthandler.motion_lock:
            if globals.motion_busy:
                return jsonify(error='Motion platform is busy.', code=ErrorCode.GENERIC, popup=True), 409
            globals.motion_busy = True
            try:
                return handler(*args, **kwargs)
            finally:
                globals.motion_busy = False
    return guarded


def guard_capture_operation(handler):
    """Reserve device state without holding the serial lock across camera exposure.

    The lamp timeout monitor must remain able to send OFF while capture is busy.
    """
    @wraps(handler)
    def guarded(*args, **kwargs):
        with porthandler.motion_lock:
            if globals.motion_busy:
                return jsonify(error='Motion platform is busy.', code=ErrorCode.GENERIC, popup=True), 409
            globals.motion_busy = True
        try:
            return handler(*args, **kwargs)
        finally:
            with porthandler.motion_lock:
                globals.motion_busy = False
    return guarded


def create_height_reference_blueprint(light_controller, filter_settings_getter, disconnect_handler):
    blueprint = Blueprint('height_reference', __name__)

    def failure(message, status):
        return jsonify(error=message, code=ErrorCode.MOTION_HEIGHT_REFERENCE_FAILED, popup=True), status

    @blueprint.route('/api/height-offset/reference', methods=['GET', 'POST'])
    def reference():
        if request.method == 'GET':
            device = porthandler.motion_platform or globals.motion_platform
            if not device or not getattr(device, 'is_open', False):
                height_offset_control.invalidate_reference()
            return jsonify(height_offset_control.status())
        data = request.get_json(silent=True)
        if not isinstance(data, dict) or set(data) != {'enabled'} or type(data['enabled']) is not bool:
            return failure('Expected a boolean enabled field.', 400)

        # Claim the same operation lock as filter/light changes. Hold it through
        # the bounded position query so no other selection can become the anchor.
        with porthandler.motion_lock:
            if globals.motion_busy:
                return failure('Motion platform is busy.', 409)
            if not data['enabled']:
                height_offset_control.invalidate_reference()
                return jsonify(height_offset_control.status())
            device = porthandler.motion_platform or globals.motion_platform
            if not device or not getattr(device, 'is_open', False):
                return failure('Motion platform is disconnected.', 503)
            if not globals.toolhead_homed or not globals.filter_revolver_homed:
                return failure('Home the platform and filter revolver first.', 409)
            globals.motion_busy = True
            try:
                offset = height_offset_control.configured_offset(
                    filter_settings_getter(), globals.filter_revolver_position,
                    light_controller.status()['active_channel'],
                )
                acknowledged, _ = porthandler.write_and_wait(device, 'M400', timeout=30.0)
                if not acknowledged:
                    height_offset_control.invalidate_reference()
                    return failure('Motion completion was not acknowledged.', 504)
                position = motioncontrols.get_toolhead_position(device, timeout=0.5, allow_busy=True)
                height_offset_control.record_combination_reference(position['z'], offset, source='anchor')
                globals.last_toolhead_pos = position
                return jsonify(height_offset_control.status())
            except (ValueError, height_offset_control.HeightOffsetCommandError) as error:
                return failure(str(error), 409)
            except OSError:
                return disconnect_handler(device, 'height reference')
            except RuntimeError as error:
                height_offset_control.invalidate_reference()
                return failure(str(error), 503)
            finally:
                globals.motion_busy = False

    return blueprint
