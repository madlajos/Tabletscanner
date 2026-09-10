import io
from flask import Flask, jsonify, request, Response, send_file
from flask_cors import CORS
import cv2
import time
import globals
from pypylon import pylon
from cameracontrol import (apply_camera_settings, 
                           validate_and_set_camera_param, validate_param,
                           get_camera_properties, stream_video,
                           load_camera_profile, apply_camera_image_geometry,
                           center_camera_axis, get_camera_image_geometry)
import porthandler
import motioncontrols
import filter_revolver
from filter_capture_series import (
    FilterCaptureSeriesCoordinator,
    capture_filename,
    capture_series_stem,
    next_capture_series_index,
    resolve_filter_targets,
)
import height_offset_control
from filter_series_camera import capture_series_frame
from height_reference_api import create_height_reference_blueprint, guard_manual_motion, guard_capture_operation
from height_offset_control import HeightOffsetCommandError
import os
import sys
import math
from datetime import datetime
import subprocess
import json
from threading import Lock
from settings_manager import (
    DEFAULT_MAX_HEIGHT_OFFSET_DOWN_MM,
    DEFAULT_MAX_HEIGHT_OFFSET_UP_MM,
    DEFAULT_FIRST_TABLET_X_MM,
    DEFAULT_FIRST_TABLET_Y_MM,
    DEFAULT_FIRST_TABLET_Z_MM,
    DEFAULT_TABLET_SPACING_MM,
    DEFAULT_LOWER_Z_BEFORE_XY_MOVE,
    DEFAULT_XY_MOVE_Z_LIMIT_MM,
    DEFAULT_SAVE_AUTOFOCUS_IMAGE,
    DEFAULT_CHECK_TABLET_PRESENCE,
    load_settings,
    save_settings,
    update_lamp_output_selectors,
    update_filter_settings,
    update_autofocus_settings,
    update_camera_combination_settings,
    get_settings,
    validate_capture_plan,
    default_autofocus_settings,
    default_filter_settings,
    validate_autofocus_settings,
    validate_filter_settings,
    validate_lamp_output_selectors,
    validate_lamp_settings,
    validate_camera_combination_settings,
    reconcile_camera_combination_settings,
    camera_filter_group,
    validate_motion_simulation_settings,
    TrayGeometryError,
    UV_LAMP_CHANNELS,
    HEIGHT_OFFSET_REFERENCE_FILTER_NAMES,
)
from light_control import (
    CaptureIlluminationError,
    LIGHT_CHANNELS,
    LampSettingsError,
    LightCommandError,
    LightConfigurationError,
    LightController,
    contains_lamp_gcode,
)
import numpy as np
from PIL import Image

from logger_config import setup_logger
from error_codes import ErrorCode, ERROR_MESSAGES
import tkinter as tk
from tkinter import filedialog
from multiprocessing import Process, Queue
import multiprocessing
from cameracontrol import converter  # if not already imported
import autofocus_main
import traceback
import bgr_main
import manual_bgr_with_check
import check_only
import under_over
import calc_color
import pipeline_steps
import pipeline_engine
import pipeline_validators
import recipe_manager
import calibration_manager
import measurement_progress
from pipeline_types import PipelineDocument
from proc_elements.scale_bar import scale_bar_overlay as _apply_scale_bar_overlay
from image_metadata import build_capture_metadata, serialize_capture_metadata


app = Flask(__name__)
app.secret_key = 'Egis'
CORS(app)
app.debug = True

setup_logger()
_EPS = 1e-6
light_controller = LightController(
    get_settings,
    lambda: porthandler.motion_platform or globals.motion_platform,
    porthandler.write_and_wait,
    operation_lock=porthandler.motion_lock,
)
bgr_capture_coordinator = FilterCaptureSeriesCoordinator()
BGR_HARDWARE_SETTLE_SECONDS = 0.5
BGR_CAMERA_RETRY_SETTLE_SECONDS = 0.25
AUTOFOCUS_HARDWARE_SETTLE_SECONDS = 0.5
AUTOFOCUS_ILLUMINATION_SETTLE_SECONDS = 0.25


def four_channel_lamp_timeout_monitor():
    """Monitor LightController deadlines without touching legacy lamp state."""
    while True:
        try:
            channel = light_controller.check_timeouts()
            if channel:
                app.logger.warning('Four-channel lamp %s switched off at its safety deadline.', channel)
        except LightConfigurationError:
            # No configured selectors yet; controller has no active channel in this state.
            pass
        except Exception:
            app.logger.exception('Four-channel lamp timeout monitor failed')
        time.sleep(0.25)

# Might need to be removed
camera_properties = None
latest_frames = None

backend_ready = False



def _normalize_path(p: str) -> str:
    """Normalize filesystem paths to use forward slashes for cross-platform consistency."""
    if not p:
        return p
    return p.replace('\\', '/')


@app.route('/favicon.ico')
def favicon():
    return '', 204


### Error handling and logging ###
@app.errorhandler(404)
def handle_not_found(error):
    """Return a clean 404 without polluting the error log."""
    app.logger.debug(f"404 Not Found: {request.path}")
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(Exception)
def handle_global_exception(error):
    error_message = str(error)
    app.logger.exception(f"Unhandled exception: {error_message}")
    
    return jsonify({
        "error": "An unexpected error occurred.",
        "details": error_message,
        "popup": True
    }), 500


def _handle_motion_usb_disconnect(ser, context: str = "operation"):
    """
    Handle USB disconnection for motion platform.
    Closes the serial port and clears global references.
    
    Args:
        ser: The serial port object
        context: String describing what operation was happening (for logging)
    
    Returns:
        tuple: (error_json, status_code) ready to return from Flask endpoint
    """
    app.logger.warning(f"Motion platform disconnected during {context} (USB error)")
    try:
        if ser:
            ser.close()
    except Exception:
        pass
    globals.motion_platform = None
    porthandler.motion_platform = None
    _reset_motion_reference_state()
    return jsonify({
        'error': ERROR_MESSAGES.get(ErrorCode.MOTIONPLATFORM_DISCONNECTED, 'Motion platform disconnected'),
        'code': ErrorCode.MOTIONPLATFORM_DISCONNECTED,
        'popup': True
    }), 503


def _reset_motion_reference_state():
    """Invalidate cached homing/revolver state when the controller reference is lost."""
    globals.homed_axes = set()
    globals.toolhead_homed = False
    globals.filter_revolver_homed = False
    globals.filter_revolver_position = None
    globals.last_toolhead_pos = {'x': None, 'y': None, 'z': None}
    height_offset_control.invalidate_reference()


def _is_serial_disconnect(exc):
    """Check if an exception is caused by a USB/serial disconnection."""
    msg = str(exc).lower()
    return any(keyword in msg for keyword in [
        'serialexception', 'writefile failed', 'permissionerror',
        'clearcommerror', 'device', 'usb'
    ])


def _is_camera_disconnect(exc):
    """Check if an exception is caused by a camera disconnection or failure."""
    messages = []
    current = exc
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        messages.append(str(current).lower())
        current = current.__cause__ or current.__context__
    msg = ' | '.join(messages)
    return any(keyword in msg for keyword in [
        'camera not ready', 'camera disconnected', 'grab failed',
        'failed to grab', 'physically removed', 'not open',
        'camera is not grabbing', 'device has been removed',
        'device was removed', 'error setting exposuretime for camera',
        'error setting gain for camera', 'error applying camera settings',
    ])


def _handle_camera_disconnect(context: str = "operation"):
    """Handle camera disconnection. Cleans up stale handle and returns (error_json, status_code)."""
    app.logger.warning(f"Camera disconnected during {context}")

    # Clean up the stale camera handle so the next connect attempt
    # doesn't short-circuit on a dead IsOpen() handle.
    cam = getattr(globals, 'camera', None)
    if cam is not None:
        try:
            if cam.IsGrabbing():
                cam.StopGrabbing()
        except Exception:
            pass
        try:
            cam.Close()
        except Exception:
            pass
        globals.camera = None
    globals.stream_running = False
    globals._cached_camera_serial = None

    return jsonify({
        'error': ERROR_MESSAGES.get(ErrorCode.CAMERA_DISCONNECTED, 'Camera disconnected'),
        'code': ErrorCode.CAMERA_DISCONNECTED,
        'popup': True
    }), 503


def retry_operation(operation, max_retries=3, wait=1, exceptions=(Exception,)):
    """
    Attempts to run 'operation' up to 'max_retries' times.
    Waits 'wait' seconds between attempts.
    Raises an exception after all attempts fail.
    """
    for attempt in range(max_retries):
        try:
            return operation()
        except exceptions as e:
            app.logger.warning("Attempt %d/%d failed: %s", attempt + 1, max_retries, e)
            time.sleep(wait)
    raise Exception("Operation failed after %d attempts" % max_retries)


### Serial Device Functions ###
# Connect/Disconnect Serial devices
def _use_virtual_motion_platform():
    return bool(get_settings().get('advanced_settings', {}).get('use_virtual_com_port', False))


def _height_offset_limits():
    advanced_settings = get_settings().get('advanced_settings', {})
    return (
        advanced_settings.get(
            'max_height_offset_up_mm',
            DEFAULT_MAX_HEIGHT_OFFSET_UP_MM,
        ),
        advanced_settings.get(
            'max_height_offset_down_mm',
            DEFAULT_MAX_HEIGHT_OFFSET_DOWN_MM,
        ),
    )


def _replace_motion_platform(use_virtual):
    """Close the current adapter and connect the selected real/virtual device."""
    with porthandler.motion_lock:
        current = porthandler.motion_platform or globals.motion_platform
        if current and getattr(current, 'is_open', False):
            try:
                light_controller.off()
            except Exception as error:
                app.logger.warning('All-off before changing motion adapter failed: %s', error)
            current.close()
        globals.motion_platform = None
        porthandler.motion_platform = None
        _reset_motion_reference_state()

        device = porthandler.connect_to_motion_platform(use_virtual=use_virtual)
        if device:
            globals.motion_platform = device
            porthandler.motion_platform = device
        return device


@app.route('/api/connect-to-motionplatform', methods=['POST'])
def connect_motionplatform():
    try:
        app.logger.info("Attempting to connect to Motion platform")
        use_virtual = _use_virtual_motion_platform()
        current = porthandler.motion_platform or globals.motion_platform
        if (
            current
            and current.is_open
            and bool(getattr(current, 'is_virtual', False)) == use_virtual
        ):
            app.logger.info("Motion platform already connected.")
            return jsonify({
                'message': 'Motion platform already connected',
                'port': current.port,
                'virtual': bool(getattr(current, 'is_virtual', False)),
            }), 200
    
        device = _replace_motion_platform(use_virtual)
        if device:
            app.logger.info("Successfully connected to Motion platform")
            return jsonify({
                'message': 'Motion platform connected',
                'port': device.port,
                'virtual': bool(getattr(device, 'is_virtual', False)),
            }), 200
        else:
            app.logger.error("Failed to connect to Motion platform: No response or incorrect ID")
            return jsonify({
                'error': ERROR_MESSAGES[ErrorCode.MOTIONPLATFORM_DISCONNECTED],
                'code': ErrorCode.MOTIONPLATFORM_DISCONNECTED,
                'popup': True
            }), 404
    except Exception as e:
        app.logger.exception("Exception occurred while connecting to Motion platform")
        return jsonify({
            'error': ERROR_MESSAGES[ErrorCode.MOTIONPLATFORM_DISCONNECTED],
            'code': ErrorCode.MOTIONPLATFORM_DISCONNECTED,
            'popup': True
        }), 500

@app.route('/api/disconnect-<device_name>', methods=['POST'])
def disconnect_serial_device(device_name):
    try:
        app.logger.info(f"Attempting to disconnect from {device_name}")
        porthandler.disconnect_serial_device(device_name)
        if device_name.lower().replace('-', '_') in ('motionplatform', 'motion_platform', 'motion'):
            _reset_motion_reference_state()
        app.logger.info(f"Successfully disconnected from {device_name}")
        return jsonify('ok')
    except Exception as e:
        app.logger.exception(f"Exception occurred while disconnecting from {device_name}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/status/serial/<device_name>', methods=['GET'])
def get_serial_device_status(device_name):
    name = device_name.lower().replace('-', '_')
    if name in ('motionplatform','motion_platform','motion'):
        ser = getattr(porthandler, 'motion_platform', None)
        if not ser or not getattr(ser, 'is_open', False):
            return jsonify({'connected': False}), 200

        # If homing or other long op is in progress, don't touch the port.
        if getattr(globals, 'motion_busy', False):
            return jsonify({
                'connected': True,
                'busy': True,
                'port': ser.port,
                'virtual': bool(getattr(ser, 'is_virtual', False)),
            }), 200

        # Deadline-bounded probe that consumes the complete M105 line under
        # the serial lock. Partial replies must not leak into M114 or motion.
        try:
            acknowledged, reply = porthandler.probe_motion_controller(ser, timeout=0.3)
            if not acknowledged:
                app.logger.debug('M105 probe was not acknowledged: %r', reply[:64])
        except (OSError, PermissionError) as e:
            # USB disconnected or permission denied
            app.logger.warning(f"Motion platform disconnected (USB error): {e}")
            try:
                ser.close()
            except Exception:
                pass
            globals.motion_platform = None
            porthandler.motion_platform = None
            _reset_motion_reference_state()
            return jsonify({'connected': False}), 200
        except Exception as e:
            app.logger.debug(f"status probe error (ignored): {e}")
        return jsonify({
            'connected': True,
            'port': ser.port,
            'virtual': bool(getattr(ser, 'is_virtual', False)),
        }), 200

    return jsonify({'error':'Invalid device name','popup':True}), 400


    
@app.route('/api/get_motion_platform_position', methods=['GET'])
def get_motion_platform_position():
    ser = porthandler.motion_platform or globals.motion_platform
    if not ser or not getattr(ser, 'is_open', False):
        return jsonify(globals.last_toolhead_pos), 200

    if getattr(globals, 'motion_busy', False):
        return jsonify(globals.last_toolhead_pos), 200

    try:
        with porthandler.motion_lock:
            pos = motioncontrols.get_toolhead_position(ser, timeout=0.3)
        # only accept numeric values
        if all(k in pos and isinstance(pos[k], (int, float)) for k in ('x','y','z')):
            globals.last_toolhead_pos = pos
        return jsonify(globals.last_toolhead_pos), 200
    except (OSError, PermissionError) as e:
        return _handle_motion_usb_disconnect(ser, "position query")
    except Exception as e:
        app.logger.warning(f"get position failed (returning cache): {e}")
        return jsonify(globals.last_toolhead_pos), 200


@app.route('/api/check_axes_homed', methods=['GET'])
def check_axes_homed():
    """Return backend-tracked homing state; an M114 reply alone does not prove homing."""
    ser = porthandler.motion_platform or globals.motion_platform
    if not ser or not getattr(ser, 'is_open', False):
        return jsonify({'x': False, 'y': False, 'z': False}), 200
    homed_axes = getattr(globals, 'homed_axes', set())
    return jsonify({axis: axis in homed_axes for axis in ('x', 'y', 'z')}), 200

    
@app.route('/api/disable_steppers', methods=['POST'])
def disable_steppers():
    """Disable all motion steppers after an acknowledged M84 command."""
    ser = globals.motion_platform
    if not ser or not getattr(ser, 'is_open', False):
        return jsonify({'error': 'Motion platform not connected', 'popup': True}), 503
    try:
        if not motioncontrols.disable_steppers(ser):
            return jsonify({'error': 'Controller did not acknowledge motor-off command', 'popup': True}), 504
        height_offset_control.invalidate_reference()
        return jsonify({'status': 'success'}), 200
    except (OSError, PermissionError):
        return _handle_motion_usb_disconnect(ser, 'disable steppers')


@app.route('/api/home_toolhead', methods=['POST', 'OPTIONS'])
def api_home_toolhead():
    if request.method == 'OPTIONS':
        return ('', 204)

    data = request.get_json(silent=True) or {}
    axes = [a.lower()[0] for a in (data.get('axes') or []) if a]
    if any(axis not in ('x', 'y', 'z', 'a') for axis in axes):
        return jsonify({'ok': False, 'error': 'Invalid homing axis.'}), 400
    requested_axes = axes or ['z', 'y', 'x', 'a']
    height_offset_control.invalidate_reference()

    ser = globals.motion_platform
    if not ser or not getattr(ser, 'is_open', False):
        return jsonify({'ok': False, 'error': 'Motion platform not connected'}), 503

    # Once a new A-axis homing attempt starts, the previous slot reference is
    # no longer safe to use unless this attempt completes successfully.
    if 'a' in requested_axes:
        globals.homed_axes.discard('a')
        globals.filter_revolver_homed = False
        globals.filter_revolver_position = None

    globals.motion_busy = True
    try:
        # Home each requested axis separately to guarantee the supplied order.
        # A is last in the default full-homing sequence.
        for axis in requested_axes:
            if not motioncontrols.home_axes(ser, axis):
                raise TimeoutError(
                    f'Motion platform {axis.upper()} homing was not acknowledged.'
                )

        homed_now = set(requested_axes)
        globals.homed_axes.update(homed_now)
        globals.toolhead_homed = all(axis in globals.homed_axes for axis in ('x', 'y', 'z'))
        if 'a' in homed_now:
            globals.filter_revolver_homed = True
            globals.filter_revolver_position = 1
            if data.get('select_autofocus_filter') is True:
                autofocus_settings = _current_autofocus_settings()
                _move_filter_revolver_to_position(
                    ser,
                    autofocus_settings['filter_position'],
                )
                app.logger.info(
                    'A-axis homed and moved directly to autofocus filter position %s.',
                    autofocus_settings['filter_position'],
                )
        
        # Query and cache the position after successful homing
        try:
            pos = motioncontrols.get_toolhead_position(ser, timeout=2.0, allow_busy=True)
            if pos and all(k in pos for k in ('x', 'y', 'z')):
                globals.last_toolhead_pos = pos
                app.logger.info(f"Position cached after homing: X={pos.get('x')}, Y={pos.get('y')}, Z={pos.get('z')}")
        except Exception as e:
            app.logger.warning(f"Could not cache position after homing: {e}")
        
        return jsonify({
            'ok': True, 
            'homed_axes': requested_axes,
            'position': globals.last_toolhead_pos
        })
    except motioncontrols.HomingRejectedError as e:
        app.logger.warning(
            "Controller rejected %s-axis homing; reply=%r",
            e.axis,
            e.reply[:256],
        )
        return jsonify({
            'ok': False,
            'error': str(e),
            'code': ErrorCode.MOTION_HOMING_FAILED,
            'popup': True
        }), 422
    except motioncontrols.HomingTimeoutError as e:
        app.logger.warning(
            "%s-axis homing exceeded %.1f seconds; reply=%r",
            e.axis,
            e.timeout,
            e.reply[:256],
        )
        return jsonify({
            'ok': False,
            'error': str(e),
            'code': ErrorCode.MOTION_HOMING_TIMEOUT,
            'popup': True
        }), 504
    except TimeoutError as e:
        app.logger.warning("Homing command timed out without closing the serial connection: %s", e)
        return jsonify({
            'ok': False,
            'error': str(e),
            'code': ErrorCode.MOTION_HOMING_TIMEOUT,
            'popup': True
        }), 504
    except filter_revolver.FilterRevolverCommandError as e:
        globals.homed_axes.discard('a')
        globals.filter_revolver_homed = False
        globals.filter_revolver_position = None
        app.logger.warning('Autofocus-filter selection after A homing failed: %s', e)
        return jsonify({
            'ok': False,
            'error': str(e),
            'code': ErrorCode.GENERIC,
            'popup': True,
        }), 504
    except (OSError, PermissionError) as e:
        app.logger.warning(f"Motion platform disconnected during homing (USB error): {e}")
        try:
            ser.close()
        except Exception:
            pass
        globals.motion_platform = None
        porthandler.motion_platform = None
        _reset_motion_reference_state()
        return jsonify({
            'ok': False, 
            'error': ERROR_MESSAGES.get(ErrorCode.MOTIONPLATFORM_DISCONNECTED, 'Motion platform disconnected'),
            'code': ErrorCode.MOTIONPLATFORM_DISCONNECTED,
            'popup': True
        }), 503
    except Exception as e:
        app.logger.exception("Homing failed")
        return jsonify({'ok': False, 'error': str(e)}), 500
    finally:
        globals.motion_busy = False


def _filter_revolver_status():
    return {
        'position': globals.filter_revolver_position,
        'homed': bool(globals.filter_revolver_homed),
        'motion_platform_homed': bool(globals.toolhead_homed),
        'busy': bool(globals.motion_busy),
    }


def _current_filter_settings():
    """Return the validated filter-wheel settings using the active Z limits."""
    max_up, max_down = _height_offset_limits()
    return validate_filter_settings(
        get_settings().get('filter_settings', default_filter_settings()),
        max_height_offset_up_mm=max_up,
        max_height_offset_down_mm=max_down,
    )


def _current_autofocus_settings(filter_settings=None):
    """Return autofocus settings validated against populated wheel slots."""
    if filter_settings is None:
        filter_settings = _current_filter_settings()
    return validate_autofocus_settings(
        get_settings().get('autofocus_settings', default_autofocus_settings()),
        filter_settings,
    )


def _move_filter_revolver_to_position(motion_platform, target_position):
    """Move to one validated slot without changing higher-level busy ownership."""
    if not globals.toolhead_homed or not globals.filter_revolver_homed:
        raise RuntimeError('Home the motion platform and filter revolver first.')
    direction, steps = filter_revolver.shortest_path(
        globals.filter_revolver_position,
        target_position,
    )
    for _ in range(steps):
        globals.filter_revolver_position = filter_revolver.rotate_one_slot(
            motion_platform,
            globals.filter_revolver_position,
            direction,
        )


def _select_configured_autofocus_hardware(
    motion_platform,
    manage_motion_busy=False,
    capture_plan_row=None,
):
    """Select and arm the saved autofocus optical/camera combination.

    The old preview acquisition is stopped only after the autofocus matrix cell
    has been applied. Acquisition restarts after the filter, camera parameters
    and illumination are all settled, so autofocus cannot consume a queued
    frame made with the previously selected exposure or gain.
    """
    filter_settings = _current_filter_settings()
    autofocus_settings = _current_autofocus_settings(filter_settings)
    light_controller.off()
    if manage_motion_busy:
        _select_measurement_filter_position(
            motion_platform,
            autofocus_settings['filter_position'],
        )
    else:
        _move_filter_revolver_to_position(
            motion_platform,
            autofocus_settings['filter_position'],
        )
    time.sleep(AUTOFOCUS_HARDWARE_SETTLE_SECONDS)

    camera_params = _apply_selected_camera_combination(
        autofocus_settings['channel'], autofocus_settings['filter_position'])
    if capture_plan_row is not None:
        applied = _apply_capture_plan_camera_settings(capture_plan_row)
        camera_params = {
            **camera_params,
            'ExposureTime': applied['exposure_time'],
            'Gain': applied['gain'],
        }
    camera = globals.camera
    if not camera or not camera.IsOpen():
        raise RuntimeError('Camera is disconnected before autofocus.')
    for setting_name in ('ExposureTime', 'Gain'):
        expected = float(camera_params[setting_name])
        actual = float(getattr(camera, setting_name).GetValue())
        if not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-6):
            raise RuntimeError(
                f'Camera {setting_name} did not reach the autofocus setting '
                f'(expected {expected:g}, actual {actual:g}).'
            )

    activation_mode = (
        autofocus_settings['brightness']
        if autofocus_settings['channel'] in UV_LAMP_CHANNELS
        else None
    )

    if not globals.grab_lock.acquire(timeout=5):
        raise RuntimeError('The camera is busy before autofocus.')
    was_grabbing = False
    try:
        was_grabbing = camera.IsGrabbing()
        if was_grabbing:
            camera.StopGrabbing()
        light_controller.activate(autofocus_settings['channel'], activation_mode)
        time.sleep(AUTOFOCUS_ILLUMINATION_SETTLE_SECONDS)
        camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    except Exception:
        try:
            light_controller.off()
        except Exception:
            pass
        if was_grabbing and camera.IsOpen() and not camera.IsGrabbing():
            camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        raise
    finally:
        globals.grab_lock.release()

    return autofocus_settings, filter_settings


def _apply_selected_height_offset(motion_platform):
    """Apply the active matrix cell when manual autofocus established zero."""
    return height_offset_control.apply_active_combination(
        motion_platform,
        _current_filter_settings(),
        light_controller.status()['active_channel'],
    )


def _camera_combination_cell(channel=None, filter_position=None):
    """Resolve the configured exposure/gain for the active optical pair."""
    channel = channel or light_controller.status()['active_channel']
    filter_position = filter_position or getattr(globals, 'filter_revolver_position', None)
    if channel is None or not isinstance(filter_position, int):
        return None
    filter_settings = _current_filter_settings()
    slots = filter_settings['slots']
    if not 1 <= filter_position <= len(slots):
        return None
    filter_key = camera_filter_group(filter_settings, filter_position)
    if filter_key is None:
        return None
    matrix = reconcile_camera_combination_settings(
        get_settings().get('camera_combination_settings'), filter_settings,
        get_settings().get('camera_params'))
    return matrix.get(filter_key, {}).get(channel)


def _apply_selected_camera_combination(channel=None, filter_position=None):
    """Apply exposure/gain and return the current camera panel values."""
    cell = _camera_combination_cell(channel, filter_position)
    settings_data = get_settings()
    current = settings_data.setdefault('camera_params', {})
    if cell is None:
        return current
    applied = {'ExposureTime': cell['exposure_time'], 'Gain': cell['gain']}
    camera = globals.camera
    if camera and camera.IsOpen():
        properties = getattr(globals, 'camera_properties', None)
        if not properties or any(name not in properties for name in applied):
            properties = get_camera_properties(camera)
            globals.camera_properties = properties
        applied = {
            name: validate_and_set_camera_param(camera, name, value, properties)
            for name, value in applied.items()
        }
    current.update(applied)
    return dict(current)


def _blue_filter_position(filter_settings):
    """Return the configured wheel position of the Kék/Blue VIS filter."""
    blue_ids = {
        definition['id']
        for definition in filter_settings.get('filters', [])
        if isinstance(definition, dict)
        and str(definition.get('name', '')).casefold()
        in HEIGHT_OFFSET_REFERENCE_FILTER_NAMES
    }
    return next(
        (position for position, filter_id in enumerate(filter_settings.get('slots', []), 1)
         if filter_id in blue_ids),
        None,
    )


def _select_blue_filter_for_vis(motion_platform):
    """Move incompatible 255/365 filters out of the optical path before VIS."""
    filter_settings = _current_filter_settings()
    current_position = globals.filter_revolver_position
    if camera_filter_group(filter_settings, current_position) not in (
        'filter_255nm', 'filter_365nm'
    ):
        return False

    blue_position = _blue_filter_position(filter_settings)
    if blue_position is None:
        raise ValueError('A configured Kék/Blue filter is required before activating VIS.')

    # Do not illuminate while the revolver is moving. If the move fails, VIS
    # is never activated and the controller remains in its safe all-off state.
    light_controller.off()
    _move_filter_revolver_to_position(motion_platform, blue_position)
    return True


app.register_blueprint(create_height_reference_blueprint(
    light_controller, _current_filter_settings, _handle_motion_usb_disconnect,
))


@app.route('/api/filter-revolver/status', methods=['GET'])
def filter_revolver_status():
    """Return the acknowledged runtime position of the physical filter revolver."""
    return jsonify(_filter_revolver_status()), 200


@app.route('/api/filter-revolver/rotate', methods=['POST'])
def rotate_filter_revolver():
    """Rotate the A (Marlin internal I) axis by one 60-degree filter slot."""
    data = request.get_json(silent=True) or {}
    direction = data.get('direction')
    if direction not in ('up', 'down'):
        return jsonify({
            'error': "Direction must be 'up' or 'down'.",
            'code': ErrorCode.GENERIC,
            'popup': True,
        }), 400

    motion_platform = porthandler.motion_platform or globals.motion_platform
    if not motion_platform or not getattr(motion_platform, 'is_open', False):
        return jsonify({
            'error': 'Motion platform not connected.',
            'code': ErrorCode.MOTIONPLATFORM_DISCONNECTED,
            'popup': True,
        }), 503

    with porthandler.motion_lock:
        if globals.motion_busy:
            return jsonify({
                'error': 'Motion platform is busy.',
                'code': ErrorCode.GENERIC,
                'popup': True,
            }), 409
        if not globals.filter_revolver_homed:
            return jsonify({
                'error': 'Home the filter revolver before rotating it.',
                'code': ErrorCode.GENERIC,
                'popup': True,
            }), 409
        globals.motion_busy = True

    try:
        globals.filter_revolver_position = filter_revolver.rotate_one_slot(
            motion_platform,
            globals.filter_revolver_position,
            direction,
        )
        height_offset = _apply_selected_height_offset(motion_platform)
        camera_params = _apply_selected_camera_combination()
        globals.motion_busy = False
        return jsonify({
            **_filter_revolver_status(),
            'height_offset': height_offset,
            'camera_params': camera_params,
        }), 200
    except filter_revolver.FilterRevolverCommandError as error:
        globals.homed_axes.discard('a')
        globals.filter_revolver_homed = False
        globals.filter_revolver_position = None
        app.logger.warning('Filter revolver rotation failed: %s', error)
        return jsonify({
            'error': str(error),
            'code': ErrorCode.GENERIC,
            'popup': True,
        }), 504
    except ValueError as error:
        app.logger.warning('Filter revolver rotation rejected: %s', error)
        return jsonify({
            'error': str(error),
            'code': ErrorCode.GENERIC,
            'popup': True,
        }), 409
    except HeightOffsetCommandError as error:
        app.logger.warning('Filter changed, but automatic height correction failed: %s', error)
        globals.motion_busy = False
        return jsonify({
            'error': str(error),
            'code': ErrorCode.MOTION_HEIGHT_OFFSET_FAILED,
            'popup': True,
            **_filter_revolver_status(),
        }), 422
    except (OSError, PermissionError):
        return _handle_motion_usb_disconnect(motion_platform, 'filter revolver rotation')
    finally:
        globals.motion_busy = False


@app.route('/api/filter-revolver/select', methods=['POST'])
def select_filter_revolver_position():
    """Move to a selected slot using the shortest sequence of acknowledged 60-degree steps."""
    data = request.get_json(silent=True) or {}
    target_position = data.get('position')
    if (
        isinstance(target_position, bool)
        or not isinstance(target_position, int)
        or target_position not in range(1, filter_revolver.SLOT_COUNT + 1)
    ):
        return jsonify({
            'error': 'Filter position must be an integer between 1 and 6.',
            'code': ErrorCode.GENERIC,
            'popup': True,
        }), 400

    motion_platform = porthandler.motion_platform or globals.motion_platform
    if not motion_platform or not getattr(motion_platform, 'is_open', False):
        return jsonify({
            'error': 'Motion platform not connected.',
            'code': ErrorCode.MOTIONPLATFORM_DISCONNECTED,
            'popup': True,
        }), 503

    with porthandler.motion_lock:
        if globals.motion_busy:
            return jsonify({
                'error': 'Motion platform is busy.',
                'code': ErrorCode.GENERIC,
                'popup': True,
            }), 409
        if not globals.filter_revolver_homed:
            return jsonify({
                'error': 'Home the filter revolver before rotating it.',
                'code': ErrorCode.GENERIC,
                'popup': True,
            }), 409
        try:
            direction, steps = filter_revolver.shortest_path(
                globals.filter_revolver_position,
                target_position,
            )
        except ValueError as error:
            return jsonify({
                'error': str(error),
                'code': ErrorCode.GENERIC,
                'popup': True,
            }), 409
        globals.motion_busy = True

    try:
        for _ in range(steps):
            globals.filter_revolver_position = filter_revolver.rotate_one_slot(
                motion_platform,
                globals.filter_revolver_position,
                direction,
            )
        height_offset = _apply_selected_height_offset(motion_platform)
        camera_params = _apply_selected_camera_combination()
        globals.motion_busy = False
        return jsonify({
            **_filter_revolver_status(),
            'direction': direction,
            'steps': steps,
            'height_offset': height_offset,
            'camera_params': camera_params,
        }), 200
    except filter_revolver.FilterRevolverCommandError as error:
        globals.homed_axes.discard('a')
        globals.filter_revolver_homed = False
        globals.filter_revolver_position = None
        app.logger.warning('Filter revolver selection failed: %s', error)
        return jsonify({
            'error': str(error),
            'code': ErrorCode.GENERIC,
            'popup': True,
        }), 504
    except ValueError as error:
        app.logger.warning('Filter revolver selection rejected: %s', error)
        return jsonify({
            'error': str(error),
            'code': ErrorCode.GENERIC,
            'popup': True,
        }), 409
    except HeightOffsetCommandError as error:
        app.logger.warning('Filter changed, but automatic height correction failed: %s', error)
        globals.motion_busy = False
        return jsonify({
            'error': str(error),
            'code': ErrorCode.MOTION_HEIGHT_OFFSET_FAILED,
            'popup': True,
            **_filter_revolver_status(),
        }), 422
    except (OSError, PermissionError):
        return _handle_motion_usb_disconnect(motion_platform, 'filter revolver selection')
    finally:
        globals.motion_busy = False

    
### Camera Functions ###
def stop_camera_stream():
    camera = globals.camera

    # Ensure the grab lock exists
    lock = getattr(globals, "grab_lock", None)
    if lock is None:
        globals.grab_lock = Lock()
        lock = globals.grab_lock

    running = bool(getattr(globals, "stream_running", False))
    if not running:
        return "Stream already stopped."

    try:
        # Signal the stream loop to stop and stop grabbing atomically
        # under the lock so the stream generator sees a consistent state.
        with lock:
            globals.stream_running = False
            if camera and camera.IsGrabbing():
                camera.StopGrabbing()
                app.logger.info("Camera stream stopped.")

        # Support both names: stream_thread (preferred) and stream_threads (legacy)
        t = getattr(globals, "stream_thread", None)
        if t is None:
            t = getattr(globals, "stream_threads", None)

        if t and hasattr(t, "is_alive") and t.is_alive():
            t.join(timeout=2)
            app.logger.info("Camera stream thread stopped.")

        # Null out both for consistency
        if hasattr(globals, "stream_thread"):
            globals.stream_thread = None
        if hasattr(globals, "stream_threads"):
            globals.stream_threads = None

        return "Camera stream stopped."
    except Exception as e:
        raise RuntimeError(f"Failed to stop camera stream: {str(e)}")

@app.route('/api/connect-camera', methods=['POST'])
def connect_camera():

    result = connect_camera_internal()
    if "error" in result:
        error_code = result.get("code", ErrorCode.GENERIC)
        result["popup"] = True
        result["error"] = ERROR_MESSAGES.get(error_code, result["error"])
        return jsonify(result), 404
    return jsonify(result), 200

@app.route('/api/disconnect-camera', methods=['POST'])
def disconnect_camera():
    try:
        stop_camera_stream()
        app.logger.info(f"Camera stream stopped before disconnecting.")
    except ValueError:
        app.logger.warning(f"Failed to stop camera stream: Invalid camera type.")
        return jsonify({"error": "Invalid camera type"}), 400
    except RuntimeError as re:
        app.logger.warning(f"Error stopping camera stream: {str(re)}")
    except Exception as e:
        app.logger.error(f"Failed to disconnect camera: {e}")
        return jsonify({"error": str(e)}), 500

    # stop_camera_stream() already stopped grabbing under the lock.
    # Just close the camera handle and clear globals.
    camera = globals.camera
    if camera and camera.IsOpen():
        camera.Close()
        app.logger.info(f"Camera closed.")

    # Clean up references
    globals.camera = None
    globals.camera_properties = {}
    globals._cached_camera_serial = None
    globals.latest_image = None
    app.logger.info(f"Camera disconnected successfully.")

    return jsonify({"status": "disconnected"}), 200

@app.route('/api/camera-name', methods=['GET'])
def get_camera_name():
    try:
        cam = getattr(globals, "camera", None)
        if not (cam and cam.IsOpen()):
            msg = "Camera not connected while trying to fetch its name."
            app.logger.warning(msg)
            return jsonify({"error": msg, "popup": True}), 400
        return jsonify({'name': cam.GetDeviceInfo().GetModelName()}), 200
    
    except Exception as e:
        app.logger.exception("Failed to get camera name")
        return jsonify({"error": "Failed to retrieve camera name", "details": str(e), "popup": True}), 500

@app.route('/api/status/camera', methods=['GET'])
def get_camera_status():
    """
    Return {"connected": bool, "streaming": bool} and
    detect if a previously-open camera was physically removed while idle.
    
    Verifies physical presence as well as the open-handle state. Some camera
    drivers keep IsOpen() true after a USB disconnect.
    """
    camera = getattr(globals, 'camera', None)
    is_streaming = bool(getattr(globals, 'stream_running', False))

    # Baseline "connected" = we have an open handle
    is_connected = bool(camera is not None)
    if is_connected:
        try:
            is_connected = camera.IsOpen()
        except Exception:
            is_connected = False

    if is_connected:
        try:
            open_serial = camera.GetDeviceInfo().GetSerialNumber()
            globals._cached_camera_serial = open_serial
        except Exception:
            open_serial = None
            globals._cached_camera_serial = None

        device_removed = False
        removal_check_available = False
        removal_check = getattr(camera, 'IsCameraDeviceRemoved', None)
        if callable(removal_check):
            removal_check_available = True
            try:
                device_removed = bool(removal_check())
            except Exception:
                device_removed = True

        present_serials = {open_serial} if open_serial else set()
        if not removal_check_available:
            # Compatibility fallback for camera APIs without the dedicated
            # removal probe. GetDeviceInfo() alone may return cached data.
            try:
                devices = pylon.TlFactory.GetInstance().EnumerateDevices()
                present_serials = set()
                for dev in devices:
                    try:
                        serial = dev.GetSerialNumber()
                        if serial:
                            present_serials.add(serial)
                    except Exception:
                        continue
            except Exception:
                present_serials = set()

        if device_removed or not open_serial or open_serial not in present_serials:
            try:
                if is_streaming:
                    try:
                        camera.StopGrabbing()
                    except Exception:
                        pass
                camera.Close()
            except Exception:
                pass

            globals.camera = None
            globals.stream_running = False
            globals._cached_camera_serial = None
            is_connected = False
            is_streaming = False
    else:
        # Camera not connected → clear cache so next connect re-enumerates
        globals._cached_camera_serial = None

    return jsonify({
        "connected": bool(is_connected),
        "streaming": bool(is_streaming),
    }), 200
    
@app.route('/api/get-camera-settings', methods=['GET'])
def get_camera_settings():
    try:
        app.logger.info("API Call: /api/get-camera-settings")
        
        settings_data = get_settings()
        camera_settings = settings_data.get('camera_params', {})
        ranges = {}
        camera = globals.camera
        if camera and camera.IsOpen():
            try:
                camera_properties = get_camera_properties(camera)
                globals.camera_properties = camera_properties
                ranges = {
                    name: camera_properties[name]
                    for name in ('ExposureTime', 'Gain', 'Gamma')
                    if name in camera_properties
                }
            except Exception as error:
                app.logger.warning('Could not query live camera setting ranges: %s', error)

        app.logger.info(f"Sending camera settings to frontend")
        return jsonify({
            "camera_params": camera_settings,
            "ranges": ranges,
        }), 200

    except Exception as e:
        app.logger.exception("Failed to get camera settings")
        return jsonify({
            "error": "Failed to read camera settings",
            "code": ErrorCode.GENERIC,
            "details": str(e),
            "popup": True
        }), 500


@app.route('/api/settings/camera/combinations', methods=['GET', 'PUT'])
def camera_combination_settings():
    """Read or replace per-filter/per-wavelength exposure and gain."""
    try:
        filter_settings = _current_filter_settings()
        if request.method == 'GET':
            matrix = reconcile_camera_combination_settings(
                get_settings().get('camera_combination_settings'), filter_settings,
                get_settings().get('camera_params'))
        else:
            matrix = validate_camera_combination_settings(
                request.get_json(silent=True), filter_settings)
            camera = globals.camera
            if camera and camera.IsOpen():
                properties = getattr(globals, 'camera_properties', None) or get_camera_properties(camera)
                globals.camera_properties = properties
                for row in matrix.values():
                    for cell in row.values():
                        for field, camera_name in (('exposure_time', 'ExposureTime'), ('gain', 'Gain')):
                            accepted = validate_param(camera_name, cell[field], properties)
                            if not math.isclose(accepted, cell[field], rel_tol=1e-9, abs_tol=1e-6):
                                limits = properties[camera_name]
                                raise ValueError(
                                    f'{camera_name} must match the camera range/increment '
                                    f"({limits['min']}..{limits['max']}, step {limits['inc']})."
                                )
            if not update_camera_combination_settings(matrix):
                raise OSError('Failed to persist camera combination settings.')
        camera_params = (_apply_selected_camera_combination() if request.method == 'PUT'
                         else dict(get_settings().get('camera_params', {})))
        ranges = {}
        camera = globals.camera
        if camera and camera.IsOpen():
            properties = getattr(globals, 'camera_properties', None) or get_camera_properties(camera)
            globals.camera_properties = properties
            ranges = {name: properties[name] for name in ('ExposureTime', 'Gain') if name in properties}
        return jsonify({
            'camera_combination_settings': matrix,
            'camera_params': camera_params,
            'ranges': ranges,
        }), 200
    except ValueError as error:
        return jsonify({'error': str(error), 'code': ErrorCode.GENERIC, 'popup': True}), 400
    except Exception as error:
        app.logger.exception('Failed to update camera combination settings')
        return jsonify({'error': str(error), 'code': ErrorCode.GENERIC, 'popup': True}), 500


def _saved_camera_image_settings():
    stored = get_settings().get('camera_image_settings', {})
    return {
        'override_enabled': bool(stored.get('override_enabled', False)),
        'width': int(stored.get('width', 4000)),
        'height': int(stored.get('height', 4000)),
        'offset_x': int(stored.get('offset_x', 0)),
        'offset_y': int(stored.get('offset_y', 0)),
    }


@app.route('/api/settings/camera/image-size', methods=['GET', 'PUT'])
def camera_image_size_settings():
    camera = globals.camera
    connected = bool(camera and camera.IsOpen())
    saved = _saved_camera_image_settings()

    if request.method == 'GET':
        geometry = None
        if connected:
            try:
                geometry = get_camera_image_geometry(camera)
            except Exception as error:
                app.logger.warning('Could not query live camera image geometry: %s', error)
        return jsonify({
            'camera_image_settings': saved if saved['override_enabled'] or not geometry else {
                'override_enabled': False,
                **geometry['values'],
            },
            'limits': geometry['limits'] if geometry else {},
            'connected': connected,
        }), 200

    try:
        payload = request.get_json(silent=True) or {}
        required = {'override_enabled', 'width', 'height', 'offset_x', 'offset_y'}
        if set(payload) != required or not isinstance(payload['override_enabled'], bool):
            raise ValueError('Camera image settings contain missing or unknown fields.')
        values = {name: payload[name] for name in ('width', 'height', 'offset_x', 'offset_y')}
        if payload['override_enabled']:
            if not connected:
                return jsonify({
                    'error': 'A kamera képgeometria felülírásához csatlakoztatott kamera szükséges.',
                    'code': ErrorCode.CAMERA_DISCONNECTED,
                    'popup': True,
                }), 503
            geometry = apply_camera_image_geometry(camera, values)
            values = geometry['values']
            limits = geometry['limits']
        else:
            # Disabling the override affects future reconnects. It deliberately
            # does not mutate the current live ROI.
            for name, value in values.items():
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not float(value).is_integer():
                    raise ValueError(f'{name} must be an integer.')
            values = {name: int(value) for name, value in values.items()}
            geometry = get_camera_image_geometry(camera) if connected else None
            limits = geometry['limits'] if geometry else {}

        normalized = {'override_enabled': payload['override_enabled'], **values}
        get_settings()['camera_image_settings'] = normalized
        if not save_settings():
            raise OSError('Failed to persist camera image settings.')
        return jsonify({
            'camera_image_settings': normalized,
            'limits': limits,
            'connected': connected,
        }), 200
    except ValueError as error:
        return jsonify({'error': str(error), 'code': ErrorCode.GENERIC, 'popup': True}), 400
    except Exception as error:
        app.logger.exception('Failed to update camera image settings')
        return jsonify({'error': str(error), 'code': ErrorCode.GENERIC, 'popup': True}), 500


@app.route('/api/settings/camera/image-size/center', methods=['POST'])
def center_camera_image():
    try:
        saved = _saved_camera_image_settings()
        if not saved['override_enabled']:
            raise ValueError('Enable camera image-size override before centering.')
        camera = globals.camera
        if not camera or not camera.IsOpen():
            return jsonify({
                'error': 'A kamera középre igazításához csatlakoztatott kamera szükséges.',
                'code': ErrorCode.CAMERA_DISCONNECTED,
                'popup': True,
            }), 503
        axis = (request.get_json(silent=True) or {}).get('axis')
        geometry = center_camera_axis(camera, axis)
        values = geometry['values']
        normalized = {'override_enabled': True, **values}
        get_settings()['camera_image_settings'] = normalized
        if not save_settings():
            raise OSError('Failed to persist centered camera image settings.')
        return jsonify({
            'camera_image_settings': normalized,
            'limits': geometry['limits'],
            'connected': True,
        }), 200
    except ValueError as error:
        return jsonify({'error': str(error), 'code': ErrorCode.GENERIC, 'popup': True}), 400
    except Exception as error:
        app.logger.exception('Failed to center camera image')
        return jsonify({'error': str(error), 'code': ErrorCode.GENERIC, 'popup': True}), 500


@app.route('/api/auto-measurement/camera-config', methods=['GET'])
def get_auto_measurement_camera_config():
    """Return current values and live Basler limits for per-row capture settings."""
    settings_data = get_settings()
    persisted = settings_data.get('camera_params', {})
    values = {
        'ExposureTime': float(persisted.get('ExposureTime', 100000.0)),
        'Gain': float(persisted.get('Gain', 0.0)),
        'Gamma': float(persisted.get('Gamma', 1.0)),
    }
    camera = globals.camera
    connected = bool(camera and camera.IsOpen())
    ranges = {}
    if connected:
        try:
            camera_properties = get_camera_properties(camera)
            globals.camera_properties = camera_properties
            ranges = {
                name: camera_properties[name]
                for name in ('ExposureTime', 'Gain', 'Gamma')
                if name in camera_properties
            }
        except Exception as error:
            app.logger.warning('Could not query Basler capture parameter ranges: %s', error)
    return jsonify({
        'connected': connected,
        'values': values,
        'ranges': ranges,
    }), 200
    
@app.route('/api/update-camera-settings', methods=['POST'])
def update_camera_settings():
    """Update one global persisted exposure, gain, or gamma setting."""
    try:
        data = request.get_json(silent=True) or {}
        setting_name = data.get('setting_name')
        setting_value = data.get('setting_value')

        if setting_name not in ('ExposureTime', 'Gain', 'Gamma'):
            return jsonify({"error": "Only ExposureTime, Gain, and Gamma are configurable.", "code": ErrorCode.GENERIC, "popup": True}), 400

        try:
            numeric_value = float(setting_value)
        except (TypeError, ValueError):
            return jsonify({"error": "Camera setting must be numeric.", "code": ErrorCode.GENERIC, "popup": True}), 400
        if not math.isfinite(numeric_value):
            return jsonify({"error": "Camera setting must be finite.", "code": ErrorCode.GENERIC, "popup": True}), 400

        camera = globals.camera
        camera_properties = globals.camera_properties
        applied_to_camera = bool(camera and camera.IsOpen())
        updated_value = numeric_value

        if applied_to_camera:
            if not camera_properties or setting_name not in camera_properties:
                camera_properties = get_camera_properties(camera)
                globals.camera_properties = camera_properties
            if setting_name not in camera_properties:
                return jsonify({
                    "error": f"{setting_name} is not supported by the connected camera.",
                    "code": ErrorCode.GENERIC,
                    "popup": True,
                }), 400
            updated_value = validate_and_set_camera_param(camera, setting_name, numeric_value, camera_properties)

        settings_data = get_settings()
        settings_data.setdefault('camera_params', {})[setting_name] = updated_value
        save_settings()
        app.logger.info("Global camera setting %s updated (applied=%s)", setting_name, applied_to_camera)
        return jsonify({
            "camera_params": settings_data['camera_params'],
            "updated_value": updated_value,
            "applied_to_camera": applied_to_camera
        }), 200

    except Exception as e:
        app.logger.exception("Failed to update camera settings")
        return jsonify({"error": str(e)}), 500


# Retired in schema v2: camera exposure and gamma are global, not per-light.
# @app.route('/api/update-camera-settings-light', methods=['POST'])
def update_camera_settings_light():
    """Compatibility route: update the schema-v2 global camera settings."""
    try:
        data = request.json
        light = data.get('light')  # 'dome' or 'bar'
        setting_name = data.get('setting_name')
        setting_value = data.get('setting_value')
        apply_to_camera = data.get('apply_to_camera', True)  # Default to True for backwards compatibility
        persist = data.get('persist', True)  # Whether to save to settings.json (False = temporary/live-only change)

        if light not in ('dome', 'bar'):
            return jsonify({"error": "Invalid light. Must be 'dome' or 'bar'."}), 400

        app.logger.info(f"[CameraSettings] Updating {light} setting {setting_name}={setting_value}, apply_to_camera={apply_to_camera}")

        updated_value = setting_value  # Default to the input value

        # Only apply to camera hardware if apply_to_camera is True
        if apply_to_camera:
            # Fetch camera and current camera_properties
            camera = globals.camera
            camera_properties = globals.camera_properties

            # Skip hardware apply if camera is not connected/open
            if not camera or not camera.IsOpen():
                app.logger.warning(f"[CameraSettings] Camera not connected; skipping hardware apply for {light} {setting_name}")
                apply_to_camera = False
            else:
                # Fallback: Refresh properties if missing
                if not camera_properties or setting_name not in camera_properties:
                    app.logger.warning("camera_properties missing or incomplete; fetching fresh values...")
                    camera_properties = get_camera_properties(camera)
                    globals.camera_properties = camera_properties

                # Apply the setting to the camera
                updated_value = validate_and_set_camera_param(
                    camera,
                    setting_name,
                    setting_value,
                    camera_properties
                )
                app.logger.info(f"[CameraSettings] \u2713 {light} {setting_name} applied to camera hardware: {updated_value}")
        else:
            app.logger.info(f"[CameraSettings] \u2713 {light} {setting_name} skipped camera hardware (only saved to settings.json)")

        # Legacy callers still provide a light name, but schema v2 has only one
        # camera setting set. Do not recreate camera_params_dome/bar here.
        if persist:
            settings_data = get_settings()
            if 'camera_params' not in settings_data:
                settings_data['camera_params'] = {}
            settings_data['camera_params'][setting_name] = updated_value
            save_settings()
            app.logger.info(f"Global camera setting {setting_name} updated through legacy {light} route")
        else:
            app.logger.info(f"{light} camera setting {setting_name} applied live only (not persisted to settings.json)")

        return jsonify({
            "message": f"{light.capitalize()} camera {setting_name} updated.",
            "updated_value": updated_value,
            "applied_to_camera": apply_to_camera,
            "persisted": persist
        }), 200

    except Exception as e:
        app.logger.exception("Failed to update light-specific camera settings")
        return jsonify({"error": str(e)}), 500


@app.route('/api/load-camera-profile', methods=['POST'])
def api_load_camera_profile():
    """Load a .pfs (Pylon Feature Set) profile onto the camera."""
    try:
        data = request.get_json() or {}
        pfs_path = data.get('path', '').strip()
        
        if not pfs_path:
            return jsonify({
                'error': 'Nincs megadva kamera profil fájl',
                'code': 'E1311',
                'popup': True
            }), 400
        
        # Normalize path for cross-platform consistency
        pfs_path = _normalize_path(pfs_path)

        # Always persist the profile path, even if the camera is not connected yet.
        settings_data = get_settings()
        if 'other_settings' not in settings_data:
            settings_data['other_settings'] = {}
        settings_data['other_settings']['camera_settings_file'] = pfs_path
        save_settings()

        camera = globals.camera
        if not camera or not camera.IsOpen():
            return jsonify({
                'success': True,
                'path': pfs_path,
                'applied': False,
                'reason': 'camera_not_connected'
            }), 200
        
        result = load_camera_profile(camera, pfs_path)
        
        if 'error' in result:
            return jsonify({
                'error': result['error'],
                'code': result.get('code', 'E1311'),
                'popup': True
            }), 400
        
        return jsonify({'success': True, 'path': pfs_path, 'applied': True}), 200
        
    except Exception as e:
        app.logger.exception("Failed to load camera profile")
        return jsonify({
            'error': f'Kamera profil betöltése sikertelen: {str(e)}',
            'code': 'E1311',
            'popup': True
        }), 500
    
    
def _clamp_axis(axis: str, target: float):
    lo, hi = globals.motion_limits[axis]
    clamped = max(lo, min(hi, target))
    return clamped, (clamped != target), lo, hi    


class XyMoveSafetyError(RuntimeError):
    """Raised when an X/Y move cannot be preceded by its configured safe Z move."""


def _xy_move_safety_settings():
    advanced = get_settings().get('advanced_settings', {})
    return (
        bool(advanced.get('lower_z_before_xy_move', DEFAULT_LOWER_Z_BEFORE_XY_MOVE)),
        float(advanced.get('xy_move_z_limit_mm', DEFAULT_XY_MOVE_Z_LIMIT_MM)),
    )


def _lower_z_before_xy_move(motion_platform):
    """Lower Z and wait for physical completion before a manual X/Y command."""
    enabled, limit = _xy_move_safety_settings()
    if not enabled:
        return None

    current_z = getattr(globals, 'last_toolhead_pos', {}).get('z')
    if current_z is None:
        try:
            position = motioncontrols.get_toolhead_position(
                motion_platform, timeout=0.5, allow_busy=True
            )
        except (OSError, PermissionError):
            raise
        except Exception as error:
            raise XyMoveSafetyError('Current Z position is unavailable.') from error
        if not isinstance(position, dict) or not isinstance(position.get('z'), (int, float)):
            raise XyMoveSafetyError('Current Z position is unavailable.')
        globals.last_toolhead_pos = position
        current_z = position['z']

    if float(current_z) <= limit + _EPS:
        return None

    acknowledged, _ = porthandler.write_and_wait(motion_platform, 'G90', timeout=2.0)
    if not acknowledged:
        raise XyMoveSafetyError('Absolute motion mode was not acknowledged.')
    acknowledged, _ = porthandler.write_and_wait(
        motion_platform, f'G1 Z{limit}', timeout=30.0
    )
    if not acknowledged:
        raise XyMoveSafetyError('Safe Z move was not acknowledged.')
    if not porthandler.write_and_wait_motion(motion_platform, 'M400', timeout=30.0):
        raise XyMoveSafetyError('Safe Z move did not complete in time.')

    globals.last_toolhead_pos['z'] = limit
    app.logger.info('Lowered Z from %.4f to %.4f before X/Y motion.', current_z, limit)
    return {'from_z': float(current_z), 'to_z': limit}
    
    
# Function to move the toolhead by a given amount (relative movement)
# Function to move the toolhead by a given amount (relative movement)
@app.route('/api/move_toolhead_relative', methods=['POST'])
@guard_manual_motion
def move_toolhead_relative():
    data = request.get_json()
    axis = data.get('axis')
    value = data.get('value')

    if axis not in ['x', 'y', 'z']:
        return jsonify({'status': 'error', 'message': 'Invalid axis'}), 400

    try:
        motion_platform = globals.motion_platform
        if motion_platform is None or not motion_platform.is_open:
            return jsonify({
                'status': 'error', 
                'message': 'Printer not connected',
                'code': ErrorCode.MOTIONPLATFORM_DISCONNECTED,
                'popup': True
            }), 503

        # Need a known current position (homed) to clamp relative moves
        curr = globals.last_toolhead_pos.get(axis) if hasattr(globals, "last_toolhead_pos") else None
        if curr is None:
            return jsonify({
                'status': 'error',
                'message': f'Axis {axis.upper()} not homed; position unknown.'
            }), 409

        target = float(curr) + float(value)
        clamped, clipped, lo, hi = _clamp_axis(axis, target)
        adj = clamped - float(curr)

        if math.isclose(adj, 0.0, abs_tol=_EPS):
            return jsonify({
                'status': 'success',
                'requested': {'axis': axis, 'delta': value},
                'sent': {'axis': axis, 'delta': 0.0},
                'clamped': {axis: bool(clipped)},
                'limits': {axis: {'min': lo, 'max': hi}},
                'message': f'Already at {axis.upper()} limit.'
            }), 200

        safety_z_move = _lower_z_before_xy_move(motion_platform) if axis in ('x', 'y') else None
        move_args = {axis: adj}
        try:
            motioncontrols.move_relative(motion_platform, **move_args)
        except (OSError, PermissionError) as e:
            return _handle_motion_usb_disconnect(motion_platform, "relative move")

        # Update cached position after successful relative move
        globals.last_toolhead_pos[axis] = clamped
        height_offset_control.invalidate_for_manual_move(z_changed=axis == 'z')

        response = {
            'status': 'success',
            'requested': {'axis': axis, 'delta': value},
            'sent': {'axis': axis, 'delta': adj},
            'clamped': {axis: bool(clipped)},
            'limits': {axis: {'min': lo, 'max': hi}}
        }
        if safety_z_move is not None:
            response['safety_z_move'] = safety_z_move
        return jsonify(response), 200

    except (OSError, PermissionError):
        return _handle_motion_usb_disconnect(globals.motion_platform, 'XY safety move')
    except XyMoveSafetyError as error:
        return jsonify({
            'status': 'error', 'error': str(error),
            'code': ErrorCode.MOTION_XY_SAFETY_FAILED, 'popup': True,
        }), 409
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

    
    
@app.route('/api/move_toolhead_absolute', methods=['POST'])
@guard_manual_motion
def move_toolhead_absolute():
    try:
        data = request.get_json() or {}
        resp, status = _move_toolhead_absolute_impl(
            x_pos=data.get('x'),
            y_pos=data.get('y'),
            z_pos=data.get('z')
        )
        return jsonify(resp), status
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500
    
    
def _run_configured_manual_autofocus(motion_platform, skip_empty_check: bool) -> dict:
    """Run the manual autofocus workflow while the caller owns ``motion_busy``."""
    # Always clear the abort flag so a previous cancelled operation cannot block this run.
    globals.autofocus_abort = False

    # Select the acknowledged filter slot first, then activate the saved
    # illumination through the four-channel interlock.
    autofocus_settings, filter_settings = _select_configured_autofocus_hardware(
        motion_platform
    )
    autofocus_camera_params = {
        'ExposureTime': float(globals.camera.ExposureTime.GetValue()),
        'Gain': float(globals.camera.Gain.GetValue()),
    }
    bgr_capture_coordinator.record_camera_params(autofocus_camera_params)
    height_offset_control.invalidate_reference()

    response = autofocus_main.autofocus_coarse(
        motion_platform,
        do_frame_touch_check=False,
        skip_empty_check=skip_empty_check,
    )
    if response.get('status') == 'OK':
        configured_offset = height_offset_control.configured_offset(
            filter_settings,
            autofocus_settings['filter_position'],
            autofocus_settings['channel'],
        )
        reference_z = height_offset_control.record_combination_reference(
            getattr(globals, 'last_toolhead_pos', {}).get('z'),
            configured_offset,
            source='anchor',
        )
        response['autofocus_reference'] = height_offset_control.status()
        response['autofocus_settings'] = autofocus_settings
        response['camera_params'] = autofocus_camera_params
        app.logger.info('Manual autofocus anchored height-offset reference set to Z=%.4f.', reference_z)
    else:
        height_offset_control.invalidate_reference()
    return response


@app.route('/api/autofocus_coarse', methods=['POST'])
def autofocus_coarse():
    motion_platform = globals.motion_platform
    if not motion_platform or not getattr(motion_platform, 'is_open', False):
        return jsonify({
            'status': 'error',
            'message': 'Motion platform not connected',
            'code': ErrorCode.MOTIONPLATFORM_DISCONNECTED,
            'popup': True,
        }), 503

    with porthandler.motion_lock:
        if globals.motion_busy:
            return jsonify({
                'status': 'error',
                'message': 'Motion platform is busy.',
                'code': ErrorCode.GENERIC,
                'popup': True,
            }), 409
        if not globals.toolhead_homed or not globals.filter_revolver_homed:
            return jsonify({
                'status': 'error',
                'message': 'Home the motion platform and filter revolver before autofocus.',
                'code': ErrorCode.GENERIC,
                'popup': True,
            }), 409
        globals.motion_busy = True

    try:
        data = request.get_json(silent=True) or {}
        response = _run_configured_manual_autofocus(
            motion_platform,
            skip_empty_check=bool(data.get('skip_empty_check', False)),
        )
        return jsonify(response)
    except filter_revolver.FilterRevolverCommandError as error:
        globals.homed_axes.discard('a')
        globals.filter_revolver_homed = False
        globals.filter_revolver_position = None
        height_offset_control.invalidate_reference()
        return jsonify({
            'status': 'error',
            'message': str(error),
            'code': ErrorCode.GENERIC,
            'popup': True,
        }), 504
    except LightConfigurationError as error:
        height_offset_control.invalidate_reference()
        return jsonify({
            'status': 'error',
            'message': str(error),
            'code': ErrorCode.GENERIC,
            'popup': True,
        }), 400
    except LightCommandError as error:
        height_offset_control.invalidate_reference()
        return jsonify({
            'status': 'error',
            'message': str(error),
            'code': ErrorCode.MOTIONPLATFORM_DISCONNECTED,
            'popup': True,
        }), 503
    except ValueError as error:
        height_offset_control.invalidate_reference()
        return jsonify({
            'status': 'error',
            'message': str(error),
            'code': ErrorCode.GENERIC,
            'popup': True,
        }), 400
    except (OSError, PermissionError) as e:
        height_offset_control.invalidate_reference()
        return _handle_motion_usb_disconnect(motion_platform, "autofocus")
    except Exception as e:
        height_offset_control.invalidate_reference()
        app.logger.exception("autofocus_coarse failed")  # logs full traceback
        return jsonify({
            'status': 'error',
            'message': str(e),
            'trace': traceback.format_exc()
        }), 500
    finally:
        globals.motion_busy = False


@app.route('/api/lights/status', methods=['GET'])
def get_lights_status():
    try:
        return jsonify({
            **light_controller.status(),
            'height_offset_reference': height_offset_control.status(),
        }), 200
    except Exception as error:
        app.logger.exception('Failed to retrieve four-channel light status')
        return jsonify({'error': str(error), 'code': ErrorCode.GENERIC, 'popup': True}), 500


@app.route('/api/lights/activate', methods=['POST'])
def activate_light():
    owns_motion = False
    try:
        data = request.get_json(silent=True) or {}
        with porthandler.motion_lock:
            if globals.motion_busy:
                return jsonify({
                    'error': 'Motion platform is busy.',
                    'code': ErrorCode.GENERIC,
                    'popup': True,
                }), 409
            globals.motion_busy = True
            owns_motion = True
        motion_platform = porthandler.motion_platform or globals.motion_platform
        filter_changed = (
            data.get('channel') == 'vis'
            and _select_blue_filter_for_vis(motion_platform)
        )
        status = light_controller.activate(data.get('channel'), data.get('mode'))
        height_offset = _apply_selected_height_offset(
            motion_platform
        )
        camera_params = _apply_selected_camera_combination()
        return jsonify({
            **status,
            'height_offset': height_offset,
            'height_offset_reference': height_offset_control.status(),
            'camera_params': camera_params,
            'filter_revolver': _filter_revolver_status() if filter_changed else None,
        }), 200
    except LampSettingsError as error:
        return jsonify({'error': str(error), 'code': ErrorCode.LAMP_SETTINGS_MISSING, 'popup': True}), 400
    except LightConfigurationError as error:
        return jsonify({'error': str(error), 'code': ErrorCode.GENERIC, 'popup': True}), 400
    except LightCommandError as error:
        return jsonify({'error': str(error), 'code': ErrorCode.MOTIONPLATFORM_DISCONNECTED, 'popup': True}), 503
    except filter_revolver.FilterRevolverCommandError as error:
        globals.homed_axes.discard('a')
        globals.filter_revolver_homed = False
        globals.filter_revolver_position = None
        app.logger.warning('Automatic blue-filter selection failed: %s', error)
        return jsonify({'error': str(error), 'code': ErrorCode.GENERIC, 'popup': True}), 504
    except ValueError as error:
        app.logger.warning('VIS activation rejected: %s', error)
        return jsonify({'error': str(error), 'code': ErrorCode.GENERIC, 'popup': True}), 409
    except (OSError, PermissionError):
        return _handle_motion_usb_disconnect(
            porthandler.motion_platform or globals.motion_platform,
            'automatic VIS filter selection',
        )
    except HeightOffsetCommandError as error:
        app.logger.warning('Light changed, but automatic height correction failed: %s', error)
        try:
            light_controller.off()
        except Exception as off_error:
            app.logger.warning('All-off after height-correction failure also failed: %s', off_error)
        return jsonify({
            'error': str(error),
            'code': ErrorCode.MOTION_HEIGHT_OFFSET_FAILED,
            'popup': True,
        }), 422
    except Exception as error:
        app.logger.exception('Failed to activate four-channel light')
        return jsonify({'error': str(error), 'code': ErrorCode.GENERIC, 'popup': True}), 500
    finally:
        if owns_motion:
            globals.motion_busy = False


@app.route('/api/lights/off', methods=['POST'])
def deactivate_light():
    try:
        data = request.get_json(silent=True) or {}
        return jsonify(light_controller.off(data.get('channel'))), 200
    except LightConfigurationError as error:
        return jsonify({'error': str(error), 'code': ErrorCode.GENERIC, 'popup': True}), 400
    except LightCommandError as error:
        return jsonify({'error': str(error), 'code': ErrorCode.MOTIONPLATFORM_DISCONNECTED, 'popup': True}), 503
    except Exception as error:
        app.logger.exception('Failed to deactivate four-channel light')
        return jsonify({'error': str(error), 'code': ErrorCode.GENERIC, 'popup': True}), 500


@app.route('/api/send_gcode', methods=['POST'])
def send_gcode():
    try:
        data = request.get_json(silent=True) or {}
        command = data.get('command')
        if not isinstance(command, str) or not command.strip():
            return jsonify({'error': 'No command provided'}), 400
        if '\r' in command or '\n' in command:
            return jsonify({'error': 'Exactly one G-code line is allowed per request.'}), 400
        if contains_lamp_gcode(command):
            return jsonify({
                'error': 'Lamp commands must use the /api/lights endpoints so the interlock is enforced.',
                'code': ErrorCode.GENERIC,
                'popup': True,
            }), 400

        ser = globals.motion_platform
        if not ser or not ser.is_open:
            return jsonify({'error': 'Motion platform not connected'}), 503

        porthandler.write(ser, command)

        return jsonify({'message': 'Command sent'}), 200
    except Exception as e:
        app.logger.exception("send_gcode failed")
        return jsonify({'error': str(e)}), 500

    # 1) Move the existing logic into a helper:

def _move_toolhead_absolute_impl(
    x_pos=None, y_pos=None, z_pos=None, *, preserve_height_reference=False
):
    motion_platform = globals.motion_platform
    if motion_platform is None or not motion_platform.is_open:
        # same 404 as before
        return {
            'status': 'error',
            'message': 'Printer not connected',
            'code': ErrorCode.MOTIONPLATFORM_DISCONNECTED,
            'popup': True
        }, 503

    requested = {}
    planned = {}
    clamped_flags = {}
    limits_out = {}

    def process_axis(ax, val):
        if val is None:
            return
        tgt = float(val)
        clamped, clipped, lo, hi = _clamp_axis(ax, tgt)
        requested[ax] = tgt
        planned[ax] = clamped
        clamped_flags[ax] = bool(clipped)
        limits_out[ax] = {'min': lo, 'max': hi}

    process_axis('x', x_pos)
    process_axis('y', y_pos)
    process_axis('z', z_pos)

    if not planned:
        return {
            'status': 'error',
            'message': 'No axes specified'
        }, 400

    all_noop = True
    curr_pos = getattr(globals, "last_toolhead_pos", {})

    # If cached position has None values, try a live M114 query to refresh
    if any(curr_pos.get(ax) is None for ax in planned):
        try:
            live_pos = motioncontrols.get_toolhead_position(motion_platform, timeout=0.4, allow_busy=True)
            if live_pos and all(k in live_pos and isinstance(live_pos[k], (int, float)) for k in ('x', 'y', 'z')):
                globals.last_toolhead_pos = live_pos
                curr_pos = live_pos
                app.logger.info(f"Refreshed cached position via M114: {live_pos}")
        except Exception as e:
            app.logger.warning(f"Live M114 query failed, using cached position: {e}")

    # Check if this is a noop (already at target). If any axis position is
    # unknown we cannot determine noop, so skip the check and send the move
    # anyway — the board enforces its own travel limits.
    position_known = True
    for ax, clamped_val in planned.items():
        curr = curr_pos.get(ax)
        if curr is None:
            position_known = False
            all_noop = False
            break
        if not math.isclose(float(curr), float(clamped_val), abs_tol=_EPS):
            all_noop = False

    if all_noop and position_known:
        return {
            'status': 'success',
            'requested': requested,
            'sent': {},
            'clamped': clamped_flags,
            'limits': limits_out,
            'message': 'Requested positions equal to current (after clamping); no move sent.'
        }, 200

    try:
        xy_moving = any(
            ax in planned and (
                curr_pos.get(ax) is None
                or not math.isclose(float(curr_pos[ax]), planned[ax], abs_tol=_EPS)
            )
            for ax in ('x', 'y')
        )
        safety_z_move = _lower_z_before_xy_move(motion_platform) if xy_moving else None
        z_after_xy = (
            safety_z_move is not None
            and 'z' in planned
            and planned['z'] > safety_z_move['to_z'] + _EPS
        )
        motioncontrols.move_to_position(
            motion_platform,
            planned.get('x'),
            planned.get('y'),
            None if z_after_xy else planned.get('z')
        )
        if z_after_xy:
            motioncontrols.move_to_position(motion_platform, z_pos=planned['z'])
    except (OSError, PermissionError) as e:
        app.logger.warning(f"Motion platform disconnected during move (USB error): {e}")
        try:
            motion_platform.close()
        except Exception:
            pass
        globals.motion_platform = None
        porthandler.motion_platform = None
        _reset_motion_reference_state()
        return {
            'status': 'error',
            'message': ERROR_MESSAGES.get(ErrorCode.MOTIONPLATFORM_DISCONNECTED, 'Motion platform disconnected'),
            'code': ErrorCode.MOTIONPLATFORM_DISCONNECTED,
            'popup': True
        }, 503
    except XyMoveSafetyError as error:
        return {
            'status': 'error', 'error': str(error),
            'code': ErrorCode.MOTION_XY_SAFETY_FAILED, 'popup': True,
        }, 409
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e)
        }, 500

    z_changed = 'z' in planned and (
        curr_pos.get('z') is None
        or not math.isclose(float(curr_pos['z']), planned['z'], abs_tol=_EPS)
    )
    # Update cached position with the planned values
    for ax, val in planned.items():
        globals.last_toolhead_pos[ax] = float(val)
    if z_changed or not preserve_height_reference:
        height_offset_control.invalidate_for_manual_move(z_changed=z_changed)

    response = {
        'status': 'success',
        'requested': requested,
        'sent': planned,
        'clamped': clamped_flags,
        'limits': limits_out
    }
    if safety_z_move is not None:
        response['safety_z_move'] = safety_z_move
    return response, 200


def _turn_on_dome_light():
    """Turn on VIS through the authoritative four-channel interlock."""
    light_controller.activate('vis')

def _turn_on_uv_dome_light():
    """Turn on dimmed 365 nm UV through the four-channel interlock."""
    light_controller.activate('uv365', 'dimmed')

def _turn_off_all_lights():
    """Turn off every physical illumination output, including error paths."""
    try:
        light_controller.off()
        return True
    except Exception as error:
        # Fall back to fixed physical selectors if settings/controller state is
        # unavailable. This is deliberately best-effort shutdown behavior.
        app.logger.warning('Controller all-off failed; retrying physical outputs: %s', error)
    ser = porthandler.motion_platform or globals.motion_platform
    if not ser or not getattr(ser, 'is_open', False):
        app.logger.error('Physical all-off retry unavailable: motion platform is disconnected.')
        return False

    failures = []
    with porthandler.motion_lock:
        for selector in range(4):
            try:
                acknowledged, _ = porthandler.write_and_wait(
                    ser, f"M106 P{selector} S0", timeout=2.0
                )
                if not acknowledged:
                    failures.append(f'P{selector}: no acknowledgement')
            except Exception as fallback_error:
                failures.append(f'P{selector}: {fallback_error}')
        if failures:
            app.logger.error('Physical all-off retry failed: %s', '; '.join(failures))
            return False

        light_controller.confirm_all_off()
        return True

def _apply_camera_settings_for_light(light: str):
    """Apply the selected optical pair, retaining legacy light aliases."""
    channel = {'dome': 'vis', 'bar': 'uv365'}.get(light, light)
    try:
        _apply_selected_camera_combination(channel)
    except Exception as error:
        app.logger.warning('Could not apply camera combination for %s: %s', channel, error)


CAPTURE_PLAN_CAMERA_FIELDS = {
    'exposure_time': 'ExposureTime',
    'gain': 'Gain',
}


def _validate_capture_plan_camera_values(capture_plan: list[dict]) -> None:
    """Reject values that the connected camera would need to clamp or round."""
    camera = globals.camera
    if not camera or not camera.IsOpen():
        return
    camera_properties = globals.camera_properties
    if not camera_properties or any(
        name not in camera_properties for name in CAPTURE_PLAN_CAMERA_FIELDS.values()
    ):
        camera_properties = get_camera_properties(camera)
        globals.camera_properties = camera_properties
    missing = [
        name for name in CAPTURE_PLAN_CAMERA_FIELDS.values()
        if name not in camera_properties
    ]
    if missing:
        raise ValueError(
            f"The connected camera does not support: {', '.join(missing)}."
        )
    for row_index, row in enumerate(capture_plan, start=1):
        for field, camera_name in CAPTURE_PLAN_CAMERA_FIELDS.items():
            accepted = validate_param(camera_name, row[field], camera_properties)
            if not math.isclose(accepted, row[field], rel_tol=1e-9, abs_tol=1e-6):
                limits = camera_properties[camera_name]
                raise ValueError(
                    f"Capture plan row {row_index} {field} must match the Basler "
                    f"range/increment ({limits['min']}..{limits['max']}, "
                    f"step {limits['inc']})."
                )


def _apply_capture_plan_camera_settings(row: dict) -> dict:
    """Apply and return the camera-normalized values for one plan row."""
    camera = globals.camera
    if not camera or not camera.IsOpen():
        raise RuntimeError('Camera is not connected.')
    camera_properties = globals.camera_properties
    if not camera_properties or any(
        name not in camera_properties for name in CAPTURE_PLAN_CAMERA_FIELDS.values()
    ):
        camera_properties = get_camera_properties(camera)
        globals.camera_properties = camera_properties
    _validate_capture_plan_camera_values([row])
    applied = {
        field: validate_and_set_camera_param(
            camera, camera_name, row[field], camera_properties
        )
        for field, camera_name in CAPTURE_PLAN_CAMERA_FIELDS.items()
    }
    for field, camera_name in CAPTURE_PLAN_CAMERA_FIELDS.items():
        if not math.isclose(applied[field], row[field], rel_tol=1e-9, abs_tol=1e-6):
            limits = camera_properties[camera_name]
            raise ValueError(
                f"{field} must match the Basler range/increment "
                f"({limits['min']}..{limits['max']}, step {limits['inc']})."
            )
    return applied


def _select_measurement_filter_position(motion_platform, target_position: int) -> None:
    """Move the homed revolver to a plan slot using acknowledged shortest-path steps."""
    with porthandler.motion_lock:
        if not globals.toolhead_homed or not globals.filter_revolver_homed:
            raise RuntimeError('Home the motion platform and filter revolver before measurement.')
        globals.motion_busy = True
    try:
        _move_filter_revolver_to_position(motion_platform, target_position)
    finally:
        globals.motion_busy = False


def _live_camera_capture_values() -> dict:
    """Read the camera values that were actually active for a capture."""
    camera = globals.camera
    values = {
        'exposure_time': None,
        'gain': None,
        'gamma': None,
    }
    if not camera or not camera.IsOpen():
        return values

    for metadata_name, camera_name in (
        ('exposure_time', 'ExposureTime'),
        ('gain', 'Gain'),
        ('gamma', 'Gamma'),
    ):
        try:
            values[metadata_name] = getattr(camera, camera_name).GetValue()
        except Exception as error:
            app.logger.warning(
                'Could not read live camera metadata value %s: %s',
                camera_name, error,
            )
    return values


def _current_capture_metadata(
    light_type: str | None,
    filter_position: int | None = None,
    requested_metadata: dict | None = None,
) -> dict:
    """Snapshot all backend-owned metadata for the image being saved."""
    if filter_position is None:
        current_filter_position = getattr(
            globals, 'filter_revolver_position', None
        )
        if isinstance(current_filter_position, int) and not isinstance(
            current_filter_position, bool
        ):
            filter_position = current_filter_position

    return build_capture_metadata(
        settings=get_settings(),
        position=dict(getattr(globals, 'last_toolhead_pos', {}) or {}),
        wavelength=_canonical_light_channel(light_type),
        filter_position=filter_position,
        camera_values=_live_camera_capture_values(),
        requested_metadata=requested_metadata,
        errors=height_offset_control.capture_errors(_canonical_light_channel(light_type), filter_position),
    )


def _save_capture_jpeg(
    image_bgr: np.ndarray,
    full_path: str,
    light_type: str | None,
    filter_position: int | None = None,
    requested_metadata: dict | None = None,
    capture_metadata: dict | None = None,
) -> dict:
    """Save a BGR image and embed the complete capture snapshot as JSON EXIF."""
    metadata = capture_metadata if capture_metadata is not None else _current_capture_metadata(
        light_type,
        filter_position=filter_position,
        requested_metadata=requested_metadata,
    )
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(image_rgb)
    exif_obj = Image.Exif()
    exif_obj[0x010E] = serialize_capture_metadata(metadata)
    pil_img.save(full_path, format='JPEG', quality=95, exif=exif_obj)
    return metadata


def _capture_and_save_image(target_folder: str, filename: str, background_subtraction: bool = False,
                            light_type: str = None, filter_position: int | None = None,
                            prepared_frame: np.ndarray | None = None,
                            capture_metadata: dict | None = None,
                            on_image_saved=None) -> list:
    """Capture image from camera and save to target folder.
    
    If background_subtraction is True, also saves a masked version.
    Also caches the latest images in globals for the /api/latest_image endpoints.
    
    Returns:
        list of saved file paths (original, and optionally masked)
    """
    # Normalize path for cross-platform consistency
    target_folder = _normalize_path(target_folder)

    # Grab frame from camera
    frame_result = prepared_frame if prepared_frame is not None else _grab_owned_camera_frame()
    
    if isinstance(frame_result, tuple):
        frame = frame_result[0]
        if frame is None:
            raise RuntimeError("Failed to grab image from camera")
    else:
        frame = frame_result
        if frame is None:
            raise RuntimeError("Failed to grab image from camera")
    
    img_cv = np.asarray(frame)
    if not isinstance(img_cv, np.ndarray) or img_cv.ndim < 2:
        raise RuntimeError("Invalid image data from camera")
    globals.latest_image = img_cv
    
    full_path = os.path.join(target_folder, f"{filename}.jpg")
    
    _save_capture_jpeg(
        img_cv,
        full_path,
        light_type,
        filter_position=filter_position,
        capture_metadata=capture_metadata,
    )
    
    saved_paths = [full_path]
    if on_image_saved is not None:
        on_image_saved(full_path, False)
    
    _cache_latest_capture(light_type, img_cv)
    
    # Background subtraction: save masked version alongside original
    if background_subtraction:
        try:
            af_contour = getattr(globals, "last_autofocus_contour", None)
            mask, kind, metrics = bgr_main.make_object_mask_from_bgr_rel(img_cv, autofocus_contour = af_contour)
            if mask is not None and np.any(mask):
                masked = bgr_main.apply_mask_zero_background(img_cv, mask)
                masked_path = os.path.join(target_folder, f"{filename}_masked.jpg")
                bgr_main.save_bgr_image_keep_exif(
                    image_bgr=masked,
                    src_image_path=full_path,
                    dst_image_path=masked_path
                )
                saved_paths.append(masked_path)
                if on_image_saved is not None:
                    on_image_saved(masked_path, True)
                _cache_latest_capture(light_type, masked, masked=True)
                app.logger.info(f"Background-subtracted image saved: {masked_path} (kind={kind})")
            else:
                app.logger.warning(f"Background subtraction found no object in {filename}")
        except Exception as e:
            app.logger.warning(f"Background subtraction failed for {filename}: {e}")
    
    return saved_paths


def _grab_owned_camera_frame(timeout_ms=5000, retries=2):
    """Grab for measurement/capture without allowing preview queue competition."""
    from cameracontrol import grab_and_convert_frame, suppress_preview_grabs
    cam = globals.camera
    if cam is None or not cam.IsOpen():
        raise RuntimeError('Camera not ready')
    with suppress_preview_grabs():
        with globals.grab_lock:
            return grab_and_convert_frame(cam, timeout_ms=timeout_ms, retries=retries)


def _canonical_light_channel(light_type):
    """Translate temporary legacy names without inferring arbitrary UV wavelengths."""
    return {'dome': 'vis', 'bar': 'uv365'}.get(light_type, light_type)


def _cache_latest_capture(light_type, image_bgr, masked=False):
    channel = _canonical_light_channel(light_type)
    if channel not in globals.latest_images:
        return
    globals.latest_images[channel]['masked' if masked else 'original'] = image_bgr.copy()


def _wait_for_motion_complete(ser, timeout=30.0):
    """Wait for motion to complete using M400 command via safe write_and_wait.
    
    M400 tells the board to finish all buffered moves before responding 'ok'.
    This guarantees the toolhead has physically stopped.
    
    Args:
        ser: serial port object
        timeout: max seconds to wait (default 30s covers long moves)
    
    Returns:
        True if motion completed, False on timeout.
    
    Raises:
        OSError / PermissionError: if USB is disconnected (caller should handle)
    """
    try:
        return porthandler.write_and_wait_motion(ser, "M400", timeout=timeout)
    except (OSError, PermissionError):
        raise  # let caller handle USB disconnect
    except Exception as e:
        app.logger.warning(f"_wait_for_motion_complete error: {e}")
        return False


def _check_devices_connected():
    """
    Check if both motion platform and camera are connected.
    Returns (motion_platform, camera, error_response, status_code).
    If error_response is not None, return it directly from the endpoint.
    """
    motion_platform = globals.motion_platform
    if not motion_platform or not getattr(motion_platform, 'is_open', False):
        return None, None, jsonify({
            'status': 'error',
            'message': 'Motion platform not connected',
            'code': ErrorCode.MOTIONPLATFORM_DISCONNECTED,
            'popup': True
        }), 503
        
    camera = globals.camera
    if not camera or not camera.IsOpen():
        return None, None, jsonify({
            'status': 'error',
            'message': 'Camera not connected',
            'code': ErrorCode.CAMERA_DISCONNECTED,
            'popup': True
        }), 503
    
    return motion_platform, camera, None, None


def _bgr_capture_result(status: str, series_index: int, saved_images: list[dict]):
    """Build the common completed/cancelled BGR series response."""
    camera_params = bgr_capture_coordinator.status().get('camera_params')
    return jsonify({
        'status': status,
        'series_index': series_index,
        'saved_images': saved_images,
        'camera_params': camera_params,
    }), 200


@app.route('/api/bgr-capture-series', methods=['POST'])
def capture_bgr_filter_series():
    """Capture RGB+UV for each requested wavelength, in request order."""
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or not isinstance(data.get('capture_id', ''), str) or len(data.get('capture_id', '')) > 128:
        return jsonify({'error': 'Invalid capture request.', 'code': ErrorCode.GENERIC, 'popup': True}), 400
    if not bgr_capture_coordinator.begin(data.get('capture_id')):
        return jsonify({
            'error': 'A BGR capture series is already running.',
            'code': ErrorCode.BGR_CAPTURE_BUSY,
            'popup': True,
        }), 409

    owns_motion = False
    saved_images = []
    series_index = 1
    selected_channel = None
    selected_mode = None
    wavelength_qualified_filenames = False
    series_completed = False
    try:
        data = request.get_json(silent=True) or {}
        raw_wavelengths = data.get('wavelengths')
        if raw_wavelengths is not None:
            if (
                not isinstance(raw_wavelengths, list)
                or not 1 <= len(raw_wavelengths) <= 4
                or any(channel not in LIGHT_CHANNELS for channel in raw_wavelengths)
                or len(set(raw_wavelengths)) != len(raw_wavelengths)
            ):
                return jsonify({
                    'error': 'Select between one and four unique, valid wavelengths.',
                    'code': ErrorCode.BGR_WAVELENGTH_SELECTION_INVALID,
                    'popup': True,
                }), 400
            capture_channels = raw_wavelengths
            wavelength_qualified_filenames = True
        else:
            # Temporary compatibility for older renderer builds.
            if data.get('mode', 'rgb') not in ('rgb', 'uv_rgb'):
                return jsonify({'error': 'Capture mode must be rgb or uv_rgb.', 'code': ErrorCode.GENERIC, 'popup': True}), 400
            capture_channels = None
        mode = data.get('mode', 'rgb')
        raw_target_folder = data.get('target_folder', '')
        target_folder = (
            _normalize_path(raw_target_folder).strip()
            if isinstance(raw_target_folder, str)
            else ''
        )
        if not target_folder or not os.path.isdir(target_folder):
            return jsonify({
                'error': 'The configured save location does not exist or is not a folder.',
                'code': ErrorCode.BGR_SAVE_FAILED,
                'popup': True,
            }), 400

        try:
            stem = capture_series_stem(target_folder)
        except (OSError, ValueError) as error:
            app.logger.warning('BGR capture save location rejected: %s', error)
            return jsonify({
                'error': str(error),
                'code': ErrorCode.BGR_SAVE_FAILED,
                'popup': True,
            }), 400

        motion_platform, _camera, error_response, error_status = _check_devices_connected()
        if error_response:
            return error_response, error_status
        if not globals.toolhead_homed or not globals.filter_revolver_homed:
            return jsonify({
                'error': 'Home the motion platform and filter revolver before BGR capture.',
                'code': ErrorCode.BGR_CAPTURE_NOT_READY,
                'popup': True,
            }), 409

        with porthandler.motion_lock:
            if globals.motion_busy:
                return jsonify({
                    'error': 'The motion platform is busy with another operation.',
                    'code': ErrorCode.BGR_CAPTURE_BUSY,
                    'popup': True,
                }), 409
            if capture_channels is None:
                selected_light = light_controller.status()
                selected_channel = selected_light['active_channel']
                selected_mode = selected_light.get('active_mode')
                if selected_channel is None:
                    return jsonify({'error': 'Activate a lamp before capture.', 'code': ErrorCode.CAPTURE_LIGHT_REQUIRED, 'popup': True}), 409
            globals.motion_busy = True
            owns_motion = True

        try:
            filter_settings_data = _current_filter_settings()
            capture_steps = []
            if capture_channels is None:
                capture_steps.extend(
                    (selected_channel, selected_mode, target)
                    for target in resolve_filter_targets(filter_settings_data, mode, selected_channel)
                )
            else:
                for channel in capture_channels:
                    channel_mode = 'dimmed' if channel in UV_LAMP_CHANNELS else None
                    channel_capture_mode = 'rgb' if channel == 'vis' else 'uv_rgb'
                    capture_steps.extend(
                        (channel, channel_mode, target)
                        for target in resolve_filter_targets(
                            filter_settings_data, channel_capture_mode, channel)
                    )
        except ValueError as error:
            return jsonify({'error': str(error), 'code': ErrorCode.BGR_FILTER_CONFIGURATION_INVALID, 'popup': True}), 400

        if not height_offset_control.status()['available']:
            if bgr_capture_coordinator.cancellation_requested():
                return _bgr_capture_result('cancelled', series_index, saved_images)
            bgr_capture_coordinator.set_autofocus_in_progress(True)
            try:
                if bgr_capture_coordinator.cancellation_requested():
                    return _bgr_capture_result('cancelled', series_index, saved_images)
                autofocus_response = _run_configured_manual_autofocus(
                    motion_platform,
                    skip_empty_check=True,
                )
            except filter_revolver.FilterRevolverCommandError as error:
                globals.homed_axes.discard('a')
                globals.filter_revolver_homed = False
                globals.filter_revolver_position = None
                height_offset_control.invalidate_reference()
                app.logger.warning('Initial BGR autofocus filter movement failed: %s', error)
                return jsonify({
                    'error': str(error),
                    'code': ErrorCode.BGR_CAPTURE_NOT_READY,
                    'popup': True,
                }), 504
            except (LampSettingsError, LightConfigurationError, ValueError) as error:
                height_offset_control.invalidate_reference()
                app.logger.warning('Initial BGR autofocus configuration failed: %s', error)
                return jsonify({
                    'error': str(error),
                    'code': ErrorCode.BGR_CAPTURE_NOT_READY,
                    'popup': True,
                }), 400
            except LightCommandError as error:
                height_offset_control.invalidate_reference()
                return jsonify({
                    'error': str(error),
                    'code': ErrorCode.MOTIONPLATFORM_DISCONNECTED,
                    'popup': True,
                }), 503
            except (OSError, PermissionError):
                height_offset_control.invalidate_reference()
                return _handle_motion_usb_disconnect(
                    motion_platform,
                    'initial BGR autofocus',
                )
            finally:
                bgr_capture_coordinator.set_autofocus_in_progress(False)

            if bgr_capture_coordinator.cancellation_requested():
                return _bgr_capture_result('cancelled', series_index, saved_images)
            if autofocus_response.get('status') != 'OK':
                error_code = autofocus_response.get('code') or ErrorCode.GENERIC
                return jsonify({
                    'error': autofocus_response.get('message') or 'Autofocus failed.',
                    'code': error_code,
                    'popup': True,
                }), 422

        try:
            # Re-evaluate after autofocus because it can be a long-running operation.
            series_index = next_capture_series_index(target_folder, stem)
        except OSError as error:
            app.logger.warning('BGR capture save location became unavailable: %s', error)
            return jsonify({
                'error': str(error),
                'code': ErrorCode.BGR_SAVE_FAILED,
                'popup': True,
            }), 400

        for selected_channel, selected_mode, target in capture_steps:
            if bgr_capture_coordinator.cancellation_requested():
                return _bgr_capture_result('cancelled', series_index, saved_images)

            try:
                light_controller.off()
                _move_filter_revolver_to_position(motion_platform, target.position)
                height_offset = height_offset_control.apply_active_combination(
                    motion_platform, filter_settings_data, selected_channel,
                )
                if not height_offset['applied']:
                    raise HeightOffsetCommandError('A valid Z reference is required for capture.')
            except filter_revolver.FilterRevolverCommandError as error:
                globals.homed_axes.discard('a')
                globals.filter_revolver_homed = False
                globals.filter_revolver_position = None
                app.logger.warning('BGR capture filter movement failed: %s', error)
                return jsonify({
                    'error': str(error),
                    'code': ErrorCode.BGR_CAPTURE_NOT_READY,
                    'popup': True,
                }), 504
            except ValueError as error:
                app.logger.warning('BGR capture filter position is invalid: %s', error)
                return jsonify({
                    'error': str(error),
                    'code': ErrorCode.BGR_CAPTURE_NOT_READY,
                    'popup': True,
                }), 409
            except HeightOffsetCommandError as error:
                app.logger.warning('BGR capture height correction failed: %s', error)
                return jsonify({
                    'error': str(error),
                    'code': ErrorCode.MOTION_HEIGHT_OFFSET_FAILED,
                    'popup': True,
                }), 422
            except (OSError, PermissionError):
                return _handle_motion_usb_disconnect(
                    motion_platform,
                    f'BGR capture filter {target.name}',
                )

            # A selected filter and its Z correction form one safe, indivisible step.
            capture_camera_params = _apply_selected_camera_combination(
                selected_channel, target.position)
            bgr_capture_coordinator.record_camera_params(capture_camera_params)
            if bgr_capture_coordinator.cancellation_requested():
                return _bgr_capture_result('cancelled', series_index, saved_images)
            # M400 confirms command completion. This quiet interval also lets
            # the revolver, Z stage and camera settings settle before the lamp
            # is activated and exposure begins.
            if bgr_capture_coordinator.wait_for_cancellation(BGR_HARDWARE_SETTLE_SECONDS):
                return _bgr_capture_result('cancelled', series_index, saved_images)

            filename = capture_filename(
                stem,
                series_index,
                target.suffix,
                selected_channel if wavelength_qualified_filenames else None,
            )
            try:
                frame = None
                for capture_attempt in range(2):
                    try:
                        frame = capture_series_frame(
                            _camera,
                            light_controller,
                            selected_channel,
                            selected_mode,
                            bgr_capture_coordinator,
                        )
                        break
                    except CaptureIlluminationError:
                        raise
                    except RuntimeError as error:
                        camera_alive = bool(_camera and _camera.IsOpen())
                        if capture_attempt or not camera_alive or bgr_capture_coordinator.cancellation_requested():
                            raise
                        app.logger.warning(
                            'Transient BGR camera grab failed for %s; retrying once: %s',
                            filename,
                            error,
                        )
                        if bgr_capture_coordinator.wait_for_cancellation(BGR_CAMERA_RETRY_SETTLE_SECONDS):
                            return _bgr_capture_result('cancelled', series_index, saved_images)
                if frame is None or bgr_capture_coordinator.cancellation_requested():
                    return _bgr_capture_result('cancelled', series_index, saved_images)
                capture_metadata = _current_capture_metadata(selected_channel, target.position)
                paths = _capture_and_save_image(
                    target_folder,
                    filename,
                    background_subtraction=False,
                    light_type=selected_channel,
                    filter_position=target.position,
                    prepared_frame=frame,
                    capture_metadata=capture_metadata,
                )
            except (CaptureIlluminationError, LightCommandError):
                raise
            except (OSError, PermissionError) as error:
                app.logger.warning('BGR image save failed for %s: %s', filename, error)
                return jsonify({
                    'error': str(error),
                    'code': ErrorCode.BGR_SAVE_FAILED,
                    'popup': True,
                }), 500
            except RuntimeError as error:
                app.logger.warning('BGR camera capture failed for %s: %s', filename, error)
                return jsonify({
                    'error': str(error),
                    'code': ErrorCode.CAMERA_DISCONNECTED,
                    'popup': True,
                }), 503

            saved_images.append({
                'filter_name': target.name,
                'suffix': target.suffix,
                'filter_position': target.position,
                'path': paths[0],
                'height_offset': height_offset,
                'wavelength': selected_channel,
                'metadata': capture_metadata,
            })
            bgr_capture_coordinator.record_image(saved_images[-1])

        if bgr_capture_coordinator.cancellation_requested():
            return _bgr_capture_result('cancelled', series_index, saved_images)
        if len(saved_images) != len(capture_steps):
            raise RuntimeError('The filter capture series did not save every required image.')
        series_completed = True
        return _bgr_capture_result('completed', series_index, saved_images)
    except CaptureIlluminationError as error:
        return jsonify({'error': str(error), 'code': ErrorCode.CAPTURE_LIGHT_TIMEOUT, 'popup': True}), 422
    except LightConfigurationError as error:
        return jsonify({'error': str(error), 'code': ErrorCode.BGR_CAPTURE_NOT_READY, 'popup': True}), 400
    except LightCommandError as error:
        return jsonify({'error': str(error), 'code': ErrorCode.MOTIONPLATFORM_DISCONNECTED, 'popup': True}), 503
    except Exception as error:
        app.logger.exception('Unexpected BGR capture series failure')
        return jsonify({
            'error': str(error),
            'code': ErrorCode.GENERIC,
            'popup': True,
        }), 500
    finally:
        try:
            _turn_off_all_lights()
        except Exception as off_error:
            app.logger.warning(
                'Could not switch off lights after filter capture: %s',
                off_error,
            )
        if series_completed:
            try:
                _select_blue_filter_for_vis(motion_platform)
                _apply_selected_height_offset(motion_platform)
                _apply_selected_camera_combination('vis')
                light_controller.activate('vis')
            except Exception as vis_error:
                app.logger.warning(
                    'Could not restore VIS with the Blue filter after completed filter capture: %s',
                    vis_error,
                )
        if owns_motion:
            globals.motion_busy = False
            globals.autofocus_abort = False
        bgr_capture_coordinator.finish()


@app.route('/api/bgr-capture-series/cancel', methods=['POST'])
def cancel_bgr_filter_series():
    """Request cooperative cancellation after the current safe hardware step."""
    requested = bgr_capture_coordinator.request_cancel()
    if requested and bgr_capture_coordinator.autofocus_in_progress():
        globals.autofocus_abort = True
    return jsonify({
        'status': 'cancellation_requested' if requested else 'idle',
    }), 200


@app.route('/api/bgr-capture-series/status', methods=['GET'])
def filter_capture_status():
    return jsonify(bgr_capture_coordinator.status())


@app.after_request
def add_capture_warnings(response):
    """Expose nonfatal motion warnings consistently to every API consumer."""
    if response.is_json:
        data = response.get_json()
        if isinstance(data, dict):
            changed = False
            if request.path == '/api/bgr-capture-series' and response.status_code >= 400 and response.status_code != 409:
                capture_status = bgr_capture_coordinator.status()
                payload = request.get_json(silent=True) or {}
                if (isinstance(payload, dict) and payload.get('capture_id')
                        and payload.get('capture_id') == capture_status['capture_id']):
                    data['saved_images'] = capture_status['saved_images']
                    changed = True
            applications = [data.get('height_offset', {})]
            applications.extend(item.get('height_offset', {}) for item in (data.get('saved_images') or []) if isinstance(item, dict))
            warnings = [item['warning'] for item in applications if isinstance(item, dict) and item.get('warning')]
            if warnings:
                data['warnings'] = warnings
                changed = True
            if changed:
                response.set_data(app.json.dumps(data))
    return response


def _format_capture_timestamp(dt: datetime) -> str:
    return dt.strftime("%m%d_%H%M%S")


def _tablet_index_to_label(tablet_index: int, grid_size: int = 10) -> str:
    try:
        index = int(tablet_index) - 1
    except (TypeError, ValueError):
        return str(tablet_index)

    if index < 0 or index >= grid_size * grid_size:
        return str(tablet_index)

    col = index % grid_size
    row = (index // grid_size) + 1
    letter = chr(65 + col)
    return f"{letter}{row}"


def _capture_image_with_light(light_type: str, measurement_folder: str, measurement_name: str, tablet_index: int, background_subtraction: bool = False) -> list:
    """
    Turn on specified light, apply camera settings, capture and save image.
    Returns list of saved file paths (original + masked if background_subtraction is on).
    
    Args:
        light_type: 'dome' or 'bar'
        measurement_folder: Directory to save image
        measurement_name: Name prefix for the image
        tablet_index: Tablet number for filename
        background_subtraction: If True, also save background-subtracted image
    """
    if light_type == 'dome':
        _turn_on_dome_light()
    else:
        _turn_on_uv_dome_light()
    
    _apply_camera_settings_for_light(light_type)
    time.sleep(0.3)  # Let light and camera settings stabilize
    
    timestamp = _format_capture_timestamp(datetime.now())
    tablet_label = _tablet_index_to_label(tablet_index)
    filename = f"{measurement_name}_{timestamp}_{tablet_label}_{light_type}"
    
    return _capture_and_save_image(measurement_folder, filename, background_subtraction=background_subtraction, light_type=light_type)


def _capture_capture_plan_row(row: dict, measurement_folder: str, measurement_name: str,
                              tablet_index: int, background_subtraction: bool = False,
                              on_image_saved=None) -> list:
    """Capture one persisted wavelength/filter row with the Octopus controller.

    The caller positions the filter revolver before this function enables the
    selected lamp channel.
    """
    wavelength = row['wavelength']
    filter_position = row['filter_position']
    mode = row['brightness'] if wavelength in ('uv255', 'uv310', 'uv365') else None
    activated = False
    try:
        _apply_capture_plan_camera_settings(row)
        light_controller.activate(wavelength, mode)
        activated = True
        time.sleep(0.3)

        timestamp = _format_capture_timestamp(datetime.now())
        tablet_label = _tablet_index_to_label(tablet_index)
        filename = f"{measurement_name}_{timestamp}_{tablet_label}_{wavelength}_filter{filter_position}"
        return _capture_and_save_image(
            measurement_folder, filename,
            background_subtraction=background_subtraction,
            light_type=wavelength,
            filter_position=filter_position,
            on_image_saved=on_image_saved,
        )
    finally:
        if activated:
            try:
                light_controller.off(wavelength)
            except Exception as error:
                app.logger.error('Could not switch off %s after capture: %s', wavelength, error)


def _measurement_capture_rows(
    capture_plan: list[dict], settings: dict, should_autofocus: bool
) -> list[tuple[int, dict]]:
    """Return indexed capture rows, treating row zero as autofocus-only."""
    auto_settings = settings.get('auto_measurement_settings', {})
    save_autofocus_image = (
        auto_settings.get('save_autofocus_image', DEFAULT_SAVE_AUTOFOCUS_IMAGE)
        if isinstance(auto_settings, dict)
        else DEFAULT_SAVE_AUTOFOCUS_IMAGE
    )
    first_capture_index = 0 if save_autofocus_image and should_autofocus else 1
    return list(enumerate(capture_plan))[first_capture_index:]


def _check_tablet_presence_enabled(settings: dict) -> bool:
    auto_settings = settings.get('auto_measurement_settings', {})
    return (
        auto_settings.get('check_tablet_presence', DEFAULT_CHECK_TABLET_PRESENCE)
        if isinstance(auto_settings, dict)
        else DEFAULT_CHECK_TABLET_PRESENCE
    )


def _prepare_tablet_presence_check(
    motion_platform,
    capture_plan: list[dict],
) -> None:
    """Restore the first-tablet optical conditions before a presence frame."""
    if not capture_plan:
        raise ValueError('A capture plan is required for the tablet-presence check.')
    autofocus_settings, filter_settings = _select_configured_autofocus_hardware(
        motion_platform,
        manage_motion_busy=True,
        capture_plan_row=capture_plan[0],
    )
    height_offset_control.apply_active_combination(
        motion_platform,
        filter_settings,
        autofocus_settings['channel'],
    )
    time.sleep(0.3)


@app.route('/api/auto_measurement/progress', methods=['GET'])
def auto_measurement_progress():
    request_id = request.args.get('request_id', '')
    progress = measurement_progress.snapshot(request_id)
    if progress is None:
        # Flask can schedule this GET before the concurrent step POST has
        # registered its progress record. This is a normal startup state, not
        # an operator-facing failure.
        return jsonify(measurement_progress.pending_snapshot(request_id)), 200
    return jsonify(progress), 200


@app.route('/api/auto_measurement/step', methods=['POST'])
def auto_measurement_step():
    """
    Process a single tablet in the auto-measurement sequence.
    This endpoint is called repeatedly by the frontend for each tablet.
    
    Sequence:
      1. Validate parameters and check devices
      2. Move to tablet X/Y → M400 wait → settle
      3. Autofocus (if enabled): coarse for first tablet, fine for subsequent
         → M400 wait → settle
      4. Capture images with selected lights (dome / bar)
      5. Turn off lights, return saved image paths
    
    Every serial command waits for board acknowledgement ('ok') before
    proceeding to ensure the BTT SKR Mini E3 is never overwhelmed.
    """
    request_id = None
    progress_outcome = 'failed'
    try:
        # Reset abort flag for each new tablet
        if globals.autofocus_abort:
            globals.autofocus_abort = False
            app.logger.info("Autofocus abort flag cleared for new tablet")
        
        data = request.get_json() or {}
        request_id = data.get('request_id')
        try:
            measurement_progress.start(request_id)
        except ValueError as error:
            return jsonify({'status': 'error', 'message': str(error)}), 400
        
        # ---------- Parse & validate parameters ----------
        tablet_index = data.get('tablet_index')
        x_pos = data.get('x')
        y_pos = data.get('y')
        z_pos = data.get('z', 20.0)
        measurement_folder = _normalize_path(data.get('measurement_folder', ''))
        measurement_name = data.get('measurement_name')
        
        autofocus_enabled = bool(data.get('autofocus', False))
        lamp_top = bool(data.get('lamp_top', False))
        lamp_side = bool(data.get('lamp_side', False))
        capture_plan = None
        if 'capture_plan' in data:
            try:
                capture_plan = validate_capture_plan(data['capture_plan'])
                _validate_capture_plan_camera_values(capture_plan)
            except ValueError as error:
                return jsonify({'status': 'error', 'message': str(error)}), 400
        is_first_tablet = bool(data.get('is_first_tablet', False))
        background_subtraction = bool(data.get('background_subtraction', False))
        
        if tablet_index is None or x_pos is None or y_pos is None:
            return jsonify({
                'status': 'error',
                'message': 'Missing required parameters (tablet_index, x, y)'
            }), 400
            
        if not measurement_folder or not measurement_name:
            return jsonify({
                'status': 'error',
                'message': 'Missing measurement_folder or measurement_name'
            }), 400
        
        if capture_plan is None:
            return jsonify({
                'status': 'error',
                'message': 'capture_plan is required. Legacy lamp_top/lamp_side requests are no longer supported.'
            }), 400
        
        # ---------- Check devices ----------
        motion_platform, camera, err_response, err_status = _check_devices_connected()
        if err_response:
            return err_response, err_status

        # Ensure folder exists
        try:
            os.makedirs(measurement_folder, exist_ok=True)
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': f'Could not create measurement folder: {e}'
            }), 400
        
        saved_images = []
        captured_plan_rows = []
        
        # =====================================================
        # STEP 1: Move to tablet position
        # =====================================================
        # If autofocus is disabled but it's the first tablet, we still move to z_pos (will autofocus)
        # If autofocus is disabled and not first tablet, we only move XY (keep Z constant)
        if autofocus_enabled or is_first_tablet:
            # Move X, Y, and Z
            move_x = float(x_pos)
            move_y = float(y_pos)
            move_z = float(z_pos)
        else:
            # Move only X and Y; leave Z unchanged
            move_x = float(x_pos)
            move_y = float(y_pos)
            move_z = None
        
        app.logger.info(f"Tablet {tablet_index}: Moving to X={move_x}, Y={move_y}, Z={move_z if move_z else 'unchanged'}")
        resp, status = _move_toolhead_absolute_impl(
            x_pos=move_x,
            y_pos=move_y,
            z_pos=move_z,
            preserve_height_reference=move_z is None,
        )
        if status != 200:
            _turn_off_all_lights()
            # If the move failed due to a device disconnect (503), propagate
            # the error code at the top level so the frontend can detect it
            # and trigger the 30-second reconnection timer.
            if status == 503 and resp.get('code'):
                return jsonify({
                    'error': resp.get('message', 'Device disconnected'),
                    'code': resp['code'],
                    'popup': True
                }), 503
            return jsonify({
                'status': 'error',
                'message': f"Move failed for tablet {tablet_index}",
                'move_response': resp
            }), status
        
        # =====================================================
        # STEP 2: Wait for motion platform to CONFIRM completion
        # =====================================================
        try:
            motion_ok = _wait_for_motion_complete(motion_platform, timeout=30.0)
            if not motion_ok:
                app.logger.warning(f"Tablet {tablet_index}: M400 timed out after move, proceeding cautiously")
        except (OSError, PermissionError) as e:
            return _handle_motion_usb_disconnect(motion_platform, f"move to tablet {tablet_index}")
        
        # Settle time: wait for vibrations to stop so camera gets a still image
        time.sleep(0.2)
        
        # =====================================================
        # STEP 3: Autofocus
        # =====================================================
        # Run autofocus if:
        #   - autofocus_enabled is True (user requested it), OR
        #   - autofocus_enabled is False BUT is_first_tablet is True (find focal plane once, then keep Z constant)
        should_autofocus = autofocus_enabled or is_first_tablet
        af_error_code = None  # Track AF error codes (E2000/E2002/E2003) for the response
        
        if should_autofocus:
            measurement_progress.set_active_plan_row(request_id, 0)
            app.logger.info(f"Tablet {tablet_index}: Starting autofocus ({'coarse' if is_first_tablet else 'fine'})")
            
            # Autofocus uses its dedicated saved light/filter selection. The
            # first plan row still supplies the operator-selected camera values.
            autofocus_settings, _ = _select_configured_autofocus_hardware(
                motion_platform,
                manage_motion_busy=True,
                capture_plan_row=capture_plan[0] if capture_plan is not None else None,
            )
            height_offset_control.invalidate_reference()
            if capture_plan is None:
                _apply_camera_settings_for_light('dome')
            app.logger.info(
                'Tablet %s: Autofocus using %s/%s with filter position %s',
                tablet_index,
                autofocus_settings['channel'],
                autofocus_settings['brightness'],
                autofocus_settings['filter_position'],
            )
            time.sleep(0.3)  # Let light and camera settings stabilize
            
            if autofocus_enabled:
                # ---- User selected autofocus ----
                # For the first tablet, calculate reference color values before autofocus
                if is_first_tablet:
                    try:
                        color_result = calc_color.calculate_median_and_span(
                            motion_platform, needed=True
                        )
                        if color_result.get('status') == 'OK':
                            ref = {k: v for k, v in color_result.items() if k != 'status'}
                            globals.color_values = ref
                            app.logger.info(f"Tablet {tablet_index}: Reference color values stored: {ref}")
                        else:
                            app.logger.warning(f"Tablet {tablet_index}: calc_color returned {color_result}")
                    except Exception as e:
                        app.logger.warning(f"Tablet {tablet_index}: calc_color error: {e}")
            
            try:
                if autofocus_enabled:
                    # Autofocus selected: run with before_auto=True for every tablet
                    af_result = autofocus_main.autofocus_coarse(
                        motion_platform, do_frame_touch_check=True, before_auto=True, debug=False
                    )
                elif is_first_tablet:
                    # Autofocus unselected, first tablet: find focal plane with skip_empty_check + no color check
                    af_result = autofocus_main.autofocus_coarse(
                        motion_platform, skip_empty_check=True, before_auto=False, debug=False
                    )
                
                af_status = af_result.get('status', 'ERROR')
                af_error_code = af_result.get('code')  # e.g. "E2000", "E2002", "E2003"
                if af_status == 'OK':
                    contour = af_result.get("final_contour") or af_result.get("contour")
                    globals.last_autofocus_contour = contour if contour else None
                    filter_settings = _current_filter_settings()
                    configured_offset = height_offset_control.configured_offset(
                        filter_settings,
                        autofocus_settings['filter_position'],
                        autofocus_settings['channel'],
                    )
                    height_offset_control.record_combination_reference(
                        getattr(globals, 'last_toolhead_pos', {}).get('z'),
                        configured_offset,
                        source='autofocus',
                    )
                    app.logger.info(f"Tablet {tablet_index}: Autofocus OK at Z={af_result.get('z_rel', '?')}")
                    
                    # When autofocus is unselected and first tablet just finished AF,
                    # calculate reference color values (camera already at focused Z)
                    if not autofocus_enabled and is_first_tablet:
                        try:
                            color_result = calc_color.calculate_median_and_span(
                                motion_platform, needed=False
                            )
                            if color_result.get('status') == 'OK':
                                ref = {k: v for k, v in color_result.items() if k != 'status'}
                                globals.color_values = ref
                                app.logger.info(f"Tablet {tablet_index}: Post-AF reference color values stored: {ref}")
                            else:
                                app.logger.warning(f"Tablet {tablet_index}: calc_color (post-AF) returned {color_result}")
                        except Exception as e:
                            app.logger.warning(f"Tablet {tablet_index}: calc_color (post-AF) error: {e}")
                elif af_status == 'ABORTED':
                    app.logger.info(f"Tablet {tablet_index}: Autofocus aborted")
                    _turn_off_all_lights()
                    return jsonify({
                        'status': 'error',
                        'message': f'Autofocus aborted for tablet {tablet_index}'
                    }), 200  # Not a server error; user stopped it
                else:
                    globals.last_autofocus_contour = None
                    app.logger.warning(f"Tablet {tablet_index}: Autofocus returned {af_status}: {af_result}")
                    # For any E2xxx tablet/quality errors, skip image capture entirely
                    if af_error_code and af_error_code.startswith('E2'):
                        app.logger.info(f"Tablet {tablet_index}: Skipping image capture (AF error {af_error_code})")
                        _turn_off_all_lights()
                        response_data = {
                            'status': 'success',
                            'tablet_index': tablet_index,
                            'saved_images': [],
                            'af_error_code': af_error_code,
                            'af_error_message': ERROR_MESSAGES.get(af_error_code, af_error_code)
                        }
                        return jsonify(response_data), 200
                    # For other AF errors, continue with image capture — the focus may still be acceptable
            except (OSError, PermissionError) as e:
                return _handle_motion_usb_disconnect(motion_platform, f"autofocus tablet {tablet_index}")
            except Exception as e:
                if _is_camera_disconnect(e):
                    _turn_off_all_lights()
                    return _handle_camera_disconnect(f"autofocus tablet {tablet_index}")
                app.logger.warning(f"Tablet {tablet_index}: Autofocus error: {e}")
                # Continue — autofocus failure should not block the measurement
            
            # Wait for autofocus motion to fully stop
            try:
                _wait_for_motion_complete(motion_platform, timeout=30.0)
            except (OSError, PermissionError) as e:
                return _handle_motion_usb_disconnect(motion_platform, f"post-autofocus tablet {tablet_index}")
            
            # Settle time after autofocus Z movements
            time.sleep(0)
            measurement_progress.set_active_plan_row(request_id, None)
        
        # =====================================================
        # STEP 3b: Manual contour detection (no AF, BGR on)
        # =====================================================
        # When autofocus did NOT run for this tablet but background
        # subtraction is enabled, we still need a fresh tablet contour.
        # The dome light produces a reliable outline, so we always use
        # dome illumination + manual_bgr even if only the bar light
        # is selected for the actual measurement image.
        if not should_autofocus and background_subtraction:
            app.logger.info(f"Tablet {tablet_index}: Getting contour via manual_bgr_with_check (no autofocus)")
            _turn_on_dome_light()
            _apply_camera_settings_for_light('dome')
            time.sleep(0.3)  # Let light and camera settings stabilize

            try:
                # --- Pre-check: run check_only before manual_bgr_with_check ---
                check_frame = _grab_owned_camera_frame(timeout_ms=5000, retries=2)

                # Greyscale difference score
                gds_result = check_only.grayscale_difference_score(check_frame)
                gds_status = gds_result.get('status', 'ERROR')
                gds_code = gds_result.get('code', '')
                if gds_status != 'OK':
                    app.logger.warning(
                        f"Tablet {tablet_index}: check_only grayscale_difference_score "
                        f"returned {gds_status} ({gds_code}) before manual_bgr"
                    )
                    if gds_code and gds_code.startswith('E2'):
                        app.logger.info(
                            f"Tablet {tablet_index}: Skipping manual_bgr + image capture "
                            f"(check_only error {gds_code})"
                        )
                        _turn_off_all_lights()
                        return jsonify({
                            'status': 'success',
                            'tablet_index': tablet_index,
                            'saved_images': [],
                            'af_error_code': gds_code,
                            'af_error_message': ERROR_MESSAGES.get(gds_code, gds_code)
                        }), 200

                # Final out-of-frame check
                oof_result = check_only.final_out_of_frame_check(check_frame)
                oof_status = oof_result.get('status', 'ERROR')
                oof_code = oof_result.get('code', '')
                if oof_status != 'OK':
                    app.logger.warning(
                        f"Tablet {tablet_index}: check_only final_out_of_frame_check "
                        f"returned {oof_status} ({oof_code}) before manual_bgr"
                    )
                    if oof_code and oof_code.startswith('E2'):
                        app.logger.info(
                            f"Tablet {tablet_index}: Skipping manual_bgr + image capture "
                            f"(check_only error {oof_code})"
                        )
                        _turn_off_all_lights()
                        return jsonify({
                            'status': 'success',
                            'tablet_index': tablet_index,
                            'saved_images': [],
                            'af_error_code': oof_code,
                            'af_error_message': ERROR_MESSAGES.get(oof_code, oof_code)
                        }), 200

                app.logger.info(f"Tablet {tablet_index}: check_only passed, proceeding with manual_bgr")

                # --- Pre-check passed: run manual_bgr_with_check ---
                mbgr_result = manual_bgr_with_check.manual_return()
                mbgr_status = mbgr_result.get('status', 'ERROR')
                if mbgr_status == 'OK':
                    contour = mbgr_result.get('final_contour')
                    globals.last_autofocus_contour = contour if contour else None
                    app.logger.info(f"Tablet {tablet_index}: manual_bgr contour obtained")
                else:
                    globals.last_autofocus_contour = None
                    mbgr_code = mbgr_result.get('code', '')
                    app.logger.warning(
                        f"Tablet {tablet_index}: manual_bgr returned {mbgr_status} ({mbgr_code})"
                    )
                    # Frame-quality errors -> skip image capture for this tablet
                    if mbgr_code and mbgr_code.startswith('E2'):
                        app.logger.info(
                            f"Tablet {tablet_index}: Skipping image capture (manual_bgr error {mbgr_code})"
                        )
                        _turn_off_all_lights()
                        return jsonify({
                            'status': 'success',
                            'tablet_index': tablet_index,
                            'saved_images': [],
                            'af_error_code': mbgr_code,
                            'af_error_message': ERROR_MESSAGES.get(mbgr_code, mbgr_code)
                        }), 200
            except (OSError, PermissionError) as e:
                _turn_off_all_lights()
                return _handle_motion_usb_disconnect(
                    motion_platform, f"manual_bgr tablet {tablet_index}"
                )
            except Exception as e:
                if _is_camera_disconnect(e):
                    _turn_off_all_lights()
                    return _handle_camera_disconnect(f"manual_bgr tablet {tablet_index}")
                globals.last_autofocus_contour = None
                app.logger.warning(f"Tablet {tablet_index}: manual_bgr failed: {e}")
                # Continue — capture images without background subtraction

        # =====================================================
        # STEP 3c: Tablet presence check (no AF, no BGR)
        # =====================================================
        # When neither autofocus nor background subtraction ran,
        # we still need to verify that a tablet is present and
        # correctly positioned. Use check_only's greyscale
        # difference score and out-of-frame detection.
        if (
            not should_autofocus
            and not background_subtraction
            and _check_tablet_presence_enabled(get_settings())
        ):
            measurement_progress.set_active_plan_row(request_id, 0)
            app.logger.info(f"Tablet {tablet_index}: Running check_only (no AF, no BGR)")
            # Re-select the same acknowledged filter/light pair and camera
            # values used to build the first-tablet reference. The previous
            # capture row may have left a UV filter selected, which makes a
            # VIS presence frame appear falsely underexposed.
            _prepare_tablet_presence_check(motion_platform, capture_plan)

            try:
                frame_bgr = _grab_owned_camera_frame(timeout_ms=5000, retries=2)

                # -- Greyscale difference score --
                gds_result = check_only.grayscale_difference_score(frame_bgr)
                gds_status = gds_result.get('status', 'ERROR')
                gds_code = gds_result.get('code', '')
                if gds_status != 'OK':
                    app.logger.warning(
                        f"Tablet {tablet_index}: check_only grayscale_difference_score "
                        f"returned {gds_status} ({gds_code})"
                    )
                    if gds_code and gds_code.startswith('E2'):
                        app.logger.info(
                            f"Tablet {tablet_index}: Skipping image capture "
                            f"(check_only error {gds_code})"
                        )
                        _turn_off_all_lights()
                        return jsonify({
                            'status': 'success',
                            'tablet_index': tablet_index,
                            'saved_images': [],
                            'af_error_code': gds_code,
                            'af_error_message': ERROR_MESSAGES.get(gds_code, gds_code)
                        }), 200

                # -- Final out-of-frame check --
                oof_result = check_only.final_out_of_frame_check(frame_bgr)
                oof_status = oof_result.get('status', 'ERROR')
                oof_code = oof_result.get('code', '')
                if oof_status != 'OK':
                    app.logger.warning(
                        f"Tablet {tablet_index}: check_only final_out_of_frame_check "
                        f"returned {oof_status} ({oof_code})"
                    )
                    if oof_code and oof_code.startswith('E2'):
                        app.logger.info(
                            f"Tablet {tablet_index}: Skipping image capture "
                            f"(check_only error {oof_code})"
                        )
                        _turn_off_all_lights()
                        return jsonify({
                            'status': 'success',
                            'tablet_index': tablet_index,
                            'saved_images': [],
                            'af_error_code': oof_code,
                            'af_error_message': ERROR_MESSAGES.get(oof_code, oof_code)
                        }), 200

                app.logger.info(f"Tablet {tablet_index}: check_only passed")

            except (OSError, PermissionError) as e:
                _turn_off_all_lights()
                return _handle_motion_usb_disconnect(
                    motion_platform, f"check_only tablet {tablet_index}"
                )
            except Exception as e:
                if _is_camera_disconnect(e):
                    _turn_off_all_lights()
                    return _handle_camera_disconnect(f"check_only tablet {tablet_index}")
                app.logger.warning(f"Tablet {tablet_index}: check_only failed: {e}")
                # Continue — check failure should not block the measurement
            finally:
                _turn_off_all_lights()
                measurement_progress.set_active_plan_row(request_id, None)

        # =====================================================
        # STEP 4: Capture images with selected lights
        # =====================================================
        if capture_plan is not None:
            capture_rows = _measurement_capture_rows(
                capture_plan, get_settings(), should_autofocus
            )
            if len(capture_rows) != len(capture_plan):
                app.logger.info(
                    'Tablet %s: Skipping autofocus-only plan row 1 capture',
                    tablet_index,
                )
            for plan_row_index, row in capture_rows:
                wavelength = row['wavelength']
                filter_position = row['filter_position']
                try:
                    measurement_progress.set_active_plan_row(request_id, plan_row_index)
                    app.logger.info(
                        'Tablet %s: Capturing %s image with filter position %s',
                        tablet_index, wavelength, filter_position,
                    )
                    _select_measurement_filter_position(motion_platform, filter_position)
                    height_offset = height_offset_control.apply_active_combination(
                        motion_platform,
                        _current_filter_settings(),
                        wavelength,
                    )
                    if height_offset.get('warning'):
                        measurement_progress.record_warning(
                            request_id, height_offset['warning']
                        )
                    if not height_offset['applied']:
                        app.logger.warning(
                            'Tablet %s: Z offset was not applied for plan row %s: %s',
                            tablet_index, plan_row_index + 1, height_offset.get('reason'),
                        )

                    def publish_saved_image(saved_path, masked):
                        measurement_progress.record_image(request_id, {
                            'path': saved_path,
                            'tablet_index': tablet_index,
                            'wavelength': wavelength,
                            'brightness': row['brightness'],
                            'filter_position': filter_position,
                            'exposure_time': row['exposure_time'],
                            'gain': row['gain'],
                            'masked': masked,
                        })

                    saved_paths = _capture_capture_plan_row(
                        row, measurement_folder, measurement_name, tablet_index,
                        background_subtraction=background_subtraction,
                        on_image_saved=publish_saved_image,
                    )
                    saved_images.extend(saved_paths)
                    captured_plan_rows.append({
                        'wavelength': wavelength,
                        'brightness': row['brightness'],
                        'filter_position': filter_position,
                        'exposure_time': row['exposure_time'],
                        'gain': row['gain'],
                        'saved_images': saved_paths,
                    })
                except (OSError, PermissionError) as error:
                    return _handle_motion_usb_disconnect(
                        motion_platform, f'{wavelength} capture tablet {tablet_index}'
                    )
                except Exception as error:
                    app.logger.error(
                        'Tablet %s: Failed to capture %s/filter %s: %s',
                        tablet_index, wavelength, filter_position, error,
                    )
                    if _is_camera_disconnect(error):
                        return _handle_camera_disconnect(f'{wavelength} capture tablet {tablet_index}')
                    if _is_serial_disconnect(error):
                        return _handle_motion_usb_disconnect(
                            motion_platform, f'{wavelength} capture tablet {tablet_index}'
                        )
                    return jsonify({
                        'status': 'error',
                        'message': f'Failed to capture {wavelength} with filter {filter_position} for tablet {tablet_index}: {error}'
                    }), 500
            measurement_progress.set_active_plan_row(request_id, None)

            # The request contained the new capture plan, so skip the legacy
            # dome/bar capture path below even when old compatibility flags are
            # also present.
            lamp_top = False
            lamp_side = False

        if lamp_top:
            try:
                app.logger.info(f"Tablet {tablet_index}: Capturing dome image")
                saved_paths = _capture_image_with_light('dome', measurement_folder, measurement_name, tablet_index, background_subtraction=background_subtraction)
                saved_images.extend(saved_paths)
                app.logger.info(f"Tablet {tablet_index}: Saved dome image(s): {saved_paths}")
            except (OSError, PermissionError) as e:
                try:
                    _turn_off_all_lights()
                except Exception:
                    pass
                return _handle_motion_usb_disconnect(motion_platform, f"dome capture tablet {tablet_index}")
            except Exception as e:
                app.logger.error(f"Tablet {tablet_index}: Failed to capture dome image: {e}")
                try:
                    _turn_off_all_lights()
                except Exception:
                    pass
                if _is_camera_disconnect(e):
                    return _handle_camera_disconnect(f"dome capture tablet {tablet_index}")
                if _is_serial_disconnect(e):
                    return _handle_motion_usb_disconnect(motion_platform, f"dome capture tablet {tablet_index}")
                return jsonify({
                    'status': 'error',
                    'message': f'Failed to capture dome image for tablet {tablet_index}: {e}'
                }), 500
        
        if lamp_side:
            try:
                # --- Bar-light exposure gate ---
                _turn_on_uv_dome_light()
                _apply_camera_settings_for_light('bar')
                time.sleep(0.3)

                bar_gate_passed = True
                bar_gate_error_code = None
                cam = globals.camera
                if cam and cam.IsOpen():
                    try:
                        gate_frame = _grab_owned_camera_frame(timeout_ms=3000)
                        gate_result = under_over.exposure_gate_from_frame(gate_frame)
                        gate_code = gate_result.get('code')
                        gate_metrics = {k: v for k, v in gate_result.items() if k not in ('status', 'code')}
                        app.logger.info(
                            f"Tablet {tablet_index}: [BAR-LIGHT GATE] status={gate_result['status']} "
                            f"code={gate_code} p95={gate_metrics.get('p95', 0):.1f} "
                            f"dr={gate_metrics.get('dr', 0):.1f} white={gate_metrics.get('white', 0):.3f}"
                        )
                        if gate_result['status'] != 'OK':
                            bar_gate_passed = False
                            bar_gate_error_code = gate_code
                            app.logger.warning(
                                f"Tablet {tablet_index}: Bar-light exposure gate failed ({gate_code}), skipping bar capture"
                            )
                    except Exception as gate_err:
                        app.logger.warning(f"Tablet {tablet_index}: Bar-light exposure gate error, proceeding: {gate_err}")

                if bar_gate_passed:
                    app.logger.info(f"Tablet {tablet_index}: Capturing bar image")
                    saved_paths = _capture_image_with_light('bar', measurement_folder, measurement_name, tablet_index, background_subtraction=background_subtraction)
                    saved_images.extend(saved_paths)
                    app.logger.info(f"Tablet {tablet_index}: Saved bar image(s): {saved_paths}")
                elif bar_gate_error_code:
                    af_error_code = bar_gate_error_code
            except (OSError, PermissionError) as e:
                try:
                    _turn_off_all_lights()
                except Exception:
                    pass
                return _handle_motion_usb_disconnect(motion_platform, f"bar capture tablet {tablet_index}")
            except Exception as e:
                app.logger.error(f"Tablet {tablet_index}: Failed to capture bar image: {e}")
                try:
                    _turn_off_all_lights()
                except Exception:
                    pass
                if _is_camera_disconnect(e):
                    return _handle_camera_disconnect(f"bar capture tablet {tablet_index}")
                if _is_serial_disconnect(e):
                    return _handle_motion_usb_disconnect(motion_platform, f"bar capture tablet {tablet_index}")
                return jsonify({
                    'status': 'error',
                    'message': f'Failed to capture bar image for tablet {tablet_index}: {e}'
                }), 500
        
        # =====================================================
        # STEP 5: Turn off lights after this tablet
        # =====================================================
        _turn_off_all_lights()
        if capture_plan is not None:
            try:
                light_controller.off()
            except Exception as error:
                app.logger.warning('Four-channel all-off after capture plan failed: %s', error)
        
        app.logger.info(f"Tablet {tablet_index}: Measurement complete ({len(saved_images)} images)")
        response_data = {
            'status': 'success',
            'tablet_index': tablet_index,
            'saved_images': saved_images,
            'captured_plan_rows': captured_plan_rows,
        }
        if af_error_code:
            response_data['af_error_code'] = af_error_code
            response_data['af_error_message'] = ERROR_MESSAGES.get(af_error_code, af_error_code)
        progress_outcome = 'completed'
        return jsonify(response_data), 200
        
    except (OSError, PermissionError) as e:
        try:
            _turn_off_all_lights()
        except Exception:
            pass
        ser = globals.motion_platform
        return _handle_motion_usb_disconnect(ser, f"auto_measurement tablet {data.get('tablet_index', '?')}")
    except Exception as e:
        try:
            _turn_off_all_lights()
        except Exception:
            pass
        app.logger.exception(f"auto_measurement_step failed: {e}")
        # Check if this is a serial/USB disconnect wrapped in another exception
        if _is_serial_disconnect(e):
            ser = globals.motion_platform
            return _handle_motion_usb_disconnect(ser, f"auto_measurement tablet {data.get('tablet_index', '?')}")
        if _is_camera_disconnect(e):
            return _handle_camera_disconnect(f"auto_measurement tablet {data.get('tablet_index', '?')}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
    finally:
        if request_id:
            measurement_progress.finish(request_id, progress_outcome)



### Video streaming Function ###
@app.route('/api/start-video-stream', methods=['GET'])
def start_video_stream():
    """
    Returns a live MJPEG response from stream_video().
    This is the *only* place we call stream_video, to avoid double-streaming.
    """
    try:
        scale_factor = float(request.args.get('scale', 0.1))

        res = connect_camera_internal()
        if "error" in res:
            app.logger.error(f"Camera connection failed: {res['error']}")
            return jsonify(res), 400

        with globals.grab_lock:
            if not globals.stream_running:
                globals.stream_running = True
                app.logger.info(f"stream_running set to True in start_video_stream")

        app.logger.info(f"Starting video stream")
        return Response(
            stream_video(scale_factor),
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )

    except ValueError as ve:
        app.logger.error(f"Invalid input: {ve}")
        return jsonify({"error": "Invalid input parameters"}), 400

    except Exception as e:
        app.logger.exception("Unexpected error in start-video-stream")
        return jsonify({"error": "Internal server error"}), 500


@app.route('/api/stop-video-stream', methods=['POST'])
def stop_video_stream():
    app.logger.info(f"Received stop request for camera stream")

    try:
        message = stop_camera_stream()
        return jsonify({"message": message}), 200
    except ValueError as ve:
        # E.g., invalid camera type
        app.logger.error(str(ve))
        return jsonify({"error": str(ve)}), 400
    except RuntimeError as re:
        app.logger.error(str(re))
        return jsonify({"error": str(re)}), 500
    except Exception as e:
        app.logger.exception(f"Unexpected exception while stopping camera stream.")
        return jsonify({"error": str(e)}), 500


def grab_camera_image():
    """
    Grabs a single frame from the camera and converts to BGR8.
    
    Returns:
        tuple: (frame_bgr, error_response, error_code) where frame_bgr is uint8 BGR array,
               or (None, error_json, error_code) on failure
    """
    try:
        lock = getattr(globals, "grab_lock", None)
        if lock is None:
            globals.grab_lock = Lock()
            lock = globals.grab_lock

        with lock:
            cam = getattr(globals, "camera", None)
            if cam is None or not cam.IsOpen():
                app.logger.error("Camera is not connected or not open.")
                return None, jsonify({
                    "error": ERROR_MESSAGES.get(ErrorCode.CAMERA_DISCONNECTED, "Camera disconnected."),
                    "code": ErrorCode.CAMERA_DISCONNECTED,
                    "popup": True
                }), 400

            if not cam.IsGrabbing():
                try:
                    cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
                    app.logger.info("Camera was not grabbing; started grabbing for still capture.")
                except Exception as e:
                    app.logger.error(f"Camera is not grabbing and failed to start: {e}")
                    return None, jsonify({
                        "error": "Camera is not grabbing and could not be started.",
                        "code": ErrorCode.CAMERA_DISCONNECTED,
                        "popup": True
                    }), 503

            # Use unified grab+convert function
            try:
                from cameracontrol import grab_and_convert_frame
                frame_bgr = grab_and_convert_frame(cam, timeout_ms=5000, retries=2)
                app.logger.info("Image grabbed and converted to BGR successfully.")
                globals.latest_image = frame_bgr
                return frame_bgr, None, None
            except RuntimeError as e:
                app.logger.error(f"Frame grab failed: {e}")
                return None, jsonify({
                    "error": "Failed to grab frame from camera",
                    "code": ErrorCode.CAMERA_DISCONNECTED,
                    "popup": True
                }), 400

    except Exception as e:
        app.logger.exception(f"Error grabbing image: {e}")
        return None, jsonify({
            "error": "Generic error during image grabbing",
            "code": ErrorCode.GENERIC,
            "popup": True
        }), 500

        
@app.route('/api/save_raw_image', methods=['POST'])
@guard_capture_operation
def save_raw_image_endpoint():
    data = request.get_json() or {}
    target_folder = _normalize_path(data.get('target_folder', ''))
    measurement_name = data.get('measurement_name')
    selected_light = light_controller.status()
    light_type = selected_light['active_channel']
    if light_type is None:
        return jsonify({'error': 'Activate a lamp before capture.', 'code': ErrorCode.CAPTURE_LIGHT_REQUIRED, 'popup': True}), 409

    if not target_folder:
        return jsonify({"message": "Cancelled"}), 200

    # Ensure folder exists
    try:
        os.makedirs(target_folder, exist_ok=True)
    except Exception as e:
        return jsonify({
            "error": f"Could not create folder: {e}",
            "code": "FOLDER_CREATION_FAILED",
            "popup": True
        }), 400

    # --- Background subtraction: obtain contour under dome light ---
    settings_data_pre = get_settings()
    bg_sub_pre = bool(settings_data_pre.get('other_settings', {}).get('background_subtraction', False))

    if bg_sub_pre:
        app.logger.info(f"Manual save: {light_type} light + BGR — obtaining contour under dome light")
        try:
            # Always switch to dome light for contour detection
            if light_type != 'vis':
                _turn_on_dome_light()
                _apply_camera_settings_for_light('dome')
                time.sleep(0.3)  # Let light and camera settings stabilize

            mbgr_result = manual_bgr_with_check.manual_return()
            mbgr_status = mbgr_result.get('status', 'ERROR')
            if mbgr_status == 'OK':
                contour = mbgr_result.get('final_contour')
                globals.last_autofocus_contour = contour if contour else None
                app.logger.info("Manual save: dome-light contour obtained for BGR")
            else:
                globals.last_autofocus_contour = None
                mbgr_code = mbgr_result.get('code', '')
                app.logger.warning(f"Manual save: manual_bgr_with_check returned {mbgr_status} ({mbgr_code})")

            # Switch back to original light for the actual capture
            if light_type != 'vis':
                light_controller.activate(light_type, selected_light['active_mode'])
                time.sleep(0.3)  # Let light and camera settings stabilize
        except Exception as e:
            app.logger.warning(f"Manual save: contour detection under dome light failed: {e}")
            globals.last_autofocus_contour = None
            # Restore original light and continue — save image without background subtraction
            if light_type != 'vis':
                try:
                    light_controller.activate(light_type, selected_light['active_mode'])
                    time.sleep(0.3)
                except Exception:
                    return jsonify({'error': 'Could not restore capture illumination.', 'code': ErrorCode.CAPTURE_LIGHT_REQUIRED, 'popup': True}), 503

    # --- Grab frame from camera ---
    img = grab_camera_image()  # your existing helper
    if isinstance(img, tuple) and len(img) == 3:
        img, capture_error, capture_status = img
        if capture_error is not None:
            return capture_error, capture_status
    try:
        light_controller.capture_remaining_seconds(light_type, selected_light['active_mode'])
    except CaptureIlluminationError as error:
        return jsonify({'error': str(error), 'code': ErrorCode.CAPTURE_LIGHT_TIMEOUT, 'popup': True}), 422

    if img is None:
        return jsonify({
            "error": "Camera disconnected or failed to grab image.",
            "code": "CAMERA_DISCONNECTED",
            "popup": True
        }), 400

    # If grab_camera_image returns a tuple/list (e.g. (frame, meta)), unwrap it
    if isinstance(img, (tuple, list)) and len(img) > 0:
        img = img[0]

    # Normalize to NumPy array
    img_cv = np.asarray(img)
    if not isinstance(img_cv, np.ndarray) or img_cv.ndim < 2:
        return jsonify({
            "error": f"Grabbed image is not a valid array (type={type(img)}, shape={getattr(img_cv, 'shape', None)})",
            "code": "INVALID_IMAGE_DATA",
            "popup": True
        }), 500

    # --- Save full-resolution image ---
    if not measurement_name:
        measurement_name = os.path.basename(os.path.normpath(target_folder)) or "measurement"

    timestamp = _format_capture_timestamp(datetime.now())
    filename = f"{timestamp}_{light_type}"
    full_path = os.path.join(target_folder, f"{filename}.jpg")

    try:
        requested_metadata = data.get('metadata')
        saved_metadata = _save_capture_jpeg(
            img_cv,
            full_path,
            light_type,
            requested_metadata=(
                requested_metadata if isinstance(requested_metadata, dict) else None
            ),
        )
    except Exception as e:
        return jsonify({
            "error": f"Could not save image: {e}",
            "code": "IMAGE_SAVE_FAILED",
            "popup": True
        }), 500

    _cache_latest_capture(light_type, img_cv)

    # Background subtraction: save masked version alongside original if enabled
    masked_path = None
    settings_data = get_settings()
    bg_sub_enabled = bool(settings_data.get('other_settings', {}).get('background_subtraction', False))
    if bg_sub_enabled:
        try:
            af_contour = getattr(globals, "last_autofocus_contour", None)
            mask, kind, metrics = bgr_main.make_object_mask_from_bgr_rel(img_cv, autofocus_contour = af_contour)
            if mask is not None and np.any(mask):
                masked = bgr_main.apply_mask_zero_background(img_cv, mask)
                base, ext = os.path.splitext(full_path)
                masked_path = f"{base}_masked{ext}"
                bgr_main.save_bgr_image_keep_exif(
                    image_bgr=masked,
                    src_image_path=full_path,
                    dst_image_path=masked_path
                )
                _cache_latest_capture(light_type, masked, masked=True)
                app.logger.info(f"Background-subtracted image saved: {masked_path} (kind={kind})")
            else:
                app.logger.warning("Background subtraction found no object in saved image")
        except Exception as e:
            app.logger.warning(f"Background subtraction failed for saved image: {e}")

    result = {
        "message": "Raw image saved",
        "path": full_path,
        "metadata": saved_metadata,
    }
    if masked_path:
        result["masked_path"] = masked_path
    return jsonify(result), 200


@app.route('/api/image-metadata', methods=['GET'])
def get_saved_image_metadata():
    path = request.args.get('path', '')
    if not os.path.isfile(path):
        return jsonify({'error': 'Image not found.', 'code': ErrorCode.GENERIC, 'popup': False}), 404
    try:
        with Image.open(path) as image:
            description = image.getexif().get(0x010E, '{}')
            if isinstance(description, bytes):
                description = description.decode('utf-8')
            metadata = json.loads(description)
        return jsonify({'metadata': metadata if isinstance(metadata, dict) else {}})
    except (ValueError, OSError, UnicodeError):
        return jsonify({'metadata': {}})


@app.route('/api/get_thumbnail', methods=['GET'])
def get_thumbnail():
    """Return a small JPEG thumbnail generated on the fly from a saved image."""
    path = request.args.get('path')
    if not path:
        return jsonify({"error": "No path specified"}), 400

    if not os.path.isfile(path):
        return jsonify({"error": "File not found"}), 404

    # Read original image from disk
    img = cv2.imread(path)
    if img is None:
        return jsonify({"error": "Could not read image file"}), 500

    # Thumbnail parameters
    max_thumb_width = 160
    max_thumb_height = 120

    # Slight blur to reduce noise before downscale
    blurred = cv2.GaussianBlur(img, (3, 3), 0)

    h, w = blurred.shape[:2]
    scale = min(max_thumb_width / w, max_thumb_height / h, 1.0)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Downscale using INTER_AREA (good for reduction)
    thumb = cv2.resize(blurred, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Encode thumbnail to JPEG in memory (no disk write)
    ok, buf = cv2.imencode(
        ".jpg",
        thumb,
        [int(cv2.IMWRITE_JPEG_QUALITY), 80]  # decent quality, small size
    )
    if not ok:
        return jsonify({"error": "Could not encode thumbnail"}), 500

    return send_file(
        io.BytesIO(buf.tobytes()),
        mimetype="image/jpeg"
    )


    
@app.route('/api/get_image', methods=['GET'])
def get_image():
    path = request.args.get('path')
    if not path:
        return jsonify({"error": "No path specified"}), 400

    if not os.path.isfile(path):
        return jsonify({"error": "File not found"}), 404

    # Assume JPEG; if you might save PNG/tiff, detect MIME type here.
    return send_file(path, mimetype='image/jpeg')



@app.route('/api/open_image', methods=['POST'])
def open_image():
    data = request.get_json() or {}
    path = data.get('path')

    if not path:
        return jsonify({"error": "No path specified"}), 400

    if not os.path.isfile(path):
        return jsonify({"success": True, "skipped": True}), 200

    try:
        # Windows-only: open with default associated app (typically Photos)
        os.startfile(path)  # type: ignore[attr-defined]
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"success": True}), 200

@app.route('/api/open_folder', methods=['POST'])
def open_folder():
    data = request.get_json() or {}
    path = data.get('path')

    if not path:
        return jsonify({"error": "No path specified"}), 400

    if not os.path.isfile(path):
        return jsonify({"error": "File not found"}), 404

    try:
        # Windows-only: open Explorer at the folder containing the file
        folder = os.path.dirname(path)
        os.startfile(folder)  # type: ignore[attr-defined]
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"success": True}), 200


@app.route('/api/delete-image', methods=['POST'])
def delete_image():
    data = request.get_json() or {}
    path = data.get('path')

    if not path:
        return jsonify({"error": "No path specified"}), 400

    if not os.path.isfile(path):
        return jsonify({"error": "File not found"}), 404

    try:
        os.remove(path)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"success": True}), 200

@app.route('/api/get-other-settings', methods=['GET'])
def get_other_settings():
    category = request.args.get('category')
    if not category:
        return jsonify({"error": "Category parameter is required."}), 400

    settings_data = get_settings()
    if category not in settings_data:
        return jsonify({"error": f"Category '{category}' not found."}), 404

    return jsonify({category: settings_data[category]}), 200


@app.route('/api/settings/lamp', methods=['GET', 'PUT'])
def lamp_settings():
    if request.method == 'GET':
        settings_data = get_settings()
        return jsonify({'lamp_settings': settings_data.get('lamp_settings', {'channels': {}})}), 200

    try:
        normalized_settings = validate_lamp_settings(request.get_json(silent=True) or {})
        settings_data = get_settings()
        lamp_settings_data = settings_data.setdefault('lamp_settings', {})
        lamp_settings_data['channels'] = normalized_settings['channels']
        if not save_settings():
            raise OSError('Failed to persist lamp settings.')
        return jsonify({'lamp_settings': lamp_settings_data}), 200
    except ValueError as error:
        return jsonify({'error': str(error), 'code': ErrorCode.GENERIC, 'popup': True}), 400
    except Exception as error:
        app.logger.exception('Failed to update lamp settings')
        return jsonify({'error': str(error), 'code': ErrorCode.GENERIC, 'popup': True}), 500


@app.route('/api/settings/lamp/advanced', methods=['GET', 'PUT'])
def advanced_lamp_settings():
    if request.method == 'GET':
        settings_data = get_settings()
        lamp_settings_data = settings_data.get('lamp_settings', {})
        return jsonify({'advanced_lamp_settings': {
            'output_selectors': lamp_settings_data.get('output_selectors', {})
        }}), 200

    try:
        normalized_settings = validate_lamp_output_selectors(request.get_json(silent=True) or {})
        with porthandler.motion_lock:
            serial_port = porthandler.motion_platform or globals.motion_platform
            if serial_port and getattr(serial_port, 'is_open', False):
                # Keep the physical all-off and the settings commit atomic with
                # respect to every activation and disconnect operation.
                light_controller.off()
            if not update_lamp_output_selectors(normalized_settings['output_selectors']):
                raise OSError('Failed to persist advanced lamp settings.')
        return jsonify({'advanced_lamp_settings': normalized_settings}), 200
    except ValueError as error:
        return jsonify({'error': str(error), 'code': ErrorCode.GENERIC, 'popup': True}), 400
    except LightCommandError as error:
        return jsonify({'error': str(error), 'code': ErrorCode.MOTIONPLATFORM_DISCONNECTED, 'popup': True}), 503
    except Exception as error:
        app.logger.exception('Failed to update advanced lamp settings')
        return jsonify({'error': str(error), 'code': ErrorCode.GENERIC, 'popup': True}), 500


@app.route('/api/settings/filter', methods=['GET', 'PUT'])
def filter_settings():
    max_up, max_down = _height_offset_limits()
    if request.method == 'GET':
        settings_data = get_settings()
        try:
            current_settings = validate_filter_settings(
                settings_data.get('filter_settings', default_filter_settings()),
                max_height_offset_up_mm=max_up,
                max_height_offset_down_mm=max_down,
            )
            return jsonify({'filter_settings': current_settings}), 200
        except ValueError as error:
            app.logger.error('Stored filter settings are invalid: %s', error)
            return jsonify({
                'error': 'Stored filter settings are invalid.',
                'code': ErrorCode.GENERIC,
                'popup': True,
            }), 500

    try:
        normalized_settings = validate_filter_settings(
            request.get_json(silent=True) or {},
            max_height_offset_up_mm=max_up,
            max_height_offset_down_mm=max_down,
        )
        with porthandler.motion_lock:
            if globals.motion_busy:
                return jsonify({'error': 'Motion platform is busy.', 'code': ErrorCode.GENERIC, 'popup': True}), 409
            previous = get_settings().get('filter_settings', default_filter_settings())
            if not update_filter_settings(normalized_settings):
                raise OSError('Failed to persist filter settings.')
            if (previous.get('slots') != normalized_settings['slots']
                    or previous.get('height_offsets_mm') != normalized_settings['height_offsets_mm']):
                height_offset_control.invalidate_reference()
        return jsonify({'filter_settings': normalized_settings}), 200
    except ValueError as error:
        return jsonify({'error': str(error), 'code': ErrorCode.GENERIC, 'popup': True}), 400
    except Exception as error:
        app.logger.exception('Failed to update filter settings')
        return jsonify({'error': str(error), 'code': ErrorCode.GENERIC, 'popup': True}), 500


@app.route('/api/settings/autofocus', methods=['GET', 'PUT'])
def autofocus_settings():
    """Read or persist the autofocus light, brightness, and populated filter slot."""
    try:
        filter_settings_data = _current_filter_settings()
        if request.method == 'GET':
            normalized_settings = _current_autofocus_settings(filter_settings_data)
            return jsonify({'autofocus_settings': normalized_settings}), 200

        normalized_settings = validate_autofocus_settings(
            request.get_json(silent=True) or {},
            filter_settings_data,
        )
        if not update_autofocus_settings(normalized_settings):
            raise OSError('Failed to persist autofocus settings.')
        return jsonify({'autofocus_settings': normalized_settings}), 200
    except ValueError as error:
        return jsonify({'error': str(error), 'code': ErrorCode.GENERIC, 'popup': True}), 400
    except Exception as error:
        app.logger.exception('Failed to update autofocus settings')
        return jsonify({'error': str(error), 'code': ErrorCode.GENERIC, 'popup': True}), 500


@app.route('/api/settings/motion/advanced', methods=['GET', 'PUT'])
def advanced_motion_settings():
    if request.method == 'GET':
        max_up, max_down = _height_offset_limits()
        advanced = get_settings().get('advanced_settings', {})
        return jsonify({
            'advanced_motion_settings': {
                'use_virtual_com_port': _use_virtual_motion_platform(),
                'lower_z_before_xy_move': advanced.get(
                    'lower_z_before_xy_move', DEFAULT_LOWER_Z_BEFORE_XY_MOVE
                ),
                'xy_move_z_limit_mm': advanced.get(
                    'xy_move_z_limit_mm', DEFAULT_XY_MOVE_Z_LIMIT_MM
                ),
                'max_height_offset_up_mm': max_up,
                'max_height_offset_down_mm': max_down,
                'first_tablet_x_mm': advanced.get('first_tablet_x_mm', DEFAULT_FIRST_TABLET_X_MM),
                'first_tablet_y_mm': advanced.get('first_tablet_y_mm', DEFAULT_FIRST_TABLET_Y_MM),
                'first_tablet_z_mm': advanced.get('first_tablet_z_mm', DEFAULT_FIRST_TABLET_Z_MM),
                'tablet_spacing_mm': advanced.get('tablet_spacing_mm', DEFAULT_TABLET_SPACING_MM),
            }
        }), 200

    try:
        if getattr(globals, 'motion_busy', False):
            return jsonify({
                'error': 'Motion platform is busy; try again after the operation finishes.',
                'code': ErrorCode.GENERIC,
                'popup': True,
            }), 409
        normalized = validate_motion_simulation_settings(request.get_json(silent=True) or {})
        existing_filter_settings = get_settings().get('filter_settings')
        if existing_filter_settings is not None:
            validate_filter_settings(
                existing_filter_settings,
                max_height_offset_up_mm=normalized['max_height_offset_up_mm'],
                max_height_offset_down_mm=normalized['max_height_offset_down_mm'],
            )
        settings_data = get_settings()
        settings_data.setdefault('advanced_settings', {}).update(normalized)
        if not save_settings():
            raise OSError('Failed to persist advanced motion settings.')

        current = porthandler.motion_platform or globals.motion_platform
        desired_virtual = normalized['use_virtual_com_port']
        if (
            current
            and getattr(current, 'is_open', False)
            and bool(getattr(current, 'is_virtual', False)) == desired_virtual
        ):
            device = current
        else:
            device = _replace_motion_platform(desired_virtual)
        return jsonify({
            'advanced_motion_settings': normalized,
            'connection': {
                'connected': bool(device and getattr(device, 'is_open', False)),
                'port': getattr(device, 'port', None),
                'virtual': bool(device and getattr(device, 'is_virtual', False)),
            },
        }), 200
    except TrayGeometryError:
        return jsonify({
            'error': 'A 10×10-es tálca koordinátáinak az X/Y mozgástartományon belül kell maradniuk.',
            'code': ErrorCode.MOTION_TRAY_OUT_OF_RANGE,
            'popup': True,
        }), 400
    except ValueError as error:
        return jsonify({'error': str(error), 'code': ErrorCode.GENERIC, 'popup': True}), 400
    except Exception as error:
        app.logger.exception('Failed to update advanced motion settings')
        return jsonify({'error': str(error), 'code': ErrorCode.GENERIC, 'popup': True}), 500

@app.route('/api/update-other-settings', methods=['POST'])
def update_other_settings():
    try:
        data = request.json
        category = data.get('category')           # e.g. 'size_limits'
        setting_name = data.get('setting_name')     # e.g. 'ng_limit'
        setting_value = data.get('setting_value')   # e.g. 123

        app.logger.info(f"Updating {category}.{setting_name} = {setting_value}")

        if category == 'auto_measurement_settings' and setting_name == 'capture_plan':
            setting_value = validate_capture_plan(setting_value)
            _validate_capture_plan_camera_values(setting_value)
        elif category == 'auto_measurement_settings' and setting_name == 'save_autofocus_image':
            if not isinstance(setting_value, bool):
                raise ValueError('save_autofocus_image must be a boolean.')
        elif category == 'auto_measurement_settings' and setting_name == 'check_tablet_presence':
            if not isinstance(setting_value, bool):
                raise ValueError('check_tablet_presence must be a boolean.')

        # Normalize path-like settings to use forward slashes
        updated_value = setting_value
        if isinstance(setting_value, str) and any(kw in setting_name.lower() for kw in ('path', 'file', 'dir', 'location', 'folder')):
            updated_value = _normalize_path(setting_value)

        # Retrieve the in-memory settings
        settings_data = get_settings()
        if category not in settings_data:
            settings_data[category] = {}

        # Update the setting in the in-memory dict
        settings_data[category][setting_name] = updated_value

        # Save the updated settings to disk
        save_settings()

        app.logger.info(f"{category}.{setting_name} updated and saved to settings.json")

        return jsonify({
            "message": f"{category}.{setting_name} updated and saved.",
            "updated_value": updated_value
        }), 200

    except ValueError as e:
        return jsonify({"error": str(e), "code": ErrorCode.GENERIC, "popup": True}), 400
    except Exception as e:
        app.logger.exception("Failed to update other settings")
        return jsonify({"error": str(e)}), 500


@app.route('/api/check-file-exists', methods=['POST'])
def check_file_exists():
    try:
        data = request.get_json(silent=True) or {}
        path = data.get('path')
        if not path:
            return jsonify({"exists": False, "error": "No path provided"}), 400

        exists = os.path.isfile(path)
        return jsonify({"exists": bool(exists)})
    except Exception as e:
        app.logger.exception("check_file_exists failed")
        return jsonify({"exists": False, "error": str(e)}), 500


@app.route('/api/check-folder-exists', methods=['POST'])
def check_folder_exists():
    try:
        data = request.get_json(silent=True) or {}
        path = data.get('path')
        if not path:
            return jsonify({"exists": False, "error": "No path provided"}), 400

        exists = os.path.isdir(path)
        return jsonify({"exists": bool(exists)})
    except Exception as e:
        app.logger.exception("check_folder_exists failed")
        return jsonify({"exists": False, "error": str(e)}), 500
    

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"ready": True}), 200

@app.route('/api/abort-autofocus', methods=['POST'])
def abort_autofocus():
    """Signal autofocus routines to abort immediately."""
    try:
        globals.autofocus_abort = True
        app.logger.info("Autofocus abort flag set")
        return jsonify({"status": "abort_signaled"}), 200
    except Exception as e:
        app.logger.exception("Error setting autofocus abort flag")
        return jsonify({"error": str(e)}), 500

@app.route('/api/turn-off-all-lights', methods=['POST'])
def turn_off_all_lights_endpoint():
    """Turn off all lights (visible and UV) immediately. Used when measurement is stopped."""
    try:
        if not _turn_off_all_lights():
            return jsonify({
                'error': 'Not every light output acknowledged OFF.',
                'code': ErrorCode.MOTIONPLATFORM_DISCONNECTED,
                'popup': True,
            }), 503
        app.logger.info("All lights turned off via endpoint")
        return jsonify({"status": "lights_off"}), 200
    except Exception as e:
        app.logger.exception("Error turning off lights")
        return jsonify({"error": str(e)}), 500

@app.route('/api/check-lamp-auto-off', methods=['GET'])
def check_lamp_auto_off():
    """Compatibility view backed by the authoritative four-channel state."""
    try:
        status = light_controller.status()
        active_channel = status['active_channel']
        return jsonify({
            "auto_turned_off": light_controller.consume_auto_off_event(),
            "dome_on": active_channel == 'vis',
            "uv_dome_on": active_channel in UV_LAMP_CHANNELS,
        }), 200
    except Exception as e:
        app.logger.exception("Error checking lamp auto-off status")
        return jsonify({"error": str(e)}), 500

@app.route('/api/select-file', methods=['GET'])
def select_file():
    """
    Opens a file selection dialog (for .pfs camera setting files) and returns the chosen path.
    """
    try:
        # Simple Tkinter-based file dialog (similar spirit to select_folder_external)
        root = tk.Tk()
        root.withdraw()  # hide the main window
        root.update()
        file_path = filedialog.askopenfilename(
            title="Select camera settings file (.pfs)",
            filetypes=[("Pylon Feature Set", "*.pfs"), ("All files", "*.*")]
        )
        root.destroy()

        if not file_path:
            file_path = ""

        # Normalize backslashes to forward slashes for JSON portability
        if file_path:
            file_path = file_path.replace('\\', '/')

        return jsonify({"file": file_path}), 200

    except Exception as e:
        app.logger.exception("File selection failed")
        return jsonify({"error": str(e)}), 500

@app.route('/api/select-folder', methods=['GET'])
def select_folder():
    folder = select_folder_external()  # opens a Tkinter folder dialog (already normalized)
    if folder is None:
        folder = ""
    # Safety: re-normalize in case the subprocess returned backslashes
    if folder:
        folder = folder.replace('\\', '/')
    return jsonify({"folder": folder})


### Internal Helper Functions ### 
def get_base_path():
    """
    Ensures all output folders like 'Results/csv_results' and 'Results/annotated_images'
    are saved next to the main NozzleScanner.exe (not inside the resources folder).
    """
    if getattr(sys, 'frozen', False):
        # If frozen, sys.executable points to .../resources/GUI_backend.exe
        return os.path.join(os.path.dirname(os.path.dirname(sys.executable)), 'Results')
    else:
        # In dev mode, simulate the same directory structure
        return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Results'))
    
def select_folder_external() -> str:
    """Open a native folder-selection dialog in a short-lived subprocess.
    
    Uses sys.executable so the same Python interpreter (and its installed
    packages, e.g. tkinter) is used.  The dialog script lives next to this
    file in the backend/ directory.
    """
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'select_folder_dialog.py')
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=15  # seconds
        )
        output = json.loads(result.stdout.strip())
        return output.get('folder', '')
    except subprocess.TimeoutExpired:
        app.logger.error("Folder selection subprocess timed out after 15s")
        return ""
    except Exception as e:
        app.logger.error(f"Folder selection failed: {e}")
        return ""

def connect_camera_internal():
    factory = pylon.TlFactory.GetInstance()
    devices = factory.EnumerateDevices()

    if not devices:
        return {
            "error": ERROR_MESSAGES.get(ErrorCode.CAMERA_DISCONNECTED, "Camera not connected."),
            "code": ErrorCode.CAMERA_DISCONNECTED,
            "popup": True
        }
        
    selected_cam = devices[0]

    # If already connected, verify the handle is truly alive by reading
    # a property.  A stale handle after USB disconnect can report IsOpen()
    # == True but throw on any real operation.
    if globals.camera and globals.camera.IsOpen():
        try:
            # Lightweight liveness check — read a hardware property
            _ = globals.camera.GetDeviceInfo().GetSerialNumber()
            globals.camera.Width.GetValue()
            return {
                "connected": True,
                "name": selected_cam.GetModelName(),
            }
        except Exception as e:
            app.logger.warning(f"Existing camera handle is stale ({e}), cleaning up for fresh reconnect")
            # Handle is dead — clean it up so we fall through to a fresh open
            try:
                if globals.camera.IsGrabbing():
                    globals.camera.StopGrabbing()
            except Exception:
                pass
            try:
                globals.camera.Close()
            except Exception:
                pass
            globals.camera = None
            globals.stream_running = False

    # Try to open the camera with a small retry loop. On some Windows setups
    # the Pylon SDK can leave the device in a transient state after frequent
    # restarts; attempt a few times and do a clean close on failure to avoid
    # leaving the camera partially opened.
    max_attempts = 3
    open_success = False
    port_name = selected_cam.GetPortName() if hasattr(selected_cam, "GetPortName") else "unknown"
    for attempt in range(1, max_attempts + 1):
        try:
            app.logger.info(f"Attempting to open camera {port_name} (try {attempt}/{max_attempts})")
            cam = pylon.InstantCamera(factory.CreateDevice(selected_cam))
            cam.Open()
            # success
            globals.camera = cam
            open_success = True
            break
        except Exception as e:
            app.logger.warning(f"Camera open attempt {attempt} failed: {e}")
            try:
                # best-effort cleanup of partial camera object
                if 'cam' in locals() and cam is not None:
                    try:
                        if cam.IsOpen():
                            cam.Close()
                    except Exception:
                        pass
                    try:
                        del cam
                    except Exception:
                        pass
            except Exception:
                pass

            # small backoff before retry
            time.sleep(0.5 * attempt)

    if not open_success:
        app.logger.exception(f"Failed to connect to camera on port {port_name} after {max_attempts} attempts")
        return {
            "error": ERROR_MESSAGES.get(ErrorCode.CAMERA_DISCONNECTED, "Camera not connected."),
            "code": ErrorCode.CAMERA_DISCONNECTED,
            "popup": True,
            "details": "Camera failed to open; try replugging the device if problem persists."
        }

    if not globals.camera.IsOpen():
        app.logger.error("Camera failed to open after connection attempt.")
        return {"error": "Camera failed to open", "popup": True}

    # Retrieve camera properties and apply settings.
    try:
        camera_properties = get_camera_properties(globals.camera)
        globals.camera_properties = camera_properties 
        settings_data = get_settings()
        apply_camera_settings(globals.camera, camera_properties, settings_data)
        
        # Cache the serial number for fast status polling (avoids EnumerateDevices)
        try:
            globals._cached_camera_serial = globals.camera.GetDeviceInfo().GetSerialNumber()
        except Exception:
            globals._cached_camera_serial = None

        # Load .pfs camera profile if configured
        pfs_path = settings_data.get('other_settings', {}).get('camera_settings_file', '')
        if pfs_path and os.path.isfile(pfs_path):
            try:
                pfs_result = load_camera_profile(globals.camera, pfs_path)
                if 'error' in pfs_result:
                    app.logger.warning(f"Failed to load .pfs profile: {pfs_result['error']}")
                else:
                    app.logger.info(f"Camera profile loaded on connect: {pfs_path}")
            except Exception as pfs_e:
                app.logger.warning(f"Error loading .pfs profile on connect: {pfs_e}")
        elif pfs_path:
            app.logger.warning(f"Configured .pfs file not found: {pfs_path}")

        image_settings = _saved_camera_image_settings()
        if image_settings['override_enabled']:
            try:
                apply_camera_image_geometry(globals.camera, image_settings)
                app.logger.info('Persisted camera image-size override applied on connect.')
            except Exception as geometry_error:
                app.logger.warning('Could not apply persisted camera image-size override: %s', geometry_error)
        
    except Exception as e:
        app.logger.warning(f"get_camera_properties failed: {e}")
        globals.camera_properties = {}

    return {
        "connected": True,
        "name": selected_cam.GetModelName(),
        "serial": selected_cam.GetSerialNumber()
    }

def start_camera_stream_internal(scale_factor=0.1):
    try:
        cam = getattr(globals, "camera", None)
        if not (cam and cam.IsOpen()):  # ← FIXED: IsOpen()
            app.logger.error("Camera not connected/open; cannot start stream.")
            return {
                "error": ERROR_MESSAGES.get(ErrorCode.CAMERA_DISCONNECTED, "Camera disconnected."),
                "code": ErrorCode.CAMERA_DISCONNECTED,
                "popup": True
            }

        lock = getattr(globals, "grab_lock", None)
        if lock is None:
            globals.grab_lock = Lock()
            lock = globals.grab_lock

        with lock:
            running = getattr(globals, "stream_running", False)
            if running and cam.IsGrabbing():
                app.logger.info("Stream already running.")
                return {"message": "Stream already running"}

            if not cam.IsGrabbing():
                app.logger.info("Camera starting grabbing.")
                cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

            globals.stream_running = True
            globals.stream_scale = scale_factor  # optional: remember current scale

        return {"message": "Video stream started successfully."}

    except Exception as e:
        app.logger.exception(f"Error starting stream: {e}")
        globals.stream_running = False
        return {"error": str(e), "popup": True}
        
def initialize_cameras():
    app.logger.info("Initializing camera...")
    
    cam = getattr(globals, "camera", None)

    if cam and cam.IsOpen():
        app.logger.info("Camera is already connected. Skipping initialization.")
        return
        
    try:
        result = connect_camera_internal()
        if result.get('connected'):
            app.logger.info(f"Successfully connected camera.")
            started = start_camera_stream_internal()
            if not started:
                app.logger.warning("Camera connected but stream did not start.")
        else:
            app.logger.error(f"Failed to connect to camera: {result.get('error')}")
    except Exception as e:
        app.logger.error(f"Error during camera initialization: {e}")


def initialize_serial_devices():
    """Initialize serial devices at startup."""
    app.logger.info("Initializing serial devices...")

    try:
        # Connect Motion Platform
        device = porthandler.connect_to_motion_platform(
            use_virtual=_use_virtual_motion_platform()
        )
        if device:
            porthandler.motion_platform = device
            globals.motion_platform = device
            app.logger.info("Motion Platform connected automatically on startup.")
        else:
            app.logger.error("Failed to auto-connect Motion Platform on startup.")
    except Exception as e:
        app.logger.error(f"Error initializing Motion Platform: {e}")
       
def shutdown_devices():
    """Clean shutdown of all devices before exit."""
    app.logger.info("Shutting down devices...")

    try:
        light_controller.off()
    except Exception as error:
        app.logger.warning('Four-channel all-off during shutdown failed: %s', error)
    
    # Close camera stream and camera
    try:
        stop_camera_stream()
        cam = globals.camera
        if cam and cam.IsOpen():
            cam.Close()
        globals.camera = None
        app.logger.info("Camera closed successfully.")
    except Exception as e:
        app.logger.debug(f"Error closing camera: {e}")
    
    # Close motion platform serial port
    try:
        ser = globals.motion_platform
        if ser and getattr(ser, 'is_open', False):
            ser.close()
            globals.motion_platform = None
            porthandler.motion_platform = None
            _reset_motion_reference_state()
            app.logger.info("Motion platform disconnected successfully.")
    except Exception as e:
        app.logger.debug(f"Error closing motion platform: {e}")      
       

### Latest Image Endpoints ###
# These endpoints serve the most recently captured images as viewable JPEGs.
# They are the ONLY endpoints exposed in the compiled (PyInstaller) build.

_LATEST_IMAGE_ALIASES = {
    # Temporary compatibility aliases for pre-v2 viewers only.
    'dome': ('vis', 'original'),
    'dome_masked': ('vis', 'masked'),
    'bar': ('uv365', 'original'),
    'bar_masked': ('uv365', 'masked'),
}

@app.route('/api/latest_image/<image_type>', methods=['GET'])
def get_latest_image(image_type):
    """Serve the latest captured image as a viewable JPEG.
    
    Valid canonical types are ``uv255``, ``uv310``, ``uv365``, and ``vis``;
    append ``_masked`` for their masked variants. ``dome`` and ``bar`` remain
    temporary aliases for VIS and UV365 respectively.
    """
    masked = image_type.endswith('_masked')
    channel = image_type[:-7] if masked else image_type
    if image_type in _LATEST_IMAGE_ALIASES:
        channel, variant = _LATEST_IMAGE_ALIASES[image_type]
    else:
        variant = 'masked' if masked else 'original'
    if channel not in globals.latest_images:
        return jsonify({
            'error': f'Unknown image type: {image_type}. '
                     'Valid canonical types: uv255, uv310, uv365, vis (optionally _masked).'
        }), 400

    img_bgr = globals.latest_images[channel][variant]
    if img_bgr is None:
        return jsonify({
            'error': f'No {image_type} image available yet. Run a measurement first.'
        }), 404
    
    # Encode BGR numpy array to JPEG bytes
    success, jpeg_buf = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not success:
        return jsonify({'error': 'Failed to encode image'}), 500
    
    return Response(
        jpeg_buf.tobytes(),
        mimetype='image/jpeg',
        headers={'Cache-Control': 'no-cache, no-store, must-revalidate'}
    )


@app.route('/api/latest_images', methods=['GET'])
def get_latest_images_status():
    """Return availability status of all latest image types."""
    status = {}
    for channel, variants in globals.latest_images.items():
        for variant, img in variants.items():
            image_type = channel if variant == 'original' else f'{channel}_masked'
            status[image_type] = {
                'available': img is not None,
                'url': f'/api/latest_image/{image_type}' if img is not None else None,
            }
    return jsonify(status), 200


# =============================================================================
# Pipeline / Recipe endpoints
# =============================================================================

@app.route('/api/pipeline/step-catalog', methods=['GET'])
def get_step_catalog():
    """Return all available step definitions for the toolbox."""
    catalog = [d.to_dict() for d in pipeline_steps.STEP_DEFINITIONS.values()]
    return jsonify({'steps': catalog}), 200


@app.route('/api/pipeline/validate', methods=['POST'])
def validate_pipeline():
    """Validate a pipeline document without executing it."""
    data = request.get_json(silent=True)
    if not data:
        return jsonify({'error': 'Hiányzó JSON törzs', 'code': ErrorCode.PIPELINE_VALIDATION_FAILED, 'popup': True}), 400

    try:
        doc = PipelineDocument.from_dict(data)
    except Exception as e:
        return jsonify({'error': f'Érvénytelen dokumentum: {e}', 'code': ErrorCode.RECIPE_INVALID_FORMAT, 'popup': True}), 400

    errors = pipeline_validators.validate_pipeline(doc)
    if errors:
        return jsonify({
            'valid': False,
            'errors': [e.to_dict() for e in errors],
        }), 200

    return jsonify({'valid': True, 'errors': []}), 200


@app.route('/api/pipeline/browse-file', methods=['GET'])
def browse_for_file():
    """Open a native file dialog for image file selection (via subprocess to avoid Tkinter thread issues)."""
    try:
        result = subprocess.run(
            ['python', 'browse_dialog.py', 'file'],
            capture_output=True, text=True, timeout=120
        )
        output = json.loads(result.stdout.strip())
        return jsonify({"path": output.get("path", "")}), 200
    except Exception as e:
        app.logger.exception("Browse file dialog failed")
        return jsonify({"path": ""}), 200


@app.route('/api/pipeline/browse-folder', methods=['GET'])
def browse_for_folder():
    """Open a native folder dialog for image folder selection (via subprocess to avoid Tkinter thread issues)."""
    try:
        result = subprocess.run(
            ['python', 'browse_dialog.py', 'folder'],
            capture_output=True, text=True, timeout=120
        )
        output = json.loads(result.stdout.strip())
        return jsonify({"path": output.get("path", "")}), 200
    except Exception as e:
        app.logger.exception("Browse folder dialog failed")
        return jsonify({"path": ""}), 200


@app.route('/api/pipeline/browse-values-file', methods=['GET'])
def browse_for_values_file():
    """Open a native file dialog for CSV/TXT explicit values import."""
    try:
        result = subprocess.run(
            ['python', 'browse_dialog.py', 'values'],
            capture_output=True, text=True, timeout=120
        )
        output = json.loads(result.stdout.strip())
        return jsonify({"path": output.get("path", "")}), 200
    except Exception:
        app.logger.exception("Browse values file dialog failed")
        return jsonify({"path": ""}), 200


@app.route('/api/pipeline/import-explicit-values', methods=['POST'])
def import_explicit_values():
    """
    Import and validate explicit values from a CSV/TXT file.
    Rules:
      - exactly one non-empty line
      - comma-separated numeric values only
      - strictly increasing sequence
    """
    payload = request.get_json(silent=True) or {}
    file_path = payload.get('path', '')
    if not isinstance(file_path, str) or not file_path.strip():
        return jsonify({"error": "Hiányzó fájl elérési útvonal."}), 400

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]

        if len(lines) != 1:
            return jsonify({"error": "A fájlnak pontosan egy nem üres sort kell tartalmaznia."}), 400

        line = lines[0]
        parts = [p.strip() for p in line.split(',')]
        if len(parts) < 1 or any(p == '' for p in parts):
            return jsonify({"error": "Érvénytelen CSV formátum. Vesszővel elválasztott számok szükségesek."}), 400

        values = []
        for p in parts:
            try:
                values.append(float(p))
            except ValueError:
                return jsonify({"error": "A fájl csak számokat tartalmazhat."}), 400

        for i in range(1, len(values)):
            if values[i] <= values[i - 1]:
                return jsonify({"error": "Az értékeknek szigorúan növekvő sorrendben kell lenniük."}), 400

        return jsonify({
            "values": values,
            "values_csv": ", ".join(str(v) for v in values),
        }), 200
    except OSError:
        return jsonify({"error": "A fájl nem olvasható."}), 400
    except Exception:
        app.logger.exception("Explicit values import failed")
        return jsonify({"error": "Váratlan hiba az import során."}), 500


@app.route('/api/pipeline/calibrations', methods=['GET'])
def list_calibrations():
    """List saved calibration equations."""
    records = calibration_manager.list_calibrations()
    return jsonify({"calibrations": records}), 200


@app.route('/api/pipeline/calibrations', methods=['POST'])
def save_calibration():
    """Save a calibration equation with metadata."""
    payload = request.get_json(silent=True) or {}
    rec, err = calibration_manager.save_calibration(payload)
    if err:
        return jsonify({"error": err}), 400
    return jsonify({"message": "Kalibráció mentve", "calibration": rec}), 200


@app.route('/api/pipeline/preview', methods=['POST'])
def preview_pipeline():
    """Execute pipeline up to a selected step and return the result image + side outputs."""
    data = request.get_json(silent=True)
    if not data:
        return jsonify({'error': 'Hiányzó JSON törzs', 'code': ErrorCode.PIPELINE_VALIDATION_FAILED, 'popup': True}), 400

    preview_step = data.get('preview_step_index', -1)
    preview_image_index = data.get('preview_image_index', 0)
    single_image_only = data.get('single_image_only', False)
    omitted_indices = data.get('omitted_indices', [])
    pipeline_data = data.get('pipeline')
    if not pipeline_data:
        return jsonify({'error': 'Hiányzó pipeline adat', 'code': ErrorCode.PIPELINE_VALIDATION_FAILED, 'popup': True}), 400

    try:
        doc = PipelineDocument.from_dict(pipeline_data)
    except Exception as e:
        return jsonify({'error': f'Érvénytelen dokumentum: {e}', 'code': ErrorCode.RECIPE_INVALID_FORMAT, 'popup': True}), 400

    single_idx = preview_image_index if single_image_only else -1
    result = pipeline_engine.execute_pipeline(
        doc,
        up_to_step=preview_step,
        single_image_index=single_idx,
        omitted_indices=omitted_indices,
    )

    if not result.success:
        error_list = [e.to_dict() for e in result.errors]
        return jsonify({
            'success': False,
            'errors': error_list,
            'executed_up_to': result.executed_up_to,
        }), 200

    # Build response: JPEG image + side outputs from the data dict
    side_outputs = pipeline_engine.extract_side_outputs(result.data)

    response_data = {
        'success': True,
        'executed_up_to': result.executed_up_to,
        'side_outputs': side_outputs,
    }

    # Encode the requested image from the data dict as preview
    # Only use circle overlay if the currently selected step IS detect_circles
    current_step_is_detect_circles = (
        preview_step >= 0 and 
        preview_step < len(doc.steps) and 
        doc.steps[preview_step].step_def_id == 'detect_circles'
    )
    
    if (result.data and current_step_is_detect_circles and 
        side_outputs.get("circle_overlay_base64") and len(side_outputs["circle_overlay_base64"]) > 0):
        # Use circle overlay image only if detect_circles is the current step
        img_idx = max(0, min(preview_image_index, len(side_outputs["circle_overlay_base64"]) - 1))
        response_data['image_base64'] = side_outputs["circle_overlay_base64"][img_idx]
        response_data['is_grayscale'] = False  # Circle overlay is always BGR
        # Get dimensions from original image for metadata
        if result.data.get("images"):
            img = result.data["images"][img_idx]
            if img is not None and hasattr(img, 'shape'):
                response_data['image_width'] = img.shape[1]
                response_data['image_height'] = img.shape[0]
    elif result.data and result.data.get("images"):
        img_idx = max(0, min(preview_image_index, len(result.data["images"]) - 1))
        img = result.data["images"][img_idx]
        if img is not None and hasattr(img, 'shape'):
            # Convert single-channel to BGR for JPEG encoding
            is_grayscale = img.ndim == 2
            if is_grayscale:
                img_enc = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                img_enc = img
            success_enc, jpeg_buf = cv2.imencode('.jpg', img_enc, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if success_enc:
                import base64
                response_data['image_base64'] = base64.b64encode(jpeg_buf.tobytes()).decode('ascii')
                response_data['image_width'] = img.shape[1]
                response_data['image_height'] = img.shape[0]
                # Signal frontend that this is grayscale (2-channel → color mapping for display)
                response_data['is_grayscale'] = is_grayscale
    
    if result.data:
        response_data['image_count'] = result.data.get("_original_count", len(result.data.get("images", [])))

    return jsonify(response_data), 200


@app.route('/api/pipeline/color-thresh-preview', methods=['POST'])
def color_thresh_live_preview():
    """Generate a live preview mask overlay for color_thresh with current parameter values."""
    try:
        payload = request.get_json(silent=True) or {}
        pipeline_dict = payload.get('pipeline', {})
        preview_index = payload.get('selected_step_index', -1)
        image_index = payload.get('preview_image_index', 0)
        current_params = payload.get('current_params', {})  # Current slider values
        
        if preview_index < 0 or not pipeline_dict or not pipeline_dict.get('steps'):
            return jsonify({'error': 'Invalid pipeline'}), 400
        
        # Run pipeline up to the color_thresh step
        try:
            doc = PipelineDocument.from_dict(pipeline_dict)
        except Exception as e:
            return jsonify({'error': f'Invalid pipeline: {e}'}), 400
        
        # Run up to selected step with current parameters
        result = pipeline_engine.execute_pipeline(doc, up_to_step=preview_index, single_image_index=image_index)
        
        if not result.success or not result.data or not result.data.get("images"):
            return jsonify({'error': 'Pipeline execution failed'}), 400
        
        # Get the input to color_thresh (the current output)
        img_idx = min(image_index, len(result.data["images"]) - 1)
        input_img = result.data["images"][img_idx]
        
        # Get metadata for color space
        space = "HSV"
        if result.data.get("meta", {}).get("select_channel"):
            space = result.data["meta"]["select_channel"].get("space", "HSV")
        
        # Build thresholds from current_params
        channel_mapping = {
            "HSV": [("H", "H_min", "H_max"), ("S", "S_min", "S_max"), ("V", "V_min", "V_max")],
            "BGR": [("B", "B_min", "B_max"), ("G", "G_min", "G_max"), ("R", "R_min", "R_max")],
            "LAB": [("L", "L_min", "L_max"), ("A", "A_min", "A_max"), ("B", "Lab_B_min", "Lab_B_max")],
            "GRAY": [("GRAY", "GRAY_min", "GRAY_max")],
        }
        
        thresholds = {}
        if space in channel_mapping:
            for ch_name, min_key, max_key in channel_mapping[space]:
                min_val = int(current_params.get(min_key, 0))
                max_val = int(current_params.get(max_key, 255))
                thresholds[ch_name] = (min_val, max_val)
        else:
            return jsonify({'error': f'Unknown color space: {space}'}), 400
        
        # Apply color threshold with current parameters
        invert = bool(current_params.get("invert", False))
        from proc_elements import color_threshold
        test_data = {
            "images": [input_img],
            "error": None,
            "count": 1,
            "meta": result.data.get("meta", {}),
            "results": {},
            "history": []
        }
        thresh_result = color_threshold(test_data, space=space, thresholds=thresholds, invert=invert)
        
        if thresh_result.get("error"):
            return jsonify({'error': f'Thresholding failed: {thresh_result["error"]}'}), 400
        
        # Get the mask overlay (already created by color_threshold)
        mask_overlays = thresh_result.get("results", {}).get("color_thresh_mask_overlays", [])
        if not mask_overlays or len(mask_overlays) == 0:
            return jsonify({'error': 'No mask overlay generated'}), 500
        
        overlay = mask_overlays[0]
        
        # Encode to base64
        success, jpeg_buf = cv2.imencode('.jpg', overlay, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if success:
            import base64
            b64_str = base64.b64encode(jpeg_buf.tobytes()).decode('ascii')
            return jsonify({
                'success': True,
                'image_base64': f'data:image/jpeg;base64,{b64_str}',
                'image_width': overlay.shape[1],
                'image_height': overlay.shape[0],
            }), 200
        else:
            return jsonify({'error': 'Failed to encode image'}), 500
        
    except Exception as e:
        app.logger.exception("color_thresh_live_preview failed")
        return jsonify({'error': f'Preview generation failed: {str(e)[:200]}'}), 500


@app.route('/api/pipeline/generate-montage', methods=['POST'])
def generate_montage():
    """
    Generate a montage grid of all currently loaded images.
    Returns Base64-encoded montage image.
    """
    try:
        from montage_utils import create_montage
        import base64
        
        payload = request.get_json(silent=True) or {}
        image_paths = payload.get('image_paths', [])
        
        if not image_paths or len(image_paths) == 0:
            return jsonify({'error': 'No images provided', 'code': 'E_NO_IMAGES', 'popup': True}), 400
        
        if len(image_paths) < 2:
            return jsonify({'error': 'Montage requires at least 2 images', 'code': 'E_FEW_IMAGES', 'popup': True}), 400
        
        # Generate montage
        montage_img = create_montage(image_paths, target_cell_width=200, target_cell_height=200, label_height=30, debug=False)
        
        if montage_img is None:
            return jsonify({'error': 'Failed to create montage', 'code': 'E_MONTAGE_FAILED', 'popup': True}), 500
        
        # Encode as JPEG Base64
        _, jpeg_data = cv2.imencode('.jpg', montage_img)
        b64_image = base64.b64encode(jpeg_data).decode('utf-8')
        
        return jsonify({
            'success': True,
            'montage_base64': b64_image,
            'image_count': len(image_paths),
            'montage_width': montage_img.shape[1],
            'montage_height': montage_img.shape[0]
        }), 200
    
    except Exception as e:
        app.logger.error(f"Montage generation error: {str(e)}")
        return jsonify({'error': f'Montage error: {str(e)[:200]}', 'code': 'E_MONTAGE_ERROR', 'popup': True}), 500


@app.route('/api/pipeline/get-step-images-montage', methods=['POST'])
def get_step_images_montage():
    """
    Generate a montage of all images processed by a specific pipeline step.
    Returns Base64-encoded montage showing all output images from that step.
    """
    try:
        import base64
        from montage_utils import calculate_grid_layout
        
        payload = request.get_json(silent=True) or {}
        pipeline_dict = payload.get('pipeline', {})
        step_index = payload.get('step_index', -1)
        
        app.logger.info(f"Montage request: step_index={step_index}, has_pipeline={bool(pipeline_dict)}")
        
        if not pipeline_dict or step_index < 0:
            return jsonify({'error': 'Invalid pipeline or step index', 'code': 'E_INVALID_REQUEST', 'popup': True}), 400
        
        # Parse pipeline
        try:
            doc = PipelineDocument.from_dict(pipeline_dict)
        except Exception as e:
            app.logger.error(f"Failed to parse pipeline: {str(e)}")
            return jsonify({'error': f'Invalid pipeline: {str(e)[:100]}', 'code': 'E_INVALID_PIPELINE', 'popup': True}), 400
        
        if step_index >= len(doc.steps):
            return jsonify({'error': 'Step index out of range', 'code': 'E_STEP_OUT_OF_RANGE', 'popup': True}), 400
        
        # Execute pipeline up to the specified step (for ALL images, not just one)
        try:
            app.logger.info(f"Executing pipeline up to step {step_index} (thumbnail mode)")
            result = pipeline_engine.execute_pipeline(doc, up_to_step=step_index, thumbnail_max_dim=400)
            app.logger.info(f"Pipeline execution result: success={result.success}, has_data={result.data is not None}")
        except Exception as e:
            app.logger.error(f"Pipeline execution failed: {str(e)}")
            return jsonify({'error': f'Pipeline execution failed: {str(e)[:100]}', 'code': 'E_PIPELINE_EXEC', 'popup': True}), 500
        
        if not result.success or not result.data or not result.data.get("images"):
            return jsonify({'error': 'No images produced by this step', 'code': 'E_NO_OUTPUT_IMAGES', 'popup': True}), 400
        
        images = result.data.get("images", [])
        if len(images) == 0:
            return jsonify({'error': 'No images to display', 'code': 'E_NO_IMAGES', 'popup': True}), 400
        
        app.logger.info(f"Processing {len(images)} images for montage")
        
        # Filter out None images and convert to BGR if needed
        valid_images = []
        for idx, img in enumerate(images):
            if img is not None and hasattr(img, 'shape') and len(img.shape) >= 2:
                # Convert grayscale to BGR for montage
                if img.ndim == 2:
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                else:
                    img_bgr = img
                valid_images.append(img_bgr)
            else:
                app.logger.warning(f"Skipping image {idx}: is_none={img is None}, has_shape={hasattr(img, 'shape')}")
        
        if len(valid_images) == 0:
            return jsonify({'error': 'No valid images to display', 'code': 'E_NO_VALID_IMAGES', 'popup': True}), 400
        
        app.logger.info(f"Got {len(valid_images)} valid images")
        
        # Calculate grid layout
        rows, cols = calculate_grid_layout(len(valid_images))
        app.logger.info(f"Grid layout: {rows}x{cols}")
        
        # Get image dimensions
        sample_img = valid_images[0]
        img_h, img_w = sample_img.shape[:2]
        
        # Create montage with labels
        cell_width = max(150, min(300, 2000 // cols))
        cell_height = max(150, min(300, 2000 // rows))
        label_height = 30
        
        grid_w = cols * (cell_width + 2) + 2
        grid_h = rows * (cell_height + label_height + 2) + 2
        
        montage = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
        montage[:] = (30, 30, 30)  # Dark background
        
        for idx, img in enumerate(valid_images):
            row = idx // cols
            col = idx % cols
            
            x = col * (cell_width + 2) + 2
            y = row * (cell_height + label_height + 2) + 2
            
            # Resize image to fit cell
            h, w = img.shape[:2]
            if w <= 0 or h <= 0:
                app.logger.warning(f"Skipping image {idx}: invalid dimensions w={w}, h={h}")
                continue
            scale = min(cell_width / w, cell_height / h)
            new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
            resized = cv2.resize(img, (new_w, new_h))
            
            # Center in cell
            offset_x = (cell_width - new_w) // 2
            offset_y = (cell_height - new_h) // 2
            montage[y+offset_y:y+offset_y+new_h, x+offset_x:x+offset_x+new_w] = resized
            
            # Draw border
            cv2.rectangle(montage, (x, y), (x+cell_width, y+cell_height), (100, 100, 100), 1)
            
            # Add label
            label = f"Img {idx+1}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            thickness = 1
            (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
            text_x = x + (cell_width - text_w) // 2
            text_y = y + cell_height + label_height - 8
            cv2.putText(montage, label, (text_x, text_y), font, font_scale, (200, 200, 200), thickness)
        
        # Encode as JPEG Base64
        _, jpeg_data = cv2.imencode('.jpg', montage, [cv2.IMWRITE_JPEG_QUALITY, 85])
        b64_image = base64.b64encode(jpeg_data).decode('utf-8')
        
        app.logger.info(f"Montage generated successfully: {montage.shape[1]}x{montage.shape[0]}")
        
        return jsonify({
            'success': True,
            'montage_base64': b64_image,
            'image_count': len(valid_images),
            'montage_width': montage.shape[1],
            'montage_height': montage.shape[0],
            'grid_rows': rows,
            'grid_cols': cols,
            'cell_width': cell_width,
            'cell_height': cell_height,
            'label_height': label_height
        }), 200
    
    except Exception as e:
        app.logger.error(f"Step images montage error: {type(e).__name__}: {str(e)}", exc_info=True)
        return jsonify({'error': f'Montage error: {str(e)[:200]}', 'code': 'E_MONTAGE_ERROR', 'popup': True}), 500


def _safe_filename_part(value: str) -> str:
    value = str(value or '')
    invalid = '<>:"/\\|?*'
    for ch in invalid:
        value = value.replace(ch, '_')
    return value


def _prepare_image_for_disk(image: np.ndarray) -> np.ndarray:
    if image is None:
        return image
    if image.dtype == np.uint8 or image.dtype == np.uint16:
        return image
    if image.dtype == np.bool_:
        return (image.astype(np.uint8) * 255)

    arr = image.astype(np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr, dtype=np.uint8)

    min_v = float(np.min(finite))
    max_v = float(np.max(finite))
    if max_v <= min_v:
        return np.zeros_like(arr, dtype=np.uint8)

    arr = (arr - min_v) / (max_v - min_v)
    arr = np.clip(arr * 255.0, 0, 255)
    return arr.astype(np.uint8)


def _next_available_path(folder: str, filename: str) -> str:
    base, ext = os.path.splitext(filename)
    candidate = os.path.join(folder, filename)
    index = 1
    while os.path.exists(candidate):
        candidate = os.path.join(folder, f"{base}_{index}{ext}")
        index += 1
    return candidate


def _is_numeric_scalar(value) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


def _table_from_value(value):
    if value is None:
        return None, None

    if isinstance(value, dict):
        if value and all(_is_numeric_scalar(v) for v in value.values()):
            rows = [[str(k), v] for k, v in value.items()]
            return ["key", "value"], rows
        return None, None

    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return ["value"], [[value.item()]]
        if value.ndim == 1:
            return [f"col_{i + 1}" for i in range(value.shape[0])], [value.tolist()]
        if value.ndim >= 2:
            arr2 = value.reshape(value.shape[0], -1)
            headers = [f"col_{i + 1}" for i in range(arr2.shape[1])]
            return headers, arr2.tolist()

    if isinstance(value, list):
        if not value:
            return None, None

        if all(isinstance(r, dict) for r in value):
            keys = []
            seen = set()
            for rec in value:
                for k in rec.keys():
                    if k not in seen:
                        seen.add(k)
                        keys.append(str(k))
            rows = [[rec.get(k, "") for k in keys] for rec in value]
            return keys, rows

        if all(isinstance(r, (list, tuple)) for r in value):
            width = max((len(r) for r in value), default=0)
            headers = [f"col_{i + 1}" for i in range(width)]
            rows = [list(r) + [""] * max(0, width - len(r)) for r in value]
            return headers, rows

        if all(_is_numeric_scalar(v) for v in value):
            return ["value"], [[v] for v in value]

    return None, None


def _pick_numeric_table(data_dict: dict):
    results = data_dict.get('results', {}) if isinstance(data_dict.get('results', {}), dict) else {}
    for key, value in results.items():
        headers, rows = _table_from_value(value)
        if headers and rows:
            return str(key), headers, rows
    return '', [], []


def _ensure_csv_filename(raw_name: str) -> str:
    name = str(raw_name or '').strip() or 'adattomb.csv'
    name = _safe_filename_part(name)
    base, ext = os.path.splitext(name)
    if ext.lower() != '.csv':
        name = f"{base or 'adattomb'}.csv"
    return name


@app.route('/api/pipeline/save-images', methods=['POST'])
def pipeline_save_images():
    """Execute pipeline up to save_images node input and write all resulting images to disk."""
    data = request.get_json(silent=True) or {}
    pipeline_data = data.get('pipeline')
    step_index = data.get('step_index')

    if pipeline_data is None or step_index is None:
        return jsonify({'error': 'Hiányzó pipeline vagy step_index mező.'}), 400

    try:
        step_index = int(step_index)
    except (TypeError, ValueError):
        return jsonify({'error': 'Érvénytelen step_index.'}), 400

    try:
        doc = PipelineDocument.from_dict(pipeline_data)
    except Exception as e:
        return jsonify({'error': f'Érvénytelen pipeline dokumentum: {e}'}), 400

    if step_index < 0 or step_index >= len(doc.steps):
        return jsonify({'error': 'A step_index tartományon kívül esik.'}), 400

    step = doc.steps[step_index]
    if step.step_def_id != 'save_images':
        return jsonify({'error': 'A kiválasztott lépés nem Kép mentése típusú.'}), 400

    params = step.param_values or {}
    output_folder = str(params.get('output_folder', '')).strip()
    name_prefix = _safe_filename_part(params.get('name_prefix', ''))
    name_suffix = _safe_filename_part(params.get('name_suffix', ''))

    if not output_folder:
        return jsonify({'error': 'A kimeneti mappa kötelező.'}), 400

    try:
        os.makedirs(output_folder, exist_ok=True)
    except Exception as e:
        return jsonify({'error': f'A mappa nem hozható létre: {e}'}), 400

    exec_result = pipeline_engine.execute_pipeline(doc, up_to_step=step_index - 1)
    if not exec_result.success:
        return jsonify({
            'error': 'A pipeline futtatása sikertelen a mentés előtt.',
            'errors': [e.to_dict() for e in exec_result.errors],
        }), 400

    run_data = exec_result.data or {}
    images = run_data.get('images', [])
    original_paths = run_data.get('_original_paths', run_data.get('paths', []))

    scale_bar_params = data.get('scale_bar_overlay')
    if scale_bar_params and isinstance(scale_bar_params, dict) and images:
        overlay_data = {'images': list(images)}
        overlay_data = _apply_scale_bar_overlay(overlay_data, **scale_bar_params)
        images = overlay_data.get('images', images)

    if not isinstance(images, list) or not images:
        return jsonify({'error': 'Nincs menthető kép a kimeneten.'}), 400

    saved_paths: list[str] = []
    for idx, img in enumerate(images):
        if img is None:
            continue

        original_name = ''
        if isinstance(original_paths, list) and idx < len(original_paths):
            original_name = os.path.basename(str(original_paths[idx]))
        if not original_name:
            original_name = f'image_{idx + 1:03d}.png'

        stem, ext = os.path.splitext(original_name)
        if not ext:
            ext = '.png'

        safe_name = f"{name_prefix}{_safe_filename_part(stem)}{name_suffix}{ext}"
        target_path = _next_available_path(output_folder, safe_name)

        image_to_save = _prepare_image_for_disk(img)
        if image_to_save is None:
            continue

        ok = cv2.imwrite(target_path, image_to_save)
        if ok:
            saved_paths.append(target_path)

    return jsonify({
        'saved_count': len(saved_paths),
        'saved_paths': saved_paths,
        'output_folder': output_folder,
    }), 200


@app.route('/api/pipeline/save-array', methods=['POST'])
def pipeline_save_array():
    """Execute pipeline before save_array node and write first numeric table to CSV."""
    data = request.get_json(silent=True) or {}
    pipeline_data = data.get('pipeline')
    step_index = data.get('step_index')

    if pipeline_data is None or step_index is None:
        return jsonify({'error': 'Hiányzó pipeline vagy step_index mező.'}), 400

    try:
        step_index = int(step_index)
    except (TypeError, ValueError):
        return jsonify({'error': 'Érvénytelen step_index.'}), 400

    try:
        doc = PipelineDocument.from_dict(pipeline_data)
    except Exception as e:
        return jsonify({'error': f'Érvénytelen pipeline dokumentum: {e}'}), 400

    if step_index < 0 or step_index >= len(doc.steps):
        return jsonify({'error': 'A step_index tartományon kívül esik.'}), 400

    step = doc.steps[step_index]
    if step.step_def_id != 'save_array':
        return jsonify({'error': 'A kiválasztott lépés nem Adattömb mentése típusú.'}), 400

    params = step.param_values or {}
    output_folder = str(params.get('output_folder', '')).strip()
    filename = _ensure_csv_filename(params.get('filename', 'adattomb.csv'))

    if not output_folder:
        return jsonify({'error': 'A mentési hely kötelező.'}), 400

    try:
        os.makedirs(output_folder, exist_ok=True)
    except Exception as e:
        return jsonify({'error': f'A mappa nem hozható létre: {e}'}), 400

    exec_result = pipeline_engine.execute_pipeline(doc, up_to_step=step_index - 1)
    if not exec_result.success:
        return jsonify({
            'error': 'A pipeline futtatása sikertelen a mentés előtt.',
            'errors': [e.to_dict() for e in exec_result.errors],
        }), 400

    run_data = exec_result.data or {}
    source_key, headers, rows = _pick_numeric_table(run_data)
    if not headers or not rows:
        return jsonify({'error': 'Nem található menthető numerikus adattömb.'}), 400

    target_path = _next_available_path(output_folder, filename)

    try:
        import csv
        with open(target_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for row in rows:
                writer.writerow(row)
    except Exception as e:
        return jsonify({'error': f'CSV mentési hiba: {e}'}), 500

    return jsonify({
        'saved_path': target_path,
        'row_count': len(rows),
        'col_count': len(headers),
        'source_key': source_key,
    }), 200


@app.route('/api/recipes', methods=['GET'])
def list_recipes():
    """List all saved recipes."""
    recipes = recipe_manager.list_recipes()
    return jsonify({'recipes': recipes}), 200


@app.route('/api/recipe-folders', methods=['GET'])
def list_recipe_folders():
    return jsonify({'folders': recipe_manager.list_recipe_folders()}), 200


@app.route('/api/recipe-folders', methods=['POST'])
def create_recipe_folder():
    data = request.get_json(silent=True) or {}
    folder, error = recipe_manager.create_recipe_folder(data.get('name', ''))
    if error:
        return jsonify({'error': error, 'code': ErrorCode.RECIPE_IO_ERROR, 'popup': True}), 400
    return jsonify({'folder': folder}), 201


@app.route('/api/recipe-folders/<folder_id>', methods=['PATCH'])
def rename_recipe_folder(folder_id):
    data = request.get_json(silent=True) or {}
    folder, error = recipe_manager.rename_recipe_folder(folder_id, data.get('name', ''))
    if error:
        return jsonify({'error': error, 'code': ErrorCode.RECIPE_IO_ERROR, 'popup': True}), 400
    return jsonify({'folder': folder}), 200


@app.route('/api/recipe-folders/<folder_id>', methods=['DELETE'])
def delete_recipe_folder(folder_id):
    success, error = recipe_manager.delete_recipe_folder(folder_id)
    if not success:
        return jsonify({'error': error, 'code': ErrorCode.RECIPE_IO_ERROR, 'popup': True}), 404
    return jsonify({'message': 'Receptmappa törölve'}), 200


@app.route('/api/recipes/<name>', methods=['GET'])
def get_recipe(name):
    """Load a recipe by name."""
    doc, error = recipe_manager.load_recipe(name)
    if error:
        return jsonify({'error': error, 'code': ErrorCode.RECIPE_NOT_FOUND, 'popup': True}), 404
    return jsonify(doc.to_dict()), 200


@app.route('/api/recipes', methods=['POST'])
def save_recipe():
    """Save a recipe."""
    data = request.get_json(silent=True)
    if not data:
        return jsonify({'error': 'Hiányzó JSON törzs', 'code': ErrorCode.RECIPE_IO_ERROR, 'popup': True}), 400

    try:
        doc = PipelineDocument.from_dict(data)
    except Exception as e:
        return jsonify({'error': f'Érvénytelen recept: {e}', 'code': ErrorCode.RECIPE_INVALID_FORMAT, 'popup': True}), 400

    success, error = recipe_manager.save_recipe(doc)
    if not success:
        return jsonify({'error': error, 'code': ErrorCode.RECIPE_IO_ERROR, 'popup': True}), 500

    return jsonify({'message': 'Recept mentve', 'name': doc.name}), 200


@app.route('/api/recipes/<name>', methods=['DELETE'])
def delete_recipe(name):
    """Delete a recipe by name."""
    success, error = recipe_manager.delete_recipe(name)
    if not success:
        return jsonify({'error': error, 'code': ErrorCode.RECIPE_NOT_FOUND, 'popup': True}), 404
    return jsonify({'message': 'Recept törölve'}), 200


@app.route('/api/recipes/<name>/description', methods=['PATCH'])
def update_recipe_description(name):
    """Update only the description of a recipe."""
    data = request.get_json(silent=True)
    if not data or 'description' not in data:
        return jsonify({'error': 'Hiányzó leírás mező', 'code': ErrorCode.RECIPE_IO_ERROR, 'popup': True}), 400
    success, error = recipe_manager.update_recipe_description(name, str(data['description']))
    if not success:
        return jsonify({'error': error, 'code': ErrorCode.RECIPE_NOT_FOUND, 'popup': True}), 404
    return jsonify({'message': 'Leírás frissítve'}), 200


@app.route('/api/recipes/<name>/folder', methods=['PATCH'])
def assign_recipe_folder(name):
    data = request.get_json(silent=True) or {}
    folder_id = data.get('folder_id')
    if folder_id is not None and not isinstance(folder_id, str):
        return jsonify({'error': 'Érvénytelen mappaazonosító', 'code': ErrorCode.RECIPE_IO_ERROR, 'popup': True}), 400
    success, error = recipe_manager.assign_recipe_folder(name, folder_id)
    if not success:
        return jsonify({'error': error, 'code': ErrorCode.RECIPE_IO_ERROR, 'popup': True}), 400
    return jsonify({'message': 'Receptmappa frissítve'}), 200


@app.route('/api/recipes/<name>/duplicate', methods=['POST'])
def duplicate_recipe(name):
    """Duplicate a recipe."""
    new_name, error = recipe_manager.duplicate_recipe(name)
    if error:
        return jsonify({'error': error, 'code': ErrorCode.RECIPE_IO_ERROR, 'popup': True}), 400
    return jsonify({'message': 'Recept másolva', 'new_name': new_name}), 200


# --- Compiled-mode route restriction ---
# When running as a standalone PyInstaller .exe (without Electron),
# only the latest_image endpoints (and health) are accessible.
# When launched by Electron, the TABLETSCANNER_FULL environment variable
# is set, enabling all routes.
_COMPILED_ALLOWED_PREFIXES = (
    '/api/latest_image/',
    '/api/latest_images',
    '/api/health',
)

@app.before_request
def _restrict_routes_in_compiled_mode():
    """In compiled (frozen) mode without TABLETSCANNER_FULL, block most endpoints."""
    if not getattr(sys, 'frozen', False):
        return None  # Development mode — allow everything

    if os.environ.get('TABLETSCANNER_FULL', '') == '1':
        return None  # Launched by Electron — allow everything

    path = request.path
    for prefix in _COMPILED_ALLOWED_PREFIXES:
        if path.startswith(prefix) or path == prefix:
            return None  # Allowed

    return jsonify({
        'error': 'This endpoint is not available in standalone mode.',
    }), 403


if __name__ == '__main__':
    multiprocessing.freeze_support()
    
    load_settings()
    initialize_cameras()
    initialize_serial_devices()
    
    import threading
    four_channel_monitor = threading.Thread(
        target=four_channel_lamp_timeout_monitor,
        daemon=True,
        name="FourChannelLampTimeoutMonitor"
    )
    four_channel_monitor.start()
    app.logger.info("Four-channel lamp timeout monitor thread started")
    
    try:
        app.run(debug=False, use_reloader=False)
    finally:
        shutdown_devices()
