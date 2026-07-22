import io
from flask import Flask, jsonify, request, Response, send_file
from flask_cors import CORS
import cv2
import time
import globals
from pypylon import pylon
from cameracontrol import (apply_camera_settings, 
                           validate_and_set_camera_param, get_camera_properties, stream_video,
                           load_camera_profile)
import porthandler
import motioncontrols
import filter_revolver
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
    load_settings,
    save_settings,
    update_lamp_output_selectors,
    update_filter_settings,
    get_settings,
    validate_capture_plan,
    default_filter_settings,
    validate_filter_settings,
    validate_lamp_output_selectors,
    validate_lamp_settings,
    validate_motion_simulation_settings,
    UV_LAMP_CHANNELS,
)
from light_control import (
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
from pipeline_types import PipelineDocument


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



GRID_SIZE_DEFAULT = 10
ORIGIN_X_DEFAULT = 20.0
ORIGIN_Y_DEFAULT = 20.0
SPACING_DEFAULT = 20.0


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


def _is_serial_disconnect(exc):
    """Check if an exception is caused by a USB/serial disconnection."""
    msg = str(exc).lower()
    return any(keyword in msg for keyword in [
        'serialexception', 'writefile failed', 'permissionerror',
        'clearcommerror', 'device', 'usb'
    ])


def _is_camera_disconnect(exc):
    """Check if an exception is caused by a camera disconnection or failure."""
    msg = str(exc).lower()
    return any(keyword in msg for keyword in [
        'camera not ready', 'camera disconnected', 'grab failed',
        'failed to grab', 'physically removed', 'not open',
        'camera is not grabbing'
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

        # Non-blocking quick probe under the lock so we don't corrupt
        # concurrent serial operations (e.g. homing, moves).
        try:
            buf = bytearray()
            deadline = time.monotonic() + 0.15
            with porthandler.motion_lock:
                ser.write(b'M105\n')
                while time.monotonic() < deadline:
                    iw = getattr(ser, 'in_waiting', 0) or 0
                    if not iw:
                        break
                    chunk = ser.read(min(iw, 64))
                    if chunk:
                        buf += chunk
                        if b"ok" in buf.lower():
                            break
            if buf and b"ok" not in buf.lower():
                app.logger.debug(f"M105 non-ok reply: {buf[:64]!r}")
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

    ser = globals.motion_platform
    if not ser or not getattr(ser, 'is_open', False):
        return jsonify({'ok': False, 'error': 'Motion platform not connected'}), 503

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
        if not globals.toolhead_homed or not globals.filter_revolver_homed:
            return jsonify({
                'error': 'Home the motion platform and filter revolver before rotating it.',
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
        globals.motion_busy = False
        return jsonify(_filter_revolver_status()), 200
    except (OSError, PermissionError):
        return _handle_motion_usb_disconnect(motion_platform, 'filter revolver rotation')
    except (TimeoutError, ValueError) as error:
        app.logger.warning('Filter revolver rotation failed: %s', error)
        return jsonify({
            'error': str(error),
            'code': ErrorCode.GENERIC,
            'popup': True,
        }), 504
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
        if not globals.toolhead_homed or not globals.filter_revolver_homed:
            return jsonify({
                'error': 'Home the motion platform and filter revolver before rotating it.',
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
        globals.motion_busy = False
        return jsonify({
            **_filter_revolver_status(),
            'direction': direction,
            'steps': steps,
        }), 200
    except (OSError, PermissionError):
        return _handle_motion_usb_disconnect(motion_platform, 'filter revolver selection')
    except (TimeoutError, ValueError) as error:
        app.logger.warning('Filter revolver selection failed: %s', error)
        return jsonify({
            'error': str(error),
            'code': ErrorCode.GENERIC,
            'popup': True,
        }), 504
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
    
    Uses a cached serial number to avoid expensive EnumerateDevices()
    calls on every 1-second poll. Only re-enumerates on cache miss or
    when the camera was previously disconnected.
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

    # Only re-enumerate devices when we need to verify physical presence.
    # Cache the serial number at connect time to avoid expensive USB
    # enumeration on every poll.
    if is_connected:
        cached_serial = getattr(globals, '_cached_camera_serial', None)
        try:
            open_serial = camera.GetDeviceInfo().GetSerialNumber()
            # Cache for future polls
            globals._cached_camera_serial = open_serial
        except Exception:
            open_serial = None
            globals._cached_camera_serial = None

        # Fast path: serial matches cache → device is still there
        if cached_serial and open_serial and cached_serial == open_serial:
            pass  # All good, skip enumeration
        else:
            # Slow path: verify via device enumeration (first poll or serial changed)
            try:
                devices = pylon.TlFactory.GetInstance().EnumerateDevices()
            except Exception:
                devices = []

            present_serials = []
            try:
                for dev in devices:
                    try:
                        present_serials.append(dev.GetSerialNumber())
                    except Exception:
                        pass
            except Exception:
                pass

            if not present_serials or (open_serial and open_serial not in present_serials):
                # Device is gone → clean up and flip to disconnected
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

        app.logger.info(f"Sending camera settings to frontend")
        return jsonify({
            "camera_params": camera_settings
        }), 200

    except Exception as e:
        app.logger.exception("Failed to get camera settings")
        return jsonify({
            "error": "Failed to read camera settings",
            "code": ErrorCode.GENERIC,
            "details": str(e),
            "popup": True
        }), 500
    
@app.route('/api/update-camera-settings', methods=['POST'])
def update_camera_settings():
    """Update one global persisted camera setting (ExposureTime or Gamma)."""
    try:
        data = request.get_json(silent=True) or {}
        setting_name = data.get('setting_name')
        setting_value = data.get('setting_value')

        if setting_name not in ('ExposureTime', 'Gamma'):
            return jsonify({"error": "Only ExposureTime and Gamma are configurable.", "code": ErrorCode.GENERIC, "popup": True}), 400

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
    
    
# Function to move the toolhead by a given amount (relative movement)
# Function to move the toolhead by a given amount (relative movement)
@app.route('/api/move_toolhead_relative', methods=['POST'])
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

        move_args = {axis: adj}
        try:
            motioncontrols.move_relative(motion_platform, **move_args)
        except (OSError, PermissionError) as e:
            return _handle_motion_usb_disconnect(motion_platform, "relative move")

        # Update cached position after successful relative move
        globals.last_toolhead_pos[axis] = clamped

        return jsonify({
            'status': 'success',
            'requested': {'axis': axis, 'delta': value},
            'sent': {'axis': axis, 'delta': adj},
            'clamped': {axis: bool(clipped)},
            'limits': {axis: {'min': lo, 'max': hi}}
        }), 200

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

    
    
@app.route('/api/move_toolhead_absolute', methods=['POST'])
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
    
    
@app.route('/api/autofocus_coarse', methods=['POST'])
def autofocus_coarse():
    try:
        # Always clear abort flag so manual AF is not blocked by a previous auto-measurement stop
        globals.autofocus_abort = False

        motion_platform = globals.motion_platform
        if not motion_platform or not getattr(motion_platform, 'is_open', False):
            return jsonify({
                'status': 'error',
                'message': 'Motion platform not connected',
                'code': ErrorCode.MOTIONPLATFORM_DISCONNECTED,
                'popup': True
            }), 503

        # --- UV-light exposure safeguard ---
        # When the UV light is active, run an under/over-exposure gate
        # before starting the (blocking) autofocus sweep.
        if light_controller.status()['active_channel'] in UV_LAMP_CHANNELS:
            cam = globals.camera
            if cam and cam.IsOpen():
                from cameracontrol import grab_and_convert_frame
                try:
                    with globals.grab_lock:
                        frame_bgr = grab_and_convert_frame(cam, timeout_ms=3000)
                    exp_result = under_over.exposure_gate_from_frame(frame_bgr)
                    exp_code = exp_result.get('code')  # None when OK, "E2302"/"E2303" on error
                    exp_metrics = {k: v for k, v in exp_result.items() if k not in ('status', 'code')}
                    app.logger.info(
                        f"[BAR-LIGHT GATE] status={exp_result['status']} code={exp_code} "
                        f"p95={exp_metrics.get('p95', 0):.1f} dr={exp_metrics.get('dr', 0):.1f} "
                        f"white={exp_metrics.get('white', 0):.3f}"
                    )
                    if exp_result['status'] != 'OK':
                        return jsonify({
                            'status': 'ERROR',
                            'code': exp_code,
                            **exp_metrics,
                        })
                except Exception as gate_err:
                    app.logger.warning(f"Bar-light exposure gate failed, proceeding: {gate_err}")

        data = request.get_json(silent=True) or {}
        skip_empty = bool(data.get('skip_empty_check', False))
        resp = autofocus_main.autofocus_coarse(motion_platform, do_frame_touch_check=False, skip_empty_check=skip_empty)
        return jsonify(resp)
    except (OSError, PermissionError) as e:
        return _handle_motion_usb_disconnect(motion_platform, "autofocus")
    except Exception as e:
        app.logger.exception("autofocus_coarse failed")  # logs full traceback
        return jsonify({
            'status': 'error',
            'message': str(e),
            'trace': traceback.format_exc()
        }), 500


@app.route('/api/lights/status', methods=['GET'])
def get_lights_status():
    try:
        return jsonify(light_controller.status()), 200
    except Exception as error:
        app.logger.exception('Failed to retrieve four-channel light status')
        return jsonify({'error': str(error), 'code': ErrorCode.GENERIC, 'popup': True}), 500


@app.route('/api/lights/activate', methods=['POST'])
def activate_light():
    try:
        data = request.get_json(silent=True) or {}
        status = light_controller.activate(data.get('channel'), data.get('mode'))
        return jsonify(status), 200
    except LampSettingsError as error:
        return jsonify({'error': str(error), 'code': ErrorCode.LAMP_SETTINGS_MISSING, 'popup': True}), 400
    except LightConfigurationError as error:
        return jsonify({'error': str(error), 'code': ErrorCode.GENERIC, 'popup': True}), 400
    except LightCommandError as error:
        return jsonify({'error': str(error), 'code': ErrorCode.MOTIONPLATFORM_DISCONNECTED, 'popup': True}), 503
    except Exception as error:
        app.logger.exception('Failed to activate four-channel light')
        return jsonify({'error': str(error), 'code': ErrorCode.GENERIC, 'popup': True}), 500


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

def _move_toolhead_absolute_impl(x_pos=None, y_pos=None, z_pos=None):
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
            live_pos = motioncontrols.get_toolhead_position(motion_platform, timeout=0.4)
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
        motioncontrols.move_to_position(
            motion_platform,
            planned.get('x'),
            planned.get('y'),
            planned.get('z')
        )
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
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e)
        }, 500

    # Update cached position with the planned values
    for ax, val in planned.items():
        globals.last_toolhead_pos[ax] = float(val)

    return {
        'status': 'success',
        'requested': requested,
        'sent': planned,
        'clamped': clamped_flags,
        'limits': limits_out
    }, 200


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
    """Apply the global camera settings used by every illumination channel.

    ``light`` is retained only for legacy callers and diagnostics.  Camera
    exposure and gamma are no longer coupled to the active lamp.
    """
    settings_data = get_settings()
    camera = globals.camera
    camera_properties = globals.camera_properties
    
    if not camera or not camera.IsOpen():
        app.logger.warning("Camera not open, cannot apply light settings")
        return
    
    if not camera_properties:
        try:
            camera_properties = get_camera_properties(camera)
            globals.camera_properties = camera_properties
        except Exception as e:
            app.logger.warning(f"Could not get camera properties: {e}")
            return
    
    light_settings = settings_data.get('camera_params', {})
    
    for setting_name, setting_value in light_settings.items():
        try:
            validate_and_set_camera_param(camera, setting_name, setting_value, camera_properties)
        except Exception as e:
            app.logger.warning(f"Could not apply {setting_name} for {light}: {e}")

def _capture_and_save_image(target_folder: str, filename: str, background_subtraction: bool = False,
                            light_type: str = None, filter_position: int | None = None) -> list:
    """Capture image from camera and save to target folder.
    
    If background_subtraction is True, also saves a masked version.
    Also caches the latest images in globals for the /api/latest_image endpoints.
    
    Returns:
        list of saved file paths (original, and optionally masked)
    """
    # Normalize path for cross-platform consistency
    target_folder = _normalize_path(target_folder)

    # Grab frame from camera
    frame_result = grab_camera_image()
    
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
    
    full_path = os.path.join(target_folder, f"{filename}.jpg")
    
    # Convert BGR -> RGB for Pillow
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    # Build EXIF metadata from settings + live camera values
    try:
        settings_data = get_settings()
        other = settings_data.get('other_settings', {})
        cam = globals.camera
        current_exposure = None
        current_gamma = None
        if cam and cam.IsOpen():
            try:
                current_exposure = cam.ExposureTime.GetValue()
            except Exception:
                pass
            try:
                current_gamma = cam.Gamma.GetValue()
            except Exception:
                pass
        meta_fields = {
            'objective': other.get('objective'),
            'spacer_rings': other.get('spacer_rings'),
            'camera_settings_file': other.get('camera_settings_file'),
            'exposure_time': current_exposure,
            'gamma': current_gamma,
            'wavelength': _canonical_light_channel(light_type),
            'filter_position': filter_position,
        }
        meta_fields = {k: v for k, v in meta_fields.items() if v is not None}
        meta_json = json.dumps(meta_fields, ensure_ascii=False) if meta_fields else None
    except Exception:
        meta_json = None

    if meta_json:
        exif_obj = Image.Exif()
        try:
            exif_obj[0x010E] = meta_json  # ImageDescription
        except Exception:
            pass
        pil_img.save(full_path, format='JPEG', quality=95, exif=exif_obj)
    else:
        pil_img.save(full_path, format='JPEG', quality=95)
    
    saved_paths = [full_path]
    
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
                _cache_latest_capture(light_type, masked, masked=True)
                app.logger.info(f"Background-subtracted image saved: {masked_path} (kind={kind})")
            else:
                app.logger.warning(f"Background subtraction found no object in {filename}")
        except Exception as e:
            app.logger.warning(f"Background subtraction failed for {filename}: {e}")
    
    return saved_paths


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
                              tablet_index: int, background_subtraction: bool = False) -> list:
    """Capture one persisted wavelength/filter row with the Octopus controller.

    The selected filter position is recorded in the filename and response.
    Automatic capture-plan positioning is intentionally not coupled to the
    separately acknowledged manual revolver controller yet.
    """
    wavelength = row['wavelength']
    filter_position = row['filter_position']
    mode = 'dimmed' if wavelength in ('uv255', 'uv310', 'uv365') else None
    activated = False
    try:
        light_controller.activate(wavelength, mode)
        activated = True
        _apply_camera_settings_for_light(wavelength)
        time.sleep(0.3)

        timestamp = _format_capture_timestamp(datetime.now())
        tablet_label = _tablet_index_to_label(tablet_index)
        filename = f"{measurement_name}_{timestamp}_{tablet_label}_{wavelength}_filter{filter_position}"
        return _capture_and_save_image(
            measurement_folder, filename,
            background_subtraction=background_subtraction,
            light_type=wavelength,
            filter_position=filter_position,
        )
    finally:
        if activated:
            try:
                light_controller.off(wavelength)
            except Exception as error:
                app.logger.error('Could not switch off %s after capture: %s', wavelength, error)


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
    try:
        # Reset abort flag for each new tablet
        if globals.autofocus_abort:
            globals.autofocus_abort = False
            app.logger.info("Autofocus abort flag cleared for new tablet")
        
        data = request.get_json() or {}
        
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
            z_pos=move_z
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
            app.logger.info(f"Tablet {tablet_index}: Starting autofocus ({'coarse' if is_first_tablet else 'fine'})")
            
            # Capture-plan measurements use VIS as their autofocus reference.
            # Legacy requests retain their existing dome-light behavior.
            if capture_plan is not None:
                light_controller.activate('vis')
            else:
                _turn_on_dome_light()
            _apply_camera_settings_for_light('dome')
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
                    af_result = autofocus_main.autofocus_coarse(motion_platform, do_frame_touch_check=True, before_auto=True)
                elif is_first_tablet:
                    # Autofocus unselected, first tablet: find focal plane with skip_empty_check + no color check
                    af_result = autofocus_main.autofocus_coarse(motion_platform, skip_empty_check=True, before_auto=False)
                
                af_status = af_result.get('status', 'ERROR')
                af_error_code = af_result.get('code')  # e.g. "E2000", "E2002", "E2003"
                if af_status == 'OK':
                    contour = af_result.get("final_contour") or af_result.get("contour")
                    globals.last_autofocus_contour = contour if contour else None
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
                from cameracontrol import grab_and_convert_frame
                cam = globals.camera
                with globals.grab_lock:
                    check_frame = grab_and_convert_frame(cam, timeout_ms=5000, retries=2)

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
        if not should_autofocus and not background_subtraction:
            app.logger.info(f"Tablet {tablet_index}: Running check_only (no AF, no BGR)")
            _turn_on_dome_light()
            _apply_camera_settings_for_light('dome')
            time.sleep(0.3)  # Let light and camera settings stabilize

            try:
                from cameracontrol import grab_and_convert_frame
                cam = globals.camera
                with globals.grab_lock:
                    frame_bgr = grab_and_convert_frame(cam, timeout_ms=5000, retries=2)

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

        # =====================================================
        # STEP 4: Capture images with selected lights
        # =====================================================
        if capture_plan is not None:
            for row in capture_plan:
                wavelength = row['wavelength']
                filter_position = row['filter_position']
                try:
                    app.logger.info(
                        'Tablet %s: Capturing %s image with filter position %s',
                        tablet_index, wavelength, filter_position,
                    )
                    saved_paths = _capture_capture_plan_row(
                        row, measurement_folder, measurement_name, tablet_index,
                        background_subtraction=background_subtraction,
                    )
                    saved_images.extend(saved_paths)
                    captured_plan_rows.append({
                        'wavelength': wavelength,
                        'filter_position': filter_position,
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
                    from cameracontrol import grab_and_convert_frame
                    try:
                        with globals.grab_lock:
                            gate_frame = grab_and_convert_frame(cam, timeout_ms=3000)
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
def save_raw_image_endpoint():
    data = request.get_json() or {}
    target_folder = _normalize_path(data.get('target_folder', ''))
    measurement_name = data.get('measurement_name')
    light_type = data.get('light_type') or 'dome'

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
            if light_type != 'dome':
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
            if light_type == 'bar':
                _turn_on_uv_dome_light()
                _apply_camera_settings_for_light('bar')
                time.sleep(0.3)  # Let light and camera settings stabilize
        except Exception as e:
            app.logger.warning(f"Manual save: contour detection under dome light failed: {e}")
            globals.last_autofocus_contour = None
            # Restore original light and continue — save image without background subtraction
            if light_type == 'bar':
                try:
                    _turn_on_uv_dome_light()
                    _apply_camera_settings_for_light('bar')
                    time.sleep(0.3)
                except Exception:
                    pass

    # --- Grab frame from camera ---
    img = grab_camera_image()  # your existing helper

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
        # Convert BGR (OpenCV) -> RGB (Pillow)
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        # Prepare EXIF metadata (ImageDescription and UserComment)
        metadata = data.get('metadata') if isinstance(data, dict) else None
        meta_json = None
        if isinstance(metadata, dict):
            meta_fields = {
                'objective': metadata.get('objective'),
                # support both possible keys used in templates
                'spacer_rings': metadata.get('spacer_rings'),
                'camera_settings_file': metadata.get('camera_settings_file'),
                'exposure_time': metadata.get('exposure_time'),
                'gamma': metadata.get('gamma')
            }
            # remove None entries
            meta_fields = {k: v for k, v in meta_fields.items() if v is not None}
            try:
                meta_json = json.dumps(meta_fields, ensure_ascii=False)
            except Exception:
                meta_json = None

        if meta_json:
            exif_obj = Image.Exif()
            # 0x010E = ImageDescription, 0x9286 = UserComment
            try:
                exif_obj[0x010E] = meta_json
            except Exception:
                pass
            pil_img.save(full_path, format='JPEG', quality=95, exif=exif_obj)
        else:
            pil_img.save(full_path, format='JPEG', quality=95)
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
        "path": full_path
    }
    if masked_path:
        result["masked_path"] = masked_path
    return jsonify(result), 200


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
        if not update_filter_settings(normalized_settings):
            raise OSError('Failed to persist filter settings.')
        return jsonify({'filter_settings': normalized_settings}), 200
    except ValueError as error:
        return jsonify({'error': str(error), 'code': ErrorCode.GENERIC, 'popup': True}), 400
    except Exception as error:
        app.logger.exception('Failed to update filter settings')
        return jsonify({'error': str(error), 'code': ErrorCode.GENERIC, 'popup': True}), 500


@app.route('/api/settings/motion/advanced', methods=['GET', 'PUT'])
def advanced_motion_settings():
    if request.method == 'GET':
        max_up, max_down = _height_offset_limits()
        return jsonify({
            'advanced_motion_settings': {
                'use_virtual_com_port': _use_virtual_motion_platform(),
                'max_height_offset_up_mm': max_up,
                'max_height_offset_down_mm': max_down,
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
    result = pipeline_engine.execute_pipeline(doc, up_to_step=preview_step, single_image_index=single_idx, omitted_indices=omitted_indices)

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
            app.logger.info(f"Executing pipeline up to step {step_index}")
            result = pipeline_engine.execute_pipeline(doc, up_to_step=step_index)
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
        _, jpeg_data = cv2.imencode('.jpg', montage)
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
