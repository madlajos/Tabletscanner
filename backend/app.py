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
import os
import sys
import math
from datetime import datetime
import subprocess
import json
from threading import Lock
from settings_manager import load_settings, save_settings, get_settings
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
import manual_bgr
import check_only



app = Flask(__name__)
app.secret_key = 'Egis'
CORS(app)
app.debug = True

setup_logger()
_EPS = 1e-6

# Might need to be removed
camera_properties = None
latest_frames = None

backend_ready = False



GRID_SIZE_DEFAULT = 10
ORIGIN_X_DEFAULT = 20.0
ORIGIN_Y_DEFAULT = 20.0
SPACING_DEFAULT = 20.0


### Error handling and logging ###
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
    return jsonify({
        'error': ERROR_MESSAGES.get(ErrorCode.MOTIONPLATFORM_DISCONNECTED, 'Motion platform disconnected'),
        'code': ErrorCode.MOTIONPLATFORM_DISCONNECTED,
        'popup': True
    }), 503


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


### Lamp Timeout Monitor ###
def lamp_timeout_monitor():
    """
    Background thread that checks every 10 seconds if lamps have been on for more than 20 seconds.
    If so, automatically turns them off and sets a flag for the frontend to detect.
    """
    LAMP_TIMEOUT_SECONDS = 300
    CHECK_INTERVAL_SECONDS = 10    # Check every 10 seconds
    
    app.logger.info("Lamp timeout monitor thread started")
    
    while True:
        try:
            time.sleep(CHECK_INTERVAL_SECONDS)
            
            current_time = time.time()
            turned_off_any = False
            
            # Check dome light
            if globals.lamp_dome_on_time is not None:
                elapsed = current_time - globals.lamp_dome_on_time
                if elapsed >= LAMP_TIMEOUT_SECONDS:
                    app.logger.info(f"Dome light auto-off after {elapsed:.0f}s of inactivity")
                    try:
                        ser = globals.motion_platform
                        if ser and ser.is_open:
                            porthandler.write_and_wait(ser, "M106 P0 S255", timeout=2.0)  # dome off
                            globals.lamp_dome_on_time = None
                            turned_off_any = True
                    except Exception as e:
                        app.logger.warning(f"Failed to auto-turn off dome light: {e}")
            
            # Check bar light
            if globals.lamp_bar_on_time is not None:
                elapsed = current_time - globals.lamp_bar_on_time
                if elapsed >= LAMP_TIMEOUT_SECONDS:
                    app.logger.info(f"Bar light auto-off after {elapsed:.0f}s of inactivity")
                    try:
                        ser = globals.motion_platform
                        if ser and ser.is_open:
                            porthandler.write_and_wait(ser, "M106 P1 S0", timeout=2.0)  # bar off
                            globals.lamp_bar_on_time = None
                            turned_off_any = True
                    except Exception as e:
                        app.logger.warning(f"Failed to auto-turn off bar light: {e}")
            
            # Set flag if any lamp was turned off
            if turned_off_any:
                globals.lamp_auto_turned_off = True
                        
        except Exception as e:
            app.logger.error(f"Lamp timeout monitor error: {e}")


### Serial Device Functions ###
# Connect/Disconnect Serial devices
@app.route('/api/connect-to-motionplatform', methods=['POST'])
def connect_motionplatform():
    try:
        app.logger.info("Attempting to connect to Motion platform")
        if porthandler.motion_platform and porthandler.motion_platform.is_open:
            app.logger.info("Motion platform already connected.")
            return jsonify({'message': 'Motion platform already connected'}), 200
    
        device = porthandler.connect_to_motion_platform()
        if device:
            porthandler.motion_platform = device
            globals.motion_platform = device
            app.logger.info("Successfully connected to Motion platform")
            return jsonify({'message': 'Motion platform connected', 'port': device.port}), 200
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
            return jsonify({'connected': True, 'busy': True, 'port': ser.port}), 200

        # Non-blocking quick probe (optional). Never reset buffers here.
        try:
            buf = bytearray()
            deadline = time.monotonic() + 0.15
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
            return jsonify({'connected': False}), 200
        except Exception as e:
            app.logger.debug(f"status probe error (ignored): {e}")
        return jsonify({'connected': True, 'port': ser.port}), 200

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
    """Check if axes are already homed by attempting to query position."""
    ser = porthandler.motion_platform or globals.motion_platform
    if not ser or not getattr(ser, 'is_open', False):
        return jsonify({'x': False, 'y': False, 'z': False}), 200

    try:
        with porthandler.motion_lock:
            pos = motioncontrols.get_toolhead_position(ser, timeout=0.3)
        # If we can get a valid position, axes are homed
        homed = all(k in pos and isinstance(pos[k], (int, float)) for k in ('x','y','z'))
        return jsonify({'x': homed, 'y': homed, 'z': homed}), 200
    except Exception as e:
        app.logger.debug(f"Could not query homed status: {e}")
        return jsonify({'x': False, 'y': False, 'z': False}), 200

    
@app.route('/api/home_toolhead', methods=['POST', 'OPTIONS'])
def api_home_toolhead():
    if request.method == 'OPTIONS':
        return ('', 204)

    data = request.get_json(silent=True) or {}
    axes = [a.lower()[0] for a in (data.get('axes') or []) if a]

    ser = globals.motion_platform
    if not ser or not getattr(ser, 'is_open', False):
        return jsonify({'ok': False, 'error': 'Motion platform not connected'}), 503

    globals.motion_busy = True
    try:
        # home all if no axes provided
        if axes:
            motioncontrols.home_axes(ser, *axes)
        else:
            motioncontrols.home_axes(ser)
        
        # Wait for homing to complete by draining any buffered responses
        # The board sends "ok" when ready, but may also send "echo:busy: processing" while working
        try:
            deadline = time.monotonic() + 20.0  # 20 second timeout for homing
            buf = bytearray()
            while time.monotonic() < deadline:
                iw = getattr(ser, 'in_waiting', 0) or 0
                if iw:
                    chunk = ser.read(min(iw, 256))
                    if chunk:
                        buf += chunk
                        # Homing complete when we see "ok" (at end of response)
                        if b"ok" in buf.lower() and (b"\n" in buf or len(buf) > 100):
                            break
                else:
                    time.sleep(0.05)
        except (OSError, PermissionError) as e:
            app.logger.warning(f"Motion platform disconnected during homing (USB error): {e}")
            try:
                ser.close()
            except Exception:
                pass
            globals.motion_platform = None
            porthandler.motion_platform = None
            return jsonify({
                'ok': False, 
                'error': ERROR_MESSAGES.get(ErrorCode.MOTIONPLATFORM_DISCONNECTED, 'Motion platform disconnected'),
                'code': ErrorCode.MOTIONPLATFORM_DISCONNECTED,
                'popup': True
            }), 503
        except Exception as e:
            app.logger.warning(f"Error waiting for homing completion: {e}")
        
        globals.toolhead_homed = True
        
        # Query and cache the position after successful homing
        try:
            pos = motioncontrols.get_toolhead_position(ser, timeout=2.0)
            if pos and all(k in pos for k in ('x', 'y', 'z')):
                globals.last_toolhead_pos = pos
                app.logger.info(f"Position cached after homing: X={pos.get('x')}, Y={pos.get('y')}, Z={pos.get('z')}")
        except Exception as e:
            app.logger.warning(f"Could not cache position after homing: {e}")
        
        return jsonify({
            'ok': True, 
            'homed_axes': axes or ['x','y','z'],
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

    
### Camera Functions ###
def stop_camera_stream():
    camera = globals.camera

    # Ensure the grab lock exists
    lock = getattr(globals, "grab_lock", None)
    if lock is None:
        globals.grab_lock = Lock()
        lock = globals.grab_lock

    # stream_running is a bool, not a dict
    running = bool(getattr(globals, "stream_running", False))
    if not running:
        return "Stream already stopped."

    try:
        # Signal all streaming loops/threads to stop
        globals.stream_running = False

        # Stop grabbing under the lock
        with lock:
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
        # Decide how you want to handle this. If invalid camera type is fatal, return here:
        return jsonify({"error": "Invalid camera type"}), 400
    except RuntimeError as re:
        app.logger.warning(f"Error stopping camera stream: {str(re)}")
        # Maybe we continue to shut down the camera anyway
    except Exception as e:
        app.logger.error(f"Failed to disconnect camera: {e}")
        return jsonify({"error": str(e)}), 500

    camera = globals.camera
    if camera and camera.IsGrabbing():
        camera.StopGrabbing()
        app.logger.info(f"Camera grabbing stopped.")

    if camera and camera.IsOpen():
        camera.Close()
        app.logger.info(f"Camera closed.")

    # Clean up references
    globals.camera = None
    camera_properties = None  # Make sure camera_properties is in scope
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
    """
    camera = getattr(globals, 'camera', None)
    is_streaming = bool(getattr(globals, 'stream_running', False))

    # Baseline "connected" = we have an open handle
    is_connected = bool(camera is not None)
    if is_connected:
        try:
            is_connected = camera.IsOpen()
        except Exception:
            # If calling IsOpen raises, treat as disconnected
            is_connected = False

    try:
        # Enumerate actual devices present now
        devices = pylon.TlFactory.GetInstance().EnumerateDevices()
    except Exception:
        devices = []

    # If we think we're connected, confirm the same device is still present
    if is_connected:
        try:
            open_serial = camera.GetDeviceInfo().GetSerialNumber()
        except Exception:
            open_serial = None

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
            # Device is gone -> clean up and flip to disconnected
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
            is_connected = False
            is_streaming = False

    return jsonify({
        "connected": bool(is_connected),
        "streaming": bool(is_streaming),
    }), 200
    
@app.route('/api/get-camera-settings', methods=['GET'])
def get_camera_settings():
    try:
        app.logger.info("API Call: /api/get-camera-settings")
        
        settings_data = get_settings()
        camera_settings_dome = settings_data.get('camera_params_dome', {})
        camera_settings_bar = settings_data.get('camera_params_bar', {})

        app.logger.info(f"Sending camera settings to frontend")
        return jsonify({
            "camera_params_dome": camera_settings_dome,
            "camera_params_bar": camera_settings_bar
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
    try:
        data = request.json
        setting_name = data.get('setting_name')
        setting_value = data.get('setting_value')

        app.logger.info(f"Updating camera setting: {setting_name} = {setting_value}")

        # Fetch camera and current camera_properties
        camera = globals.camera
        camera_properties = globals.camera_properties

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

        # NOTE: Direct camera settings endpoint deprecated. Use /api/update-camera-settings-light instead.
        # For now, log and return error.
        app.logger.warning("Direct camera settings update called; this endpoint is deprecated. Use /api/update-camera-settings-light")
        return jsonify({
            "error": "This endpoint is deprecated. Use /api/update-camera-settings-light instead.",
            "code": ErrorCode.GENERIC,
            "popup": False
        }), 400

    except Exception as e:
        app.logger.exception("Failed to update camera settings")
        return jsonify({"error": str(e)}), 500


@app.route('/api/update-camera-settings-light', methods=['POST'])
def update_camera_settings_light():
    """Update light-specific camera settings (Dome or Bar)"""
    try:
        data = request.json
        light = data.get('light')  # 'dome' or 'bar'
        setting_name = data.get('setting_name')
        setting_value = data.get('setting_value')
        apply_to_camera = data.get('apply_to_camera', True)  # Default to True for backwards compatibility

        if light not in ('dome', 'bar'):
            return jsonify({"error": "Invalid light. Must be 'dome' or 'bar'."}), 400

        app.logger.info(f"[CameraSettings] Updating {light} setting {setting_name}={setting_value}, apply_to_camera={apply_to_camera}")

        updated_value = setting_value  # Default to the input value

        # Only apply to camera hardware if apply_to_camera is True
        if apply_to_camera:
            # Fetch camera and current camera_properties
            camera = globals.camera
            camera_properties = globals.camera_properties

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

        # Always persist the value in settings.json under the light-specific section
        settings_data = get_settings()
        category = f'camera_params_{light}'
        if category not in settings_data:
            settings_data[category] = {}
        settings_data[category][setting_name] = updated_value
        save_settings()

        app.logger.info(f"{light} camera setting {setting_name} updated and saved to settings.json")

        return jsonify({
            "message": f"{light.capitalize()} camera {setting_name} updated and saved.",
            "updated_value": updated_value,
            "applied_to_camera": apply_to_camera
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
        
        resp = autofocus_main.autofocus_coarse(motion_platform)
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


@app.route('/api/send_gcode', methods=['POST'])
def send_gcode():
    try:
        data = request.get_json()
        command = data.get('command')
        if not command:
            return jsonify({'error': 'No command provided'}), 400

        ser = globals.motion_platform
        if not ser or not ser.is_open:
            return jsonify({'error': 'Motion platform not connected'}), 503

        porthandler.write(ser, command)
        
        # Track lamp on/off state for 5-minute auto-off feature
        cmd_upper = command.strip().upper()
        if 'M106 P0 S0' in cmd_upper:
            # Dome light ON (inverted logic: S0 = on)
            globals.lamp_dome_on_time = time.time()
            app.logger.debug("Dome light turned ON, timestamp tracked")
        elif 'M106 P0 S255' in cmd_upper:
            # Dome light OFF
            globals.lamp_dome_on_time = None
            app.logger.debug("Dome light turned OFF, timestamp cleared")
        elif 'M106 P1 S255' in cmd_upper:
            # Bar light ON
            globals.lamp_bar_on_time = time.time()
            app.logger.debug("Bar light turned ON, timestamp tracked")
        elif 'M106 P1 S0' in cmd_upper:
            # Bar light OFF
            globals.lamp_bar_on_time = None
            app.logger.debug("Bar light turned OFF, timestamp cleared")
        
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
    """Turn on dome light (M106 P0 S0) and turn off bar light (M106 P1 S0)."""
    ser = globals.motion_platform
    if ser and ser.is_open:
        porthandler.write_and_wait(ser, "M106 P1 S0", timeout=2.0)   # bar off
        porthandler.write_and_wait(ser, "M106 P0 S0", timeout=2.0)   # dome on (S0 = on for this inverted setup)
        # Track dome light on time for 5-minute auto-off
        globals.lamp_dome_on_time = time.time()
        globals.lamp_bar_on_time = None

def _turn_on_bar_light():
    """Turn on bar light (M106 P1 S255) and turn off dome light (M106 P0 S255)."""
    ser = globals.motion_platform
    if ser and ser.is_open:
        porthandler.write_and_wait(ser, "M106 P0 S255", timeout=2.0)  # dome off
        porthandler.write_and_wait(ser, "M106 P1 S255", timeout=2.0)  # bar on
        # Track bar light on time for 5-minute auto-off
        globals.lamp_bar_on_time = time.time()
        globals.lamp_dome_on_time = None

def _turn_off_all_lights():
    """Turn off both lights. Silently ignores errors if serial port is disconnected."""
    ser = globals.motion_platform
    if ser and ser.is_open:
        try:
            porthandler.write_and_wait(ser, "M106 P0 S255", timeout=2.0)  # dome off
        except Exception:
            pass
        try:
            porthandler.write_and_wait(ser, "M106 P1 S0", timeout=2.0)    # bar off
        except Exception:
            pass
    # Clear lamp on times
    globals.lamp_dome_on_time = None
    globals.lamp_bar_on_time = None

def _apply_camera_settings_for_light(light: str):
    """Apply camera settings for specific light (dome or bar)."""
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
    
    category = f'camera_params_{light}'
    light_settings = settings_data.get(category, {})
    
    for setting_name, setting_value in light_settings.items():
        try:
            validate_and_set_camera_param(camera, setting_name, setting_value, camera_properties)
        except Exception as e:
            app.logger.warning(f"Could not apply {setting_name} for {light}: {e}")

def _capture_and_save_image(target_folder: str, filename: str, background_subtraction: bool = False) -> list:
    """Capture image from camera and save to target folder.
    
    If background_subtraction is True, also saves a masked version.
    
    Returns:
        list of saved file paths (original, and optionally masked)
    """
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
    pil_img.save(full_path, format='JPEG', quality=95)
    
    saved_paths = [full_path]
    
    # Background subtraction: save masked version alongside original
    if background_subtraction:
        try:
            af_contour = getattr(globals, "last_autofocus_contour", None)
            print(af_contour)
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
                app.logger.info(f"Background-subtracted image saved: {masked_path} (kind={kind})")
            else:
                app.logger.warning(f"Background subtraction found no object in {filename}")
        except Exception as e:
            app.logger.warning(f"Background subtraction failed for {filename}: {e}")
    
    return saved_paths


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
    return dt.strftime("%m%d_%H%M")


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
        _turn_on_bar_light()
    
    _apply_camera_settings_for_light(light_type)
    time.sleep(0.3)  # Let light and camera settings stabilize
    
    timestamp = _format_capture_timestamp(datetime.now())
    tablet_label = _tablet_index_to_label(tablet_index)
    filename = f"{measurement_name}_{timestamp}_{tablet_label}_{light_type}"
    
    return _capture_and_save_image(measurement_folder, filename, background_subtraction=background_subtraction)


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
        measurement_folder = data.get('measurement_folder')
        measurement_name = data.get('measurement_name')
        
        autofocus_enabled = bool(data.get('autofocus', False))
        lamp_top = bool(data.get('lamp_top', False))
        lamp_side = bool(data.get('lamp_side', False))
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
        
        if not lamp_top and not lamp_side:
            return jsonify({
                'status': 'error',
                'message': 'At least one light must be selected'
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
            
            # Turn on dome light and apply dome camera settings for autofocus
            _turn_on_dome_light()
            _apply_camera_settings_for_light('dome')
            time.sleep(0.3)  # Let light and camera settings stabilize
            
            try:
                if is_first_tablet:
                    # First tablet: full coarse+fine scan to find the focal plane
                    af_result = autofocus_main.autofocus_coarse(motion_platform)
                else:
                    # Subsequent tablets: fine-only around previous best Z
                    af_result = autofocus_main.autofocus_coarse(motion_platform)
                
                af_status = af_result.get('status', 'ERROR')
                af_error_code = af_result.get('code')  # e.g. "E2000", "E2002", "E2003"
                if af_status == 'OK':
                    contour = af_result.get("final_contour") or af_result.get("contour")
                    globals.last_autofocus_contour = contour if contour else None
                    app.logger.info(f"Tablet {tablet_index}: Autofocus OK at Z={af_result.get('z_rel', '?')}")
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
                    # For tablet-quality errors, skip image capture entirely
                    if af_error_code in ('E2000', 'E2002', 'E2003', 'E2004'):
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
            app.logger.info(f"Tablet {tablet_index}: Getting contour via manual_bgr (no autofocus)")
            _turn_on_dome_light()
            _apply_camera_settings_for_light('dome')
            time.sleep(0.3)  # Let light and camera settings stabilize

            try:
                mbgr_result = manual_bgr.manual_return()
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
                    if mbgr_code in ('E2000', 'E2002', 'E2003', 'E2004', 'E2104', 'E2105', 'E2106'):
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
                    if gds_code in ('E2000', 'E2002', 'E2003', 'E2005'):
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
                    if oof_code in ('E2004', 'E2010', 'E2012', 'E2013', 'E2014'):
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
                app.logger.info(f"Tablet {tablet_index}: Capturing bar image")
                saved_paths = _capture_image_with_light('bar', measurement_folder, measurement_name, tablet_index, background_subtraction=background_subtraction)
                saved_images.extend(saved_paths)
                app.logger.info(f"Tablet {tablet_index}: Saved bar image(s): {saved_paths}")
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
        
        app.logger.info(f"Tablet {tablet_index}: Measurement complete ({len(saved_images)} images)")
        response_data = {
            'status': 'success',
            'tablet_index': tablet_index,
            'saved_images': saved_images
        }
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
    target_folder = data.get('target_folder')
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

    # --- Bar light + background subtraction: obtain contour under dome light ---
    settings_data_pre = get_settings()
    bg_sub_pre = bool(settings_data_pre.get('other_settings', {}).get('background_subtraction', False))

    if light_type == 'bar' and bg_sub_pre:
        app.logger.info("Manual save: bar light + BGR — obtaining contour under dome light")
        try:
            _turn_on_dome_light()
            _apply_camera_settings_for_light('dome')
            time.sleep(0.3)  # Let light and camera settings stabilize

            mbgr_result = manual_bgr.manual_return()
            mbgr_status = mbgr_result.get('status', 'ERROR')
            if mbgr_status == 'OK':
                contour = mbgr_result.get('final_contour')
                globals.last_autofocus_contour = contour if contour else None
                app.logger.info("Manual save: dome-light contour obtained for BGR")
            else:
                globals.last_autofocus_contour = None
                mbgr_code = mbgr_result.get('code', '')
                app.logger.warning(f"Manual save: manual_bgr returned {mbgr_status} ({mbgr_code})")

            # Switch back to bar light for the actual capture
            _turn_on_bar_light()
            _apply_camera_settings_for_light('bar')
            time.sleep(0.3)  # Let light and camera settings stabilize
        except Exception as e:
            app.logger.warning(f"Manual save: contour detection under dome light failed: {e}")
            globals.last_autofocus_contour = None
            # Restore bar light and continue — save image without background subtraction
            try:
                _turn_on_bar_light()
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

    # Background subtraction: save masked version alongside original if enabled
    masked_path = None
    settings_data = get_settings()
    bg_sub_enabled = bool(settings_data.get('other_settings', {}).get('background_subtraction', False))
    if bg_sub_enabled:
        try:
            af_contour = getattr(globals, "last_autofocus_contour", None)
            print(af_contour)
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

@app.route('/api/update-other-settings', methods=['POST'])
def update_other_settings():
    try:
        data = request.json
        category = data.get('category')           # e.g. 'size_limits'
        setting_name = data.get('setting_name')     # e.g. 'ng_limit'
        setting_value = data.get('setting_value')   # e.g. 123

        app.logger.info(f"Updating {category}.{setting_name} = {setting_value}")

        # Retrieve the in-memory settings
        settings_data = get_settings()
        if category not in settings_data:
            settings_data[category] = {}

        # For consistency, you could add validation or conversion here if needed.
        # For now, we'll just pass the new value as is.
        updated_value = setting_value

        # Update the setting in the in-memory dict
        settings_data[category][setting_name] = updated_value

        # Save the updated settings to disk
        save_settings()

        app.logger.info(f"{category}.{setting_name} updated and saved to settings.json")

        return jsonify({
            "message": f"{category}.{setting_name} updated and saved.",
            "updated_value": updated_value
        }), 200

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
    """Turn off all lights (dome and bar) immediately. Used when measurement is stopped."""
    try:
        _turn_off_all_lights()
        app.logger.info("All lights turned off via endpoint")
        return jsonify({"status": "lights_off"}), 200
    except Exception as e:
        app.logger.exception("Error turning off lights")
        return jsonify({"error": str(e)}), 500

@app.route('/api/check-lamp-auto-off', methods=['GET'])
def check_lamp_auto_off():
    """
    Check if lamps were automatically turned off by the timeout monitor.
    Returns the flag state and clears it.
    """
    try:
        auto_off = globals.lamp_auto_turned_off
        if auto_off:
            globals.lamp_auto_turned_off = False  # Clear the flag after reading
        return jsonify({
            "auto_turned_off": auto_off,
            "dome_on": globals.lamp_dome_on_time is not None,
            "bar_on": globals.lamp_bar_on_time is not None
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

        return jsonify({"file": file_path}), 200

    except Exception as e:
        app.logger.exception("File selection failed")
        return jsonify({"error": str(e)}), 500

@app.route('/api/select-folder', methods=['GET'])
def select_folder():
    folder = select_folder_external()  # opens a Tkinter folder dialog
    if folder is None:
        folder = ""
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
    try:
        result = subprocess.run(
            ['python', 'select_folder_dialog.py'], 
            capture_output=True, 
            text=True,
            timeout=15  # seconds
        )
        output = json.loads(result.stdout.strip())
        return output['folder']
    except Exception as e:
        print("Folder selection failed:", e)
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
        device = porthandler.connect_to_motion_platform()
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
            app.logger.info("Motion platform disconnected successfully.")
    except Exception as e:
        app.logger.debug(f"Error closing motion platform: {e}")      
       

if __name__ == '__main__':
    multiprocessing.freeze_support()
    
    load_settings()
    initialize_cameras()
    initialize_serial_devices()
    
    # Start lamp timeout monitor thread
    import threading
    lamp_monitor = threading.Thread(target=lamp_timeout_monitor, daemon=True, name="LampTimeoutMonitor")
    lamp_monitor.start()
    globals.lamp_timeout_thread = lamp_monitor
    app.logger.info("Lamp timeout monitor thread started")
    
    try:
        app.run(debug=False, use_reloader=False)
    finally:
        shutdown_devices()