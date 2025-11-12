from flask import Flask, jsonify, request, Response
from flask_cors import CORS
import cv2
import time
import globals
from pypylon import pylon
from cameracontrol import (apply_camera_settings, 
                           validate_and_set_camera_param, get_camera_properties)
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

from logger_config import setup_logger
from error_codes import ErrorCode, ERROR_MESSAGES
import tkinter as tk
from tkinter import filedialog
from multiprocessing import Process, Queue
import multiprocessing

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
            if buf:
                app.logger.debug(f"M105 non-ok reply: {buf[:64]!r}")
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
    except Exception as e:
        app.logger.warning(f"get position failed (returning cache): {e}")
        return jsonify(globals.last_toolhead_pos), 200


    
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
        globals.toolhead_homed = True
        return jsonify({'ok': True, 'homed_axes': axes or ['x','y','z']})
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
        camera_settings = settings_data.get('camera_params', {})

        if not camera_settings:
            app.logger.warning("No settings found for Camera.")
            return jsonify({
                "error": "No settings found for Camera",
                "code": ErrorCode.GENERIC,
                "popup": True
            }), 404

        app.logger.info(f"Sending camera settings to frontend: {camera_settings}")
        return jsonify(camera_settings), 200

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

        # Persist the validated value in settings.json
        settings_data = get_settings()
        settings_data['camera_params'][setting_name] = updated_value
        save_settings()

        app.logger.info(f"Camera setting {setting_name} updated and saved to settings.json")

        return jsonify({
            "message": f"Camera {setting_name} updated and saved.",
            "updated_value": updated_value
        }), 200

    except Exception as e:
        app.logger.exception("Failed to update camera settings")
        return jsonify({"error": str(e)}), 500
    
    
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
            return jsonify({'status': 'error', 'message': 'Printer not connected'}), 404

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
        motioncontrols.move_relative(motion_platform, **move_args)

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
        data = request.get_json()
        x_pos = data.get('x')
        y_pos = data.get('y')
        z_pos = data.get('z')

        motion_platform = globals.motion_platform
        if motion_platform is None or not motion_platform.is_open:
            return jsonify({'status': 'error', 'message': 'Printer not connected'}), 404

        requested = {}
        planned = {}
        clamped_flags = {}
        limits_out = {}

        # Helper to process one axis if provided
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
            return jsonify({'status': 'error', 'message': 'No axes specified'}), 400

        # If all provided axes clamp to their current values, skip sending a move
        all_noop = True
        curr_pos = getattr(globals, "last_toolhead_pos", {})
        for ax, clamped_val in planned.items():
            curr = curr_pos.get(ax)
            if curr is None:
                # Not homed axis; we cannot trust absolute; require homing first
                return jsonify({'status': 'error',
                                'message': f'Axis {ax.upper()} not homed; position unknown.'}), 409
            if not math.isclose(float(curr), float(clamped_val), abs_tol=_EPS):
                all_noop = False

        if all_noop:
            return jsonify({
                'status': 'success',
                'requested': requested,
                'sent': {},
                'clamped': clamped_flags,
                'limits': limits_out,
                'message': 'Requested positions equal to current (after clamping); no move sent.'
            }), 200

        # Send only the axes we plan to change
        motioncontrols.move_to_position(
            motion_platform,
            planned.get('x'),
            planned.get('y'),
            planned.get('z')
        )

        return jsonify({
            'status': 'success',
            'requested': requested,
            'sent': planned,
            'clamped': clamped_flags,
            'limits': limits_out
        }), 200

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


### Video streaming Function ###
@app.route('/api/start-video-stream', methods=['GET'])
def start_video_stream():
    """
    Returns a live MJPEG response from generate_frames().
    This is the *only* place we call generate_frames, to avoid double-streaming.
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
            generate_frames(scale_factor),
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

def generate_frames(scale_factor=0.1):
    app.logger.info(f"Generating frames with scale factor {scale_factor}")
    camera = globals.camera
    if not camera:
        app.logger.error(f"Camera is not connected.")
        return
    if not camera.IsGrabbing():
        app.logger.info(f"Camera starting grabbing.")
        camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    try:
        while globals.stream_running:
            with globals.grab_lock:
                grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                if grab_result.GrabSucceeded():
                    image = grab_result.Array
                    if scale_factor != 1.0:
                        width = int(image.shape[1] * scale_factor)
                        height = int(image.shape[0] * scale_factor)
                        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
                    success, frame = cv2.imencode('.jpg', image)
                    if success:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame.tobytes() + b'\r\n')
                grab_result.Release()
    except Exception as e:
        app.logger.error(f"Error in video stream: {e}")
        if "Device has been removed" in str(e):
            globals.stream_running = False
            if camera and camera.IsOpen():
                try:
                    camera.StopGrabbing()
                    camera.Close()
                except Exception as close_err:
                    app.logger.error(f"Failed to close camera after unplug: {close_err}")
            # FIX: clear the correct reference (was `globals.cameras = None`)
            globals.camera = None
    finally:
        app.logger.info(f"Camera streaming thread stopped.")

def grab_camera_image():
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

            # Retry grabbing the image up to 10 times.
            grab_result = retry_operation(
                lambda: attempt_frame_grab(cam),
                max_retries=10,
                wait=1
            )

            if grab_result is None:
                app.logger.error("Grab result is None for camera")
                return None, jsonify({
                    "error": ERROR_MESSAGES.get(ErrorCode.CAMERA_DISCONNECTED),
                    "code": ErrorCode.CAMERA_DISCONNECTED,
                    "popup": True
                }), 400

            try:
                if not grab_result.GrabSucceeded():
                    app.logger.error("Grab result unsuccessful for camera")
                    return None, jsonify({
                        "error": ERROR_MESSAGES.get(ErrorCode.CAMERA_DISCONNECTED),
                        "code": ErrorCode.CAMERA_DISCONNECTED,
                        "popup": True
                    }), 400
            except Exception as e:
                app.logger.exception(f"Exception while checking GrabSucceeded: {e}")
                return None, jsonify({
                    "error": ERROR_MESSAGES.get(ErrorCode.CAMERA_DISCONNECTED, "Camera disconnected."),
                    "code": ErrorCode.CAMERA_DISCONNECTED,
                    "popup": True
                }), 400

            app.logger.info("Image grabbed successfully.")
            image = grab_result.Array
            globals.latest_image = image.copy()
            grab_result.Release()
            
            return image, None, None

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
    target_folder = data.get("target_folder", "")
    if not target_folder:
        return jsonify({"message": "Cancelled"}), 200

    try:
        os.makedirs(target_folder, exist_ok=True)
    except Exception as e:
        app.logger.exception(f"Failed to create folder '{target_folder}': {e}")
        return jsonify({
            "error": "Failed to create target folder",
            "code": ErrorCode.GENERIC,
            "popup": True
        }), 500
    
    now = datetime.now().strftime("%Y%m%d%H%M%S")

    image, *_ = grab_camera_image()
    if image is None:
        app.logger.error("Grab result is None for camera")
        return jsonify({
            "error": ERROR_MESSAGES.get(ErrorCode.CAMERA_DISCONNECTED, "Camera disconnected"),
            "code": ErrorCode.CAMERA_DISCONNECTED,
            "popup": True
        }), 400

    fn = os.path.join(target_folder, f"{now}.jpg")
    try:
        cv2.imwrite(fn, image)
    except Exception as e:
        app.logger.exception(f"Failed to write image '{fn}': {e}")
        return jsonify({
            "error": "Failed to save image",
            "code": ErrorCode.GENERIC,
            "popup": True
        }), 500

    return jsonify({
        "message": "Raw image saved",
        "path": fn
    }), 200

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
    

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"ready": True}), 200

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
    
def attempt_frame_grab(camera):
    # Attempt to retrieve the image result.
    grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
    # Check if the frame grab was successful.
    if not grab_result.GrabSucceeded():
        grab_result.Release()
        raise Exception(f"Grab result unsuccessful for camera")
    return grab_result

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

    # If already connected, return info.
    if globals.camera and globals.camera.IsOpen():
        return {
            "connected": True,
            "name": selected_cam.GetModelName(),
        }

    try:
        globals.camera = pylon.InstantCamera(factory.CreateDevice(selected_cam))
        globals.camera.Open()
        
    except Exception as e:
        # Use GetPortName() if available; otherwise, fallback.
        port_name = selected_cam.GetPortName() if hasattr(selected_cam, "GetPortName") else "unknown"
        app.logger.exception(f"Failed to connect to camera on port {port_name}: {e}")
        return {
            "error": ERROR_MESSAGES.get(ErrorCode.CAMERA_DISCONNECTED, "Camera not connected."),
            "code": ErrorCode.CAMERA_DISCONNECTED,
            "popup": True
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
        
if __name__ == '__main__':
    multiprocessing.freeze_support()
    
    load_settings()
    initialize_cameras()
    initialize_serial_devices()
    
    app.run(debug=False, use_reloader=False)