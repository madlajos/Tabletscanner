from flask import Flask, jsonify, request, Response
from flask_cors import CORS
import cv2
import time
import globals
from pypylon import pylon
from cameracontrol import (apply_camera_settings, 
                           validate_and_set_camera_param, get_camera_properties)
import porthandler
import os
import sys
import pyodbc
from datetime import datetime
import subprocess
import json
from threading import Lock
from settings_manager import load_settings, save_settings, get_settings
import numpy as np

from logger_config import setup_logger
setup_logger()
from error_codes import ErrorCode, ERROR_MESSAGES
import tkinter as tk
from tkinter import filedialog
from multiprocessing import Process, Queue
import multiprocessing

app = Flask(__name__)
app.secret_key = 'Egis'
CORS(app)
app.debug = True


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
@app.route('/api/connect-to-turntable', methods=['POST'])
def connect_turntable():
    try:
        app.logger.info("Attempting to connect to Turntable")
        if porthandler.turntable and porthandler.turntable.is_open:
            app.logger.info("Turntable already connected.")
            return jsonify({'message': 'Turntable already connected'}), 200

        device = porthandler.connect_to_turntable()
        if device:
            porthandler.turntable = device
            app.logger.info("Successfully connected to Turntable")
            return jsonify({'message': 'Turntable connected', 'port': device.port}), 200
        else:
            app.logger.error("Failed to connect to Turntable: No response or incorrect ID")
            return jsonify({
                'error': ERROR_MESSAGES[ErrorCode.TURNTABLE_DISCONNECTED],
                'code': ErrorCode.TURNTABLE_DISCONNECTED,
                'popup': True
            }), 404
    except Exception as e:
        app.logger.exception("Exception occurred while connecting to Turntable")
        return jsonify({
            'error': ERROR_MESSAGES[ErrorCode.TURNTABLE_DISCONNECTED],
            'code': ErrorCode.TURNTABLE_DISCONNECTED,
            'popup': True
        }), 500

@app.route('/api/connect-to-barcode', methods=['POST'])
def connect_barcode_scanner():
    try:
        app.logger.info("Attempting to connect to Barcode Scanner")
        # Always attempt a fresh connection.
        device = porthandler.connect_to_barcode_scanner()
        if device:
            app.logger.info("Successfully connected to Barcode Scanner")
            return jsonify({'message': 'Barcode Scanner connected', 'port': device.port}), 200
        else:
            app.logger.error("Failed to connect Barcode Scanner: Device not found")
            return jsonify({
                'error': ERROR_MESSAGES[ErrorCode.BARCODE_DISCONNECTED],
                'code': ErrorCode.BARCODE_DISCONNECTED,
                'popup': True
            }), 404
    except Exception as e:
        app.logger.exception("Exception occurred while connecting to Barcode Scanner")
        return jsonify({
            'error': ERROR_MESSAGES[ErrorCode.BARCODE_DISCONNECTED],
            'code': ErrorCode.BARCODE_DISCONNECTED,
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
    app.logger.debug(f"Received status request for device: {device_name}")
    device = None
    if device_name.lower() == 'turntable':
        device = porthandler.turntable
    elif device_name.lower() in ['barcode', 'barcodescanner']:
        device = porthandler.barcode_scanner
    else:
        app.logger.error("Invalid device name")
        return jsonify({'error': 'Invalid device name', 'popup': True}), 400

    if device and device.is_open:
        if device_name.lower() == 'turntable':
            if not porthandler.turntable_waiting_for_done:
                try:
                    device.write(b'IDN?\n')
                    response = device.read(10).decode(errors='ignore').strip()
                    if response:
                        app.logger.debug(f"{device_name} is responsive on port {device.port}")
                        return jsonify({'connected': True, 'port': device.port})
                except Exception as e:
                    app.logger.warning(f"{device_name} is unresponsive, disconnecting. Error: {str(e)}")
                    porthandler.disconnect_serial_device(device_name)
                    return jsonify({
                        'connected': False,
                        'error': f"{device_name.capitalize()} unresponsive",
                        'code': ErrorCode.TURNTABLE_DISCONNECTED,
                        'popup': True
                    }), 400
        elif device_name.lower() in ['barcode', 'barcodescanner']:
            from serial.tools import list_ports
            available_ports = [port.device for port in list_ports.comports()]
            if device.port in available_ports:
                app.logger.debug(f"{device_name} is connected on port {device.port}")
                return jsonify({'connected': True, 'port': device.port})
            else:
                app.logger.warning(f"{device_name} appears to be disconnected (port not found).")
                return jsonify({
                    'connected': False,
                    'error': "Barcode Scanner disconnected",
                    'code': ErrorCode.BARCODE_DISCONNECTED,
                    'popup': True
                }), 400

        app.logger.debug(f"{device_name} is connected on port {device.port}")
        return jsonify({'connected': True, 'port': device.port})

    app.logger.warning(f"{device_name} appears to be disconnected.")
    # For turntable, return error with code.
    if device_name.lower() == 'turntable':
        return jsonify({
            'connected': False,
            'error': f"{device_name.capitalize()} appears to be disconnected",
            'code': ErrorCode.TURNTABLE_DISCONNECTED,
            'popup': True
        }), 400
    else:
        return jsonify({
            'connected': False,
            'error': f"{device_name.capitalize()} appears to be disconnected",
            'code': ErrorCode.BARCODE_DISCONNECTED,
            'popup': True
        }), 400
    
### Camera-related functions ###
def stop_camera_stream(camera_type):
    if camera_type not in globals.cameras:
        raise ValueError(f"Invalid camera type: {camera_type}")

    camera = globals.cameras.get(camera_type)

    with globals.grab_locks[camera_type]:
        if not globals.stream_running.get(camera_type, False):
            return "Stream already stopped."

        try:
            globals.stream_running[camera_type] = False
            if camera and camera.IsGrabbing():
                camera.StopGrabbing()
                app.logger.info(f"{camera_type.capitalize()} camera stream stopped.")

            if globals.stream_threads.get(camera_type) and globals.stream_threads[camera_type].is_alive():
                globals.stream_threads[camera_type].join(timeout=2)
                app.logger.info(f"{camera_type.capitalize()} stream thread stopped.")

            globals.stream_threads[camera_type] = None
            return f"{camera_type.capitalize()} stream stopped."
        except Exception as e:
            raise RuntimeError(f"Failed to stop {camera_type} stream: {str(e)}")

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
    camera_type = request.args.get('type')

    if camera_type not in globals.cameras or globals.cameras[camera_type] is None:
        app.logger.warning(f"{camera_type.capitalize()} camera is already disconnected or not initialized")
        return jsonify({"status": "already disconnected"}), 200

    try:
        stop_camera_stream(camera_type)
        app.logger.info(f"{camera_type.capitalize()} stream stopped before disconnecting.")
    except ValueError:
        app.logger.warning(f"Failed to stop {camera_type} stream: Invalid camera type.")
        # Decide how you want to handle this. If invalid camera type is fatal, return here:
        return jsonify({"error": "Invalid camera type"}), 400
    except RuntimeError as re:
        app.logger.warning(f"Error stopping {camera_type} stream: {str(re)}")
        # Maybe we continue to shut down the camera anyway
    except Exception as e:
        app.logger.error(f"Failed to disconnect {camera_type} camera: {e}")
        return jsonify({"error": str(e)}), 500

    camera = globals.cameras.get(camera_type, None)
    if camera and camera.IsGrabbing():
        camera.StopGrabbing()
        app.logger.info(f"{camera_type.capitalize()} camera grabbing stopped.")

    if camera and camera.IsOpen():
        camera.Close()
        app.logger.info(f"{camera_type.capitalize()} camera closed.")

    # Clean up references
    globals.cameras[camera_type] = None
    camera_properties[camera_type] = None  # Make sure camera_properties is in scope
    app.logger.info(f"{camera_type.capitalize()} camera disconnected successfully.")

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
    camera_type = request.args.get('type')
    if camera_type not in globals.cameras:
        app.logger.error(f"Invalid camera type: {camera_type}")
        return jsonify({
            "error": ERROR_MESSAGES.get(ErrorCode.GENERIC, "Invalid camera type specified."),
            "code": ErrorCode.GENERIC,
            "popup": True
        }), 400

    camera = globals.cameras.get(camera_type)
    is_connected = camera is not None and camera.IsOpen()
    is_streaming = globals.stream_running.get(camera_type, False)

    factory = pylon.TlFactory.GetInstance()
    devices = factory.EnumerateDevices()

    return jsonify({
        "connected": is_connected,
        "streaming": is_streaming
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
        camera_type = data.get('camera_type')
        setting_name = data.get('setting_name')
        setting_value = data.get('setting_value')

        app.logger.info(f"Updating {camera_type} camera setting: {setting_name} = {setting_value}")

        # Apply the setting to the camera
        updated_value = validate_and_set_camera_param(
            globals.cameras[camera_type],
            setting_name,
            setting_value,
            camera_properties[camera_type],
            camera_type
        )

        settings_data = get_settings()
        settings_data['camera_params'][camera_type][setting_name] = updated_value
        save_settings()

        app.logger.info(f"{camera_type.capitalize()} camera setting {setting_name} updated and saved to settings.json")

        return jsonify({
            "message": f"{camera_type.capitalize()} camera {setting_name} updated and saved.",
            "updated_value": updated_value
        }), 200

    except Exception as e:
        app.logger.exception("Failed to update camera settings")
        return jsonify({"error": str(e)}), 500


### Video streaming Function ###
@app.route('/api/start-video-stream', methods=['GET'])
def start_video_stream():
    """
    Returns a live MJPEG response from generate_frames(camera_type).
    This is the *only* place we call generate_frames, to avoid double-streaming.
    """
    try:
        camera_type = request.args.get('type')
        scale_factor = float(request.args.get('scale', 0.1))

        if not camera_type or camera_type not in globals.cameras:
            app.logger.error(f"Invalid or missing camera type: {camera_type}")
            return jsonify({"error": "Invalid or missing camera type"}), 400

         # Ensure the camera is connected
        res = connect_camera_internal(camera_type)
        if "error" in res:
            app.logger.error(f"Camera connection failed: {res['error']}")
            return jsonify(res), 400

        with globals.grab_locks[camera_type]:
            if not globals.stream_running.get(camera_type, False):
                globals.stream_running[camera_type] = True
                app.logger.info(f"stream_running[{camera_type}] set to True in start_video_stream")

        app.logger.info(f"Starting video stream for {camera_type}")
        return Response(
            generate_frames(camera_type, scale_factor),
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
    camera_type = request.args.get('type')
    app.logger.info(f"Received stop request for {camera_type}")

    try:
        message = stop_camera_stream(camera_type)
        return jsonify({"message": message}), 200
    except ValueError as ve:
        # E.g., invalid camera type
        app.logger.error(str(ve))
        return jsonify({"error": str(ve)}), 400
    except RuntimeError as re:
        app.logger.error(str(re))
        return jsonify({"error": str(re)}), 500
    except Exception as e:
        app.logger.exception(f"Unexpected exception while stopping {camera_type} stream.")
        return jsonify({"error": str(e)}), 500

def generate_frames(camera_type, scale_factor=0.1):
    app.logger.info(f"Generating frames for {camera_type} with scale factor {scale_factor}")
    camera = globals.cameras.get(camera_type)
    if not camera:
        app.logger.error(f"{camera_type.capitalize()} camera is not connected.")
        return
    if not camera.IsGrabbing():
        app.logger.info(f"{camera_type.capitalize()} camera starting grabbing.")
        camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    try:
        while globals.stream_running[camera_type]:
            with globals.grab_locks[camera_type]:
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
        app.logger.error(f"Error in {camera_type} video stream: {e}")
        if "Device has been removed" in str(e):
            globals.stream_running[camera_type] = False
            if camera and camera.IsOpen():
                try:
                    camera.StopGrabbing()
                    camera.Close()
                except Exception as close_err:
                    app.logger.error(f"Failed to close {camera_type} after unplug: {close_err}")
            globals.cameras[camera_type] = None
    finally:
        app.logger.info(f"{camera_type.capitalize()} camera streaming thread stopped.")


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
    
def attempt_frame_grab(camera, camera_type):
    # Attempt to retrieve the image result.
    grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
    # Check if the frame grab was successful.
    if not grab_result.GrabSucceeded():
        grab_result.Release()
        raise Exception(f"Grab result unsuccessful for camera {camera_type}")
    return grab_result

def get_db_connection():
    from settings_manager import get_settings
    settings_data = get_settings()
    sql_config = settings_data.get("sql_server")
    if not sql_config:
        raise Exception("SQL Server configuration not found in settings.")

    server = sql_config.get("server")
    database = sql_config.get("db_name")
    username = sql_config.get("username")
    password = sql_config.get("password")

    connection_string = (
        'DRIVER={ODBC Driver 17 for SQL Server};'
        'SERVER=tcp:' + server + ',1433;'
        'DATABASE=' + database + ';'
        'UID=' + username + ';'
        'PWD=' + password +';'
        'Pooling=False;'
    )

    try:
        # Attempt to connect with a short timeout.
        conn = pyodbc.connect(connection_string, timeout=15)
        return conn
    except Exception as e:
        # Log error and re-raise if needed.
        raise Exception("Failed to connect to SQL Server: " + str(e))

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
        if not (cam and cam.isOpen()):
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
        # Connect turntable as before.
        device = porthandler.connect_to_turntable()
        if device:
            porthandler.turntable = device
            app.logger.info("Turntable connected automatically on startup.")
        else:
            app.logger.error("Failed to auto-connect turntable on startup.")
    except Exception as e:
        app.logger.error(f"Error initializing turntable: {e}")

    try:
        # Connect barcode scanner similarly.
        device = porthandler.connect_to_barcode_scanner()
        if device:
            app.logger.info("Barcode scanner connected automatically on startup.")
        else:
            app.logger.error("Failed to auto-connect barcode scanner on startup.")
    except Exception as e:
        app.logger.error(f"Error initializing barcode scanner: {e}")

        
if __name__ == '__main__':
    multiprocessing.freeze_support()
    
    load_settings()
    initialize_cameras()
    initialize_serial_devices()
    
    app.run(debug=False, use_reloader=False)