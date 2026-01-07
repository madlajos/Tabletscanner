import cv2
import sys
from typing import Optional
from queue import Queue, Empty
from pypylon import pylon
import logging
import datetime
import os
import time
import requests
import threading

from globals import app, stream_running, stream_thread, camera
from logger_config import CameraError

opencv_display_format = 'BGR8'

def abort(reason: str, return_code: int = 1, usage: bool = False):
    app.logger.error(reason)
    sys.exit(return_code)

def parse_args() -> Optional[str]:
    args = sys.argv[1:]
    argc = len(args)

    for arg in args:
        if arg in ('/h', '-h'):
            sys.exit(0)

    if argc > 1:
        abort(reason="Invalid number of arguments. Abort.", return_code=2, usage=True)

    return None if argc == 0 else args[0]

# Function to continuously generate and send frames
def stream_video(scale_factor: float = 1.0, jpeg_quality: int = 80):
    cam = getattr(globals, "camera", None)
    if not (cam and cam.IsOpen()):
        app.logger.error("Camera is not open.")
        return  # end generator immediately

    app.logger.info("Camera streaming thread started.")
    globals.stream_running = True
    
    # Ensure a single grab lock exists; TODO: Untested.
    lock = getattr(globals, "grab_lock", None)
    if lock is None:
        from threading import Lock
        globals.grab_lock = Lock()
        lock = globals.grab_lock

    try:
        while getattr(globals, "stream_running", False):
            try:
                # --- Grab inside the lock (short critical section) ---
                with lock:
                    cam = getattr(globals, "camera", None)
                    if not (cam and cam.IsOpen()):
                        app.logger.warning("Camera closed during streaming.")
                        break

                    grab_result = cam.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                    if not grab_result.GrabSucceeded():
                        app.logger.warning("Grab did not succeed.")
                        grab_result.Release()
                        continue

                    # Copy the array before releasing the result buffer
                    image = grab_result.Array.copy()
                    grab_result.Release()
                    
                if scale_factor and scale_factor != 1.0:
                        h, w = image.shape[:2]
                        new_w = max(1, int(w * scale_factor))
                        new_h = max(1, int(h * scale_factor))
                        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

                ok, frame = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)])
                if not ok:
                    app.logger.error("Failed to encode frame.")
                    continue
 
            
            except Exception as e:
                        app.logger.error(f"Error in video stream loop: {e}")
                        # If device was removed, bail out; otherwise loop continues
                        if "Device has been removed" in str(e):
                            break
                        continue
    finally:
        globals.stream_running = False
        app.logger.info("Camera streaming generator stopped.")
        
def get_camera(camera_id: str) -> pylon.InstantCamera:
    factory = pylon.TlFactory.GetInstance()
    devices = factory.EnumerateDevices()

    app.logger.info("Connected devices:")
    for device in devices:
        app.logger.info(f"Device Model: {device.GetModelName()}, Serial Number: {device.GetSerialNumber()}")

    # Search for the requested camera ID
    for device in devices:
        if device.GetSerialNumber() == camera_id:
            try:
                camera = pylon.InstantCamera(factory.CreateDevice(device))
                camera.Open()
                if not camera.IsOpen():
                    raise CameraError(f"Camera '{camera_id}' failed to open after creation.")
                return camera
            except Exception as e:
                app.logger.error(f"Failed to open camera {camera_id}: {e}")
                raise CameraError(f"Failed to open Camera '{camera_id}'.") from e

    raise CameraError(f"Failed to access Camera '{camera_id}'. Available devices: " +
                      f"{[device.GetSerialNumber() for device in devices]}")

def setup_camera(camera: pylon.InstantCamera, camera_params: dict):
    try:
        # It is usually not necessary to call camera.Open() here if already open
        # But if needed, check and open:
        if not camera.IsOpen():
            camera.Open()

        camera.Width.SetValue(round(camera_params['Width']))
        app.logger.info(f"Set Image Width to {round(camera_params['Width'])}")

        camera.Height.SetValue(round(camera_params['Height']))
        app.logger.info(f"Set Image Height to {round(camera_params['Height'])}")

        camera.AcquisitionFrameRateEnable.SetValue(True)
        camera.AcquisitionFrameRate.SetValue(round(camera_params['FrameRate']))
        app.logger.info(f"Set AcquisitionFrameRate to {round(camera_params['FrameRate'])}")

        camera.ExposureTime.SetValue(round(camera_params['ExposureTime']))
        app.logger.info(f"Set ExposureTime to {round(camera_params['ExposureTime'])}")

        camera.Gain.SetValue(round(camera_params['Gain']))
        app.logger.info(f"Set Gain to {round(camera_params['Gain'])}")

    except Exception as e:
        app.logger.error(f"Error setting camera parameters: {e}")
        raise CameraError("Error setting camera parameters.") from e


def setup_pixel_format(camera: pylon.InstantCamera):
    try:
        if camera.PixelFormat.GetValue() != opencv_display_format:
            camera.PixelFormat.SetValue(opencv_display_format)
    except Exception as e:
        app.logger.error(f"Error setting pixel format: {e}")
        raise CameraError("Failed to set pixel format.") from e
        
def start_streaming(camera: pylon.InstantCamera):
    handler = Handler()
    try:
        camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        while camera.IsGrabbing():
            grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if grab_result.GrabSucceeded():
                image = grab_result.Array
                cv2.imshow("Stream", image)
                if cv2.waitKey(1) == 13:  # Enter key to exit streaming loop
                    break
                handler.save_frame(grab_result)  # Pass grab_result or image based on implementation
            grab_result.Release()
    except Exception as e:
        app.logger.error(f"Error during streaming: {e}")
        raise CameraError("Streaming error encountered.") from e
    finally:
        try:
            camera.StopGrabbing()
            camera.Close()
        except Exception as e:
            app.logger.error(f"Error stopping streaming: {e}")
        
def stop_streaming(camera: pylon.InstantCamera):
    try:
        if camera.IsGrabbing():
            camera.StopGrabbing()
            app.logger.info(f"Stopped streaming for camera: {camera.GetDeviceInfo().GetSerialNumber()}")
        else:
            app.logger.info(f"Camera {camera.GetDeviceInfo().GetSerialNumber()} is not currently streaming.")
    except Exception as e:
        app.logger.error(f"Error stopping streaming for camera: {e}")
        raise CameraError("Failed to stop streaming.") from e

class Handler:
    def __init__(self, folder_selected):
        self.display_queue = Queue(10)
        self.save_next_frame = False
        self.folder_selected = folder_selected
        self.saved_image_path = None

    def get_image(self):
        try:
            return self.display_queue.get(timeout=1)
        except Empty: 
            return None

    def set_save_next_frame(self):
        self.save_next_frame = True

    def get_latest_image_name(self):
        return self.saved_image_path

    def save_frame(self, frame):
        try:
            # If frame is a GrabResult, convert it to an OpenCV image.
            # You might need to check if frame has an 'Array' attribute.
            if hasattr(frame, 'Array'):
                frame_np = frame.Array
            else:
                frame_np = frame
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            filename = os.path.join(self.folder_selected, f"IMG_{timestamp}.jpg")
            cv2.imwrite(filename, frame_np)
            self.saved_image_path = f"IMG_{timestamp}.jpg"
            app.logger.info(f"Image saved as: {filename}")
        except Exception as e:
            app.logger.error(f"Error saving frame: {e}")
            # Optionally, raise an exception if saving the image is critical.
            raise

    def __call__(self, cam: pylon.InstantCamera, stream, frame: pylon.GrabResult):
        try:
            if frame.GrabSucceeded():
                app.logger.debug(f"{cam} acquired frame.")
                if frame.get_pixel_format() == opencv_display_format:
                    display = frame
                else:
                    display = frame.convert_pixel_format(opencv_display_format)
                self.display_queue.put(display.as_opencv_image(), True)

                if self.save_next_frame:
                    self.save_frame(frame)
                    self.save_next_frame = False

                cam.queue_frame(frame)
                return frame
        except Exception as e:
            app.logger.error(f"Handler error: {e}")
            raise

def get_camera_properties(camera: pylon.InstantCamera) -> dict:
    properties = {}

    if hasattr(camera, 'GainSelector'):
        try:
            camera.GainSelector.SetValue("All")
            app.logger.info("GainSelector set to 'All'")
        except Exception as e:
            app.logger.warning(f"Could not set GainSelector to 'All': {e}")

    def safe_get(prop_name, accessor):
        try:
            properties[prop_name] = accessor()
            app.logger.info(f"Loaded property '{prop_name}': {properties[prop_name]}")
        except Exception as e:
            app.logger.warning(f"Could not load property '{prop_name}': {e}")

    safe_get('Width', lambda: {
        'min': camera.Width.GetMin(),
        'max': camera.Width.GetMax(),
        'inc': camera.Width.GetInc()
    })

    safe_get('Height', lambda: {
        'min': camera.Height.GetMin(),
        'max': camera.Height.GetMax(),
        'inc': camera.Height.GetInc()
    })

    safe_get('OffsetX', lambda: {
        'min': camera.OffsetX.GetMin(),
        'max': camera.OffsetX.GetMax(),
        'inc': camera.OffsetX.GetInc()
    })

    safe_get('OffsetY', lambda: {
        'min': camera.OffsetY.GetMin(),
        'max': camera.OffsetY.GetMax(),
        'inc': camera.OffsetY.GetInc()
    })

    safe_get('ExposureTime', lambda: {
        'min': camera.ExposureTime.GetMin(),
        'max': camera.ExposureTime.GetMax(),
        'inc': camera.ExposureTime.GetInc()
    })

    safe_get('Gain', lambda: {
        'min': camera.Gain.GetMin(),
        'max': camera.Gain.GetMax(),
        'inc': 0.01
    })

    if hasattr(camera, 'Gamma'):
        safe_get('Gamma', lambda: {
            'min': camera.Gamma.GetMin(),
            'max': camera.Gamma.GetMax(),
            'inc': 0.01
        })

    safe_get('FrameRate', lambda: {
        'min': camera.AcquisitionFrameRate.GetMin(),
        'max': camera.AcquisitionFrameRate.GetMax(),
        'inc': 0.01
    })

    return properties
def validate_param(param_name: str, param_value: float, properties: dict) -> float:
    try:
        param_value = float(param_value)
    except Exception as e:
        raise ValueError(f"Invalid parameter value for {param_name}: {e}")
    
    prop = properties.get(param_name)
    if not prop:
        raise KeyError(f"Property '{param_name}' not found in camera properties.")

    min_value = prop['min']
    max_value = prop['max']
    increment = prop['inc'] or 1  # Default to 1 if None

    if param_value < min_value:
        adjusted_value = min_value
    elif param_value > max_value:
        # Safely clamp to the largest valid value below max
        steps = int((max_value - min_value) // increment)
        adjusted_value = min_value + steps * increment
    else:
        steps = round((param_value - min_value) / increment)
        adjusted_value = min_value + steps * increment

    adjusted_value = round(adjusted_value, 6)  # high precision to avoid overflow

    if adjusted_value != param_value:
        app.logger.info(
            f"Adjusted {param_name}: {param_value} → {adjusted_value} "
            f"(min={min_value}, max={max_value}, inc={increment})"
        )
    return adjusted_value

 
def apply_camera_settings(camera, camera_properties, settings):
    app.logger.info(f"Loaded settings in apply_camera_settings: {settings}")

    camera_settings = settings.get('camera', {}) or {}

    if not camera_settings:
        app.logger.warning("No camera settings found under settings['camera']. Skipping apply.")
        return
    
    if not camera or not camera.IsOpen():
        app.logger.warning("Camera is not open. Cannot apply settings.")
        return

    app.logger.info(f"Applying camera settings: {camera_settings}")

    try:
        for setting_name, setting_value in camera_settings.items():
            app.logger.info(f"Setting {setting_name} = {setting_value}")
            validate_and_set_camera_param(
                camera,
                setting_name,
                setting_value,
                camera_properties                
                )
        app.logger.info("Camera settings applied successfully.")
        
    except Exception as e:
        app.logger.error(f"Failed to apply camera settings: {e}")
        raise CameraError("Error applying camera settings.") from e

def validate_and_set_camera_param(camera, param_name: str, param_value: float, properties: dict):
    global stream_running, stream_thread
    valid_value = validate_param(param_name, param_value, properties)

    try:
        was_streaming = stream_running

        # Stop streaming if changing critical dimensions
        if param_name in ['Width', 'Height'] and was_streaming:
            stream_running = False
            if camera.IsGrabbing():
                camera.StopGrabbing()
                app.logger.info(f"Camera stream stopped to apply {param_name} change.")
            if stream_thread and stream_thread.is_alive():
                stream_thread.join(timeout=2)
                app.logger.info("Camera stream thread joined.")
            stream_thread = None
            time.sleep(0.5)

        if not camera.IsOpen():
            camera.Open()
            app.logger.info(f"Camera reopened to apply {param_name}.")

        # Apply the validated parameter
        if param_name == 'Width':
            camera.Width.SetValue(valid_value)
        elif param_name == 'Height':
            camera.Height.SetValue(valid_value)
        elif param_name == 'OffsetX':
            camera.OffsetX.SetValue(valid_value)
        elif param_name == 'OffsetY':
            camera.OffsetY.SetValue(valid_value)
        elif param_name == 'ExposureTime':
            camera.ExposureTime.SetValue(valid_value)
        elif param_name == 'FrameRate':
            camera.AcquisitionFrameRateEnable.SetValue(True)
            camera.AcquisitionFrameRate.SetValue(valid_value)
        elif param_name == 'Gain':
            camera.Gain.SetValue(valid_value)
        elif param_name == 'Gamma':
            camera.Gamma.SetValue(valid_value)

        # Optional reversals
        camera.ReverseX.SetValue(True)
        camera.ReverseY.SetValue(True)

        app.logger.info(f"Camera {param_name} successfully set to {valid_value}")

        # Restart streaming if we had to stop it
        if param_name in ['Width', 'Height'] and was_streaming:
            stream_running = True
            stream_thread = threading.Thread(target=stream_video, args=(1.0, 80))
            stream_thread.start()
            app.logger.info(f"Camera stream restarted after {param_name} change.")

    except Exception as e:
        app.logger.error(f"Failed to set {param_name} for Camera: {e}")
        raise CameraError(f"Error setting {param_name} for Camera.") from e

    return valid_value

def notify_stream_status(camera_type: str, is_streaming: bool):
    try:
        response = requests.post(f'http://localhost:4200/api/stream-status', json={
            'camera_type': camera_type,
            'is_streaming': is_streaming
        })
        if response.status_code == 200:
            app.logger.info(f"Stream status for {camera_type} updated to {is_streaming}")
        else:
            app.logger.warning(f"Failed to update stream status for {camera_type}: {response.status_code}")
    except Exception as e:
        app.logger.error(f"Error notifying frontend about stream status: {e}")