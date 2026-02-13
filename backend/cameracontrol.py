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
import globals
from globals import app

from logger_config import CameraError


CAMERA_PIXEL_FORMAT = "BayerGR10p"   # what the camera outputs
OPENCV_PIXEL_TYPE = pylon.PixelType_BGR8packed  # what OpenCV receives

converter = pylon.ImageFormatConverter()
converter.OutputPixelFormat = OPENCV_PIXEL_TYPE
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned


opencv_display_format = 'BGR8'


def load_camera_profile(camera, pfs_path: str) -> dict:
    """
    Load a .pfs (Pylon Feature Set) file onto the camera.
    
    Args:
        camera: Basler InstantCamera object (must be open)
        pfs_path: Path to the .pfs file
        
    Returns:
        dict: {"success": True} or {"error": "message", "code": "E1311"}
    """
    import time
    import globals
    
    if not camera or not camera.IsOpen():
        return {
            "error": "Kamera nincs csatlakoztatva",
            "code": "E1311"
        }
    
    if not pfs_path:
        return {
            "error": "Nincs megadva kamera profil fájl",
            "code": "E1311"
        }
    
    if not os.path.isfile(pfs_path):
        return {
            "error": f"Kamera profil fájl nem található: {pfs_path}",
            "code": "E1311"
        }
    
    try:
        # Stop video stream if running (critical to avoid state conflicts)
        was_streaming = getattr(globals, "stream_running", False)
        if was_streaming:
            globals.stream_running = False
            app.logger.info("Stopped video stream to load .pfs profile")
            # Give stream loop time to exit gracefully
            time.sleep(0.3)
        
        # Stop grabbing if active (required for profile loading)
        was_grabbing = camera.IsGrabbing()
        if was_grabbing:
            camera.StopGrabbing()
            app.logger.info("Stopped grabbing to load .pfs profile")
        
        # Load the .pfs file
        pylon.FeaturePersistence.Load(pfs_path, camera.GetNodeMap(), True)
        app.logger.info(f"Camera profile loaded successfully: {pfs_path}")
        
        # Restart grabbing if it was active
        if was_grabbing:
            camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            app.logger.info("Restarted grabbing after loading .pfs profile")
        
        # Restart streaming if it was active
        if was_streaming:
            globals.stream_running = True
            app.logger.info("Restarted video stream after loading .pfs profile")
        
        return {"success": True}
        
    except Exception as e:
        app.logger.error(f"Failed to load camera profile {pfs_path}: {e}")
        return {
            "error": f"Kamera profil betöltése sikertelen: {str(e)}",
            "code": "E1311"
        }


def grab_and_convert_frame(camera, timeout_ms=5000, retries=2):
    """
    Unified frame grab and conversion function.
    
    Grabs a frame from the camera, converts BayerGR10p -> BGR8, and releases the grab result.
    The returned frame is BGR8 (uint8, HxWx3) ready for OpenCV processing.
    
    Args:
        camera: Basler InstantCamera object (must be open and grabbing)
        timeout_ms: Timeout in milliseconds for frame retrieval
        
    Returns:
        frame_bgr: NumPy array (HxWx3, uint8, BGR format)
        
    Raises:
        RuntimeError: If camera not open/grabbing, or grab fails
    """
    if not (camera and camera.IsOpen()):
        raise RuntimeError("Camera not open")
    
    if not camera.IsGrabbing():
        camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    last_error = None
    attempts = max(1, int(retries) + 1)

    for attempt in range(attempts):
        grab_result = None
        try:
            grab_result = camera.RetrieveResult(int(timeout_ms), pylon.TimeoutHandling_ThrowException)
            if not grab_result.GrabSucceeded():
                last_error = RuntimeError("Frame grab unsuccessful")
                continue

            # Convert BayerGR10p (or raw Bayer) -> BGR8 for OpenCV
            frame_bgr = converter.Convert(grab_result).GetArray()

            # Return a copy so the frame persists after release
            return frame_bgr.copy()
        except Exception as e:
            last_error = e
        finally:
            try:
                if grab_result is not None:
                    grab_result.Release()
            except Exception:
                pass

        if attempt < attempts - 1:
            time.sleep(0.05)

    if last_error:
        raise RuntimeError(str(last_error))
    raise RuntimeError("Frame grab unsuccessful")


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

def stream_video(scale_factor: float = 1.0, jpeg_quality: int = 80):
    """
    Generator that yields multipart JPEG frames from the camera.
    
    Uses grab_and_convert_frame() for unified BayerGR10p -> BGR8 conversion.
    All frames are converted to BGR immediately after grabbing for consistency.
    
    Args:
        scale_factor: Resize factor (1.0 = no resize)
        jpeg_quality: JPEG encoding quality (1-100)
        
    Yields:
        bytes: Multipart JPEG frame ready for HTTP streaming
    """
    cam = getattr(globals, "camera", None)
    if not (cam and cam.IsOpen()):
        app.logger.error("Camera is not open.")
        return

    if not cam.IsGrabbing():
        cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    app.logger.info("Camera streaming generator started.")
    globals.stream_running = True

    # Ensure a single grab lock exists
    lock = getattr(globals, "grab_lock", None)
    if lock is None:
        from threading import Lock
        globals.grab_lock = Lock()
        lock = globals.grab_lock

    # Rate-limit error logging: track consecutive errors and last log time
    consecutive_errors = 0
    last_error_log_time = 0.0
    ERROR_LOG_INTERVAL = 5.0  # Log at most once every 5 seconds per error type

    try:
        while getattr(globals, "stream_running", False):
            try:
                # Grab and convert inside the lock (short critical section)
                with lock:
                    cam = getattr(globals, "camera", None)
                    if not (cam and cam.IsOpen()):
                        app.logger.warning("Camera closed during streaming.")
                        break

                    # Use unified grab+convert function
                    image_bgr = grab_and_convert_frame(cam, timeout_ms=5000)

                # Reset error counter on successful frame
                consecutive_errors = 0

                # Resize outside the lock
                if scale_factor and scale_factor != 1.0:
                    h, w = image_bgr.shape[:2]
                    new_w = max(1, int(w * scale_factor))
                    new_h = max(1, int(h * scale_factor))
                    image_bgr = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

                # Encode to JPEG
                ok, frame = cv2.imencode(
                    ".jpg",
                    image_bgr,
                    [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
                )
                if not ok:
                    app.logger.error("Failed to encode frame.")
                    continue

                # Yield multipart JPEG
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame.tobytes() + b"\r\n"
                )

            except Exception as e:
                error_str = str(e).lower()
                app.logger.debug(f"Error in video stream loop: {e}")

                # Camera physically disconnected — stop the stream
                if "removed" in error_str or "device has been" in error_str:
                    app.logger.error(f"Camera disconnected during streaming: {e}")
                    break

                consecutive_errors += 1

                # Rate-limit warning logs to avoid log spam
                now = time.time()
                if "not grabbing" not in error_str:
                    if now - last_error_log_time >= ERROR_LOG_INTERVAL:
                        suppressed = f" ({consecutive_errors} errors since last log)" if consecutive_errors > 1 else ""
                        app.logger.warning(f"Video stream error: {e}{suppressed}")
                        last_error_log_time = now

                # Back off to avoid tight-looping on persistent errors
                backoff = min(0.5, 0.05 * consecutive_errors)
                time.sleep(backoff)
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

    except Exception as e:
        app.logger.error(f"Error setting camera parameters: {e}")
        raise CameraError("Error setting camera parameters.") from e


def setup_pixel_format(camera: pylon.InstantCamera):
    try:
        current = camera.PixelFormat.GetValue()
        if current != CAMERA_PIXEL_FORMAT:
            camera.PixelFormat.SetValue(CAMERA_PIXEL_FORMAT)
            app.logger.info(f"PixelFormat set: {current} -> {CAMERA_PIXEL_FORMAT}")
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
                image = converter.Convert(grab_result).GetArray()
                cv2.imshow("Stream", image)
                if cv2.waitKey(1) == 13:  # Enter key to exit streaming loop
                    break
                handler.save_frame(image)  # Pass grab_result or image based on implementation
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

    if hasattr(camera, 'Gamma'):
        safe_get('Gamma', lambda: {
            'min': camera.Gamma.GetMin(),
            'max': camera.Gamma.GetMax(),
            'inc': 0.01
        })

    if hasattr(camera, 'Gain'):
        safe_get('Gain', lambda: {
            'min': camera.Gain.GetMin(),
            'max': camera.Gain.GetMax(),
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

    # Try to get light-specific settings (camera_params_dome or camera_params_bar)
    # If neither exists, fall back to old 'camera' section for backwards compatibility
    camera_settings = settings.get('camera_params_dome', {}) or settings.get('camera_params_bar', {}) or settings.get('camera', {}) or {}

    if not camera_settings:
        app.logger.warning("No camera settings found. Skipping apply.")
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

def _apply_camera_param(camera, param_name: str, valid_value):
    """Apply a single validated parameter value to the camera hardware."""
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
    elif param_name == 'Gamma':
        camera.Gamma.SetValue(valid_value)
    elif param_name == 'Gain':
        camera.Gain.SetValue(valid_value)

    # Optional reversals
    camera.ReverseX.SetValue(True)
    camera.ReverseY.SetValue(True)


def validate_and_set_camera_param(camera, param_name: str, param_value: float, properties: dict):
    valid_value = validate_param(param_name, param_value, properties)

    MAX_RETRIES = 3
    RETRY_DELAY = 0.15  # seconds between retries

    try:
        was_streaming = globals.stream_running

        # Stop streaming if changing critical dimensions
        if param_name in ['Width', 'Height'] and was_streaming:
            globals.stream_running = False
            if camera.IsGrabbing():
                camera.StopGrabbing()
                app.logger.info(f"Camera stream stopped to apply {param_name} change.")
            if globals.stream_thread and globals.stream_thread.is_alive():
                globals.stream_thread.join(timeout=2)
                app.logger.info("Camera stream thread joined.")
            globals.stream_thread = None
            time.sleep(0.5)

        if not camera.IsOpen():
            camera.Open()
            app.logger.info(f"Camera reopened to apply {param_name}.")

        # Acquire grab_lock so we don't collide with the video stream's
        # frame grabs – the USB bus can't handle concurrent register
        # writes and frame retrieval, causing TimeoutExceptions.
        lock = getattr(globals, "grab_lock", None)
        if lock is None:
            from threading import Lock
            globals.grab_lock = Lock()
            lock = globals.grab_lock

        last_error = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                with lock:
                    _apply_camera_param(camera, param_name, valid_value)
                last_error = None
                break  # success
            except Exception as e:
                last_error = e
                is_timeout = "timeout" in str(e).lower() or "TimeoutException" in type(e).__name__
                if is_timeout and attempt < MAX_RETRIES:
                    app.logger.warning(
                        f"Timeout setting {param_name} (attempt {attempt}/{MAX_RETRIES}), retrying..."
                    )
                    time.sleep(RETRY_DELAY)
                else:
                    break

        if last_error is not None:
            raise last_error

        app.logger.info(f"Camera {param_name} successfully set to {valid_value}")

        # Restart streaming if we had to stop it
        if param_name in ['Width', 'Height'] and was_streaming:
            globals.stream_running = True
            globals.stream_thread = threading.Thread(target=stream_video, args=(1.0, 80))
            globals.stream_thread.start()
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