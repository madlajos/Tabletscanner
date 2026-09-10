
import numpy as np
import time
import cv2
import globals
from motioncontrols import move_relative, get_toolhead_position  # (get_toolhead_position lehet unused, hagyom)
import porthandler


def wait_motion_done(motion_platform):
    porthandler.write_and_wait_motion(motion_platform, "M400", timeout=30.0)


def move_to_virtual_z(motion_platform, current_z, target_z, settle_s=0):
    # !!! MOZGÁS: NEM NYÚLTUNK !!!
    dz = float(target_z) - float(current_z)
    print("Menj " + str(target_z) + " pozícióra")
    if abs(dz) > 1e-9:
        move_relative(motion_platform, z=dz)
        time.sleep(float(settle_s))
        wait_motion_done(motion_platform)
    return float(target_z)


def acquire_frame(timeout_ms=2000, retries=2):
    cam = globals.camera
    if cam is None or not cam.IsOpen():
        raise RuntimeError("Camera not ready")
    from cameracontrol import grab_and_convert_frame, suppress_preview_grabs
    with suppress_preview_grabs():
        with globals.grab_lock:
            return grab_and_convert_frame(cam, timeout_ms=timeout_ms, retries=retries)


def _err(code: str, **extra):
    d = {"status": "ERROR", "code": str(code)}
    if extra:
        d.update(extra)
    return d


def _ok(**extra):
    d = {"status": "OK"}
    if extra:
        d.update(extra)
    return d


def calculate_median_and_span(motion_platform, z_max=40, frame_scale=0.1, grab_timeout_ms=2000, needed=False):
    # start
    if needed is True:
        current_z = 0.0
        current_z = move_to_virtual_z(motion_platform, current_z, float(z_max))
        time.sleep(1)

    # first frame
    first_frame = acquire_frame(timeout_ms=grab_timeout_ms)
    if frame_scale is not None and float(frame_scale) != 1.0:
        first_frame = cv2.resize(
            first_frame, None,
            fx=float(frame_scale), fy=float(frame_scale),
            interpolation=cv2.INTER_AREA
        )

    try:
        # gray
        if first_frame.ndim == 3 and first_frame.shape[2] >= 3:
            gray = cv2.cvtColor(first_frame[..., :3], cv2.COLOR_BGR2GRAY)
        else:
            gray = first_frame.astype(np.uint8)

        # Otsu mask
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # masked stats
        values = gray[mask > 0]
        if values.size == 0:
            return _err("EMPTY_MASK", message="Otsu mask is empty, no pixels selected")

        gray_med = float(np.median(values))
        gray_min = float(np.min(values))
        gray_max = float(np.max(values))
        gray_span = float(gray_max - gray_min)

        return _ok(
            gray_median=gray_med,
            gray_min=gray_min,
            gray_max=gray_max,
            gray_span=gray_span,
        )

    except Exception as e:
        return _err("CALCULATION_FAILED", message=str(e))
