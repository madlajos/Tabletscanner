"""Fresh, illuminated still frames with bounded acquisition and live-stream arbitration."""

import math

from pypylon import pylon

import globals
from cameracontrol import grab_and_convert_frame, suppress_preview_grabs
from light_control import CaptureIlluminationError


LAMP_SETTLE_SECONDS = 0.25


@suppress_preview_grabs()
def capture_series_frame(camera, lights, channel, mode, coordinator):
    """Return one copied BGR frame, or None on cooperative cancellation.

    Stop/restart acquisition to discard preview frames exposed under an earlier
    filter or in darkness. After activating the lamp, acquire and discard one
    guard frame before returning the next frame. This makes the wait inherently
    follow the configured exposure time and ensures the saved exposure starts
    only after one complete illuminated camera cycle. The lamp is activated only
    after obtaining grab_lock; neither lock contention, filter motion nor JPEG
    saving consumes its window. The normal timeout monitor stays enabled for the
    entire operation.
    """
    if not globals.grab_lock.acquire(timeout=5):
        raise RuntimeError('The camera is busy with another acquisition.')
    was_grabbing = False
    try:
        if not camera or not camera.IsOpen():
            raise RuntimeError('Camera is disconnected.')
        was_grabbing = camera.IsGrabbing()
        if was_grabbing:
            camera.StopGrabbing()
        if coordinator.cancellation_requested():
            return None
        exposure_seconds = float(camera.ExposureTime.GetValue()) / 1_000_000
        if not math.isfinite(exposure_seconds) or not 0 < exposure_seconds < 29:
            raise CaptureIlluminationError('The exposure must be shorter than 29 seconds.')
        lights.activate(channel, mode)
        remaining = lights.capture_remaining_seconds(channel, mode)
        # Two complete acquisitions (guard + saved frame) must fit without
        # renewing or bypassing a UV thermal deadline. Keep additional time for
        # lamp settling, sensor readout and normal thread scheduling.
        if remaining is not None and (2 * exposure_seconds) + LAMP_SETTLE_SECONDS + 0.3 >= remaining:
            raise CaptureIlluminationError('The exposure does not fit the configured UV safety timeout.')
        if coordinator.wait_for_cancellation(LAMP_SETTLE_SECONDS):
            return None

        camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

        remaining = lights.capture_remaining_seconds(channel, mode)
        guard_timeout_seconds = min(
            30.,
            exposure_seconds + 2.,
            remaining - exposure_seconds - 0.2 if remaining is not None else 30.,
        )
        if guard_timeout_seconds <= exposure_seconds:
            raise CaptureIlluminationError('Insufficient illumination time remains for the guard exposure.')
        grab_and_convert_frame(
            camera,
            timeout_ms=max(1, math.ceil(guard_timeout_seconds * 1000)),
            retries=0,
        )

        # A UV deadline may expire while RetrieveResult is blocked. Validate it
        # after the guard frame and again after the frame that will be saved.
        lights.capture_remaining_seconds(channel, mode)
        if coordinator.cancellation_requested():
            return None

        remaining = lights.capture_remaining_seconds(channel, mode)
        timeout_seconds = min(30., exposure_seconds + 2., remaining - 0.1 if remaining is not None else 30.)
        if timeout_seconds <= exposure_seconds:
            raise CaptureIlluminationError('Insufficient illumination time remains for the exposure.')
        frame = grab_and_convert_frame(
            camera,
            timeout_ms=max(1, math.ceil(timeout_seconds * 1000)),
            retries=0,
        )
        # Never save a dark/partial frame if the monitor expired or a lamp was lost.
        lights.capture_remaining_seconds(channel, mode)
        return frame
    finally:
        try:
            try:
                if camera and camera.IsOpen() and camera.IsGrabbing():
                    camera.StopGrabbing()
            finally:
                lights.off()
        finally:
            try:
                if was_grabbing and camera and camera.IsOpen():
                    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            finally:
                globals.grab_lock.release()
