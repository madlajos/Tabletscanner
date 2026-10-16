"""Cooperative door pauses owned by one automatic-measurement request.

Only the request's bounded motion/acquisition phases install a guard. Preview
workers and the thermal monitor remain independent. A trip restarts the current
phase, never a completed capture row or an entire tablet with saved images.
"""

from contextvars import ContextVar
from collections import OrderedDict
from threading import Event, Lock
from time import monotonic

from light_control import DoorCaptureInterrupted, DoorOpenError


_current = ContextVar('measurement_door_guard', default=None)
_cancel_lock = Lock()
_cancelled_requests = OrderedDict()


def cancel_measurement_request(request_id):
    """Remember a stop even when its tablet POST has not reached Flask yet."""
    if not isinstance(request_id, str) or not 1 <= len(request_id) <= 128:
        raise ValueError('Invalid measurement request ID.')
    with _cancel_lock:
        _cancelled_requests[request_id] = monotonic() + 3600
        _cancelled_requests.move_to_end(request_id)
        while len(_cancelled_requests) > 128:
            _cancelled_requests.popitem(last=False)


def measurement_request_cancelled(request_id):
    if not isinstance(request_id, str):
        return False
    with _cancel_lock:
        deadline = _cancelled_requests.get(request_id)
        if deadline is None:
            return False
        if monotonic() >= deadline:
            _cancelled_requests.pop(request_id, None)
            return False
        return True


class MeasurementCancelled(RuntimeError):
    pass


class MeasurementDoorInterrupted(RuntimeError):
    pass


def acquisition_checkpoint():
    guard = _current.get()
    if guard is not None:
        guard.checkpoint()


def before_serial_command(command):
    parts = command.upper().split(maxsplit=1)
    if parts and parts[0] in ('G0', 'G1', 'G28'):
        acquisition_checkpoint()


class MeasurementDoorGuard:
    def __init__(self, serial_port, lights, refresh, publish_pause, cancelled,
                 wait=None):
        self.serial_port = serial_port
        self.lights = lights
        self.refresh = refresh
        self.publish_pause = publish_pause
        self.cancelled = cancelled
        self.wait = wait or Event().wait
        self.trip = None

    def cancellation_requested(self):
        return self.cancelled()

    def wait_for_cancellation(self, seconds):
        self.wait(seconds)
        return self.cancelled()

    def _check_cancelled(self):
        if self.cancelled():
            raise MeasurementCancelled('Measurement stopped while waiting for the doors.')

    def checkpoint(self):
        self._check_cancelled()
        doors = self.refresh(self.serial_port)
        if not doors['known'] or doors['uv_blocked'] or doors['trip_count'] != self.trip:
            raise MeasurementDoorInterrupted('Door state changed during measurement.')

    def wait_closed(self, interrupted=False):
        self._check_cancelled()
        doors = self.refresh(self.serial_port)
        paused = (interrupted or not doors['known'] or doors['uv_blocked']
                  or (self.trip is not None and doors['trip_count'] != self.trip))
        if paused:
            self.publish_pause(True)
            # OFF is allowed even while state is unknown. Never disable the
            # normal thermal monitor, nor hold serial/camera locks while waiting.
            self.lights.off()
        while not doors['known'] or doors['uv_blocked']:
            self._check_cancelled()
            self.wait(0.1)
            doors = self.refresh(self.serial_port)
        self._check_cancelled()
        self.trip = doors['trip_count']
        if paused:
            self.publish_pause(False)

    def run(self, operation, *, prepare=None, validate_after=True):
        self.wait_closed()
        while True:
            token = _current.set(self)
            try:
                self._check_cancelled()
                if prepare is not None:
                    prepare()
                result = operation()
                if validate_after:
                    self.checkpoint()
                return result
            except (MeasurementDoorInterrupted, DoorCaptureInterrupted, DoorOpenError):
                pass
            except MeasurementCancelled:
                raise
            except Exception:
                # Saving can have committed a valid image already. Only the
                # explicit acquisition/door exceptions above may retry it.
                if not validate_after:
                    raise
                # Older adapters can wrap an interrupted motion in their own
                # exception. Recheck the phase's trip before classifying it.
                try:
                    self.checkpoint()
                except MeasurementDoorInterrupted:
                    pass
                else:
                    raise
            finally:
                _current.reset(token)
            self.wait_closed(interrupted=True)
