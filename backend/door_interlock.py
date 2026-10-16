"""Firmware door reports, retained across serial drains and fragmented reads."""
import re
import threading

_REPORT = re.compile(rb'^TS_DOORS FRONT:([01]) TOP:([01]) UV_BLOCKED:([01]) TRIP:(\d{1,10})\r?$')


def unknown_door_status():
    return {'known': False, 'front_open': None, 'top_open': None,
            'uv_blocked': True, 'trip_count': None}


class DoorInterlockState:
    def __init__(self):
        self._lock = threading.RLock()
        self._serial = None
        self._pending = b''
        self._state = None
        self._revision = 0

    def feed(self, serial_port, data):
        with self._lock:
            if serial_port is not self._serial:
                self._serial = serial_port
                self._state = None
                self._pending = b''
            lines = (self._pending + bytes(data)).split(b'\n')
            self._pending = lines.pop()[-512:]
            for line in lines:
                match = _REPORT.fullmatch(line)
                if not match:
                    if line.startswith(b'TS_DOORS') or line.rstrip(b'\r') == b'start':
                        self._state = None
                    continue
                front, top, blocked, trips = map(int, match.groups())
                if blocked != bool(front or top) or trips > 0xffffffff:
                    self._state = None
                    continue
                self._state = {
                    'known': True, 'front_open': bool(front), 'top_open': bool(top),
                    'uv_blocked': bool(blocked), 'trip_count': trips,
                }
                self._revision += 1

    def revision(self):
        with self._lock:
            return self._revision

    def invalidate(self, serial_port):
        with self._lock:
            if serial_port is self._serial:
                self._state = None
                self._pending = b''

    def snapshot(self, serial_port):
        with self._lock:
            if serial_port and getattr(serial_port, 'is_open', False) and serial_port is self._serial and self._state:
                return dict(self._state)
            return unknown_door_status()
