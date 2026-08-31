"""In-process serial-port simulation for hardware-free motion UI testing.

The adapter intentionally exposes the small pyserial-compatible surface used by
TabletScanner. It models Marlin position/mode state and acknowledges the BTT
Octopus commands used by the application without requiring a Windows COM-port
driver.
"""

import re
import threading


_AXIS_VALUE_RE = re.compile(r'(?:^|\s)([XYZA])\s*(-?\d+(?:\.\d+)?)', re.IGNORECASE)


class VirtualOctopusSerial:
    port = 'VIRTUAL_BTT_OCTOPUS'
    baudrate = 115200
    timeout = 0.2
    write_timeout = 0.5
    is_virtual = True

    def __init__(self):
        self.is_open = True
        self._lock = threading.RLock()
        self._input = bytearray()
        self._absolute_mode = True
        self._position = {'X': 0.0, 'Y': 0.0, 'Z': 0.0, 'A': 0.0}
        self._limits = {
            'X': (0.0, 175.0),
            'Y': (0.0, 165.0),
            'Z': (0.0, 40.0),
            # The physical filter revolver is circular and has no software
            # travel seam. The backend periodically normalizes its coordinate.
            'A': (float('-inf'), float('inf')),
        }
        self._light_pwm = {}
        self.command_history = []

    @property
    def in_waiting(self):
        with self._lock:
            return len(self._input)

    def write(self, data):
        if not self.is_open:
            raise OSError('Virtual COM port is closed')
        if isinstance(data, str):
            data = data.encode('ascii', 'ignore')
        with self._lock:
            for raw_command in bytes(data).decode('ascii', 'ignore').splitlines():
                command = raw_command.strip()
                if command:
                    self.command_history.append(command)
                    self._input.extend(self._execute(command))
        return len(data)

    def read(self, size=1):
        with self._lock:
            count = min(max(int(size), 0), len(self._input))
            result = bytes(self._input[:count])
            del self._input[:count]
            return result

    def readline(self):
        with self._lock:
            newline = self._input.find(b'\n')
            size = newline + 1 if newline >= 0 else len(self._input)
            return self.read(size)

    def flush(self):
        return None

    def reset_input_buffer(self):
        with self._lock:
            self._input.clear()

    def reset_output_buffer(self):
        return None

    def close(self):
        with self._lock:
            self.is_open = False
            self._input.clear()

    def open(self):
        with self._lock:
            self.is_open = True

    def _execute(self, command):
        upper = command.upper()
        code = upper.split(maxsplit=1)[0]

        if code == 'M115':
            return (
                b'FIRMWARE_NAME:Marlin TS-LIGHT-V3-P0F0-P1F1-P2HE0-P3HE1-LOCK-RHOME '
                b'MACHINE_TYPE:Tablet Scanner BTT Octopus V1.1\nok\n'
            )
        if code == 'M105':
            return b'ok T:25.00 /0.00 @:0\n'
        if code == 'M114':
            pos = self._position
            return f"X:{pos['X']:.3f} Y:{pos['Y']:.3f} Z:{pos['Z']:.3f} E:0.000\nok\n".encode('ascii')
        if code == 'G90':
            self._absolute_mode = True
        elif code == 'G91':
            self._absolute_mode = False
        elif code == 'G28':
            axes = [axis for axis in ('X', 'Y', 'Z', 'A') if re.search(rf'(?:^|\s){axis}(?:\s|$)', upper)]
            for axis in axes or ('X', 'Y', 'Z'):
                # X/Z home to their native minimum and Y homes to its native
                # maximum. Each uses a 2 mm post-home backoff.
                if axis == 'Y':
                    self._position[axis] = self._limits[axis][1] - 2.0
                else:
                    self._position[axis] = 2.0 if axis in ('X', 'Z') else 0.0
        elif code in ('G0', 'G1'):
            for axis, raw_value in _AXIS_VALUE_RE.findall(upper):
                value = float(raw_value)
                target = value if self._absolute_mode else self._position[axis] + value
                low, high = self._limits[axis]
                self._position[axis] = max(low, min(high, target))
        elif code == 'G92':
            for axis, raw_value in _AXIS_VALUE_RE.findall(upper):
                self._position[axis] = float(raw_value)
        elif code == 'M106':
            selector = re.search(r'(?:^|\s)P(\d+)', upper)
            pwm = re.search(r'(?:^|\s)S(\d+)', upper)
            if selector and pwm:
                selector_number = int(selector.group(1))
                pwm_value = max(0, min(255, int(pwm.group(1))))
                if pwm_value:
                    for output in range(4):
                        self._light_pwm[output] = 0
                self._light_pwm[selector_number] = pwm_value

        # Marlin acknowledges supported control commands and benign unknown
        # commands alike. This mirrors firmware behaviour needed by raw G-code
        # testing while keeping all state changes inside this process.
        return b'ok\n'
