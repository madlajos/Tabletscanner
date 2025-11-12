import porthandler
import time
import re
import globals
from flask import jsonify
from typing import Dict
import logging
log = logging.getLogger(__name__)

_POS_RE = re.compile(r'X:\s*(-?\d+(?:\.\d+)?)\s+Y:\s*(-?\d+(?:\.\d+)?)\s+Z:\s*(-?\d+(?:\.\d+)?)', re.I)


class Printer:
    def __init__(self, port):
        self.port = port

def home_axes(motion_platform, *axes):
    # If no axes are specified, home all axes
    if not axes:
        axes = ['X', 'Y', 'Z']
    else:
        # Ensure each axis has a space between them
        axes = [axis.upper() for axis in axes]

    axes_str = " ".join(axes)
    command = f"G28{axes_str}"

    try:
        porthandler.write(motion_platform, command)
    except Exception as e:
        print(f"An error occured while sending the Homing command to the Motion platform: {e}")

def disable_steppers(motion_platform, *axes):
    # If no axes are specified, disable all steppers
    if not axes:
        axes = ['X', 'Y', 'Z']
    else:
        # Ensure each axis has a space between them
        axes = [axis.upper() for axis in axes]
    
    axes_str = " ".join(axes)
    command = f"M84{axes_str}"

    try:
        porthandler.write(motion_platform, command)
    except Exception as e:
        print(f"An error occured while sending the Disable Steppers command to the Motion platform: {e}")

def get_toolhead_position(ser, timeout: float = 0.3) -> Dict[str, float]:
    """
    Sends M114 and returns {"x":..., "y":..., "z":...} with a hard overall timeout.
    """
    # Quick status probe to drain/flush noisy buffers when NOT busy.
    if not globals.motion_busy:
        try:
            # Clear any stale bytes first to avoid mixing with the M105 we send now.
            try:
                if hasattr(ser, "reset_input_buffer"):
                    ser.reset_input_buffer()
                else:
                    iw = getattr(ser, "in_waiting", 0) or 0
                    if iw:
                        ser.read(iw)
            except Exception:
                pass

            buf = bytearray()
            deadline = time.monotonic() + 0.15
            with porthandler.motion_lock:
                ser.write(b'M105\n')

            # Read briefly; typical reply: b"ok T:25.00 /0.00 @:0\n"
            while time.monotonic() < deadline:
                iw = getattr(ser, "in_waiting", 0) or 0
                if iw:
                    chunk = ser.read(min(iw, 256))
                    if chunk:
                        buf += chunk
                    # Stop once we see 'ok' or a newline; we don't need the temp value
                    if b"ok" in buf.lower() or b"\n" in buf:
                        break
                else:
                    time.sleep(0.01)

            # Only log if the reply looks truly unexpected (no 'ok' seen)
            if buf and b"ok" not in buf.lower():
                log.debug(f"M105 unexpected reply: {buf[:64]!r}")
        except Exception as e:
            log.debug(f"M105 probe error (ignored): {e}")

        # Hard flush again so the upcoming M114 parse isn't polluted by M105
        try:
            if hasattr(ser, "reset_input_buffer"):
                ser.reset_input_buffer()
            else:
                iw = getattr(ser, "in_waiting", 0) or 0
                if iw:
                    ser.read(iw)
        except Exception:
            pass
    else:
        # During long ops (e.g. homing), avoid issuing M114; let caller use cache.
        raise RuntimeError("Motion platform busy")

    # Query current position.
    with porthandler.motion_lock:
        ser.write(b'M114\n')

    end = time.monotonic() + timeout
    buf = bytearray()

    while time.monotonic() < end:
        iw = getattr(ser, "in_waiting", 0) or 0
        if iw:
            chunk = ser.read(min(iw, 256))
            if chunk:
                buf += chunk
                # Heuristic: break once a typical M114 line is complete.
                if (b"X:" in buf and b"Y:" in buf and b"Z:" in buf) and (b"\n" in buf or b"ok" in buf.lower()):
                    break
        else:
            time.sleep(0.01)

    s = buf.decode("ascii", "ignore")

    # Try regex first if available.
    try:
        m = _POS_RE.search(s)  # e.g. r"X:([-0-9.]+).*?Y:([-0-9.]+).*?Z:([-0-9.]+)"
    except NameError:
        m = None

    if m:
        x, y, z = float(m.group(1)), float(m.group(2)), float(m.group(3))
        return {"x": x, "y": y, "z": z}

    # Fallback parser (handles lines like: "X:10.00 Y:20.00 Z:30.00 E:...").
    try:
        pos = parse_position(s)
        if all(k in pos for k in ("x", "y", "z")):
            return {"x": float(pos["x"]), "y": float(pos["y"]), "z": float(pos["z"])}
    except Exception:
        pass

    # No parseable coordinates found.
    raise RuntimeError("No M114 position in reply")


def parse_position(response):
    # Example response: "X:10.00 Y:20.00 Z:30.00 E:0.00 Count X:8100 Y:0 Z:4320"
    position = {}
    lines = response.split('\n')
    
    for line in lines:
        if "X:" in line and "Y:" in line and "Z:" in line:
            for axis in ['X', 'Y', 'Z']:
                start = line.find(axis + ":")
                if start != -1:
                    end = line.find(" ", start)
                    value = line[start+2:end]
                    position[axis.lower()] = float(value)
    
    return position


# Moves the Motion platform toolhead to a specified location. If the Z coordinate is not given,
# it remains unchanged.
# motioncontrols.py

def move_to_position(motion_platform, x_pos=None, y_pos=None, z_pos=None):
    # Absolute mode
    porthandler.write(motion_platform, "G90")

    parts = []
    if x_pos is not None:
        parts.append(f"X{x_pos}")
    if y_pos is not None:
        parts.append(f"Y{y_pos}")
    if z_pos is not None:
        parts.append(f"Z{z_pos}")

    if not parts:
        return  # nothing to do

    move_command = "G1 " + " ".join(parts)

    try:
        porthandler.write(motion_platform, move_command)
    except Exception as e:
        print(f"Error occurred while sending move to position command: {e}")


# Moves the Motion platform by the specified values.
# Can be called with 1-3 arguements, like move_relative(printer, x=1, y=1)
def move_relative(motion_platform, x=None, y=None, z=None):
    # Set the printer to use relative coordinates
    porthandler.write(motion_platform, "G91")

    # Construct the move command with the specified distances
    move_command = "G1"

    # Add x-axis movement if provided
    if x is not None:
        move_command += f" X{x}"

    # Add y-axis movement if provided
    if y is not None:
        move_command += f" Y{y}"

    # Add z-axis movement if provided
    if z is not None:
        move_command += f" Z{z}"

    try:
        # Send the move command to the printer
        porthandler.write(motion_platform, move_command)
    except Exception as e:
        print(f"Error occurred while sending move command: {e}")