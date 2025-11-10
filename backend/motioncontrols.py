import porthandler
import time
import re
import app
from flask import jsonify
from typing import Dict

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
    # If you have a shared serial lock, use it here:
    if not globals.motion_busy:
        try:
            buf = bytearray()
            deadline = time.monotonic() + 0.15
            with porthandler.motion_lock:
                ser.write(b'M105\n')
                # read incoming bytes until "ok" or timeout...
                ...
            if buf:
                app.logger.debug(f"M105 non-ok reply: {buf[:64]!r}")
        except Exception as e:
            app.logger.debug(f"status probe error (ignored): {e}")
        return jsonify({'connected': True, 'port': ser.port}), 200
    end = time.monotonic() + timeout
    buf = bytearray()

    while time.monotonic() < end:
        iw = getattr(ser, "in_waiting", 0) or 0
        if iw:
            chunk = ser.read(min(iw, 128))
            if chunk:
                buf += chunk
                if b"ok" in buf.lower():
                    break
        else:
            time.sleep(0.01)

    s = buf.decode("ascii", "ignore")
    m = _POS_RE.search(s)
    if not m:
        # Fallback: some firmwares print without labels: "X:1.23 Y:4.56 Z:7.89" still matches
        raise RuntimeError("No M114 position in reply")

    x, y, z = float(m.group(1)), float(m.group(2)), float(m.group(3))
    return {"x": x, "y": y, "z": z}

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
def move_to_position(motion_platform, x_pos, y_pos, z_pos = None):
    # Set the printer to use absolute coordinates
    porthandler.write(motion_platform, "G90")

    # Construct the move command with the specified coordinates
    move_command = f"G1 X{x_pos} Y{y_pos}"
    
    # If a z-coordinate is provided, include it in the move command
    if z_pos is not None:
        move_command += f" Z{z_pos}"

    try:
        # Send the move command to the printer
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