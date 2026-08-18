import porthandler
import re
import globals
from flask import jsonify
from typing import Dict
import logging
log = logging.getLogger(__name__)

_POS_RE = re.compile(r'X:\s*(-?\d+(?:\.\d+)?)\s+Y:\s*(-?\d+(?:\.\d+)?)\s+Z:\s*(-?\d+(?:\.\d+)?)', re.I)

A_AXIS_HOMING_TIMEOUT_SECONDS = 60.0
DEFAULT_HOMING_TIMEOUT_SECONDS = 30.0
Y_NATIVE_MAX_MM = 175.0


def logical_y_to_native(y: float) -> float:
    """Map operator Y (increasing away from home) to Marlin's CoreXY Y."""
    return Y_NATIVE_MAX_MM - float(y)


def native_y_to_logical(y: float) -> float:
    """Map Marlin's max-homed CoreXY Y to the operator coordinate system."""
    return Y_NATIVE_MAX_MM - float(y)


class HomingError(RuntimeError):
    def __init__(self, axis: str, timeout: float, reply: bytes):
        self.axis = axis
        self.timeout = timeout
        self.reply = reply
        super().__init__(f'{axis} axis homing failed.')


class HomingRejectedError(HomingError):
    pass


class HomingTimeoutError(HomingError):
    pass


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
    command = f"G28 {axes_str}"
    timeout_seconds = (
        A_AXIS_HOMING_TIMEOUT_SECONDS
        if axes == ['A']
        else DEFAULT_HOMING_TIMEOUT_SECONDS
    )

    try:
        # Hold the shared serial lock until Marlin acknowledges completion so
        # no status probe or second motion command can interleave with G28.
        acknowledged, reply = porthandler.write_and_wait(
            motion_platform,
            command,
            timeout=timeout_seconds,
            failure_markers=(b'homing failed', b'error:'),
        )
        if acknowledged:
            return True
        if b'homing failed' in reply.lower() or b'error:' in reply.lower():
            raise HomingRejectedError(axes_str, timeout_seconds, reply)
        raise HomingTimeoutError(axes_str, timeout_seconds, reply)
    except HomingError:
        raise
    except (OSError, PermissionError):
        raise  # let caller handle USB disconnect
    except Exception as e:
        log.error(f"Error sending homing command to Motion platform: {e}")
        return False

def disable_steppers(motion_platform, *axes):
    # If no axes are specified, disable all steppers
    if not axes:
        axes = ['X', 'Y', 'Z']
    else:
        # Ensure each axis has a space between them
        axes = [axis.upper() for axis in axes]
    
    axes_str = " ".join(axes)
    command = f"M84 {axes_str}"

    try:
        ok, reply = porthandler.write_and_wait(motion_platform, command, timeout=2.0)
        if not ok:
            log.warning(f"Disable steppers command '{command}' was not acknowledged")
        return ok
    except (OSError, PermissionError):
        raise  # let caller handle USB disconnect
    except Exception as e:
        log.error(f"Error sending Disable Steppers command: {e}")
        return False

def get_toolhead_position(ser, timeout: float = 0.3, allow_busy: bool = False) -> Dict[str, float]:
    """
    Sends M114 and returns {"x":..., "y":..., "z":...} with a hard overall timeout.
    """
    if globals.motion_busy and not allow_busy:
        # During long ops (e.g. homing), avoid issuing M114; let caller use cache.
        raise RuntimeError("Motion platform busy")

    # The shared command reader owns the lock until Marlin's acknowledgement,
    # so connection polling cannot consume part of this M114 response.
    acknowledged, reply = porthandler.write_and_wait(
        ser,
        'M114',
        timeout=timeout,
        expect=b'ok',
    )
    if not acknowledged:
        log.debug('M114 was not acknowledged: %r', reply[:128])

    s = reply.decode("ascii", "ignore")

    # Try regex first if available.
    try:
        m = _POS_RE.search(s)  # e.g. r"X:([-0-9.]+).*?Y:([-0-9.]+).*?Z:([-0-9.]+)"
    except NameError:
        m = None

    if m:
        x, y, z = float(m.group(1)), float(m.group(2)), float(m.group(3))
        return {"x": x, "y": native_y_to_logical(y), "z": z}

    # Fallback parser (handles lines like: "X:10.00 Y:20.00 Z:30.00 E:...").
    try:
        pos = parse_position(s)
        if all(k in pos for k in ("x", "y", "z")):
            return {
                "x": float(pos["x"]),
                "y": native_y_to_logical(pos["y"]),
                "z": float(pos["z"]),
            }
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
    parts = []
    if x_pos is not None:
        parts.append(f"X{x_pos}")
    if y_pos is not None:
        parts.append(f"Y{logical_y_to_native(y_pos)}")
    if z_pos is not None:
        parts.append(f"Z{z_pos}")

    if not parts:
        return  # nothing to do

    try:
        # Set absolute mode and wait for board acknowledgement
        ok, _ = porthandler.write_and_wait(motion_platform, "G90", timeout=2.0)
        if not ok:
            log.warning("G90 did not receive 'ok' (continuing)")

        # Send the move command and wait for board acknowledgement
        move_command = "G1 " + " ".join(parts)
        ok, _ = porthandler.write_and_wait(motion_platform, move_command, timeout=30.0)
        if not ok:
            log.warning(f"Move command '{move_command}' timed out waiting for 'ok'")

    except (OSError, PermissionError):
        raise  # let caller handle USB disconnect
    except Exception as e:
        log.error(f"Error occurred while sending move to position command: {e}")


# Moves the Motion platform by the specified values.
# Can be called with 1-3 arguements, like move_relative(printer, x=1, y=1)
def move_relative(motion_platform, x=None, y=None, z=None):
    # Construct the move command with the specified distances
    move_command = "G1"

    # Add x-axis movement if provided
    if x is not None:
        move_command += f" X{x}"

    # Add y-axis movement if provided
    if y is not None:
        move_command += f" Y{-float(y)}"

    # Add z-axis movement if provided
    if z is not None:
        move_command += f" Z{z}"

    try:
        # Set relative mode and wait for board acknowledgement
        ok, _ = porthandler.write_and_wait(motion_platform, "G91", timeout=2.0)
        if not ok:
            log.warning("G91 did not receive 'ok' (continuing)")

        # Send the move command and wait for board acknowledgement
        ok, _ = porthandler.write_and_wait(motion_platform, move_command, timeout=30.0)
        if not ok:
            log.warning(f"Relative move '{move_command}' timed out waiting for 'ok'")

    except (OSError, PermissionError):
        raise  # let caller handle USB disconnect
    except Exception as e:
        log.error(f"Error occurred while sending move command: {e}")
