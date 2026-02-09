import serial
import serial.tools.list_ports
import logging
import time
import threading
import globals

log = logging.getLogger(__name__)

# Global serial device variables
motion_platform = None
motion_platform_waiting_for_done = False
motion_lock = threading.RLock()

def connect_to_serial_device(device_name, identification_command, expected_response, vid, pid):
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        logging.error(f"No COM ports found while looking for {device_name}.")
        return None

    matching_ports = [p for p in ports if (p.vid == vid and p.pid == pid)]
    if not matching_ports:
        logging.warning(f"No ports found with VID=0x{vid:04x} PID=0x{pid:04x} for {device_name}.")
        return None

    logging.info(f"Found {len(matching_ports)} candidate port(s) for {device_name} by VID/PID.")

    for port_info in matching_ports:
        ser = None
        try:
            logging.info(f"Trying {port_info.device} for {device_name}.")
            # Short timeouts so we never block the Flask thread
            ser = serial.Serial(port_info.device, baudrate=115200, timeout=0.2, write_timeout=0.5)
            time.sleep(0.2)  # tiny settle (STM32 CDC can spew right after open)

            if identification_command:
                try:
                    ser.reset_input_buffer()
                    ser.reset_output_buffer()
                except Exception:
                    pass

                # Send M115 and read with a tiny deadline; don't hang on readline()
                ser.write((identification_command + '\n').encode('ascii', 'ignore'))
                ser.flush()

                ok = False
                deadline = time.monotonic() + 1.0  # ~1s total
                buf_lines = []
                while time.monotonic() < deadline:
                    iw = 0
                    try:
                        iw = ser.in_waiting
                    except Exception:
                        iw = 0
                    if iw:
                        line = ser.readline().decode(errors='ignore').strip()
                        if line:
                            buf_lines.append(line)
                            if expected_response.lower() in line.lower():
                                ok = True
                                break
                    else:
                        time.sleep(0.02)

                logging.info(f"Received response from {port_info.device}: {buf_lines!r}")
                if not ok:
                    logging.warning(f"Unexpected M115 reply on {port_info.device}, expected '{expected_response}'.")
                    ser.close()
                    continue  # try next candidate

            # Success: set final runtime options and disable auto temp reports once
            try:
                ser.timeout = 0.2
                ser.write_timeout = 0.5
                # Optional: ser.inter_byte_timeout = 0.1
                ser.reset_input_buffer()
                ser.reset_output_buffer()
                ser.write(b"M155 S0\n")  # stop auto temperature spam that corrupts reads
                ser.flush()
                time.sleep(0.1)
                # drain immediate reply without blocking
                while True:
                    iw = getattr(ser, "in_waiting", 0) or 0
                    if not iw:
                        break
                    _ = ser.read(iw)
            except Exception as e:
                logging.warning(f"Post-connect init failed (continuing): {e}")

            logging.info(f"Connected to {device_name} on port {port_info.device}")
            return ser

        except Exception as e:
            logging.exception(f"Exception while trying to connect to {device_name} on {port_info.device}: {e}")
            if ser and ser.is_open:
                ser.close()

    logging.error(f"Failed to connect to {device_name}. No matching ports responded correctly.")
    return None


def connect_to_motion_platform():
    """
    Connects to the motion platform using its known identification command and VID/PID.
    """
    motion_platform = globals.motion_platform
    if motion_platform and motion_platform.is_open:
        logging.info("Motion platform is already connected.")
        return motion_platform

    ser = connect_to_serial_device(
        device_name="Motion Platform",
        identification_command="M115",
        expected_response="FIRMWARE_NAME:Marlin",
        vid=0x0483, pid=0x5740
    )
    if ser is None:
        logging.error("Motion platform device not found or did not respond correctly.")
        return None

    # Assign global (already initialized above)
    globals.motion_platform = ser

    # Turn off lights on connect
    try:
        write(ser, "M106 P0 S255")  # Dome light off
        write(ser, "M106 P1 S0")    # Bar light off
    except Exception as e:
        logging.warning(f"Failed to turn off lights on connect: {e}")

    return ser

def disconnect_serial_device(device_name):
    """
    Forcefully disconnects the specified serial device ('motion_platform').
    """
    logging.info(f"Attempting to disconnect {device_name}")

    try:
        if device_name.lower() == 'motion_platform' and globals.motion_platform is not None:
            if globals.motion_platform.is_open:
                globals.motion_platform.close()  # Close port safely
            globals.motion_platform = None  # Remove reference
            logging.info("Motion platform disconnected successfully.")
        else:
            logging.warning(f"{device_name} was not connected.")
    except Exception as e:
        logging.error(f"Error while disconnecting {device_name}: {e}")
        


def drain_serial_buffer(ser, timeout=0.15):
    """
    Drain (discard) all pending bytes from the serial input buffer.
    Returns the drained bytes for diagnostic purposes.
    """
    drained = bytearray()
    try:
        if hasattr(ser, "reset_input_buffer"):
            # Read what's there first, then reset
            iw = getattr(ser, "in_waiting", 0) or 0
            if iw:
                drained += ser.read(iw)
            ser.reset_input_buffer()
        else:
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                iw = getattr(ser, "in_waiting", 0) or 0
                if not iw:
                    break
                chunk = ser.read(min(iw, 512))
                if chunk:
                    drained += chunk
    except Exception as e:
        logging.debug(f"drain_serial_buffer error (ignored): {e}")
    return bytes(drained)


def write_and_wait(ser, command, timeout=5.0, expect=b"ok"):
    """
    Send a G-code command to the BTT SKR Mini E3 board and wait for the
    expected response (default: 'ok').

    This is the **safe** way to send commands: it acquires the motion_lock,
    drains stale data, sends the command, and blocks until the board
    acknowledges or the timeout expires.

    Args:
        ser:      serial.Serial object (motion_platform)
        command:  G-code string, e.g. "M400" or "G1 X10"
        timeout:  max seconds to wait for the expected response
        expect:   bytes to look for in the reply (case-insensitive)

    Returns:
        (True, reply_bytes)  – if expected response was seen
        (False, reply_bytes) – if timeout expired without seeing it

    Raises:
        OSError / PermissionError – if USB is disconnected (caller should handle)
    """
    if not ser or not getattr(ser, 'is_open', False):
        raise OSError("Serial port not open")

    cmd_bytes = (command.strip() + "\n").encode("ascii", "ignore")
    expect_lower = expect.lower()

    with motion_lock:
        # 1. Drain any stale data so we don't confuse old replies with new ones
        drain_serial_buffer(ser, timeout=0.1)

        # 2. Send the command
        ser.write(cmd_bytes)
        ser.flush()

        # 3. Wait for expected response
        buf = bytearray()
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            iw = getattr(ser, "in_waiting", 0) or 0
            if iw:
                chunk = ser.read(min(iw, 512))
                if chunk:
                    buf += chunk
                    if expect_lower in buf.lower():
                        return True, bytes(buf)
            else:
                time.sleep(0.01)

    logging.debug(f"write_and_wait timeout for '{command.strip()}': got {buf[:128]!r}")
    return False, bytes(buf)


def write_and_wait_motion(ser, command, timeout=30.0):
    """
    Convenience wrapper for motion commands (G0/G1/G28/M400) that need
    longer timeouts. The BTT board sends 'echo:busy: processing' while
    working on long moves and 'ok' when done.

    Returns True if motion completed, False on timeout.
    """
    ok, _ = write_and_wait(ser, command, timeout=timeout, expect=b"ok")
    return ok


def write(device, data):
    """
    Low-level write: sends command without waiting for a reply.
    Prefer write_and_wait() for any command where confirmation matters.
    """
    if isinstance(data, tuple):
        command = "{},{}".format(*data)
    else:
        command = data + "\n"

    if isinstance(device, serial.Serial):
        with motion_lock:
            device.write(command.encode())
            device.flush()
    else:
        print("Invalid device type")
