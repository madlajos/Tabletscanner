import serial
import serial.tools.list_ports
import logging
import time
import threading
import re
import globals
from virtual_octopus import VirtualOctopusSerial
from door_interlock import DoorInterlockState

log = logging.getLogger(__name__)

# Global serial device variables
motion_platform = None
motion_platform_waiting_for_done = False
motion_lock = threading.RLock()
EXPECTED_FIRMWARE_MARKER = "TS-LIGHT-V3-P0F0-P1F1-P2HE0-P3HE1-LOCK-RHOME-DOORS1"
door_interlock = DoorInterlockState()

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
                acknowledged, reply = write_and_wait(ser, identification_command, timeout=1.0)
                identity = reply.decode('ascii', errors='ignore')
                ok = acknowledged and re.search(
                    rf'(?<![\w-]){re.escape(expected_response)}(?![\w-])', identity
                )
                if not ok:
                    logging.warning(f"Unexpected M115 reply on {port_info.device}, expected '{expected_response}'.")
                    ser.close()
                    continue  # try next candidate

            # Success: set final runtime options and disable auto temp reports once
            try:
                ser.timeout = 0.2
                ser.write_timeout = 0.5
                # Optional: ser.inter_byte_timeout = 0.1
                ser.reset_output_buffer()
                write_and_wait(ser, 'M155 S0', timeout=0.5)
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


def connect_to_motion_platform(use_virtual=False):
    """
    Connects to the motion platform using its known identification command and VID/PID.
    """
    global motion_platform
    current_platform = globals.motion_platform
    if current_platform and current_platform.is_open:
        if bool(getattr(current_platform, 'is_virtual', False)) == bool(use_virtual):
            motion_platform = current_platform
            logging.info("Motion platform is already connected.")
            return current_platform
        disconnect_serial_device('motion_platform')

    if use_virtual:
        virtual_platform = VirtualOctopusSerial()
        refresh_door_status(virtual_platform)
        with motion_lock:
            globals.motion_platform = virtual_platform
            motion_platform = virtual_platform
        logging.info("Connected to virtual BTT Octopus motion platform.")
        return virtual_platform

    ser = connect_to_serial_device(
        device_name="Motion Platform",
        identification_command="M115",
        expected_response=EXPECTED_FIRMWARE_MARKER,
        vid=0x0483, pid=0x5740
    )
    if ser is None:
        logging.error("Motion platform device not found or did not respond correctly.")
        return None

    # Turn off every M106 lamp output on connect. The selector order is
    # P0=FAN0/VIS, P1=FAN1/365, P2=HE0/255, P3=HE1/310.
    off_failures = []
    with motion_lock:
        for selector in range(4):
            try:
                acknowledged, _ = write_and_wait(ser, f"M106 P{selector} S0", timeout=2.0)
                if not acknowledged:
                    off_failures.append(f'P{selector}: no acknowledgement')
            except Exception as error:
                off_failures.append(f'P{selector}: {error}')
        if off_failures:
            logging.error('Refusing motion connection because lamp all-off failed: %s', '; '.join(off_failures))
            try:
                ser.close()
            finally:
                globals.motion_platform = None
                motion_platform = None
            return None
        try:
            doors_known = refresh_door_status(ser)['known']
        except Exception:
            log.exception('Initial door status query failed')
            doors_known = False
        if not doors_known:
            logging.error('Refusing motion connection: firmware door status is unavailable.')
            ser.close()
            globals.motion_platform = None
            motion_platform = None
            return None
        # Publish the port only after every physical output acknowledged OFF.
        globals.motion_platform = ser
        motion_platform = ser

    return ser

def disconnect_serial_device(device_name):
    """
    Forcefully disconnects the specified serial device ('motion_platform').
    """
    global motion_platform
    logging.info(f"Attempting to disconnect {device_name}")

    try:
        normalized_name = device_name.lower().replace('-', '_')
        platform = globals.motion_platform or motion_platform
        if normalized_name in ('motionplatform', 'motion_platform', 'motion') and platform is not None:
            # Prevent an activation from interleaving after an OFF but before
            # close. write_and_wait safely reacquires this RLock.
            with motion_lock:
                try:
                    if platform.is_open:
                        for selector in range(4):
                            try:
                                acknowledged, _ = write_and_wait(
                                    platform, f"M106 P{selector} S0", timeout=2.0
                                )
                                if not acknowledged:
                                    logging.warning('P%s did not acknowledge OFF before disconnect.', selector)
                            except Exception as error:
                                logging.warning('Failed to turn off P%s before disconnect: %s', selector, error)
                finally:
                    try:
                        if getattr(platform, 'is_open', False):
                            platform.close()
                    finally:
                        globals.motion_platform = None
                        motion_platform = None
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
    with motion_lock:
        drained = bytearray()
        try:
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                iw = getattr(ser, "in_waiting", 0) or 0
                if not iw:
                    break
                chunk = ser.read(min(iw, 512))
                if chunk:
                    drained += chunk
                    door_interlock.feed(ser, chunk)
        except Exception as e:
            logging.debug(f"drain_serial_buffer error (ignored): {e}")
        return bytes(drained)


def write_and_wait(ser, command, timeout=5.0, expect=b"ok", failure_markers=(), drain=True):
    """
    Send a G-code command to the BTT Octopus board and wait for the
    expected response (default: 'ok').

    This is the **safe** way to send commands: it acquires the motion_lock,
    drains stale data, sends the command, and blocks until the board
    acknowledges or the timeout expires.

    Args:
        ser:      serial.Serial object (motion_platform)
        command:  G-code string, e.g. "M400" or "G1 X10"
        timeout:  max seconds to wait for the expected response
        expect:   bytes to look for in the reply (case-insensitive)
        failure_markers: reply fragments that end the wait as a failed command

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
    failure_markers_lower = tuple(marker.lower() for marker in failure_markers)

    # The checkpoint raises to its owning phase; it never waits with this lock.
    from measurement_safety import before_serial_command
    before_serial_command(command)
    with motion_lock:
        # 1. Drain any stale data so we don't confuse old replies with new ones
        if drain:
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
                    door_interlock.feed(ser, chunk)
                    reply_lower = buf.lower()
                    if any(marker in reply_lower for marker in failure_markers_lower):
                        logging.warning(
                            "Controller rejected '%s': %r",
                            command.strip(),
                            bytes(buf[:256]),
                        )
                        return False, bytes(buf)
                    # Consume a complete acknowledgement line, including door
                    # rejection/status text that precedes it.
                    complete_lines = reply_lower.split(b'\n')[:-1]
                    completed = (
                        any(re.fullmatch(rb'ok(?:\s[^\r\n]*)?\r?', line) for line in complete_lines)
                        if expect_lower == b'ok'
                        else any(expect_lower in line + b'\n' for line in complete_lines)
                    )
                    if completed:
                        return b'ts_door_open' not in reply_lower, bytes(buf)
            else:
                time.sleep(0.01)

    logging.debug(f"write_and_wait timeout for '{command.strip()}': got {buf[:128]!r}")
    return False, bytes(buf)


def refresh_door_status(ser, timeout=0.5):
    """Request authoritative state; a missing report invalidates cached safety state."""
    with motion_lock:
        # Drain first so a stale unsolicited report cannot satisfy this query.
        drain_serial_buffer(ser)
        before = door_interlock.revision()
        try:
            acknowledged, _ = write_and_wait(ser, 'M119', timeout=timeout, drain=False)
        except Exception:
            door_interlock.invalidate(ser)
            raise
        if not acknowledged or door_interlock.revision() == before:
            door_interlock.invalidate(ser)
        return door_interlock.snapshot(ser)


def poll_door_interlock():
    """Background polling never queues behind or interleaves with a motion owner."""
    if not motion_lock.acquire(blocking=False):
        return
    try:
        ser = motion_platform or globals.motion_platform
        if ser and getattr(ser, 'is_open', False) and not globals.motion_busy:
            refresh_door_status(ser, timeout=0.2)
    finally:
        motion_lock.release()


def probe_motion_controller(ser, timeout=0.3):
    """Send M105 and consume its complete response line under ``motion_lock``.

    Marlin commonly replies with ``ok T:...\n``. Waiting only until the two
    acknowledgement bytes arrive can leave the temperature suffix in the
    buffer, where it corrupts the next command response. A newline is therefore
    the completion marker; the completed line must also contain ``ok``.
    """
    completed, reply = write_and_wait(
        ser,
        'M105',
        timeout=timeout,
    )
    return completed and b'ok' in reply.lower(), reply


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

    if device is not None and getattr(device, 'is_open', False) and callable(getattr(device, 'write', None)):
        with motion_lock:
            device.write(command.encode())
            device.flush()
    else:
        raise OSError("Serial port not open")
