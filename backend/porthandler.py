import serial
import serial.tools.list_ports
import logging
import time
import threading
import globals

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
        
        
        

def write_turntable(command, timeout=10, expect_response=True):
    global turntable, turntable_waiting_for_done

    if turntable is None or not turntable.is_open:
        raise Exception("Turntable is not connected or available.")

    formatted_command = f"{command}\n"
    turntable.reset_input_buffer()
    turntable.reset_output_buffer()
    turntable.write(formatted_command.encode())
    turntable.flush()
    logging.info(f"Command sent to turntable: {formatted_command.strip()}")

    # If we do not expect a DONE response, return immediately.
    if not expect_response:
        return True

    turntable_waiting_for_done = True
    start_time = time.time()
    received_data = ""

    while time.time() - start_time < timeout:
        if turntable.in_waiting > 0:
            received_chunk = turntable.read(turntable.in_waiting).decode(errors='ignore')
            received_data += received_chunk
            logging.info(f"Received from turntable: {received_chunk.strip()}")

            if "DONE" in received_data:
                logging.info("Turntable movement completed successfully.")
                turntable_waiting_for_done = False
                return True
        time.sleep(0.05)

    logging.warning("Timeout waiting for 'DONE' signal from turntable.")
    turntable_waiting_for_done = False
    return False

def query_turntable(command, timeout=5):
    """
    Sends a query command to the turntable and returns its reply as a string.
    """
    global turntable
    if turntable is None or not turntable.is_open:
        raise Exception("Turntable is not connected or available.")
    
    formatted_command = f"{command}\n"
    turntable.reset_input_buffer()
    turntable.reset_output_buffer()
    turntable.write(formatted_command.encode())
    turntable.flush()
    logging.info(f"Query sent to turntable: {formatted_command.strip()}")

    start_time = time.time()
    received_data = ""
    while time.time() - start_time < timeout:
        if turntable.in_waiting > 0:
            received_chunk = turntable.read(turntable.in_waiting).decode(errors='ignore')
            received_data += received_chunk
            logging.info(f"Received from turntable: {received_chunk.strip()}")
            # Assume the response ends with a newline.
            if "\n" in received_data:
                # Return the first line from the response.
                return received_data.strip().split("\n")[0]
        time.sleep(0.05)

    logging.warning("Timeout waiting for turntable query response.")
    return None


def write(device, data):
    if isinstance(data, tuple):
        command = "{},{}".format(*data)
    else:
        command = data + "\n"

    if isinstance(device, serial.Serial):
        device.write(command.encode())
    else:
        print("Invalid device type")
