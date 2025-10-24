import serial
import serial.tools.list_ports
import logging
import time
import threading
import globals

# Global serial device variables
turntable = None
barcode_scanner = None
turntable_waiting_for_done = False

def connect_to_serial_device(device_name, identification_command, expected_response, vid, pid):
    """
    Attempt to connect to a serial device by scanning for a matching VID/PID.
    Optionally, send an identification command and compare the response.
    Returns:
        serial.Serial instance if successful, or None.
    """
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        logging.error(f"No COM ports found while looking for {device_name}.")
        return None

    # Filter ports by matching VID/PID.
    matching_ports = [
        port for port in ports
        if (port.vid == vid and port.pid == pid)
    ]
    if not matching_ports:
        logging.warning(f"No ports found with VID=0x{vid:04x} PID=0x{pid:04x} for {device_name}.")
        return None

    logging.info(f"Found {len(matching_ports)} candidate port(s) for {device_name} by VID/PID.")
    
    for port_info in matching_ports:
        serial_port = None
        try:
            logging.info(f"Trying {port_info.device} for {device_name}.")
            serial_port = serial.Serial(port_info.device, baudrate=115200, timeout=1)

            if identification_command:
                # Send the identification command.
                serial_port.write((identification_command + '\n').encode())
                response = serial_port.readline().decode(errors='ignore').strip()
                logging.info(f"Received response from {port_info.device}: '{response}'")

                if response != expected_response:
                    logging.warning(f"Unexpected response '{response}' on {port_info.device}")
                    serial_port.close()
                    continue  # Try next candidate
            # If no identification command is required, we assume the connection is valid.
            logging.info(f"Connected to {device_name} on port {port_info.device}")
            return serial_port

        except Exception as e:
            logging.exception(
                f"Exception while trying to connect to {device_name} on {port_info.device}: {e}"
            )
            if serial_port and serial_port.is_open:
                serial_port.close()

    logging.error(f"Failed to connect to {device_name}. No matching ports responded correctly.")
    return None


def connect_to_turntable():
    """
    Connects to the turntable using its known identification command and VID/PID.
    """
    global turntable
    if turntable and turntable.is_open:
        logging.info("Turntable is already connected.")
        return turntable

    identification_command = "IDN?"
    expected_response = "TTBL"
    # For the turntable, VID/PID are hard-coded.
    turntable = connect_to_serial_device(
        device_name="Turntable",
        identification_command=identification_command,
        expected_response=expected_response,
        vid=0x2E8A,  # Example VID for turntable
        pid=0x0003   # Example PID for turntable
    )
    if turntable is None:
        logging.error("Turntable device not found or did not respond correctly.")
        return None
    return turntable


def connect_to_barcode_scanner():
    """
    Connects to the barcode scanner (QD2100) using its VID/PID.
    If successful, assigns the scanner to `barcode_scanner` and starts the listener thread.
    """
    global barcode_scanner

    # Check if already connected
    if barcode_scanner and barcode_scanner.is_open:
        logging.info("Barcode scanner is already connected.")
        return barcode_scanner

    logging.info("Attempting to connect to barcode scanner...")
    barcode_scanner = connect_to_serial_device(
        device_name="BarcodeScanner",
        identification_command="",
        expected_response="",
        vid=0x05F9,
        pid=0x4204
    )

    if barcode_scanner and barcode_scanner.is_open:
        logging.info("Barcode scanner connected successfully.")
        # Start barcode listener only if not already running
        if not any(t.name == "BarcodeListener" for t in threading.enumerate()):
            threading.Thread(target=barcode_scanner_listener, name="BarcodeListener", daemon=True).start()
        return barcode_scanner

    logging.error("Failed to connect to barcode scanner.")
    return None

def barcode_scanner_listener():
    """Continuously read barcode scanner data and update globals.latest_barcode."""
    global barcode_scanner  # Ensure we're modifying the global barcode_scanner variable

    while True:
        if barcode_scanner and barcode_scanner.is_open:
            try:
                # Read barcode data
                line = barcode_scanner.readline().decode(errors='ignore').strip()
                if line:
                    globals.latest_barcode = line
                    logging.info(f"Barcode scanned: {line}")

            except serial.SerialException as e:
                logging.error(f"SerialException reading barcode scanner: {e}")
                barcode_scanner.close()
                barcode_scanner = None
                logging.warning("Barcode scanner disconnected! Attempting reconnection...")

                # Attempt reconnection loop
                while barcode_scanner is None:
                    try:
                        barcode_scanner = connect_to_barcode_scanner()
                        if barcode_scanner and barcode_scanner.is_open:
                            logging.info("Barcode scanner reconnected successfully.")
                            break  # Exit the loop on success
                    except Exception as e:
                        logging.error(f"Reconnection attempt failed: {e}")
                        time.sleep(2)  # Wait before retrying

        time.sleep(0.5)  # Poll every 500ms

def disconnect_serial_device(device_name):
    """
    Forcefully disconnects the specified serial device ('turntable' or 'barcode').
    """
    global turntable, barcode_scanner
    logging.info(f"Attempting to disconnect {device_name}")

    try:
        if device_name.lower() == 'turntable' and turntable is not None:
            if turntable.is_open:
                turntable.close()  # Close port safely
            turntable = None  # Remove reference
            logging.info("Turntable disconnected successfully.")
        elif device_name.lower() in ['barcode', 'barcodescanner'] and barcode_scanner is not None:
            if barcode_scanner.is_open:
                barcode_scanner.close()  # Close port safely
            barcode_scanner = None  # Remove reference
            logging.info("Barcode scanner disconnected successfully.")
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

def write_barcode_scanner(data):
    """
    Sends data to the barcode scanner if needed.
    Typically, barcode scanners stream data automatically when a barcode is scanned.
    This function is provided if you later need to send configuration commands.
    """
    global barcode_scanner
    if barcode_scanner is None or not barcode_scanner.is_open:
        raise Exception("Barcode scanner is not connected or available.")

    try:
        if isinstance(data, str):
            formatted_data = f"{data}\n"
        else:
            formatted_data = str(data) + "\n"
        barcode_scanner.write(formatted_data.encode())
        barcode_scanner.flush()
        logging.info(f"Data sent to barcode scanner: {formatted_data.strip()}")
    except Exception as e:
        logging.error(f"Error writing to barcode scanner: {str(e)}")
        raise