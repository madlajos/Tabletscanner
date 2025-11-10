import porthandler
import time

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

def get_toolhead_position(motion_platform, retries=3, delay=1):
    if motion_platform is not None:
        for attempt in range(retries):
            try:
                # Clear the input buffer
                motion_platform.reset_input_buffer()
                
                # Send the M114 command to get the current position
                porthandler.write(motion_platform, "M114")
                response = ""
                while True:
                    line = motion_platform.readline().decode().strip()
                    if line == "ok":
                        break
                    response += line + "\n"
                
                if response:
                    # Parse the response to extract X, Y, and Z positions
                    position = parse_position(response)
                    return position
                else:
                    print("No response from motion platform.")
                    if attempt < retries - 1:
                        time.sleep(delay)
            except Exception as e:
                print(f"An error occurred while getting toolhead position from motion platform (attempt {attempt + 1}): {e}")
                if attempt < retries - 1:
                    time.sleep(delay)
        print("Failed to get toolhead position position after retries.")
        return None
    else:
        print("Invalid device type or motion platform is not connected")
    return None

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