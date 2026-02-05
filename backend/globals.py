import threading
from flask import Flask

app = Flask(__name__)

camera = None
stream_running = False
stream_thread = None
grab_lock = threading.Lock()
latest_image = None


motion_platform = None
motion_busy = False
last_toolhead_pos = {"x": None, "y": None, "z": None}
toolhead_x_pos = "?"
toolhead_y_pos = "?"
toolhead_z_pos = "?"
toolhead_homed = False 
last_best_z = None
autofocus_abort = False  # Flag to abort autofocus if measurement is stopped

motion_limits = {
    "x": (0.0, 175.0),
    "y": (0.0, 175.0),
    "z": (0.0, 30.0),
}