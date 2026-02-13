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
last_autofocus_contour = None  # Contour from autofocus or manual_bgr for background subtraction

motion_limits = {
    "x": (0.0, 175.0),
    "y": (0.0, 175.0),
    "z": (0.0, 30.0),
}

# Lamp tracking for 5-minute auto-off
lamp_dome_on_time = None  # Timestamp when dome light was turned on (None if off)
lamp_bar_on_time = None   # Timestamp when bar light was turned on (None if off)
lamp_auto_turned_off = False  # Flag to signal frontend that lamps were auto-turned off
lamp_timeout_thread = None  # Background thread for monitoring lamp timeouts