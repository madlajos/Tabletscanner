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

# Latest captured images (BGR numpy arrays) keyed by canonical illumination channel.
# ``dome``/``bar`` aliases are handled at the API boundary for temporary legacy clients.
latest_images = {
    'uv255': {'original': None, 'masked': None},
    'uv310': {'original': None, 'masked': None},
    'uv365': {'original': None, 'masked': None},
    'vis': {'original': None, 'masked': None},
}
last_toolhead_pos = {"x": None, "y": None, "z": None}
toolhead_x_pos = "?"
toolhead_y_pos = "?"
toolhead_z_pos = "?"
toolhead_homed = False
homed_axes = set()
filter_revolver_homed = False
filter_revolver_position = None
last_best_z = None
autofocus_abort = False  # Flag to abort autofocus if measurement is stopped
last_autofocus_contour = None  # Contour from autofocus or manual_bgr for background subtraction
color_values = None  # Reference color values from calc_color (used by autofocus before_auto check)

motion_limits = {
    "x": (0.0, 175.0),
    "y": (0.0, 175.0),
    "z": (0.0, 30.0),
}

# Lamp tracking for auto-off (visible: 5 min, UV: 30 s normal / 5 s high-power)
lamp_dome_on_time = None  # Timestamp when visible (dome) light was turned on (None if off)
lamp_uv_dome_on_time = None       # Timestamp when UV light was turned on
lamp_uv_dome_high_power = False   # True if UV was turned on with S255 (5 s timeout)
lamp_auto_turned_off = False  # Flag to signal frontend that lamps were auto-turned off
lamp_timeout_thread = None  # Background thread for monitoring lamp timeouts
