import threading
from flask import Flask

app = Flask(__name__)

camera = None
stream_running = False
stream_thread = None
grab_lock = threading.Lock()
# Suppress preview retrieval only while an operation acquires its own frames.
preview_grab_suppression_lock = threading.Lock()
preview_grab_suppression_count = 0
latest_preview_multipart_frame = None
# Frames acquired by autofocus/measurement are published here while those
# operations own the camera.  The MJPEG generator can display them without
# taking an additional frame from the camera queue.
latest_owned_preview_image = None
latest_owned_preview_sequence = 0
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
autofocus_reference_z = None
autofocus_applied_offset_mm = 0.0
height_reference_source = None
height_reference_offset_mm = 0.0
height_offset_application = None
autofocus_abort = False  # Flag to abort autofocus if measurement is stopped
last_autofocus_contour = None  # Contour from autofocus or manual_bgr for background subtraction
color_values = None  # Reference color values from calc_color (used by autofocus before_auto check)

motion_limits = {
    "x": (0.0, 175.0),
    "y": (0.0, 165.0),
    "z": (0.0, 40.0),
}

