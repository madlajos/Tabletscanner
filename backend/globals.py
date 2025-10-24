import threading
from flask import Flask

app = Flask(__name__)

camera = None
stream_running = False
stream_thread = None
grab_lock = threading.Lock()
latest_image = None

#TODO: Modify these to motionplatform
turntable_position = "?"
turntable_homed = False 