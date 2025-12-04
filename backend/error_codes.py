import json
import os

class ErrorCode:
    CAMERA_DISCONNECTED = "E1111"
    MOTIONPLATFORM_DISCONNECTED = "E1201"
    GENERIC = "GENERIC"
    CAMERA_STATUS_UNKNOWN = "E9999"

def load_error_messages():
    file_path = os.path.join(os.path.dirname(__file__), "error_messages.json")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

ERROR_MESSAGES = load_error_messages()