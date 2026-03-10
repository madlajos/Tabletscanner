import json
import os

class ErrorCode:
    CAMERA_DISCONNECTED = "E1111"
    MOTIONPLATFORM_DISCONNECTED = "E1201"
    EXPOSURE_UNDER = "E2302"
    EXPOSURE_OVER = "E2303"
    GENERIC = "GENERIC"
    CAMERA_STATUS_UNKNOWN = "E9999"

    # Pipeline / Recipe error codes
    PIPELINE_VALIDATION_FAILED = "E3001"
    PIPELINE_TYPE_INCOMPATIBLE = "E3002"
    PIPELINE_PARAM_OUT_OF_RANGE = "E3003"
    PIPELINE_SOURCE_NOT_FOUND = "E3004"
    PIPELINE_STEP_EXECUTION_FAILED = "E3005"
    RECIPE_IO_ERROR = "E3006"
    RECIPE_NOT_FOUND = "E3007"
    RECIPE_INVALID_FORMAT = "E3008"

def load_error_messages():
    file_path = os.path.join(os.path.dirname(__file__), "error_messages.json")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

ERROR_MESSAGES = load_error_messages()