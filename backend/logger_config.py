# logger_config.py
import logging
from logging.handlers import RotatingFileHandler
import sys
import os

def get_base_path():
    # In frozen mode, sys.executable gives the path of the exe;
    # os.path.dirname(sys.executable) returns its folder.
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(__file__)

DEFAULT_SETTINGS_PATH = os.path.join(get_base_path(), 'settings.json')


def setup_logger():
    logger = logging.getLogger()  # root logger
    logger.setLevel(logging.DEBUG)

    # Formatter for all handlers
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    # Console handler (DEBUG and above)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (WARNING and above)
    log_path = os.path.join(get_base_path(), 'zoltek_backend.log')
    file_handler = RotatingFileHandler(log_path, maxBytes=10485760, backupCount=1)
    file_handler.setLevel(logging.WARNING)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


class CameraError(Exception):
    """Exception raised for camera-related errors."""
    pass

class SerialError(Exception):
    """Exception raised for serial device errors."""
    pass
