import json
import os
import threading
import logging
import sys

# In case multiple threads read/write settings at once:
_settings_lock = threading.Lock()

# This variable will hold our settings after we load from JSON
_cached_settings = {}

def get_base_path():
    # In frozen mode, sys.executable gives the path of the exe;
    # os.path.dirname(sys.executable) returns its folder.
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(__file__)

DEFAULT_SETTINGS_PATH = os.path.join(get_base_path(), 'settings.json')

def load_settings(settings_path=DEFAULT_SETTINGS_PATH):
    global _cached_settings
    with _settings_lock:
        try:
            with open(settings_path, 'r') as file:
                _cached_settings = json.load(file)
            logging.info(f"Settings loaded from {settings_path}")
        except FileNotFoundError:
            error_message = f"Settings file not found at {settings_path}"
            logging.error(error_message)
            # Return an empty dict as fallback
            _cached_settings = {}
        except json.JSONDecodeError:
            error_message = "Invalid JSON format in settings file."
            logging.error(error_message)
            _cached_settings = {}
        except Exception as e:
            error_message = f"Failed to load settings: {e}"
            logging.error(error_message)
            _cached_settings = {}
    return _cached_settings

def save_settings(settings_path=DEFAULT_SETTINGS_PATH):
    global _cached_settings
    with _settings_lock:
        try:
            with open(settings_path, 'w') as file:
                json.dump(_cached_settings, file, indent=4)
            logging.info("Settings saved successfully.")
            return True
        except Exception as e:
            error_message = f"Failed to save settings: {e}"
            logging.error(error_message)
            return False

def get_settings() -> dict:
    """Returns the current in-memory settings dict."""
    return _cached_settings

def set_settings(new_settings: dict):
    """Replaces the entire settings dictionary in memory."""
    global _cached_settings
    with _settings_lock:
        _cached_settings = new_settings
