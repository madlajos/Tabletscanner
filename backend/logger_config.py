# logger_config.py
import logging
from logging.handlers import RotatingFileHandler
import sys
import os

def get_base_path():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(__file__)

DEFAULT_SETTINGS_PATH = os.path.join(get_base_path(), 'settings.json')

def setup_logger():
    root = logging.getLogger()  # root logger

    # Make this idempotent
    if getattr(root, "_configured_by_app", False):
        return root

    root.setLevel(logging.DEBUG)

    # Start fresh: remove any pre-existing handlers to avoid duplicates
    for h in list(root.handlers):
        root.removeHandler(h)

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    # Console handler (DEBUG+)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    root.addHandler(ch)

    # File handler (WARNING+)
    log_path = os.path.join(get_base_path(), 'zoltek_backend.log')
    fh = RotatingFileHandler(log_path, maxBytes=10_485_760, backupCount=1, encoding="utf-8")
    fh.setLevel(logging.WARNING)
    fh.setFormatter(formatter)
    root.addHandler(fh)

    # Ensure Flask and Werkzeug DO NOT have their own handlers; let them propagate to root
    for name in ("flask.app", "werkzeug"):
        lg = logging.getLogger(name)
        lg.setLevel(logging.DEBUG)  # keep detail, rely on root handlers for output
        lg.propagate = True
        for h in list(lg.handlers):
            lg.removeHandler(h)

    root._configured_by_app = True
    return root

class CameraError(Exception):
    pass

class SerialError(Exception):
    pass
