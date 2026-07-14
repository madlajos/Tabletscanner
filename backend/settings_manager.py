import json
import os
import threading
import logging
import sys
import copy
import math
import shutil
import tempfile
import re

# In case multiple threads read/write settings at once:
_settings_lock = threading.RLock()

# This variable will hold our settings after we load from JSON
_cached_settings = {}

def get_base_path():
    # In frozen mode, sys.executable gives the path of the exe;
    # os.path.dirname(sys.executable) returns its folder.
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(__file__)

DEFAULT_SETTINGS_PATH = os.path.join(get_base_path(), 'settings.json')
SETTINGS_SCHEMA_VERSION = 2
UV_LAMP_CHANNELS = ('uv255', 'uv310', 'uv365')
LIGHT_CHANNELS = (*UV_LAMP_CHANNELS, 'vis')
FILTER_POSITIONS = (1, 2, 3, 4, 5, 6)
ADVANCED_LAMP_CHANNELS = (*UV_LAMP_CHANNELS, 'vis')
LAMP_SETTING_FIELDS = (
    'dim_percent',
    'full_percent',
    'dim_timeout_seconds',
    'full_timeout_seconds',
)


def _finite_number(value, default):
    """Return a finite numeric value without accepting booleans as numbers."""
    if isinstance(value, bool):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def validate_lamp_settings(payload):
    """Validate and normalize the editable UV lamp configuration payload."""
    if not isinstance(payload, dict) or not isinstance(payload.get('channels'), dict):
        raise ValueError('Lamp settings must contain a channels object.')

    incoming_channels = payload['channels']
    if set(incoming_channels) != set(UV_LAMP_CHANNELS):
        raise ValueError('Lamp settings must include exactly uv255, uv310, and uv365.')

    normalized_channels = {}
    for channel in UV_LAMP_CHANNELS:
        values = incoming_channels[channel]
        if not isinstance(values, dict) or set(values) != set(LAMP_SETTING_FIELDS):
            raise ValueError(f'{channel} must include all brightness and timeout values.')

        normalized = {}
        for field in ('dim_percent', 'full_percent'):
            value = _finite_number(values[field], None)
            if value is None or not 10 <= value <= 100:
                raise ValueError(f'{channel}.{field} must be between 10 and 100.')
            normalized[field] = value
        for field in ('dim_timeout_seconds', 'full_timeout_seconds'):
            value = _finite_number(values[field], None)
            if value is None or value <= 0:
                raise ValueError(f'{channel}.{field} must be a positive number of seconds.')
            normalized[field] = value
        normalized_channels[channel] = normalized

    return {'channels': normalized_channels}


def validate_capture_plan(value):
    """Validate the persisted ordered wavelength/filter measurement plan."""
    if not isinstance(value, list) or not value:
        raise ValueError('Capture plan must contain at least one row.')

    normalized_rows = []
    for row in value:
        if not isinstance(row, dict):
            raise ValueError('Each capture plan row must be an object.')
        wavelength = row.get('wavelength')
        filter_position = row.get('filter_position')
        if wavelength not in LIGHT_CHANNELS:
            raise ValueError('Capture plan contains an unknown wavelength.')
        if isinstance(filter_position, bool) or not isinstance(filter_position, int) or filter_position not in FILTER_POSITIONS:
            raise ValueError('Capture plan filter position must be an integer from 1 to 6.')
        normalized_rows.append({'wavelength': wavelength, 'filter_position': filter_position})
    return normalized_rows


def validate_lamp_output_selectors(payload):
    """Validate the configurable M106 output selectors, e.g. P0 or P3."""
    if not isinstance(payload, dict) or not isinstance(payload.get('output_selectors'), dict):
        raise ValueError('Advanced lamp settings must contain an output_selectors object.')

    selectors = payload['output_selectors']
    if set(selectors) != set(ADVANCED_LAMP_CHANNELS):
        raise ValueError('Output selectors must include uv255, uv310, uv365, and vis.')

    normalized = {}
    for channel in ADVANCED_LAMP_CHANNELS:
        value = selectors[channel]
        if not isinstance(value, str) or not re.fullmatch(r'P\d+', value.strip(), flags=re.IGNORECASE):
            raise ValueError(f'{channel} selector must have the format P followed by a non-negative number.')
        normalized[channel] = value.strip().upper()
    return {'output_selectors': normalized}


def migrate_settings(settings):
    """Migrate a settings dictionary to schema v2 without mutating the input.

    Schema v2 makes camera exposure and gamma global and adds placeholders for the
    four-channel capture plan and UV lamp configuration. UV configuration remains
    empty until an operator supplies safe values through the future settings UI.
    """
    if not isinstance(settings, dict):
        raise ValueError('Settings root must be a JSON object.')

    migrated = copy.deepcopy(settings)
    current_version = migrated.get('settings_schema_version', 1)
    if current_version == SETTINGS_SCHEMA_VERSION:
        other_settings = migrated.get('other_settings')
        if isinstance(other_settings, dict) and 'settings_preset_name' in other_settings:
            other_settings.pop('settings_preset_name')
            return migrated, True
        return migrated, False
    if not isinstance(current_version, int) or current_version < 1 or current_version > SETTINGS_SCHEMA_VERSION:
        raise ValueError(f'Unsupported settings schema version: {current_version!r}')

    # Old dome settings are the intended source for the new VIS/global camera
    # behaviour. Bar settings are only a fallback for installations without dome.
    camera_source = migrated.get('camera_params')
    if not isinstance(camera_source, dict):
        camera_source = migrated.get('camera_params_dome')
    if not isinstance(camera_source, dict):
        camera_source = migrated.get('camera_params_bar')
    if not isinstance(camera_source, dict):
        camera_source = {}

    migrated['camera_params'] = {
        'ExposureTime': _finite_number(camera_source.get('ExposureTime'), 100000.0),
        'Gamma': _finite_number(camera_source.get('Gamma'), 1.0),
    }
    migrated.pop('camera_params_dome', None)
    migrated.pop('camera_params_bar', None)

    other_settings = migrated.get('other_settings')
    if isinstance(other_settings, dict):
        other_settings.pop('settings_preset_name', None)

    auto_measurement = migrated.get('auto_measurement_settings')
    if not isinstance(auto_measurement, dict):
        auto_measurement = {}
    auto_measurement.setdefault('capture_plan', [
        {'wavelength': 'vis', 'filter_position': 1}
    ])
    migrated['auto_measurement_settings'] = auto_measurement

    lamp_settings = migrated.get('lamp_settings')
    if not isinstance(lamp_settings, dict):
        lamp_settings = {}
    channels = lamp_settings.get('channels')
    if not isinstance(channels, dict):
        channels = {}
    lamp_settings['channels'] = channels
    migrated['lamp_settings'] = lamp_settings
    migrated['settings_schema_version'] = SETTINGS_SCHEMA_VERSION
    return migrated, True


def _backup_path(settings_path):
    return f'{settings_path}.v1.bak'


def _write_settings_atomic(settings_path, settings):
    """Atomically replace settings_path with UTF-8 JSON in its existing directory."""
    directory = os.path.dirname(os.path.abspath(settings_path))
    fd, temp_path = tempfile.mkstemp(prefix='.settings-', suffix='.tmp', dir=directory, text=True)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as file:
            json.dump(settings, file, indent=4, ensure_ascii=False)
            file.write('\n')
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp_path, settings_path)
    except Exception:
        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            pass
        raise

def load_settings(settings_path=DEFAULT_SETTINGS_PATH):
    global _cached_settings
    with _settings_lock:
        try:
            with open(settings_path, 'r', encoding='utf-8') as file:
                loaded_settings = json.load(file)
            _cached_settings, migrated = migrate_settings(loaded_settings)
            if migrated:
                backup_path = _backup_path(settings_path)
                if loaded_settings.get('settings_schema_version', 1) < SETTINGS_SCHEMA_VERSION and not os.path.exists(backup_path):
                    shutil.copy2(settings_path, backup_path)
                _write_settings_atomic(settings_path, _cached_settings)
                logging.info('Settings migrated to schema v%s.', SETTINGS_SCHEMA_VERSION)
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
            _write_settings_atomic(settings_path, _cached_settings)
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
