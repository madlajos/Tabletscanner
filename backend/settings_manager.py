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
SETTINGS_SCHEMA_VERSION = 3
UV_LAMP_CHANNELS = ('uv255', 'uv310', 'uv365')
LIGHT_CHANNELS = (*UV_LAMP_CHANNELS, 'vis')
FILTER_POSITIONS = (1, 2, 3, 4, 5, 6)
FILTER_COLOR_PATTERN = re.compile(r'^#[0-9a-fA-F]{6}$')
FILTER_ID_PATTERN = re.compile(r'^[A-Za-z0-9_-]{1,100}$')
MAX_CONFIGURED_FILTERS = 100
DEFAULT_MAX_HEIGHT_OFFSET_UP_MM = 5.0
DEFAULT_MAX_HEIGHT_OFFSET_DOWN_MM = -5.0
ADVANCED_LAMP_CHANNELS = (*UV_LAMP_CHANNELS, 'vis')
OCTOPUS_LIGHT_OUTPUT_SELECTORS = {
    'uv255': 'P2',
    'uv310': 'P3',
    'uv365': 'P1',
    'vis': 'P0',
}
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


def default_filter_settings():
    """Return the empty six-position filter-revolver configuration."""
    return {'filters': [], 'slots': [None] * len(FILTER_POSITIONS)}


def validate_filter_settings(
    payload,
    max_height_offset_up_mm=100.0,
    max_height_offset_down_mm=-100.0,
):
    """Validate filter definitions and the six selected revolver slots."""
    if not isinstance(payload, dict) or set(payload) != {'filters', 'slots'}:
        raise ValueError('Filter settings must contain filters and slots.')
    filters = payload['filters']
    slots = payload['slots']
    if not isinstance(filters, list) or not isinstance(slots, list) or len(slots) != len(FILTER_POSITIONS):
        raise ValueError('Filter settings must contain six slots.')
    if len(filters) > MAX_CONFIGURED_FILTERS:
        raise ValueError(f'At most {MAX_CONFIGURED_FILTERS} filters can be configured.')

    normalized_filters = []
    ids = set()
    normalized_names = set()
    for definition in filters:
        if not isinstance(definition, dict) or set(definition) != {
            'id', 'name', 'wavelength_range', 'height_offset_mm', 'color'
        }:
            raise ValueError('Each filter must include id, name, wavelength_range, height_offset_mm, and color.')
        filter_id = definition['id'].strip() if isinstance(definition['id'], str) else ''
        name = definition['name'].strip() if isinstance(definition['name'], str) else ''
        wavelength_range = definition['wavelength_range'].strip() if isinstance(definition['wavelength_range'], str) else ''
        height_offset_mm = _finite_number(definition['height_offset_mm'], None)
        color = definition['color'].strip() if isinstance(definition['color'], str) else ''
        normalized_name = name.casefold()
        if not FILTER_ID_PATTERN.fullmatch(filter_id) or filter_id in ids:
            raise ValueError('Filter IDs must be unique identifiers.')
        if not name or len(name) > 100 or normalized_name in normalized_names:
            raise ValueError('Filter names must be unique and no longer than 100 characters.')
        if not wavelength_range or len(wavelength_range) > 100:
            raise ValueError('Filter wavelength range is required and must be no longer than 100 characters.')
        if (
            height_offset_mm is None
            or not max_height_offset_down_mm <= height_offset_mm <= max_height_offset_up_mm
        ):
            raise ValueError(
                'Filter height offset must be between '
                f'{max_height_offset_down_mm:g} and {max_height_offset_up_mm:g} mm.'
            )
        if not FILTER_COLOR_PATTERN.fullmatch(color):
            raise ValueError('Filter color must be a six-digit hexadecimal color.')
        ids.add(filter_id)
        normalized_names.add(normalized_name)
        normalized_filters.append({
            'id': filter_id,
            'name': name,
            'wavelength_range': wavelength_range,
            'height_offset_mm': height_offset_mm,
            'color': color.lower(),
        })

    normalized_slots = []
    for slot in slots:
        if slot is not None and (not isinstance(slot, str) or slot not in ids):
            raise ValueError('Each selected slot must reference a configured filter.')
        normalized_slots.append(slot)
    return {'filters': normalized_filters, 'slots': normalized_slots}


def validate_lamp_output_selectors(payload):
    """Validate the mapping fixed by the approved firmware identity marker."""
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
    if normalized != OCTOPUS_LIGHT_OUTPUT_SELECTORS:
        raise ValueError(
            'Output selectors must match the approved firmware mapping: '
            'uv255=P2, uv310=P3, uv365=P1, vis=P0.'
        )
    return {'output_selectors': normalized}


def validate_motion_simulation_settings(payload):
    """Validate the advanced motion and filter-height preferences."""
    required_fields = {
        'use_virtual_com_port',
        'max_height_offset_up_mm',
        'max_height_offset_down_mm',
    }
    if not isinstance(payload, dict) or set(payload) != required_fields:
        raise ValueError('Advanced motion settings contain missing or unknown fields.')
    enabled = payload['use_virtual_com_port']
    if not isinstance(enabled, bool):
        raise ValueError('use_virtual_com_port must be a boolean.')
    max_up = _finite_number(payload['max_height_offset_up_mm'], None)
    max_down = _finite_number(payload['max_height_offset_down_mm'], None)
    if max_up is None or max_up <= 0:
        raise ValueError('Maximum upward height offset must be a positive number.')
    if max_down is None or max_down >= 0:
        raise ValueError('Maximum downward height offset must be a negative number.')
    return {
        'use_virtual_com_port': enabled,
        'max_height_offset_up_mm': max_up,
        'max_height_offset_down_mm': max_down,
    }


def migrate_settings(settings):
    """Migrate settings without mutating the input.

    Schema v2 made camera settings global and introduced the four-channel lamp
    model. Schema v3 installs the selector order verified on the physical
    Octopus controller so stale packaged settings cannot rotate wavelengths.
    """
    if not isinstance(settings, dict):
        raise ValueError('Settings root must be a JSON object.')

    migrated = copy.deepcopy(settings)
    current_version = migrated.get('settings_schema_version', 1)
    if current_version == SETTINGS_SCHEMA_VERSION:
        changed = False
        other_settings = migrated.get('other_settings')
        if isinstance(other_settings, dict) and 'settings_preset_name' in other_settings:
            other_settings.pop('settings_preset_name')
            changed = True

        # Schema-v3 builds all share one mapping-specific firmware marker. A
        # rotated v3 file may have been written by an earlier development build;
        # repair it on every load instead of trusting the version number alone.
        lamp_settings = migrated.get('lamp_settings')
        if not isinstance(lamp_settings, dict):
            lamp_settings = {}
            migrated['lamp_settings'] = lamp_settings
            changed = True
        channels = lamp_settings.get('channels')
        if not isinstance(channels, dict):
            lamp_settings['channels'] = {}
            changed = True
        if lamp_settings.get('output_selectors') != OCTOPUS_LIGHT_OUTPUT_SELECTORS:
            lamp_settings['output_selectors'] = copy.deepcopy(OCTOPUS_LIGHT_OUTPUT_SELECTORS)
            changed = True
        return migrated, changed
    if not isinstance(current_version, int) or current_version < 1 or current_version > SETTINGS_SCHEMA_VERSION:
        raise ValueError(f'Unsupported settings schema version: {current_version!r}')

    if current_version < 2:
        # Old dome settings are the intended source for the new VIS/global camera
        # behaviour. Bar settings are only a fallback when dome is unavailable.
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

    if current_version < 2:
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
    if current_version < 3:
        lamp_settings['output_selectors'] = copy.deepcopy(OCTOPUS_LIGHT_OUTPUT_SELECTORS)
    migrated['lamp_settings'] = lamp_settings
    migrated['settings_schema_version'] = SETTINGS_SCHEMA_VERSION
    return migrated, True


def _backup_path(settings_path, schema_version):
    return f'{settings_path}.v{schema_version}.bak'


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
                source_version = loaded_settings.get('settings_schema_version', 1)
                backup_path = _backup_path(settings_path, source_version)
                if source_version < SETTINGS_SCHEMA_VERSION and not os.path.exists(backup_path):
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


def update_lamp_output_selectors(output_selectors, settings_path=DEFAULT_SETTINGS_PATH):
    """Atomically persist the validated, firmware-locked selector mapping."""
    global _cached_settings
    normalized_selectors = validate_lamp_output_selectors({
        'output_selectors': output_selectors
    })['output_selectors']
    with _settings_lock:
        lamp_settings = _cached_settings.setdefault('lamp_settings', {'channels': {}})
        missing = object()
        previous_selectors = lamp_settings.get('output_selectors', missing)
        lamp_settings['output_selectors'] = copy.deepcopy(normalized_selectors)
        try:
            _write_settings_atomic(settings_path, _cached_settings)
            logging.info('Lamp output selectors saved successfully.')
            return True
        except Exception as error:
            if previous_selectors is missing:
                lamp_settings.pop('output_selectors', None)
            else:
                lamp_settings['output_selectors'] = previous_selectors
            logging.error('Failed to save lamp output selectors: %s', error)
            return False


def update_filter_settings(filter_settings, settings_path=DEFAULT_SETTINGS_PATH):
    """Persist validated filter settings while holding the settings lock."""
    global _cached_settings
    with _settings_lock:
        missing = object()
        previous_settings = _cached_settings.get('filter_settings', missing)
        _cached_settings['filter_settings'] = copy.deepcopy(filter_settings)
        try:
            _write_settings_atomic(settings_path, _cached_settings)
            logging.info("Filter settings saved successfully.")
            return True
        except Exception as error:
            if previous_settings is missing:
                _cached_settings.pop('filter_settings', None)
            else:
                _cached_settings['filter_settings'] = previous_settings
            logging.error("Failed to save filter settings: %s", error)
            return False


def get_settings() -> dict:
    """Returns the current in-memory settings dict."""
    return _cached_settings

def set_settings(new_settings: dict):
    """Replaces the entire settings dictionary in memory."""
    global _cached_settings
    with _settings_lock:
        _cached_settings = new_settings
