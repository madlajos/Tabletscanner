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
SETTINGS_SCHEMA_VERSION = 9
UV_LAMP_CHANNELS = ('uv255', 'uv310', 'uv365')
LIGHT_CHANNELS = (*UV_LAMP_CHANNELS, 'vis')
AUTOFOCUS_BRIGHTNESS_MODES = ('dimmed', 'full')
FILTER_POSITIONS = (1, 2, 3, 4, 5, 6)
FILTER_COLOR_PATTERN = re.compile(r'^#[0-9a-fA-F]{6}$')
FILTER_ID_PATTERN = re.compile(r'^[A-Za-z0-9_-]{1,100}$')
MAX_CONFIGURED_FILTERS = 100
EMPTY_FILTER_KEY = 'empty'
HEIGHT_OFFSET_REFERENCE_FILTER_NAMES = frozenset(('kék', 'blue'))
VIS_INCOMPATIBLE_FILTER_NAMES = frozenset(('255nm', '265nm', '365nm'))
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
DEFAULT_CAPTURE_EXPOSURE_TIME = 100000.0
DEFAULT_CAPTURE_GAIN = 0.0
DEFAULT_CAPTURE_GAMMA = 1.0
DEFAULT_FIRST_TABLET_X_MM = 2.9
DEFAULT_FIRST_TABLET_Y_MM = 0.0
DEFAULT_FIRST_TABLET_Z_MM = 20.0
DEFAULT_TABLET_SPACING_MM = 18.3
DEFAULT_CAMERA_IMAGE_WIDTH = 4000
DEFAULT_CAMERA_IMAGE_HEIGHT = 4000
TRAY_GRID_SIZE = 10
X_TRAVEL_MAX_MM = 175.0
Y_TRAVEL_MAX_MM = 165.0
Z_TRAVEL_MAX_MM = 40.0
# Allow sub-millimetre calibration rounding at the tray edge. Motion commands
# remain clamped to the exact per-axis limits.
TRAY_EDGE_CALIBRATION_TOLERANCE_MM = 0.5


class TrayGeometryError(ValueError):
    """Raised when the configured 10 x 10 tray exceeds the motion envelope."""


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
    """Validate the persisted ordered wavelength/filter/camera measurement plan."""
    if not isinstance(value, list) or not value:
        raise ValueError('Capture plan must contain at least one row.')

    normalized_rows = []
    for row in value:
        if not isinstance(row, dict):
            raise ValueError('Each capture plan row must be an object.')
        wavelength = row.get('wavelength')
        filter_position = row.get('filter_position')
        exposure_time = _finite_number(row.get('exposure_time'), None)
        gain = _finite_number(row.get('gain'), None)
        gamma = _finite_number(row.get('gamma'), None)
        if wavelength not in LIGHT_CHANNELS:
            raise ValueError('Capture plan contains an unknown wavelength.')
        if isinstance(filter_position, bool) or not isinstance(filter_position, int) or filter_position not in FILTER_POSITIONS:
            raise ValueError('Capture plan filter position must be an integer from 1 to 6.')
        if exposure_time is None or exposure_time <= 0:
            raise ValueError('Capture plan exposure time must be a positive finite number.')
        if gain is None or gain < 0:
            raise ValueError('Capture plan gain must be a non-negative finite number.')
        if gamma is None or gamma <= 0:
            raise ValueError('Capture plan gamma must be a positive finite number.')
        normalized_rows.append({
            'wavelength': wavelength,
            'filter_position': filter_position,
            'exposure_time': exposure_time,
            'gain': gain,
            'gamma': gamma,
        })
    # Row 1 is the fixed autofocus reference. Normalize older/direct API
    # payloads so the persisted and runtime plan cannot bypass the UI lock.
    normalized_rows[0]['wavelength'] = 'vis'
    normalized_rows[0]['filter_position'] = 1
    return normalized_rows


def default_autofocus_settings():
    """Return the legacy-safe manual/automatic autofocus hardware selection."""
    return {
        'channel': 'vis',
        'brightness': 'full',
        'filter_position': 1,
    }


def validate_autofocus_settings(payload, filter_settings=None):
    """Validate the configured autofocus illumination and physical filter slot."""
    required_fields = {'channel', 'brightness', 'filter_position'}
    if not isinstance(payload, dict) or set(payload) != required_fields:
        raise ValueError(
            'Autofocus settings must contain channel, brightness, and filter_position.'
        )

    channel = payload['channel']
    brightness = payload['brightness']
    filter_position = payload['filter_position']
    if channel not in LIGHT_CHANNELS:
        raise ValueError('Autofocus settings contain an unknown light channel.')
    if brightness not in AUTOFOCUS_BRIGHTNESS_MODES:
        raise ValueError('Autofocus brightness must be dimmed or full.')
    if channel == 'vis' and brightness != 'full':
        raise ValueError('VIS autofocus illumination is available only at full brightness.')
    if (
        isinstance(filter_position, bool)
        or not isinstance(filter_position, int)
        or filter_position not in FILTER_POSITIONS
    ):
        raise ValueError('Autofocus filter position must be an integer from 1 to 6.')

    if filter_settings is not None:
        slots = filter_settings.get('slots') if isinstance(filter_settings, dict) else None
        if not isinstance(slots, list) or len(slots) != len(FILTER_POSITIONS):
            raise ValueError('Filter settings must contain six slots.')
        # Slot 1 is the fixed empty reference. Every other selectable position
        # must currently contain a configured filter on the physical wheel.
        if filter_position != 1 and slots[filter_position - 1] is None:
            raise ValueError('The selected autofocus filter position is empty.')

    return {
        'channel': channel,
        'brightness': brightness,
        'filter_position': filter_position,
    }


def default_filter_settings():
    """Return the empty six-position filter-revolver configuration."""
    return {
        'filters': [],
        'slots': [None] * len(FILTER_POSITIONS),
        'height_offsets_mm': {
            EMPTY_FILTER_KEY: {channel: 0.0 for channel in LIGHT_CHANNELS},
        },
    }


def validate_filter_settings(
    payload,
    max_height_offset_up_mm=100.0,
    max_height_offset_down_mm=-100.0,
):
    """Validate filter definitions and the six selected revolver slots."""
    if not isinstance(payload, dict) or set(payload) != {'filters', 'slots', 'height_offsets_mm'}:
        raise ValueError('Filter settings must contain filters, slots, and height_offsets_mm.')
    filters = payload['filters']
    slots = payload['slots']
    height_offsets = payload['height_offsets_mm']
    if not isinstance(filters, list) or not isinstance(slots, list) or len(slots) != len(FILTER_POSITIONS):
        raise ValueError('Filter settings must contain six slots.')
    if not isinstance(height_offsets, dict):
        raise ValueError('Filter height offsets must be an object.')
    if len(filters) > MAX_CONFIGURED_FILTERS:
        raise ValueError(f'At most {MAX_CONFIGURED_FILTERS} filters can be configured.')

    normalized_filters = []
    ids = set()
    normalized_names = set()
    for definition in filters:
        if not isinstance(definition, dict) or set(definition) != {
            'id', 'name', 'wavelength_range', 'color'
        }:
            raise ValueError('Each filter must include id, name, wavelength_range, and color.')
        filter_id = definition['id'].strip() if isinstance(definition['id'], str) else ''
        name = definition['name'].strip() if isinstance(definition['name'], str) else ''
        wavelength_range = definition['wavelength_range'].strip() if isinstance(definition['wavelength_range'], str) else ''
        color = definition['color'].strip() if isinstance(definition['color'], str) else ''
        normalized_name = name.casefold()
        if (
            not FILTER_ID_PATTERN.fullmatch(filter_id)
            or filter_id == EMPTY_FILTER_KEY
            or filter_id in ids
        ):
            raise ValueError('Filter IDs must be unique identifiers.')
        if not name or len(name) > 100 or normalized_name in normalized_names:
            raise ValueError('Filter names must be unique and no longer than 100 characters.')
        if not wavelength_range or len(wavelength_range) > 100:
            raise ValueError('Filter wavelength range is required and must be no longer than 100 characters.')
        if not FILTER_COLOR_PATTERN.fullmatch(color):
            raise ValueError('Filter color must be a six-digit hexadecimal color.')
        ids.add(filter_id)
        normalized_names.add(normalized_name)
        normalized_filters.append({
            'id': filter_id,
            'name': name,
            'wavelength_range': wavelength_range,
            'color': color.lower(),
        })

    normalized_slots = []
    for slot in slots:
        if slot is not None and (not isinstance(slot, str) or slot not in ids):
            raise ValueError('Each selected slot must reference a configured filter.')
        normalized_slots.append(slot)
    # Slot 1 is the physical no-filter position used by autofocus.
    normalized_slots[0] = None

    expected_offset_keys = {EMPTY_FILTER_KEY, *ids}
    if set(height_offsets) != expected_offset_keys:
        raise ValueError('Height offsets must contain the empty position and every configured filter exactly once.')
    normalized_height_offsets = {}
    vis_incompatible_filter_ids = {
        definition['id']
        for definition in normalized_filters
        if re.sub(r'[\s_-]+', '', definition['name']).casefold()
        in VIS_INCOMPATIBLE_FILTER_NAMES
    }
    reference_filter_ids = {
        definition['id']
        for definition in normalized_filters
        if definition['name'].casefold() in HEIGHT_OFFSET_REFERENCE_FILTER_NAMES
    }
    for filter_key in expected_offset_keys:
        row = height_offsets[filter_key]
        if not isinstance(row, dict) or set(row) != set(LIGHT_CHANNELS):
            raise ValueError('Each height-offset row must include uv255, uv310, uv365, and vis.')
        normalized_row = {}
        for channel in LIGHT_CHANNELS:
            value = (
                0.0
                if channel == 'vis' and (
                    filter_key in vis_incompatible_filter_ids
                    or filter_key in reference_filter_ids
                )
                else _finite_number(row[channel], None)
            )
            if value is None or not max_height_offset_down_mm <= value <= max_height_offset_up_mm:
                raise ValueError(
                    'Each height offset must be between '
                    f'{max_height_offset_down_mm:g} and {max_height_offset_up_mm:g} mm.'
                )
            normalized_row[channel] = value
        normalized_height_offsets[filter_key] = normalized_row

    # VIS with the configured blue filter is the height-matrix zero. The
    # physical empty-filter/VIS combination remains independently calibratable.
    return {
        'filters': normalized_filters,
        'slots': normalized_slots,
        'height_offsets_mm': normalized_height_offsets,
    }


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
    """Validate advanced motion, tray geometry, and filter-height preferences."""
    required_fields = {
        'use_virtual_com_port',
        'max_height_offset_up_mm',
        'max_height_offset_down_mm',
        'first_tablet_x_mm',
        'first_tablet_y_mm',
        'first_tablet_z_mm',
        'tablet_spacing_mm',
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
    first_x = _finite_number(payload['first_tablet_x_mm'], None)
    first_y = _finite_number(payload['first_tablet_y_mm'], None)
    first_z = _finite_number(payload['first_tablet_z_mm'], None)
    spacing = _finite_number(payload['tablet_spacing_mm'], None)
    if first_x is None or not 0 <= first_x <= X_TRAVEL_MAX_MM:
        raise ValueError('First tablet X must be between 0 and 175 mm.')
    if first_y is None or not 0 <= first_y <= Y_TRAVEL_MAX_MM:
        raise ValueError('First tablet Y must be between 0 and 165 mm.')
    if first_z is None or not 0 <= first_z <= Z_TRAVEL_MAX_MM:
        raise ValueError('First tablet Z must be between 0 and 40 mm.')
    if spacing is None or spacing <= 0:
        raise ValueError('Tablet spacing must be a positive number.')
    last_x = first_x + (TRAY_GRID_SIZE - 1) * spacing
    last_y = first_y + (TRAY_GRID_SIZE - 1) * spacing
    tray_x_limit = X_TRAVEL_MAX_MM + TRAY_EDGE_CALIBRATION_TOLERANCE_MM
    tray_y_limit = Y_TRAVEL_MAX_MM + TRAY_EDGE_CALIBRATION_TOLERANCE_MM
    if last_x > tray_x_limit or last_y > tray_y_limit:
        raise TrayGeometryError(
            'The 10 x 10 tray coordinates must stay within the X/Y travel limits.'
        )
    return {
        'use_virtual_com_port': enabled,
        'max_height_offset_up_mm': max_up,
        'max_height_offset_down_mm': max_down,
        'first_tablet_x_mm': first_x,
        'first_tablet_y_mm': first_y,
        'first_tablet_z_mm': first_z,
        'tablet_spacing_mm': spacing,
    }


def migrate_settings(settings):
    """Migrate settings without mutating the input.

    Schema v2 made camera settings global and introduced the four-channel lamp
    model. Schema v3 installs the selector order verified on the physical
    Octopus controller so stale packaged settings cannot rotate wavelengths.
    Schema v8 expands the former per-filter height value into a
    filter-by-wavelength matrix. The current calibration zero is blue-filter/VIS.
    Schema v9 adds a validated autofocus light/brightness/filter selection.
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
        if not isinstance(migrated.get('autofocus_settings'), dict):
            migrated['autofocus_settings'] = default_autofocus_settings()
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
            'Gain': _finite_number(camera_source.get('Gain'), DEFAULT_CAPTURE_GAIN),
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
        camera_params = migrated.get('camera_params', {})
        auto_measurement.setdefault('capture_plan', [
            {
                'wavelength': 'vis',
                'filter_position': 1,
                'exposure_time': _finite_number(
                    camera_params.get('ExposureTime'), DEFAULT_CAPTURE_EXPOSURE_TIME
                ),
                'gain': _finite_number(camera_params.get('Gain'), DEFAULT_CAPTURE_GAIN),
                'gamma': _finite_number(camera_params.get('Gamma'), DEFAULT_CAPTURE_GAMMA),
            }
        ])
        migrated['auto_measurement_settings'] = auto_measurement

    if current_version < 4:
        camera_params = migrated.get('camera_params')
        if not isinstance(camera_params, dict):
            camera_params = {}
        default_exposure = _finite_number(
            camera_params.get('ExposureTime'), DEFAULT_CAPTURE_EXPOSURE_TIME
        )
        default_gamma = _finite_number(camera_params.get('Gamma'), DEFAULT_CAPTURE_GAMMA)
        auto_measurement = migrated.get('auto_measurement_settings')
        if not isinstance(auto_measurement, dict):
            auto_measurement = {}
        capture_plan = auto_measurement.get('capture_plan')
        if not isinstance(capture_plan, list) or not capture_plan:
            capture_plan = [{'wavelength': 'vis', 'filter_position': 1}]
        enriched_plan = []
        for row in capture_plan:
            if not isinstance(row, dict):
                enriched_plan.append(row)
                continue
            enriched_row = copy.deepcopy(row)
            enriched_row.setdefault('exposure_time', default_exposure)
            enriched_row.setdefault('gamma', default_gamma)
            enriched_plan.append(enriched_row)
        auto_measurement['capture_plan'] = enriched_plan
        migrated['auto_measurement_settings'] = auto_measurement

    if current_version < 5:
        advanced_settings = migrated.get('advanced_settings')
        if not isinstance(advanced_settings, dict):
            advanced_settings = {}
        auto_measurement = migrated.get('auto_measurement_settings')
        if not isinstance(auto_measurement, dict):
            auto_measurement = {}
        advanced_settings.setdefault(
            'first_tablet_x_mm',
            _finite_number(auto_measurement.pop('first_tablet_x', None), DEFAULT_FIRST_TABLET_X_MM),
        )
        advanced_settings.setdefault(
            'first_tablet_y_mm',
            _finite_number(auto_measurement.pop('first_tablet_y', None), DEFAULT_FIRST_TABLET_Y_MM),
        )
        advanced_settings.setdefault(
            'first_tablet_z_mm',
            _finite_number(auto_measurement.pop('first_tablet_z', None), DEFAULT_FIRST_TABLET_Z_MM),
        )
        advanced_settings.setdefault(
            'tablet_spacing_mm',
            _finite_number(auto_measurement.pop('tablet_spacing', None), DEFAULT_TABLET_SPACING_MM),
        )
        migrated['advanced_settings'] = advanced_settings
        migrated['auto_measurement_settings'] = auto_measurement

    if current_version < 6:
        camera_params = migrated.get('camera_params')
        if not isinstance(camera_params, dict):
            camera_params = {}
        migrated['camera_image_settings'] = {
            'override_enabled': False,
            'width': int(_finite_number(camera_params.pop('Width', None), DEFAULT_CAMERA_IMAGE_WIDTH)),
            'height': int(_finite_number(camera_params.pop('Height', None), DEFAULT_CAMERA_IMAGE_HEIGHT)),
            'offset_x': int(_finite_number(camera_params.pop('OffsetX', None), 0)),
            'offset_y': int(_finite_number(camera_params.pop('OffsetY', None), 0)),
        }
        migrated['camera_params'] = camera_params

    if current_version < 7:
        camera_params = migrated.get('camera_params')
        if not isinstance(camera_params, dict):
            camera_params = {}
        default_gain = _finite_number(camera_params.get('Gain'), DEFAULT_CAPTURE_GAIN)
        camera_params['Gain'] = default_gain
        migrated['camera_params'] = camera_params

        auto_measurement = migrated.get('auto_measurement_settings')
        if not isinstance(auto_measurement, dict):
            auto_measurement = {}
        capture_plan = auto_measurement.get('capture_plan')
        if not isinstance(capture_plan, list) or not capture_plan:
            capture_plan = [{
                'wavelength': 'vis',
                'filter_position': 1,
                'exposure_time': _finite_number(
                    camera_params.get('ExposureTime'), DEFAULT_CAPTURE_EXPOSURE_TIME
                ),
                'gamma': _finite_number(
                    camera_params.get('Gamma'), DEFAULT_CAPTURE_GAMMA
                ),
            }]
        enriched_plan = []
        for row in capture_plan:
            if not isinstance(row, dict):
                enriched_plan.append(row)
                continue
            enriched_row = copy.deepcopy(row)
            enriched_row.setdefault('gain', default_gain)
            enriched_plan.append(enriched_row)
        auto_measurement['capture_plan'] = enriched_plan
        migrated['auto_measurement_settings'] = auto_measurement

    if current_version < 8:
        legacy_filter_settings = migrated.get('filter_settings')
        if not isinstance(legacy_filter_settings, dict):
            legacy_filter_settings = {'filters': [], 'slots': [None] * len(FILTER_POSITIONS)}
        legacy_filters = legacy_filter_settings.get('filters')
        if not isinstance(legacy_filters, list):
            legacy_filters = []

        migrated_filters = []
        height_offsets = {
            EMPTY_FILTER_KEY: {channel: 0.0 for channel in LIGHT_CHANNELS},
        }
        for definition in legacy_filters:
            if not isinstance(definition, dict):
                migrated_filters.append(definition)
                continue
            migrated_definition = copy.deepcopy(definition)
            legacy_offset = _finite_number(
                migrated_definition.pop('height_offset_mm', None),
                0.0,
            )
            migrated_filters.append(migrated_definition)
            filter_id = migrated_definition.get('id')
            if isinstance(filter_id, str) and filter_id:
                # Preserve the old calibration on every wavelength until the
                # operator refines the new per-combination values.
                height_offsets[filter_id] = {
                    channel: legacy_offset for channel in LIGHT_CHANNELS
                }

        migrated['filter_settings'] = {
            'filters': migrated_filters,
            'slots': copy.deepcopy(
                legacy_filter_settings.get('slots', [None] * len(FILTER_POSITIONS))
            ),
            'height_offsets_mm': height_offsets,
        }

    if current_version < 9:
        migrated['autofocus_settings'] = default_autofocus_settings()

    lamp_settings = migrated.get('lamp_settings')
    if not isinstance(lamp_settings, dict):
        lamp_settings = {}
    channels = lamp_settings.get('channels')
    if not isinstance(channels, dict):
        channels = {}
    lamp_settings['channels'] = channels
    if current_version < 3 or lamp_settings.get('output_selectors') != OCTOPUS_LIGHT_OUTPUT_SELECTORS:
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
        previous_autofocus_settings = _cached_settings.get('autofocus_settings', missing)
        autofocus_settings = (
            default_autofocus_settings()
            if previous_autofocus_settings is missing
            else copy.deepcopy(previous_autofocus_settings)
        )
        try:
            _cached_settings['autofocus_settings'] = validate_autofocus_settings(
                autofocus_settings,
                filter_settings,
            )
        except ValueError:
            # Removing the selected wheel entry must not leave autofocus aimed
            # at an empty physical slot. Fall back to the established safe
            # empty-filter/VIS selection in the same atomic settings write.
            _cached_settings['autofocus_settings'] = default_autofocus_settings()
        try:
            _write_settings_atomic(settings_path, _cached_settings)
            logging.info("Filter settings saved successfully.")
            return True
        except Exception as error:
            if previous_settings is missing:
                _cached_settings.pop('filter_settings', None)
            else:
                _cached_settings['filter_settings'] = previous_settings
            if previous_autofocus_settings is missing:
                _cached_settings.pop('autofocus_settings', None)
            else:
                _cached_settings['autofocus_settings'] = previous_autofocus_settings
            logging.error("Failed to save filter settings: %s", error)
            return False


def update_autofocus_settings(autofocus_settings, settings_path=DEFAULT_SETTINGS_PATH):
    """Persist validated autofocus settings while holding the settings lock."""
    global _cached_settings
    with _settings_lock:
        missing = object()
        previous_settings = _cached_settings.get('autofocus_settings', missing)
        _cached_settings['autofocus_settings'] = copy.deepcopy(autofocus_settings)
        try:
            _write_settings_atomic(settings_path, _cached_settings)
            logging.info('Autofocus settings saved successfully.')
            return True
        except Exception as error:
            if previous_settings is missing:
                _cached_settings.pop('autofocus_settings', None)
            else:
                _cached_settings['autofocus_settings'] = previous_settings
            logging.error('Failed to save autofocus settings: %s', error)
            return False


def get_settings() -> dict:
    """Returns the current in-memory settings dict."""
    return _cached_settings

def set_settings(new_settings: dict):
    """Replaces the entire settings dictionary in memory."""
    global _cached_settings
    with _settings_lock:
        _cached_settings = new_settings
