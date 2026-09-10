"""Build and serialize captured JPEG ImageDescription EXIF metadata."""

from __future__ import annotations

from typing import Any, Mapping
import json
import math

from settings_manager import DEFAULT_FIRST_TABLET_X_MM, DEFAULT_FIRST_TABLET_Y_MM, DEFAULT_TABLET_SPACING_MM


def tray_position_label(settings, position):
    """Label actual XY coordinates only when they match a configured tray center."""
    geometry = settings.get('advanced_settings', {})
    try:
        x, y = float(position['x']), float(position['y'])
        x0 = float(geometry.get('first_tablet_x_mm', DEFAULT_FIRST_TABLET_X_MM))
        y0 = float(geometry.get('first_tablet_y_mm', DEFAULT_FIRST_TABLET_Y_MM))
        spacing = float(geometry.get('tablet_spacing_mm', DEFAULT_TABLET_SPACING_MM))
        if not all(math.isfinite(value) for value in (x, y, x0, y0, spacing)) or spacing <= 0:
            return None
        col, row = round((x - x0) / spacing), round((y - y0) / spacing)
        if not (0 <= col < 10 and 0 <= row < 10):
            return None
        if abs(x - (x0 + col * spacing)) > .01 or abs(y - (y0 + row * spacing)) > .01:
            return None
        return f'{chr(65 + col)}{row + 1}'
    except (KeyError, TypeError, ValueError, AttributeError):
        return None


def _filter_for_position(
    filter_settings: Mapping[str, Any] | None,
    filter_position: int | None,
) -> Mapping[str, Any] | None:
    """Return the configured filter definition assigned to a one-based slot."""
    if (
        not isinstance(filter_settings, Mapping)
        or isinstance(filter_position, bool)
        or not isinstance(filter_position, int)
    ):
        return None

    slots = filter_settings.get("slots")
    filters = filter_settings.get("filters")
    slot_index = filter_position - 1
    if (
        not isinstance(slots, list)
        or not isinstance(filters, list)
        or slot_index < 0
        or slot_index >= len(slots)
    ):
        return None

    filter_id = slots[slot_index]
    return next(
        (
            definition
            for definition in filters
            if isinstance(definition, Mapping) and definition.get("id") == filter_id
        ),
        None,
    )


def _rounded_number(value, decimals):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return value
    rounded = round(float(value), decimals)
    return int(rounded) if rounded.is_integer() else rounded


def build_capture_metadata(
    *,
    settings: Mapping[str, Any] | None,
    position: Mapping[str, Any] | None,
    wavelength: str | None,
    filter_position: int | None,
    camera_values: Mapping[str, Any] | None,
    requested_metadata: Mapping[str, Any] | None = None,
    errors: list[str] | None = None,
) -> dict[str, Any]:
    """Build capture metadata from authoritative backend state.

    ``requested_metadata`` is retained as a compatibility fallback for older
    manual-save clients. Runtime settings, live coordinates, and live camera
    values take precedence when available.
    """
    settings = settings if isinstance(settings, Mapping) else {}
    position = position if isinstance(position, Mapping) else {}
    camera_values = camera_values if isinstance(camera_values, Mapping) else {}
    requested_metadata = (
        requested_metadata if isinstance(requested_metadata, Mapping) else {}
    )
    other = settings.get("other_settings")
    other = other if isinstance(other, Mapping) else {}

    configured_profile = other.get("camera_settings_file")
    if configured_profile is None:
        configured_profile = requested_metadata.get("camera_settings_file")

    selected_filter = _filter_for_position(
        settings.get("filter_settings"), filter_position
    )

    def configured_or_requested(key: str) -> Any:
        value = other.get(key)
        return value if value is not None else requested_metadata.get(key)

    def live_or_requested(live_key: str, request_key: str) -> Any:
        value = camera_values.get(live_key)
        return value if value is not None else requested_metadata.get(request_key)

    return {
        "objective": configured_or_requested("objective"),
        "spacer_rings": configured_or_requested("spacer_rings"),
        # Keep the legacy key for existing metadata consumers.
        "camera_settings_file": configured_profile,
        "camera_profile": configured_profile,
        "x": _rounded_number(position.get("x"), 4),
        "y": _rounded_number(position.get("y"), 4),
        "z": _rounded_number(position.get("z"), 4),
        "wavelength": wavelength,
        "Errors": list(errors or []),
        "tray_position": tray_position_label(settings, position),
        "filter_position": filter_position,
        "filter_wavelength": (
            selected_filter.get("wavelength_range") if selected_filter else None
        ),
        "filter_name": selected_filter.get("name") if selected_filter else None,
        "exposure_time": _rounded_number(live_or_requested("exposure_time", "exposure_time"), 0),
        "gain": _rounded_number(live_or_requested("gain", "gain"), 4),
        "gamma": _rounded_number(live_or_requested("gamma", "gamma"), 4),
    }


def serialize_capture_metadata(metadata: dict) -> str:
    """Return JSON that survives Pillow's ASCII EXIF ImageDescription field.

    Pillow replaces non-ASCII characters in a string-valued ImageDescription
    with question marks. JSON Unicode escapes keep the EXIF payload ASCII while
    ``json.loads`` restores the original operator-facing text.
    """
    return json.dumps(metadata, ensure_ascii=True)
