"""Build the JSON stored in captured JPEG ImageDescription EXIF metadata."""

from __future__ import annotations

from typing import Any, Mapping


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


def build_capture_metadata(
    *,
    settings: Mapping[str, Any] | None,
    position: Mapping[str, Any] | None,
    wavelength: str | None,
    filter_position: int | None,
    camera_values: Mapping[str, Any] | None,
    requested_metadata: Mapping[str, Any] | None = None,
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
        "x": position.get("x"),
        "y": position.get("y"),
        "z": position.get("z"),
        "wavelength": wavelength,
        "filter_position": filter_position,
        "filter_wavelength": (
            selected_filter.get("wavelength_range") if selected_filter else None
        ),
        "filter_name": selected_filter.get("name") if selected_filter else None,
        "exposure_time": live_or_requested("exposure_time", "exposure_time"),
        "gain": live_or_requested("gain", "gain"),
        "gamma": live_or_requested("gamma", "gamma"),
    }
