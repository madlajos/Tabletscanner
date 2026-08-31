"""Autofocus-referenced Z correction for filter/illumination combinations."""

import math

import globals
import porthandler
from settings_manager import EMPTY_FILTER_KEY, LIGHT_CHANNELS


class HeightOffsetCommandError(RuntimeError):
    """The requested automatic Z correction could not be completed safely."""


def invalidate_reference() -> None:
    """Disable automatic offsets until manual autofocus succeeds again."""
    globals.autofocus_reference_z = None
    globals.autofocus_applied_offset_mm = 0.0


def record_reference(z_position) -> float:
    """Record the focused blue-filter/VIS Z coordinate as the zero point."""
    try:
        reference_z = float(z_position)
    except (TypeError, ValueError) as error:
        raise ValueError('Autofocus did not produce a valid Z reference.') from error
    if not math.isfinite(reference_z):
        raise ValueError('Autofocus did not produce a finite Z reference.')
    z_min, z_max = globals.motion_limits['z']
    if not z_min <= reference_z <= z_max:
        raise ValueError('Autofocus Z reference is outside the configured travel range.')
    globals.autofocus_reference_z = reference_z
    globals.autofocus_applied_offset_mm = 0.0
    return reference_z


def record_combination_reference(z_position, configured_offset_mm) -> float:
    """Rebase a focused filter/light combination onto the height-matrix zero.

    Matrix offsets are calibrated relative to blue-filter/VIS. When autofocus
    uses a different combination, its focused Z is therefore ``zero + offset``.
    Store the derived zero so subsequent filter/light changes continue to use
    the existing matrix correctly.
    """
    try:
        focused_z = float(z_position)
        applied_offset = float(configured_offset_mm)
    except (TypeError, ValueError) as error:
        raise ValueError('Autofocus did not produce a valid Z reference.') from error
    if not math.isfinite(focused_z) or not math.isfinite(applied_offset):
        raise ValueError('Autofocus did not produce a finite Z reference.')

    reference_z = record_reference(focused_z - applied_offset)
    globals.autofocus_applied_offset_mm = applied_offset
    return reference_z


def status() -> dict:
    reference_z = getattr(globals, 'autofocus_reference_z', None)
    return {
        'available': isinstance(reference_z, (int, float)) and math.isfinite(reference_z),
        'reference_z': reference_z,
        'applied_offset_mm': getattr(globals, 'autofocus_applied_offset_mm', 0.0),
    }


def configured_offset(filter_settings: dict, filter_position: int, channel: str) -> float:
    """Resolve one validated matrix value for the active physical combination."""
    if channel not in LIGHT_CHANNELS:
        raise HeightOffsetCommandError('No known illumination channel is active.')
    if isinstance(filter_position, bool) or not isinstance(filter_position, int):
        raise HeightOffsetCommandError('The filter position is not known.')

    slots = filter_settings.get('slots') if isinstance(filter_settings, dict) else None
    matrix = filter_settings.get('height_offsets_mm') if isinstance(filter_settings, dict) else None
    if not isinstance(slots, list) or filter_position < 1 or filter_position > len(slots):
        raise HeightOffsetCommandError('The filter position is not configured.')
    if not isinstance(matrix, dict):
        raise HeightOffsetCommandError('The height-offset matrix is not configured.')

    filter_key = slots[filter_position - 1] or EMPTY_FILTER_KEY
    row = matrix.get(filter_key)
    if not isinstance(row, dict) or channel not in row:
        raise HeightOffsetCommandError('The selected filter/light height offset is missing.')
    try:
        offset = float(row[channel])
    except (TypeError, ValueError) as error:
        raise HeightOffsetCommandError('The selected height offset is invalid.') from error
    if not math.isfinite(offset):
        raise HeightOffsetCommandError('The selected height offset is invalid.')
    return offset


def apply_active_combination(motion_platform, filter_settings: dict, channel: str | None) -> dict:
    """Move Z to reference + selected offset, or report why offsets are inactive.

    Callers own the higher-level ``globals.motion_busy`` operation flag. Serial
    commands are still serialized by ``porthandler.write_and_wait``.
    """
    reference_z = getattr(globals, 'autofocus_reference_z', None)
    if not isinstance(reference_z, (int, float)) or not math.isfinite(reference_z):
        return {'applied': False, 'reason': 'autofocus_required'}
    if channel is None:
        return {'applied': False, 'reason': 'no_active_light'}

    offset = configured_offset(
        filter_settings,
        getattr(globals, 'filter_revolver_position', None),
        channel,
    )
    target_z = float(reference_z) + offset
    z_min, z_max = globals.motion_limits['z']
    if not z_min <= target_z <= z_max:
        raise HeightOffsetCommandError(
            f'The autofocus reference plus offset would move Z outside {z_min:g}-{z_max:g} mm.'
        )

    current_z = getattr(globals, 'last_toolhead_pos', {}).get('z')
    if isinstance(current_z, (int, float)) and math.isclose(
        float(current_z), target_z, abs_tol=1e-6
    ):
        globals.autofocus_applied_offset_mm = offset
        return {'applied': True, 'offset_mm': offset, 'target_z': target_z, 'moved': False}

    for command, timeout in (
        ('G90', 2.0),
        (f'G1 Z{target_z:.4f}', 30.0),
        ('M400', 30.0),
    ):
        acknowledged, reply = porthandler.write_and_wait(
            motion_platform,
            command,
            timeout=timeout,
        )
        if not acknowledged:
            raise HeightOffsetCommandError(
                f'Automatic height correction was not acknowledged for {command!r}; '
                f'controller reply: {reply[:256]!r}'
            )

    globals.last_toolhead_pos['z'] = target_z
    globals.autofocus_applied_offset_mm = offset
    return {'applied': True, 'offset_mm': offset, 'target_z': target_z, 'moved': True}
