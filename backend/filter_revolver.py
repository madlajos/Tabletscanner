"""Acknowledged motion commands for the six-position filter revolver."""

import logging

import porthandler


log = logging.getLogger(__name__)

SLOT_COUNT = 6
STEP_DEGREES = 60
AXIS_NAME = 'A'


def next_position(current_position: int, direction: str) -> int:
    """Return the next one-based slot for an up/counter-clockwise or down/clockwise step."""
    if current_position not in range(1, SLOT_COUNT + 1):
        raise ValueError('Current filter position must be between 1 and 6.')
    if direction not in ('up', 'down'):
        raise ValueError("Direction must be 'up' or 'down'.")
    offset = -1 if direction == 'up' else 1
    return ((current_position - 1 + offset) % SLOT_COUNT) + 1


def shortest_path(current_position: int, target_position: int) -> tuple[str, int]:
    """Return the direction and number of 60-degree steps to a target slot."""
    if current_position not in range(1, SLOT_COUNT + 1):
        raise ValueError('Current filter position must be between 1 and 6.')
    if target_position not in range(1, SLOT_COUNT + 1):
        raise ValueError('Target filter position must be between 1 and 6.')
    down_steps = (target_position - current_position) % SLOT_COUNT
    up_steps = (current_position - target_position) % SLOT_COUNT
    return ('down', down_steps) if down_steps <= up_steps else ('up', up_steps)


def rotate_one_slot(motion_platform, current_position: int, direction: str) -> int:
    """Rotate exactly one slot and return the new position after all commands are acknowledged."""
    target_position = next_position(current_position, direction)
    delta = -STEP_DEGREES if direction == 'up' else STEP_DEGREES

    with porthandler.motion_lock:
        # Marlin's configured A/I soft range is 0..360 degrees. Rebase only at
        # the wrap boundary so the requested physical move remains exactly 60°.
        if current_position == 1 and direction == 'up':
            ok, _ = porthandler.write_and_wait(
                motion_platform, f'G92 {AXIS_NAME}360', timeout=2.0
            )
            if not ok:
                raise TimeoutError('Filter revolver coordinate rebase was not acknowledged.')

        ok, _ = porthandler.write_and_wait(motion_platform, 'G91', timeout=2.0)
        if not ok:
            raise TimeoutError('Relative positioning mode was not acknowledged.')

        move_ok = False
        restore_ok = False
        try:
            move_ok = porthandler.write_and_wait_motion(
                motion_platform,
                f'G1 {AXIS_NAME}{delta}',
                timeout=30.0,
            )
            if move_ok:
                move_ok = porthandler.write_and_wait_motion(
                    motion_platform,
                    'M400',
                    timeout=30.0,
                )
        finally:
            restore_ok, _ = porthandler.write_and_wait(
                motion_platform, 'G90', timeout=2.0
            )

        if not move_ok:
            raise TimeoutError('Filter revolver movement did not complete.')
        if not restore_ok:
            raise TimeoutError('Absolute positioning mode could not be restored.')

        if current_position == SLOT_COUNT and direction == 'down':
            ok, _ = porthandler.write_and_wait(
                motion_platform, f'G92 {AXIS_NAME}0', timeout=2.0
            )
            if not ok:
                raise TimeoutError('Filter revolver coordinate normalization was not acknowledged.')

    log.info(
        'Filter revolver moved from slot %s to slot %s (%s%d degrees).',
        current_position,
        target_position,
        '+' if delta > 0 else '',
        delta,
    )
    return target_position
