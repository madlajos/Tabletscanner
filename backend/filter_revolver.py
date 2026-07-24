"""Acknowledged motion commands for the six-position filter revolver."""

import logging

import porthandler


log = logging.getLogger(__name__)

SLOT_COUNT = 6
STEP_DEGREES = 60
AXIS_NAME = 'A'
MOVE_FEEDRATE_DEGREES_PER_MINUTE = 90 * 60
CONTROL_TIMEOUT_SECONDS = 2.0
MOTION_TIMEOUT_SECONDS = 30.0


class FilterRevolverCommandError(RuntimeError):
    """A command was sent, but the revolver controller did not acknowledge it."""

    def __init__(self, stage: str, command: str, reply: bytes):
        self.stage = stage
        self.command = command
        self.reply = reply
        reply_preview = repr(reply[:256]) if reply else '<no response>'
        super().__init__(
            f'Filter revolver {stage} failed for {command!r}; controller reply: {reply_preview}'
        )


def _require_ack(motion_platform, stage: str, command: str, timeout: float) -> None:
    acknowledged, reply = porthandler.write_and_wait(
        motion_platform,
        command,
        timeout=timeout,
    )
    if acknowledged:
        log.debug(
            'Filter revolver %s acknowledged for %r: %r',
            stage,
            command,
            reply[:256],
        )
        return

    log.warning(
        'Filter revolver %s was not acknowledged for %r; reply=%r',
        stage,
        command,
        reply[:256],
    )
    raise FilterRevolverCommandError(stage, command, reply)


def next_position(current_position: int, direction: str) -> int:
    """Return the next slot: screen-right/up advances, screen-left/down goes back."""
    if current_position not in range(1, SLOT_COUNT + 1):
        raise ValueError('Current filter position must be between 1 and 6.')
    if direction not in ('up', 'down'):
        raise ValueError("Direction must be 'up' or 'down'.")
    offset = 1 if direction == 'up' else -1
    return ((current_position - 1 + offset) % SLOT_COUNT) + 1


def shortest_path(current_position: int, target_position: int) -> tuple[str, int]:
    """Return the direction and number of 60-degree steps to a target slot."""
    if current_position not in range(1, SLOT_COUNT + 1):
        raise ValueError('Current filter position must be between 1 and 6.')
    if target_position not in range(1, SLOT_COUNT + 1):
        raise ValueError('Target filter position must be between 1 and 6.')
    up_steps = (target_position - current_position) % SLOT_COUNT
    down_steps = (current_position - target_position) % SLOT_COUNT
    return ('up', up_steps) if up_steps <= down_steps else ('down', down_steps)


def rotate_one_slot(motion_platform, current_position: int, direction: str) -> int:
    """Rotate exactly one slot and return the new position after all commands are acknowledged."""
    target_position = next_position(current_position, direction)
    delta = -STEP_DEGREES if direction == 'up' else STEP_DEGREES

    with porthandler.motion_lock:
        # Marlin's configured A/I soft range is 0..360 degrees. Rebase only at
        # the wrap boundary so the requested physical move remains exactly 60°.
        _require_ack(
            motion_platform,
            'relative-mode setup',
            'G91',
            CONTROL_TIMEOUT_SECONDS,
        )

        movement_error = None
        try:
            _require_ack(
                motion_platform,
                'movement command',
                f'G1 {AXIS_NAME}{delta} F{MOVE_FEEDRATE_DEGREES_PER_MINUTE}',
                MOTION_TIMEOUT_SECONDS,
            )
            _require_ack(
                motion_platform,
                'movement completion',
                'M400',
                MOTION_TIMEOUT_SECONDS,
            )
        except FilterRevolverCommandError as error:
            movement_error = error
        finally:
            try:
                _require_ack(
                    motion_platform,
                    'absolute-mode restore',
                    'G90',
                    CONTROL_TIMEOUT_SECONDS,
                )
            except FilterRevolverCommandError:
                if movement_error is None:
                    raise
                log.exception(
                    'Absolute positioning mode restoration also failed after a revolver movement error.'
                )

        if movement_error is not None:
            raise movement_error

        if target_position == 1:
            _require_ack(
                motion_platform,
                'coordinate normalization',
                f'G92 {AXIS_NAME}0',
                CONTROL_TIMEOUT_SECONDS,
            )

    log.info(
        'Filter revolver moved from slot %s to slot %s (%s%d degrees).',
        current_position,
        target_position,
        '+' if delta > 0 else '',
        delta,
    )
    return target_position
