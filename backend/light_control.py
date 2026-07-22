"""Thread-safe four-channel illumination controller.

Physical M106 selectors are fixed by the approved firmware identity marker and
are never accepted from a frontend lamp command.
"""

from dataclasses import dataclass
from contextlib import nullcontext
import re
import threading
import time

from settings_manager import UV_LAMP_CHANNELS, validate_lamp_output_selectors, validate_lamp_settings

LIGHT_CHANNELS = (*UV_LAMP_CHANNELS, 'vis')
UV_MODES = ('dimmed', 'full')
_LAMP_GCODE_PATTERN = re.compile(r'M0*10[67](?!\d)', flags=re.IGNORECASE)


def contains_lamp_gcode(command):
    """Return whether one raw G-code payload could address a lamp output."""
    if not isinstance(command, str):
        return False
    return bool(_LAMP_GCODE_PATTERN.search(re.sub(r'\s+', '', command)))


class LightConfigurationError(ValueError):
    pass


class LampSettingsError(LightConfigurationError):
    """The UV lamp brightness/timeout configuration is not usable."""
    pass


class LightCommandError(RuntimeError):
    pass


@dataclass(frozen=True)
class ActiveLight:
    channel: str
    mode: str | None
    deadline: float | None


class LightController:
    def __init__(
        self,
        settings_getter,
        serial_getter,
        command_writer,
        clock=time.monotonic,
        operation_lock=None,
    ):
        self._settings_getter = settings_getter
        self._serial_getter = serial_getter
        self._command_writer = command_writer
        self._clock = clock
        self._operation_lock = operation_lock
        self._lock = threading.RLock()
        self._active: ActiveLight | None = None
        self._active_serial = None
        self._auto_turned_off: list[str] = []
        self._auto_off_event_pending = False

    def status(self):
        with self._lock:
            active = self._active
            if active:
                try:
                    current_serial = self._serial_getter()
                except Exception:
                    current_serial = None
                if current_serial is not self._active_serial or not getattr(current_serial, 'is_open', False):
                    self._active = None
                    self._active_serial = None
                    active = None
            return {
                'active_channel': active.channel if active else None,
                'active_mode': active.mode if active else None,
                'channels': {channel: bool(active and active.channel == channel) for channel in LIGHT_CHANNELS},
                'auto_turned_off': list(self._auto_turned_off),
            }

    def consume_auto_off_event(self):
        """Return and clear the compatibility event without touching hardware."""
        with self._lock:
            pending = self._auto_off_event_pending
            self._auto_off_event_pending = False
            return pending

    def confirm_all_off(self):
        """Synchronize cached state after an independently acknowledged all-off."""
        with self._lock:
            self._active = None
            self._active_serial = None

    def activate(self, channel, mode=None):
        if channel not in LIGHT_CHANNELS:
            raise LightConfigurationError('Unknown light channel.')
        if channel in UV_LAMP_CHANNELS and mode not in UV_MODES:
            raise LightConfigurationError('UV channels require dimmed or full mode.')
        if channel == 'vis' and mode is not None:
            raise LightConfigurationError('VIS does not support a brightness mode.')

        with self._operation_guard(), self._lock:
            # Validate safety-critical UV configuration before output routing so
            # incomplete brightness/timeout values receive their dedicated error.
            lamp_settings = self._uv_lamp_settings() if channel in UV_LAMP_CHANNELS else None
            selectors = self._selectors()

            # Never trust cached software state as proof that another physical
            # output is off. A backend restart, raw G-code, failed response, or
            # stale packaged settings can desynchronize it from the board. Turn
            # off every selector and require every acknowledgement before
            # energizing the requested channel.
            self._send_all_off(selectors)
            self._active = None
            self._active_serial = None

            if channel == 'vis':
                pwm = 255
                deadline = None
            else:
                channel_settings = lamp_settings['channels'][channel]
                prefix = 'dim' if mode == 'dimmed' else 'full'
                pwm = round(255 * channel_settings[f'{prefix}_percent'] / 100)
                deadline = self._clock() + channel_settings[f'{prefix}_timeout_seconds']

            active_serial = self._send(channel, selectors, pwm)
            self._active = ActiveLight(channel, mode, deadline)
            self._active_serial = active_serial
            return self.status()

    def off(self, channel=None):
        if channel is not None and channel not in LIGHT_CHANNELS:
            raise LightConfigurationError('Unknown light channel.')
        with self._operation_guard(), self._lock:
            selectors = self._selectors()
            # A channel-specific UI toggle is still a physical all-off. Cached
            # state cannot prove that another MOSFET was not left energized by
            # an earlier process, failed request, or incompatible build.
            self._send_all_off(selectors)
            self._active = None
            self._active_serial = None
            return self.status()

    def check_timeouts(self):
        with self._operation_guard(), self._lock:
            if not self._active or self._active.deadline is None or self._clock() < self._active.deadline:
                return None
            channel = self._active.channel
            self._send_all_off(self._selectors())
            self._active = None
            self._active_serial = None
            self._auto_turned_off.append(channel)
            self._auto_turned_off = self._auto_turned_off[-10:]
            self._auto_off_event_pending = True
            return channel

    def _uv_lamp_settings(self):
        settings = self._settings_getter() or {}
        try:
            return validate_lamp_settings(settings.get('lamp_settings', {}))
        except ValueError as error:
            raise LampSettingsError(str(error)) from error

    def _selectors(self):
        settings = self._settings_getter() or {}
        try:
            return validate_lamp_output_selectors({
                'output_selectors': settings.get('lamp_settings', {}).get('output_selectors', {})
            })['output_selectors']
        except ValueError as error:
            raise LightConfigurationError(str(error)) from error

    def _send(self, channel, selectors, pwm):
        selector = selectors[channel]
        try:
            serial_port = self._serial_getter()
            if not serial_port or not getattr(serial_port, 'is_open', False):
                raise LightCommandError('Motion platform is not connected.')
            acknowledged, _ = self._command_writer(serial_port, f'M106 {selector} S{pwm}', timeout=2.0)
            if not acknowledged:
                raise LightCommandError(f'Controller did not acknowledge {channel} command.')
            return serial_port
        except LightCommandError:
            raise
        except Exception as error:
            raise LightCommandError(f'{channel} command failed: {error}') from error

    def _send_off(self, channel, selectors):
        self._send(channel, selectors, 0)

    def _send_all_off(self, selectors):
        """Attempt every OFF command, then fail without energizing anything."""
        failures = []
        # Use physical selector order so traces and board indicator LEDs always
        # show the fixed P0..P3 clearing sequence.
        for configured_channel in sorted(LIGHT_CHANNELS, key=lambda channel: selectors[channel]):
            try:
                self._send_off(configured_channel, selectors)
            except LightCommandError as error:
                failures.append(f'{configured_channel}: {error}')
        if failures:
            raise LightCommandError('Failed to turn off all light outputs: ' + '; '.join(failures))

    def _operation_guard(self):
        return self._operation_lock if self._operation_lock is not None else nullcontext()
