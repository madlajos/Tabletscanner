"""Thread-safe four-channel illumination controller.

Physical M106 selectors are operator-configured values, never frontend input.
"""

from dataclasses import dataclass
import threading
import time

from settings_manager import UV_LAMP_CHANNELS, validate_lamp_output_selectors, validate_lamp_settings

LIGHT_CHANNELS = (*UV_LAMP_CHANNELS, 'vis')
UV_MODES = ('dimmed', 'full')


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
    def __init__(self, settings_getter, serial_getter, command_writer, clock=time.monotonic):
        self._settings_getter = settings_getter
        self._serial_getter = serial_getter
        self._command_writer = command_writer
        self._clock = clock
        self._lock = threading.RLock()
        self._active: ActiveLight | None = None
        self._auto_turned_off: list[str] = []

    def status(self):
        with self._lock:
            active = self._active
            return {
                'active_channel': active.channel if active else None,
                'active_mode': active.mode if active else None,
                'channels': {channel: bool(active and active.channel == channel) for channel in LIGHT_CHANNELS},
                'auto_turned_off': list(self._auto_turned_off),
            }

    def activate(self, channel, mode=None):
        if channel not in LIGHT_CHANNELS:
            raise LightConfigurationError('Unknown light channel.')
        if channel in UV_LAMP_CHANNELS and mode not in UV_MODES:
            raise LightConfigurationError('UV channels require dimmed or full mode.')
        if channel == 'vis' and mode is not None:
            raise LightConfigurationError('VIS does not support a brightness mode.')

        with self._lock:
            # Validate safety-critical UV configuration before output routing so
            # incomplete brightness/timeout values receive their dedicated error.
            lamp_settings = self._uv_lamp_settings() if channel in UV_LAMP_CHANNELS else None
            selectors = self._selectors()
            if self._active and self._active.channel != channel:
                self._send_off(self._active.channel, selectors)
                self._active = None

            if channel == 'vis':
                pwm = 255
                deadline = None
            else:
                channel_settings = lamp_settings['channels'][channel]
                prefix = 'dim' if mode == 'dimmed' else 'full'
                pwm = round(255 * channel_settings[f'{prefix}_percent'] / 100)
                deadline = self._clock() + channel_settings[f'{prefix}_timeout_seconds']

            self._send(channel, selectors, pwm)
            self._active = ActiveLight(channel, mode, deadline)
            return self.status()

    def off(self, channel=None):
        if channel is not None and channel not in LIGHT_CHANNELS:
            raise LightConfigurationError('Unknown light channel.')
        with self._lock:
            selectors = self._selectors()
            if channel is None:
                for configured_channel in LIGHT_CHANNELS:
                    self._send_off(configured_channel, selectors)
                self._active = None
            elif self._active and self._active.channel == channel:
                self._send_off(channel, selectors)
                self._active = None
            return self.status()

    def check_timeouts(self):
        with self._lock:
            if not self._active or self._active.deadline is None or self._clock() < self._active.deadline:
                return None
            channel = self._active.channel
            self._send_off(channel, self._selectors())
            self._active = None
            self._auto_turned_off.append(channel)
            self._auto_turned_off = self._auto_turned_off[-10:]
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
        serial_port = self._serial_getter()
        if not serial_port or not getattr(serial_port, 'is_open', False):
            raise LightCommandError('Motion platform is not connected.')
        acknowledged, _ = self._command_writer(serial_port, f'M106 {selector} S{pwm}', timeout=2.0)
        if not acknowledged:
            raise LightCommandError(f'Controller did not acknowledge {channel} command.')

    def _send_off(self, channel, selectors):
        self._send(channel, selectors, 0)
