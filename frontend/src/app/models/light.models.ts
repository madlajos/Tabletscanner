/** Canonical identifiers used by the four-channel illumination contract. */
export type LightChannel = 'uv255' | 'uv310' | 'uv365' | 'vis';

export type UvBrightnessMode = 'dimmed' | 'full';

export type LightActivationMode = UvBrightnessMode | null;

export interface UvLampSettings {
  dim_percent: number;
  full_percent: number;
  dim_timeout_seconds: number;
  full_timeout_seconds: number;
}

export type UvLampSettingsByChannel = Record<Exclude<LightChannel, 'vis'>, UvLampSettings>;

export interface LampSettings {
  channels: Partial<UvLampSettingsByChannel>;
}

export interface AdvancedLampSettings {
  output_selectors: Record<LightChannel, string>;
}

export interface LightStatus {
  active_channel: LightChannel | null;
  active_mode: LightActivationMode;
  channels: Record<LightChannel, boolean>;
  auto_turned_off: LightChannel[];
}

export interface CapturePlanRow {
  id: string;
  wavelength: LightChannel;
  filter_position: 1 | 2 | 3 | 4 | 5 | 6;
}

export interface CaptureRequestRow {
  wavelength: LightChannel;
  filter_position: number;
}

export const LIGHT_CHANNELS: readonly LightChannel[] = ['uv255', 'uv310', 'uv365', 'vis'];

export const LIGHT_CHANNEL_LABELS: Readonly<Record<LightChannel, string>> = {
  uv255: '255 nm',
  uv310: '310 nm',
  uv365: '365 nm',
  vis: 'VIS'
};

export const LIGHT_ASSET_FILENAMES: Readonly<Record<LightChannel, { on: string; off: string }>> = {
  uv255: { on: '255nm_on.svg', off: '255nm_off.svg' },
  uv310: { on: '310nm_on.svg', off: '310nm_off.svg' },
  uv365: { on: '365nm_on.svg', off: '365nm_off.svg' },
  vis: { on: 'vis_on.svg', off: 'vis_off.svg' }
};
