import { HeightReferenceStatus } from './height-reference.models';
import { HeightOffsetApplication, FilterRevolverStatus } from './filter-settings.models';

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

export interface DoorInterlockStatus {
  known: boolean;
  front_open: boolean | null;
  top_open: boolean | null;
  uv_blocked: boolean;
  trip_count: number | null;
}

export const UNKNOWN_DOOR_INTERLOCK: DoorInterlockStatus = {
  known: false, front_open: null, top_open: null, uv_blocked: true, trip_count: null,
};

export function doorsAllowUv(doors: DoorInterlockStatus | undefined): boolean {
  return doors?.known === true && doors.front_open === false
    && doors.top_open === false && doors.uv_blocked === false;
}

export interface LightStatus {
  door_interlock?: DoorInterlockStatus;
  active_channel: LightChannel | null;
  active_mode: LightActivationMode;
  channels: Record<LightChannel, boolean>;
  auto_turned_off: LightChannel[];
  height_offset?: HeightOffsetApplication;
  height_offset_reference?: HeightReferenceStatus;
  camera_params?: Record<string, number>;
  filter_revolver?: FilterRevolverStatus | null;
}

export interface CapturePlanRow {
  id: string;
  wavelength: LightChannel;
  brightness: UvBrightnessMode;
  filter_position: 1 | 2 | 3 | 4 | 5 | 6;
  exposure_time: number;
  gain: number;
  exposure_time_text: string;
  gain_text: string;
}

export interface CaptureRequestRow {
  wavelength: LightChannel;
  brightness: UvBrightnessMode;
  filter_position: number;
  exposure_time: number;
  gain: number;
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
