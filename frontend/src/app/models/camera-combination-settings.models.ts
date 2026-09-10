import { HeightOffsetChannel } from './filter-settings.models';

export interface CameraCombinationCell {
  exposure_time: number;
  gain: number;
}

export type CameraCombinationRow = Record<HeightOffsetChannel, CameraCombinationCell>;
export type CameraCombinationSettings = Record<string, CameraCombinationRow>;

export interface CameraParameterLimit {
  min: number;
  max: number;
  inc: number;
}

export interface CameraCombinationSettingsResponse {
  camera_combination_settings: CameraCombinationSettings;
  camera_params: Record<string, number>;
  ranges: Partial<Record<'ExposureTime' | 'Gain', CameraParameterLimit>>;
}
