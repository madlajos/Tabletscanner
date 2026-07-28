export interface CameraIntegerLimit {
  min: number;
  max: number;
  inc: number;
}

export interface CameraImageSettings {
  override_enabled: boolean;
  width: number;
  height: number;
  offset_x: number;
  offset_y: number;
}

export interface CameraImageSettingsResponse {
  camera_image_settings: CameraImageSettings;
  limits: Partial<Record<'width' | 'height' | 'offset_x' | 'offset_y', CameraIntegerLimit>>;
  connected: boolean;
}
