export interface AdvancedMotionSettings {
  use_virtual_com_port: boolean;
  max_height_offset_up_mm: number;
  max_height_offset_down_mm: number;
}

export interface MotionConnectionStatus {
  connected: boolean;
  port: string | null;
  virtual: boolean;
}
