export interface AdvancedMotionSettings {
  use_virtual_com_port: boolean;
  max_height_offset_up_mm: number;
  max_height_offset_down_mm: number;
  first_tablet_x_mm: number;
  first_tablet_y_mm: number;
  first_tablet_z_mm: number;
  tablet_spacing_mm: number;
}

export interface MotionConnectionStatus {
  connected: boolean;
  port: string | null;
  virtual: boolean;
}
