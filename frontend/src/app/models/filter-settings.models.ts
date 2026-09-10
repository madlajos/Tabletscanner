export interface FilterDefinition {
  id: string;
  name: string;
  wavelength_range: string;
  color: string;
}

export type HeightOffsetChannel = 'uv255' | 'uv310' | 'uv365' | 'vis';
export type HeightOffsetRow = Record<HeightOffsetChannel, number>;

export interface HeightOffsetApplication {
  requested_z?: number;
  missing_offset_mm?: number;
  warning?: import('./capture-metadata.models').CaptureWarning;
  applied: boolean;
  reason?: 'autofocus_required' | 'no_active_light';
  offset_mm?: number;
  target_z?: number;
  moved?: boolean;
}

export interface FilterSettings {
  filters: FilterDefinition[];
  slots: Array<string | null>;
  height_offsets_mm: Record<string, HeightOffsetRow>;
}

export interface FilterRevolverStatus {
  position: number | null;
  homed: boolean;
  motion_platform_homed: boolean;
  busy: boolean;
  height_offset?: HeightOffsetApplication;
  camera_params?: Record<string, number>;
}

export type FilterRevolverDirection = 'up' | 'down';

export interface FilterRevolverMoveResponse extends FilterRevolverStatus {
  direction: FilterRevolverDirection;
  steps: number;
}
