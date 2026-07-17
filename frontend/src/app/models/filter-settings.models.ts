export interface FilterDefinition {
  id: string;
  name: string;
  wavelength_range: string;
  height_offset_mm: number;
  color: string;
}

export interface FilterSettings {
  filters: FilterDefinition[];
  slots: Array<string | null>;
}

export interface FilterRevolverStatus {
  position: number | null;
  homed: boolean;
  motion_platform_homed: boolean;
  busy: boolean;
}

export type FilterRevolverDirection = 'up' | 'down';

export interface FilterRevolverMoveResponse extends FilterRevolverStatus {
  direction: FilterRevolverDirection;
  steps: number;
}
