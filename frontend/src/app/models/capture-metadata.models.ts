export interface CaptureMetadata {
  wavelength?: string | null;
  filter_name?: string | null;
  filter_position?: number | null;
  filter_wavelength?: string | null;
  tray_position?: string | null;
  Errors?: string[];
}

export interface CaptureWarning {
  id: string;
  code: string;
  missing_offset_mm: number;
  target_z: number;
}

export function wavelengthLabel(wavelength?: string | null): string {
  const labels: Record<string, string> = { vis: 'VIS', uv255: '255 nm', uv310: '310 nm', uv365: '365 nm' };
  return wavelength ? labels[wavelength] || wavelength : '—';
}
