export interface HeightReferenceStatus {
  available: boolean;
  reference_z: number | null;
  applied_offset_mm: number;
  source: 'autofocus' | 'anchor' | null;
  baseline_offset_mm: number;
}
