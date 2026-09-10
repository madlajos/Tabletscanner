export type IntensityMode = 'per_image' | 'pooled' | 'grouped';
export interface IntensityStatistics {
  min: number; max: number; mean: number; median: number; std: number; pixel_count: number;
  dynamic_range: number | null;
  [key: string]: number | null;
}
export interface IntensitySummary {
  mode: IntensityMode;
  groups: { label: string; image_indices: number[]; sample_count: number; channels: (IntensityStatistics | null)[] }[];
  samples: { label: string; image_index: number; channels: (IntensityStatistics | null)[] }[];
}
