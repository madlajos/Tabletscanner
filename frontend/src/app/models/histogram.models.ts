export interface HistogramSummary {
  mode: 'per_image' | 'pooled' | 'grouped';
  samples: { label: string; image_index: number; histogram: number[]; stats: Record<string, number> }[];
  groups: { label: string; indices: number[]; sample_count: number; histogram: number[] }[];
}
