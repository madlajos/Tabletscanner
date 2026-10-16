import { PipelineDocument, PreviewResponse } from './pipeline.models';

export type InlineAnalysisMode = 'each_tablet' | 'measurement_end';
export type InlineInputScope = 'tablet' | 'cumulative';
export type InlineResultStatus = 'queued' | 'running' | 'completed' | 'failed';

export interface InlineAnalysisError {
  id: string;
  code: string;
  error: string;
  popup: boolean;
  tablet_id: string | null;
  result_id: string | null;
}

export interface InlineAnalysisTablet {
  tablet_id: string;
  tablet_index: number | string;
  label: string;
  image_count: number;
  status: 'waiting' | InlineResultStatus;
  result_id: string | null;
}

export interface InlineAnalysisResult {
  result_id: string;
  kind: 'tablet' | 'cumulative';
  tablet_id: string | null;
  label: string;
  status: InlineResultStatus;
  image_count: number;
  input_count: number;
  thumbnail_base64?: string;
  output_images?: InlineOutputImage[];
  revision: number;
}

export interface InlineOutputImage {
  image_index: number;
  tablet_id: string | null;
  wavelength?: string;
  channel?: string | null;
  filter_name?: string | null;
}

export interface InlineAnalysisSession {
  session_id: string;
  recipe_name: string;
  measurement_name: string;
  mode: InlineAnalysisMode;
  input_scope: InlineInputScope;
  state: 'collecting' | 'finishing' | 'completed' | 'cancelled';
  pipeline: PipelineDocument;
  tablets: InlineAnalysisTablet[];
  results: InlineAnalysisResult[];
  errors: InlineAnalysisError[];
  revision: number;
  pipeline_revision: number;
}

export interface InlineAnalysisConfiguration {
  mode?: InlineAnalysisMode;
  input_scope?: InlineInputScope;
  pipeline?: PipelineDocument;
}

export interface InlineAnalysisPreview {
  result_id: string | null;
  image_index: number;
  loading: boolean;
  response: PreviewResponse | null;
}
