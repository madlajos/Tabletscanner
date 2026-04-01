/**
 * Pipeline domain model types — mirrors backend pipeline_types.py
 */

export type DataType = 'IMAGE' | 'GRAYSCALE' | 'MASK' | 'SCALAR' | 'HISTOGRAM' | 'CONTOURS';

export type ParamType = 'int' | 'float' | 'bool' | 'enum' | 'string' | 'file_path';

export interface ParamSchema {
  name: string;
  label: string;
  type: ParamType;
  default: any;
  required?: boolean;
  min?: number;
  max?: number;
  step?: number;
  options?: string[];
  description?: string;
  odd_only?: boolean;
}

export interface StepDefinition {
  id: string;
  name: string;
  category: string;
  description: string;
  icon: string;
  input_type: DataType;
  output_type: DataType;
  params: ParamSchema[];
  side_output_types: Record<string, string>;
  required_preceding_steps?: string[];
  secondary_inputs?: string[];
}

export interface StepInstance {
  instance_id: string;
  step_def_id: string;
  param_values: Record<string, any>;
  order: number;
}

export interface PipelineDocument {
  schema_version: number;
  name: string;
  description: string;
  steps: StepInstance[];
  created_at: string;
  modified_at: string;
}

export interface StepError {
  step_index: number;
  step_def_id: string;
  error_code: string;
  message: string;
  param_name?: string;
}

export interface ValidationResponse {
  valid: boolean;
  errors: StepError[];
}

export interface PreviewResponse {
  success: boolean;
  errors?: StepError[];
  executed_up_to: number;
  side_outputs?: Record<string, any>;
  image_base64?: string;
  image_width?: number;
  image_height?: number;
  image_count?: number;
  is_grayscale?: boolean;
}

export interface RecipeSummary {
  name: string;
  description: string;
  step_count: number;
  modified_at: string;
}

export function createStepInstance(stepDefId: string, order: number, defaults?: Record<string, any>): StepInstance {
  return {
    instance_id: crypto.randomUUID(),
    step_def_id: stepDefId,
    param_values: defaults ? { ...defaults } : {},
    order,
  };
}

export function createEmptyPipeline(name: string = ''): PipelineDocument {
  return {
    schema_version: 1,
    name,
    description: '',
    steps: [],
    created_at: '',
    modified_at: '',
  };
}
