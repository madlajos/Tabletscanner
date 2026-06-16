import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { map } from 'rxjs/operators';
import { BASE_URL } from '../api-config';
import {
  StepDefinition,
  PipelineDocument,
  ValidationResponse,
  PreviewResponse,
  RecipeSummary,
} from '../models/pipeline.models';

export interface CalibrationRecord {
  id: string;
  name: string;
  equation: string;
  comment?: string;
  x_name?: string;
  y_name?: string;
  y_key?: string;
  model?: string;
  degree?: number;
  coefficients?: number[];
  x_min?: number;
  x_max?: number;
  created_at?: string;
}

export interface SaveImagesResponse {
  saved_count: number;
  saved_paths: string[];
  output_folder: string;
}

export interface SaveArrayResponse {
  saved_path: string;
  row_count: number;
  col_count: number;
  source_key: string;
}

export interface MontageResponse {
  success: boolean;
  montage_base64: string;
  image_count: number;
  montage_width: number;
  montage_height: number;
  grid_rows: number;
  grid_cols: number;
  cell_width: number;
  cell_height: number;
  label_height: number;
}

@Injectable({ providedIn: 'root' })
export class RecipeService {
  constructor(private http: HttpClient) {}

  /** Fetch step catalog for the toolbox. */
  getStepCatalog(): Observable<StepDefinition[]> {
    return this.http
      .get<{ steps: StepDefinition[] }>(`${BASE_URL}/pipeline/step-catalog`)
      .pipe(map((res) => res.steps));
  }

  /** Validate pipeline without executing. */
  validatePipeline(doc: PipelineDocument): Observable<ValidationResponse> {
    return this.http.post<ValidationResponse>(`${BASE_URL}/pipeline/validate`, doc);
  }

  /** Execute pipeline up to a step and get preview image + side outputs. */
  previewStep(
    pipeline: PipelineDocument,
    previewStepIndex: number,
    previewImageIndex: number = 0,
    singleImageOnly: boolean = false,
    omittedIndices: number[] = [],
    scaleBarOverlay?: Record<string, any> | null,
  ): Observable<PreviewResponse> {
    return this.http.post<PreviewResponse>(`${BASE_URL}/pipeline/preview`, {
      pipeline,
      preview_step_index: previewStepIndex,
      preview_image_index: previewImageIndex,
      single_image_only: singleImageOnly,
      omitted_indices: omittedIndices,
      scale_bar_overlay: scaleBarOverlay ?? undefined,
    });
  }

  /** Execute batch image saving for a save_images node. */
  savePipelineImages(
    pipeline: PipelineDocument,
    stepIndex: number,
    scaleBarOverlay?: Record<string, any> | null,
  ): Observable<SaveImagesResponse> {
    return this.http.post<SaveImagesResponse>(`${BASE_URL}/pipeline/save-images`, {
      pipeline,
      step_index: stepIndex,
      scale_bar_overlay: scaleBarOverlay ?? undefined,
    });
  }

  /** Execute CSV save for a save_array node. */
  savePipelineArray(pipeline: PipelineDocument, stepIndex: number): Observable<SaveArrayResponse> {
    return this.http.post<SaveArrayResponse>(`${BASE_URL}/pipeline/save-array`, {
      pipeline,
      step_index: stepIndex,
    });
  }

  /** Open native file dialog for image selection. */
  browseFile(): Observable<{ path: string }> {
    return this.http.get<{ path: string }>(`${BASE_URL}/pipeline/browse-file`);
  }

  /** Open native folder dialog for image folder selection. */
  browseFolder(): Observable<{ path: string }> {
    return this.http.get<{ path: string }>(`${BASE_URL}/pipeline/browse-folder`);
  }

  /** Open native file dialog for explicit values CSV/TXT import. */
  browseValuesFile(): Observable<{ path: string }> {
    return this.http.get<{ path: string }>(`${BASE_URL}/pipeline/browse-values-file`);
  }

  /** Import and validate explicit values from CSV/TXT file. */
  importExplicitValues(path: string): Observable<{ values: number[]; values_csv: string }> {
    return this.http.post<{ values: number[]; values_csv: string }>(
      `${BASE_URL}/pipeline/import-explicit-values`,
      { path }
    );
  }

  listCalibrations(): Observable<CalibrationRecord[]> {
    return this.http
      .get<{ calibrations: CalibrationRecord[] }>(`${BASE_URL}/pipeline/calibrations`)
      .pipe(map((res) => res.calibrations ?? []));
  }

  saveCalibration(payload: {
    name: string;
    equation: string;
    comment?: string;
    x_name?: string;
    y_name?: string;
    y_key?: string;
    model?: string;
    degree?: number;
    coefficients?: number[];
    x_min?: number;
    x_max?: number;
  }): Observable<{ message: string; calibration: CalibrationRecord }> {
    return this.http.post<{ message: string; calibration: CalibrationRecord }>(
      `${BASE_URL}/pipeline/calibrations`,
      payload,
    );
  }

  /** List saved recipes. */
  listRecipes(): Observable<RecipeSummary[]> {
    return this.http
      .get<{ recipes: RecipeSummary[] }>(`${BASE_URL}/recipes`)
      .pipe(map((res) => res.recipes));
  }

  /** Load a recipe by name. */
  loadRecipe(name: string): Observable<PipelineDocument> {
    return this.http.get<PipelineDocument>(`${BASE_URL}/recipes/${encodeURIComponent(name)}`);
  }

  /** Save a recipe. */
  saveRecipe(doc: PipelineDocument): Observable<{ message: string; name: string }> {
    return this.http.post<{ message: string; name: string }>(`${BASE_URL}/recipes`, doc);
  }

  /** Delete a recipe. */
  deleteRecipe(name: string): Observable<{ message: string }> {
    return this.http.delete<{ message: string }>(
      `${BASE_URL}/recipes/${encodeURIComponent(name)}`
    );
  }

  /** Update only the description of a recipe. */
  updateRecipeDescription(name: string, description: string): Observable<{ message: string }> {
    return this.http.patch<{ message: string }>(
      `${BASE_URL}/recipes/${encodeURIComponent(name)}/description`,
      { description }
    );
  }

  /** Duplicate a recipe. */
  duplicateRecipe(name: string): Observable<{ message: string; new_name: string }> {
    return this.http.post<{ message: string; new_name: string }>(
      `${BASE_URL}/recipes/${encodeURIComponent(name)}/duplicate`,
      {}
    );
  }

  /** Generate a montage from multiple image paths. */
  generateMontage(imagePaths: string[]): Observable<{ montage_base64: string }> {
    return this.http.post<{ montage_base64: string }>(
      `${BASE_URL}/pipeline/generate-montage`,
      { image_paths: imagePaths }
    );
  }

  /** Generate a montage of all images processed by a specific pipeline step. */
  getStepImagesMontage(pipeline: PipelineDocument, stepIndex: number): Observable<MontageResponse> {
    return this.http.post<MontageResponse>(
      `${BASE_URL}/pipeline/get-step-images-montage`,
      { pipeline, step_index: stepIndex }
    );
  }
}
