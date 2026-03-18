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
  ): Observable<PreviewResponse> {
    return this.http.post<PreviewResponse>(`${BASE_URL}/pipeline/preview`, {
      pipeline,
      preview_step_index: previewStepIndex,
      preview_image_index: previewImageIndex,
      single_image_only: singleImageOnly,
      omitted_indices: omittedIndices,
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
}
