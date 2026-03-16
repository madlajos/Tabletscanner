import { Injectable } from '@angular/core';
import { BehaviorSubject, Subject, debounceTime } from 'rxjs';
import {
  StepDefinition,
  StepInstance,
  StepError,
  PipelineDocument,
  PreviewResponse,
  createStepInstance,
  createEmptyPipeline,
} from '../models/pipeline.models';
import { RecipeService } from './recipe.service';

@Injectable({ providedIn: 'root' })
export class PipelineStateService {
  /** Step catalog loaded from backend. */
  private stepCatalogSubject = new BehaviorSubject<StepDefinition[]>([]);
  stepCatalog$ = this.stepCatalogSubject.asObservable();

  /** Current pipeline being edited. */
  private pipelineSubject = new BehaviorSubject<PipelineDocument>(createEmptyPipeline());
  pipeline$ = this.pipelineSubject.asObservable();

  /** Selected step index (-1 = none). */
  private selectedStepIndexSubject = new BehaviorSubject<number>(-1);
  selectedStepIndex$ = this.selectedStepIndexSubject.asObservable();

  /** Validation errors. */
  private validationErrorsSubject = new BehaviorSubject<StepError[]>([]);
  validationErrors$ = this.validationErrorsSubject.asObservable();

  /** Preview loading state. */
  private previewLoadingSubject = new BehaviorSubject<boolean>(false);
  previewLoading$ = this.previewLoadingSubject.asObservable();

  /** Preview image (base64 data URL). */
  private previewImageSubject = new BehaviorSubject<string | null>(null);
  previewImage$ = this.previewImageSubject.asObservable();

  /** Side outputs from the pipeline execution (accumulated). */
  private sideOutputsSubject = new BehaviorSubject<Record<string, any>>({});
  sideOutputs$ = this.sideOutputsSubject.asObservable();

  /** Omitted data point indices (from graph viewer). */
  private omittedPointsSubject = new BehaviorSubject<{ indices: Set<number>; imageNames: string[] }>({ indices: new Set(), imageNames: [] });
  omittedPoints$ = this.omittedPointsSubject.asObservable();

  /** Preview image index (which image to show when multiple loaded). */
  private previewImageIndexSubject = new BehaviorSubject<number>(0);
  previewImageIndex$ = this.previewImageIndexSubject.asObservable();

  /** Total image count from last preview. */
  private imageCountSubject = new BehaviorSubject<number>(0);
  imageCount$ = this.imageCountSubject.asObservable();

  /** Image dimensions from last preview. */
  private imageDimsSubject = new BehaviorSubject<{ w: number; h: number }>({ w: 0, h: 0 });
  imageDims$ = this.imageDimsSubject.asObservable();

  /** Emitted when pipeline changes (debounced for preview). */
  private pipelineChangedSubject = new Subject<void>();

  /** Emitted when a chart should be maximized in the preview area. */
  private maximizeGraphSubject = new Subject<{ data: any; omittedIndices: Set<number> }>();
  maximizeGraph$ = this.maximizeGraphSubject.asObservable();

  /** Recipe dirty flag. */
  private dirtySubject = new BehaviorSubject<boolean>(false);
  dirty$ = this.dirtySubject.asObservable();

  /** Current recipe name. */
  private recipeNameSubject = new BehaviorSubject<string>('');
  recipeName$ = this.recipeNameSubject.asObservable();

  /** Steps that aggregate across all images and must not use single-image mode. */
  private readonly AGGREGATING_STEPS = new Set([
    'fit_curve', 'predict_node', 'add_sequence_values',
  ]);

  constructor(private recipeService: RecipeService) {
    // Auto-preview on pipeline change (debounced)
    this.pipelineChangedSubject.pipe(debounceTime(400)).subscribe(() => {
      // Skip auto-preview for fit_curve and mask_rect_roi (manual play/apply button)
      const idx = this.selectedStepIndexSubject.value;
      const pipeline = this.getPipeline();
      if (idx >= 0 && idx < pipeline.steps.length) {
        const defId = pipeline.steps[idx].step_def_id;
        if (defId === 'fit_curve' || defId === 'mask_rect_roi') {
          return;
        }
      }
      this.requestPreview();
    });
  }

  // --- Catalog ---

  loadCatalog(): void {
    this.recipeService.getStepCatalog().subscribe({
      next: (catalog) => this.stepCatalogSubject.next(catalog),
      error: (err) => console.error('Failed to load step catalog:', err),
    });
  }

  getStepDefinition(stepDefId: string): StepDefinition | undefined {
    return this.stepCatalogSubject.value.find((d) => d.id === stepDefId);
  }

  // --- Pipeline manipulation ---

  addStep(stepDefId: string, atIndex?: number): void {
    const defn = this.getStepDefinition(stepDefId);
    if (!defn) return;

    const pipeline = this.getPipeline();
    const defaults: Record<string, any> = {};
    for (const p of defn.params) {
      defaults[p.name] = p.default;
    }

    const idx = atIndex ?? pipeline.steps.length;
    const steps = [...pipeline.steps];

    // Auto-add secondary input steps before this step
    let insertOffset = 0;
    if (defn.secondary_inputs?.length) {
      for (const secId of defn.secondary_inputs) {
        const secDefn = this.getStepDefinition(secId);
        if (secDefn) {
          const secDefaults: Record<string, any> = {};
          for (const p of secDefn.params) {
            secDefaults[p.name] = p.default;
          }
          const secInst = createStepInstance(secId, idx + insertOffset, secDefaults);
          steps.splice(idx + insertOffset, 0, secInst);
          insertOffset++;
        }
      }
    }

    const inst = createStepInstance(stepDefId, idx + insertOffset, defaults);
    steps.splice(idx + insertOffset, 0, inst);
    this.updateSteps(steps);
    this.selectStep(idx + insertOffset);
  }

  removeStep(index: number): void {
    const pipeline = this.getPipeline();
    if (index < 0 || index >= pipeline.steps.length) return;

    const steps = [...pipeline.steps];
    const stepToRemove = steps[index];
    const defn = this.getStepDefinition(stepToRemove.step_def_id);

    // Collect indices to remove (main step + its secondary inputs)
    const indicesToRemove = new Set<number>([index]);
    if (defn?.secondary_inputs?.length) {
      for (const secId of defn.secondary_inputs) {
        for (let j = index - 1; j >= 0; j--) {
          if (steps[j].step_def_id === secId && !indicesToRemove.has(j)) {
            indicesToRemove.add(j);
            break;
          }
        }
      }
    }

    // Remove from highest index first to preserve lower indices
    const sortedIndices = Array.from(indicesToRemove).sort((a, b) => b - a);
    for (const idx of sortedIndices) {
      steps.splice(idx, 1);
    }
    this.updateSteps(steps);

    // Adjust selection
    const selected = this.selectedStepIndexSubject.value;
    if (selected >= steps.length) {
      this.selectStep(steps.length - 1);
    } else if (indicesToRemove.has(selected)) {
      this.selectStep(Math.min(Math.min(...indicesToRemove), steps.length - 1));
    }
  }

  moveStep(fromIndex: number, toIndex: number): void {
    const pipeline = this.getPipeline();
    if (fromIndex < 0 || fromIndex >= pipeline.steps.length) return;
    if (toIndex < 0 || toIndex >= pipeline.steps.length) return;

    const defn = this.getStepDefinition(pipeline.steps[fromIndex].step_def_id);
    const steps = [...pipeline.steps];

    // Collect the main step and its secondary inputs as a group
    const groupIndices = [fromIndex];
    if (defn?.secondary_inputs?.length) {
      const secondarySet = new Set(defn.secondary_inputs);
      for (let j = fromIndex - 1; j >= 0; j--) {
        if (secondarySet.has(steps[j].step_def_id)) {
          groupIndices.unshift(j);
          secondarySet.delete(steps[j].step_def_id);
          if (secondarySet.size === 0) break;
        }
      }
    }

    // Extract the group (in order)
    const group = groupIndices.map(i => steps[i]);
    // Remove from highest index first
    for (let k = groupIndices.length - 1; k >= 0; k--) {
      steps.splice(groupIndices[k], 1);
    }

    // Adjust target index for removed items before it
    let adjustedTo = toIndex;
    for (const gi of groupIndices) {
      if (gi < toIndex) adjustedTo--;
    }

    // Insert group (secondaries first, then main)
    steps.splice(adjustedTo, 0, ...group);
    this.updateSteps(steps);
    this.selectStep(adjustedTo + group.length - 1); // Select the main step
  }

  updateParams(index: number, paramValues: Record<string, any>): void {
    const pipeline = this.getPipeline();
    if (index < 0 || index >= pipeline.steps.length) return;

    const steps = pipeline.steps.map((s, i) => {
      if (i === index) {
        return { ...s, param_values: { ...paramValues } };
      }
      return s;
    });
    this.updateSteps(steps);
  }

  selectStep(index: number): void {
    this.selectedStepIndexSubject.next(index);
    this.previewImageIndexSubject.next(0);
    // For ROI step, preview the previous step to show the input image
    const pipeline = this.getPipeline();
    if (index >= 0 && index < pipeline.steps.length &&
        pipeline.steps[index].step_def_id === 'mask_rect_roi' && index > 0) {
      this.requestPreviewForStep(index - 1);
    } else {
      // Trigger preview for the newly selected step
      this.pipelineChangedSubject.next();
    }
  }

  /** Preview a specific step index (used to show input image for ROI editing). */
  requestPreviewForStep(stepIndex: number, forceAllImages: boolean = false): void {
    const pipeline = this.getPipeline();
    const imageIndex = this.previewImageIndexSubject.value;

    if (pipeline.steps.length === 0 || stepIndex < 0) {
      this.previewImageSubject.next(null);
      return;
    }

    const step = pipeline.steps[stepIndex];
    const isAggregating = this.AGGREGATING_STEPS.has(step.step_def_id);
    const singleImageOnly = !forceAllImages && !isAggregating;
    const omittedArr = Array.from(this.omittedPointsSubject.value.indices);

    this.previewLoadingSubject.next(true);
    this.recipeService.previewStep(pipeline, stepIndex, imageIndex, singleImageOnly, omittedArr).subscribe({
      next: (res: PreviewResponse) => {
        this.previewLoadingSubject.next(false);
        if (res.success) {
          this.validationErrorsSubject.next([]);
          if (res.image_base64) {
            this.previewImageSubject.next('data:image/jpeg;base64,' + res.image_base64);
          } else {
            this.previewImageSubject.next(null);
          }
          this.imageCountSubject.next(res.image_count ?? 0);
          if (res.image_width && res.image_height) {
            this.imageDimsSubject.next({ w: res.image_width, h: res.image_height });
          }
        }
      },
      error: () => { this.previewLoadingSubject.next(false); },
    });
  }

  // --- Pipeline state helpers ---

  getPipeline(): PipelineDocument {
    return this.pipelineSubject.value;
  }

  getSelectedStepIndex(): number {
    return this.selectedStepIndexSubject.value;
  }

  getImageCount(): number {
    return this.imageCountSubject.value;
  }

  getImageDims(): { w: number; h: number } {
    return this.imageDimsSubject.value;
  }

  newPipeline(): void {
    this.pipelineSubject.next(createEmptyPipeline());
    this.selectedStepIndexSubject.next(-1);
    this.validationErrorsSubject.next([]);
    this.previewImageSubject.next(null);
    this.sideOutputsSubject.next({});
    this.dirtySubject.next(false);
    this.recipeNameSubject.next('');
  }

  loadPipeline(doc: PipelineDocument): void {
    this.pipelineSubject.next(doc);
    this.recipeNameSubject.next(doc.name);
    this.dirtySubject.next(false);
    this.selectedStepIndexSubject.next(doc.steps.length > 0 ? 0 : -1);
    this.pipelineChangedSubject.next();
  }

  private updateSteps(steps: StepInstance[]): void {
    // Re-index
    steps.forEach((s, i) => (s.order = i));
    const pipeline = { ...this.getPipeline(), steps };
    this.pipelineSubject.next(pipeline);
    this.dirtySubject.next(true);
    this.pipelineChangedSubject.next();
  }

  // --- Preview ---

  requestPreview(forceAllImages: boolean = false): void {
    const pipeline = this.getPipeline();
    const stepIndex = this.selectedStepIndexSubject.value;
    const imageIndex = this.previewImageIndexSubject.value;

    if (pipeline.steps.length === 0 || stepIndex < 0) {
      this.previewImageSubject.next(null);
      this.sideOutputsSubject.next({});
      this.imageCountSubject.next(0);
      return;
    }

    const step = pipeline.steps[stepIndex];
    const isAggregating = this.AGGREGATING_STEPS.has(step.step_def_id);
    const singleImageOnly = !forceAllImages && !isAggregating;

    // Pass omitted indices for curve fitting
    const omittedArr = Array.from(this.omittedPointsSubject.value.indices);

    this.previewLoadingSubject.next(true);
    this.recipeService.previewStep(pipeline, stepIndex, imageIndex, singleImageOnly, omittedArr).subscribe({
      next: (res: PreviewResponse) => {
        this.previewLoadingSubject.next(false);
        if (res.success) {
          this.validationErrorsSubject.next([]);
          if (res.image_base64) {
            this.previewImageSubject.next('data:image/jpeg;base64,' + res.image_base64);
          } else {
            this.previewImageSubject.next(null);
          }
          this.sideOutputsSubject.next(res.side_outputs || {});
          this.imageCountSubject.next(res.image_count ?? 0);
          if (res.image_width && res.image_height) {
            this.imageDimsSubject.next({ w: res.image_width, h: res.image_height });
          }
        } else {
          this.validationErrorsSubject.next(res.errors || []);
          this.previewImageSubject.next(null);
          this.imageCountSubject.next(0);
        }
      },
      error: (err) => {
        this.previewLoadingSubject.next(false);
        console.error('Preview failed:', err);
      },
    });
  }

  // --- Validation ---

  validate(): void {
    const pipeline = this.getPipeline();
    this.recipeService.validatePipeline(pipeline).subscribe({
      next: (res) => {
        this.validationErrorsSubject.next(res.errors || []);
      },
      error: (err) => console.error('Validation failed:', err),
    });
  }

  /** Get validation errors for a specific step. */
  getStepErrors(stepIndex: number): StepError[] {
    return this.validationErrorsSubject.value.filter((e) => e.step_index === stepIndex);
  }

  // --- Image navigation ---

  setPreviewImageIndex(index: number): void {
    const count = this.imageCountSubject.value;
    const clamped = Math.max(0, Math.min(index, count - 1));
    if (clamped !== this.previewImageIndexSubject.value) {
      this.previewImageIndexSubject.next(clamped);
      this.requestPreview();
    }
  }

  getPreviewImageIndex(): number {
    return this.previewImageIndexSubject.value;
  }

  resetPreviewImageIndex(): void {
    this.previewImageIndexSubject.next(0);
  }

  /** Reorder images in the load_image step by setting an explicit file_order param. */
  reorderLoadedImages(newOrder: number[]): void {
    const pipeline = this.getPipeline();
    const loadStepIdx = pipeline.steps.findIndex(s => s.step_def_id === 'load_image');
    if (loadStepIdx < 0) return;
    const step = pipeline.steps[loadStepIdx];
    const updated = { ...step.param_values, file_order: newOrder.join(',') };
    this.updateParams(loadStepIdx, updated);
  }

  /** Notify about omitted data points from the graph viewer. */
  notifyOmittedPoints(indices: Set<number>, imageNames: string[]): void {
    this.omittedPointsSubject.next({ indices: new Set(indices), imageNames: [...imageNames] });
  }

  getOmittedPoints(): { indices: Set<number>; imageNames: string[] } {
    return this.omittedPointsSubject.value;
  }

  /** Request maximizing a chart in the preview area. */
  requestMaximizeGraph(data: any, omittedIndices: Set<number>): void {
    this.maximizeGraphSubject.next({ data, omittedIndices });
  }
}
