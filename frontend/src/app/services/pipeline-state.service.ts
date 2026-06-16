import { Injectable } from '@angular/core';
import { BehaviorSubject, Subject, debounceTime } from 'rxjs';
import {
  StepDefinition,
  StepInstance,
  StepError,
  PipelineDocument,
  DataType,
  PreviewResponse,
  createStepInstance,
  createEmptyPipeline,
} from '../models/pipeline.models';
import { RecipeService } from './recipe.service';

type PortDirection = 'source' | 'transform' | 'sink';

interface StepIoOverride {
  direction?: PortDirection;
  inputType?: DataType | null;
  outputType?: DataType | null;
}

const STEP_IO_OVERRIDES: Record<string, StepIoOverride> = {
  // Source nodes: no primary input in the main chain.
  load_image: { direction: 'source', inputType: null },
  add_sequence_values: { direction: 'source', inputType: null },
  // Sink node: explicit save action as final step.
  save_images: { direction: 'sink', outputType: null },
  save_array: { direction: 'sink', outputType: null },
  // Explicit grayscale transform override to keep drag/drop compatibility stable.
  robust_stretch_gamma: { inputType: 'GRAYSCALE', outputType: 'GRAYSCALE' },
};

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

  /** Toolbox preview step id (single-click in toolbox). */
  private toolboxPreviewStepIdSubject = new BehaviorSubject<string | null>(null);
  toolboxPreviewStepId$ = this.toolboxPreviewStepIdSubject.asObservable();

  /** Validation errors. */
  private validationErrorsSubject = new BehaviorSubject<StepError[]>([]);
  validationErrors$ = this.validationErrorsSubject.asObservable();

  /** Preview loading state. */
  private previewLoadingSubject = new BehaviorSubject<boolean>(false);
  previewLoading$ = this.previewLoadingSubject.asObservable();

  /** Preview image (base64 data URL). */
  private previewImageSubject = new BehaviorSubject<string | null>(null);
  previewImage$ = this.previewImageSubject.asObservable();

  /** Optional overlay/result image shown beside the base preview. */
  private previewImageOverrideSubject = new BehaviorSubject<string | null>(null);
  previewImageOverride$ = this.previewImageOverrideSubject.asObservable();

  /** Multi-panel dual-map preview (gray/RGB originals + N component results + optional sub-classification). */
  private dualMapPreviewSubject = new BehaviorSubject<{
    grayBase: string | null;
    grayOverlays: string[];
    rgbBase: string | null;
    rgbOverlays: string[];
    subBase: string | null;
    subOverlays: string[];
    subLabel: string;
  } | null>(null);
  dualMapPreview$ = this.dualMapPreviewSubject.asObservable();

  setDualMapPreview(state: {
    grayBase: string | null;
    grayOverlays: string[];
    rgbBase: string | null;
    rgbOverlays: string[];
    subBase: string | null;
    subOverlays: string[];
    subLabel: string;
  } | null): void {
    this.dualMapPreviewSubject.next(state);
  }

  /** Whether preview image is grayscale. */
  private previewImageIsGrayscaleSubject = new BehaviorSubject<boolean>(false);
  previewImageIsGrayscale$ = this.previewImageIsGrayscaleSubject.asObservable();

  /** Manually set the overlay/result image without replacing the base preview. */
  setPreviewImageOverride(imageDataUrl: string | null, isGrayscale: boolean = false): void {
    this.previewImageOverrideSubject.next(imageDataUrl);
  }

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

  /** Current scalebar overlay configuration used for export/save. */
  private scaleBarExportParamsSubject = new BehaviorSubject<Record<string, any> | null>(null);
  scaleBarExportParams$ = this.scaleBarExportParamsSubject.asObservable();

  /** Emitted when pipeline changes (debounced for preview). */
  private pipelineChangedSubject = new Subject<void>();

  /** Emitted when a chart should be maximized in the preview area. */
  private maximizeGraphSubject = new Subject<{ data: any; omittedIndices: Set<number>; sourceStepIndex: number }>();
  maximizeGraph$ = this.maximizeGraphSubject.asObservable();

  /** Emitted when expanded chart (scatter or PCA) should be shown. */
  private expandedChartSubject = new Subject<{ data: any; type: 'scatter' | 'pca'; title: string }>();
  expandedChart$ = this.expandedChartSubject.asObservable();

  /** Recipe dirty flag. */
  private dirtySubject = new BehaviorSubject<boolean>(false);
  dirty$ = this.dirtySubject.asObservable();

  /** Current recipe name. */
  private recipeNameSubject = new BehaviorSubject<string>('');
  recipeName$ = this.recipeNameSubject.asObservable();

  /** Steps that aggregate across all images and must not use single-image mode. */
  private readonly AGGREGATING_STEPS = new Set([
    'fit_curve', 'predict_node', 'add_sequence_values', 'histogram_pca', 'detect_circles',
    'dual_map',   // needs all images to auto-detect gray/RGB pairs
  ]);

  private shouldPreviewRoiOutput(step: StepInstance | null): boolean {
    if (!step || step.step_def_id !== 'mask_rect_roi') return false;
    return step.param_values?.['output_mode'] === 'crop' || step.param_values?.['apply_mask'] !== false;
  }

  constructor(private recipeService: RecipeService) {
    // Auto-preview on pipeline change (debounced)
    this.pipelineChangedSubject.pipe(debounceTime(400)).subscribe(() => {
      // Skip auto-preview for fit_curve (manual play/apply button)
      const idx = this.selectedStepIndexSubject.value;
      const pipeline = this.getPipeline();
      if (idx >= 0 && idx < pipeline.steps.length) {
        const defId = pipeline.steps[idx].step_def_id;
        if (defId === 'fit_curve') {
          return;
        }
        // For ROI step: show crop output too, but the editor overlay will be disabled on the preview side.
        if (defId === 'mask_rect_roi') {
          const roiStep = pipeline.steps[idx];
          if (this.shouldPreviewRoiOutput(roiStep)) {
            this.requestPreview();
          } else if (idx > 0) {
            this.requestPreviewForStep(idx - 1);
          } else {
            this.requestPreview();
          }
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
    if (!this.canInsertStepAtFlatIndex(stepDefId, idx)) return;

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
    this.clearToolboxPreviewStep();
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

    if (!this.isPrimaryChainCompatible(steps)) {
      return;
    }

    this.updateSteps(steps);
    this.selectStep(adjustedTo + group.length - 1); // Select the main step
  }

  /** Check if inserting a step at a flat pipeline index keeps the chain compatible. */
  canInsertStepAtFlatIndex(stepDefId: string, flatIndex: number): boolean {
    const defn = this.getStepDefinition(stepDefId);
    if (!defn) return false;

    // add_sequence_values is intended as a secondary input for fit_curve,
    // not as a standalone main-chain step.
    if (stepDefId === 'add_sequence_values') return false;

    const pipeline = this.getPipeline();
    const idx = Math.max(0, Math.min(flatIndex, pipeline.steps.length));
    const simulated = [...pipeline.steps];

    // Simulate auto-added secondary input steps for a main step insert.
    let insertOffset = 0;
    if (defn.secondary_inputs?.length) {
      for (const secId of defn.secondary_inputs) {
        const secDefn = this.getStepDefinition(secId);
        if (!secDefn) continue;
        simulated.splice(idx + insertOffset, 0, this.createTemporaryStep(secId));
        insertOffset++;
      }
    }
    simulated.splice(idx + insertOffset, 0, this.createTemporaryStep(stepDefId));

    return this.isPrimaryChainCompatible(simulated);
  }

  /** Check if inserting a step at a main-chain index is valid. */
  canInsertStepAtMainIndex(stepDefId: string, mainIndex: number): boolean {
    const mainChain = this.getMainChain(this.getPipeline().steps);
    const flatIndex = mainIndex < mainChain.length
      ? mainChain[mainIndex].pipelineIndex
      : this.getPipeline().steps.length;
    return this.canInsertStepAtFlatIndex(stepDefId, flatIndex);
  }

  /** Check if moving a main-chain node from one main index to another is valid. */
  canMoveMainStep(fromMainIndex: number, toMainIndex: number): boolean {
    const pipeline = this.getPipeline();
    const mainChain = this.getMainChain(pipeline.steps);
    if (fromMainIndex < 0 || fromMainIndex >= mainChain.length) return false;
    if (toMainIndex < 0 || toMainIndex >= mainChain.length) return false;
    if (fromMainIndex === toMainIndex) return true;

    const fromFlat = mainChain[fromMainIndex].pipelineIndex;
    const toFlat = mainChain[toMainIndex].pipelineIndex;

    const defn = this.getStepDefinition(pipeline.steps[fromFlat].step_def_id);
    const steps = [...pipeline.steps];

    const groupIndices = [fromFlat];
    if (defn?.secondary_inputs?.length) {
      const secondarySet = new Set(defn.secondary_inputs);
      for (let j = fromFlat - 1; j >= 0; j--) {
        if (secondarySet.has(steps[j].step_def_id)) {
          groupIndices.unshift(j);
          secondarySet.delete(steps[j].step_def_id);
          if (secondarySet.size === 0) break;
        }
      }
    }

    const group = groupIndices.map((i) => steps[i]);
    for (let k = groupIndices.length - 1; k >= 0; k--) {
      steps.splice(groupIndices[k], 1);
    }

    let adjustedTo = toFlat;
    for (const gi of groupIndices) {
      if (gi < toFlat) adjustedTo--;
    }
    steps.splice(adjustedTo, 0, ...group);
    return this.isPrimaryChainCompatible(steps);
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
    this.clearToolboxPreviewStep();
    const pipeline = this.getPipeline();
    const selectedStep = index >= 0 && index < pipeline.steps.length ? pipeline.steps[index] : null;
    const selectedDef = selectedStep ? this.getStepDefinition(selectedStep.step_def_id) : undefined;
    this.selectedStepIndexSubject.next(index);
    if (selectedDef?.id !== 'save_images' && selectedDef?.id !== 'save_array') {
      this.previewImageIndexSubject.next(0);
    }
    // For ROI step, preview the previous step to show the input image
    if (index >= 0 && index < pipeline.steps.length &&
        pipeline.steps[index].step_def_id === 'mask_rect_roi' && index > 0) {
      const roiStep = pipeline.steps[index];
      if (this.shouldPreviewRoiOutput(roiStep)) {
        this.requestPreview();
      } else {
        // Show input image when masking is disabled for interactive drawing.
        this.requestPreviewForStep(index - 1);
      }
    } else if (index >= 0 && index < pipeline.steps.length &&
               (pipeline.steps[index].step_def_id === 'save_images' ||
                pipeline.steps[index].step_def_id === 'save_array') && index > 0) {
      this.requestPreviewForStep(index - 1);
    } else if (index >= 0 && index < pipeline.steps.length &&
               pipeline.steps[index].step_def_id === 'scale_bar_overlay' && index > 0) {
      this.requestPreviewForStep(index - 1);
    } else if (index >= 0 && index < pipeline.steps.length &&
               pipeline.steps[index].step_def_id === 'fit_curve' && index > 0) {
      // Fit curve runs manually, but inspector still needs upstream outputs for dynamic Y options.
      this.requestPreviewForStep(index - 1);
    } else if (index >= 0 && index < pipeline.steps.length &&
               pipeline.steps[index].step_def_id === 'predict_node' && index > 0) {
      // Predict node needs upstream intensity_stats for Y field dropdown.
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
      this.previewImageOverrideSubject.next(null);
      this.dualMapPreviewSubject.next(null);
      return;
    }

    const step = pipeline.steps[stepIndex];
    const isAggregating = this.AGGREGATING_STEPS.has(step.step_def_id);
    const singleImageOnly = !forceAllImages && !isAggregating;
    const selectedStep = this.selectedStepIndexSubject.value >= 0 && this.selectedStepIndexSubject.value < pipeline.steps.length
      ? pipeline.steps[this.selectedStepIndexSubject.value]
      : null;
    const scaleBarOverlay =
      selectedStep?.step_def_id === 'save_images' || selectedStep?.step_def_id === 'save_array'
        ? this.scaleBarExportParamsSubject.value
        : null;
    const omittedArr = Array.from(this.omittedPointsSubject.value.indices);

    this.previewLoadingSubject.next(true);
    this.previewImageOverrideSubject.next(null);
    this.dualMapPreviewSubject.next(null);
    this.recipeService.previewStep(pipeline, stepIndex, imageIndex, singleImageOnly, omittedArr, scaleBarOverlay).subscribe({
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
          this.previewImageIsGrayscaleSubject.next(res.is_grayscale ?? false);
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

  setScaleBarExportParams(params: Record<string, any> | null): void {
    this.scaleBarExportParamsSubject.next(params ? { ...params } : null);
  }

  getScaleBarExportParams(): Record<string, any> | null {
    const params = this.scaleBarExportParamsSubject.value;
    return params ? { ...params } : null;
  }

  getStepOutputType(stepIndex: number): DataType | null {
    const pipeline = this.getPipeline();
    if (stepIndex < 0 || stepIndex >= pipeline.steps.length) {
      return null;
    }

    const step = pipeline.steps[stepIndex];
    const defn = this.getStepDefinition(step.step_def_id);
    if (!defn) {
      return null;
    }

    return this.getOutputType(defn, step);
  }

  getImageCount(): number {
    return this.imageCountSubject.value;
  }

  getImageDims(): { w: number; h: number } {
    return this.imageDimsSubject.value;
  }

  showExpandedChart(data: any, type: 'scatter' | 'pca', title: string): void {
    this.expandedChartSubject.next({ data, type, title });
  }

  newPipeline(): void {
    this.pipelineSubject.next(createEmptyPipeline());
    this.selectedStepIndexSubject.next(-1);
    this.clearToolboxPreviewStep();
    this.validationErrorsSubject.next([]);
    this.previewImageSubject.next(null);
    this.previewImageOverrideSubject.next(null);
    this.dualMapPreviewSubject.next(null);
    this.sideOutputsSubject.next({});
    this.dirtySubject.next(false);
    this.recipeNameSubject.next('');
  }

  loadPipeline(doc: PipelineDocument): void {
    this.pipelineSubject.next(doc);
    this.recipeNameSubject.next(doc.name);
    this.dirtySubject.next(false);
    this.clearToolboxPreviewStep();
    this.selectedStepIndexSubject.next(doc.steps.length > 0 ? 0 : -1);
    this.previewImageOverrideSubject.next(null);
    this.dualMapPreviewSubject.next(null);
    this.pipelineChangedSubject.next();
  }

  setToolboxPreviewStep(stepDefId: string | null): void {
    this.toolboxPreviewStepIdSubject.next(stepDefId);
  }

  clearToolboxPreviewStep(): void {
    this.toolboxPreviewStepIdSubject.next(null);
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
      this.previewImageOverrideSubject.next(null);
      this.dualMapPreviewSubject.next(null);
      this.sideOutputsSubject.next({});
      this.imageCountSubject.next(0);
      return;
    }

    const step = pipeline.steps[stepIndex];
    const previewStepIndex =
      step.step_def_id === 'save_images' || step.step_def_id === 'save_array'
        ? Math.max(0, stepIndex - 1)
        : stepIndex;
    const previewStep = pipeline.steps[previewStepIndex];
    const isAggregating = this.AGGREGATING_STEPS.has(previewStep.step_def_id);
    const singleImageOnly = !forceAllImages && !isAggregating;
    const scaleBarOverlay =
      step.step_def_id === 'save_images' || step.step_def_id === 'save_array'
        ? this.scaleBarExportParamsSubject.value
        : null;

    // Pass omitted indices for curve fitting
    const omittedArr = Array.from(this.omittedPointsSubject.value.indices);

    this.previewLoadingSubject.next(true);
    this.previewImageOverrideSubject.next(null);
    this.dualMapPreviewSubject.next(null);
    this.recipeService.previewStep(pipeline, previewStepIndex, imageIndex, singleImageOnly, omittedArr, scaleBarOverlay).subscribe({
      next: (res: PreviewResponse) => {
        this.previewLoadingSubject.next(false);
        if (res.success) {
          this.validationErrorsSubject.next([]);
          if (res.image_base64) {
            this.previewImageSubject.next('data:image/jpeg;base64,' + res.image_base64);
          } else {
            this.previewImageSubject.next(null);
          }
          this.previewImageIsGrayscaleSubject.next(res.is_grayscale ?? false);
          this.sideOutputsSubject.next(res.side_outputs || {});
          this.imageCountSubject.next(res.image_count ?? 0);
          if (res.image_width && res.image_height) {
            this.imageDimsSubject.next({ w: res.image_width, h: res.image_height });
          }
        } else {
          this.validationErrorsSubject.next(res.errors || []);
          this.previewImageSubject.next(null);
          this.previewImageOverrideSubject.next(null);
          this.dualMapPreviewSubject.next(null);
          this.previewImageIsGrayscaleSubject.next(false);
          this.imageCountSubject.next(0);
          this.sideOutputsSubject.next({});
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
      const stepIndex = this.selectedStepIndexSubject.value;
      const pipeline = this.getPipeline();
      if (stepIndex >= 0 && stepIndex < pipeline.steps.length) {
        const selected = pipeline.steps[stepIndex];
        if (selected.step_def_id === 'fit_curve' && stepIndex > 0) {
          // Fit curve is a manual action. While paging images, only refresh upstream context.
          this.requestPreviewForStep(stepIndex - 1);
          return;
        }
        if (selected.step_def_id === 'mask_rect_roi' && stepIndex > 0) {
          if (this.shouldPreviewRoiOutput(selected)) {
            this.requestPreview();
          } else {
            // ROI step: preview input image for drawing, ROI overlay handles visualization
            this.requestPreviewForStep(stepIndex - 1);
          }
          return;
        }
      }
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
  requestMaximizeGraph(data: any, omittedIndices: Set<number>, sourceStepIndex: number): void {
    this.maximizeGraphSubject.next({ data, omittedIndices, sourceStepIndex });
  }

  private createTemporaryStep(stepDefId: string): StepInstance {
    return {
      instance_id: `tmp-${stepDefId}-${Math.random().toString(36).slice(2, 9)}`,
      step_def_id: stepDefId,
      param_values: {},
      order: -1,
    };
  }

  private getSecondaryIndices(steps: StepInstance[]): Set<number> {
    const secondary = new Set<number>();
    for (let i = 0; i < steps.length; i++) {
      const defn = this.getStepDefinition(steps[i].step_def_id);
      if (!defn?.secondary_inputs?.length) continue;
      for (const secId of defn.secondary_inputs) {
        for (let j = i - 1; j >= 0; j--) {
          if (steps[j].step_def_id === secId && !secondary.has(j)) {
            secondary.add(j);
            break;
          }
        }
      }
    }
    return secondary;
  }

  private getMainChain(steps: StepInstance[]): Array<{ step: StepInstance; definition?: StepDefinition; pipelineIndex: number }> {
    const secondary = this.getSecondaryIndices(steps);
    const main: Array<{ step: StepInstance; definition?: StepDefinition; pipelineIndex: number }> = [];
    for (let i = 0; i < steps.length; i++) {
      if (secondary.has(i)) continue;
      main.push({
        step: steps[i],
        definition: this.getStepDefinition(steps[i].step_def_id),
        pipelineIndex: i,
      });
    }
    return main;
  }

  private getDirection(defn: StepDefinition): PortDirection {
    return STEP_IO_OVERRIDES[defn.id]?.direction ?? 'transform';
  }

  private getInputType(defn: StepDefinition): DataType | null {
    const override = STEP_IO_OVERRIDES[defn.id];
    if (override && Object.prototype.hasOwnProperty.call(override, 'inputType')) {
      return override.inputType ?? null;
    }
    return defn.input_type ?? null;
  }

  private getOutputType(defn: StepDefinition, step?: StepInstance): DataType | null {
    if (defn.id === 'calculate_histograms') {
      return 'HISTOGRAM';
    }

    if (defn.id === 'calculate_intensity_stats') {
      return 'SCALAR';
    }

    if (defn.id === 'histogram_equalization') {
      const selected = String(step?.param_values?.['output_mode'] ?? '').trim();
      if (selected === 'histogram') return 'HISTOGRAM';
      if (selected === 'image') return 'GRAYSCALE';
      const fallback = defn.params.find((p) => p.name === 'output_mode')?.default;
      if (fallback === 'histogram') return 'HISTOGRAM';
      return 'GRAYSCALE';
    }

    if (defn.id === 'apply_threshold') {
      const selected = String(step?.param_values?.['mode'] ?? '').trim();
      if (selected === 'trunc' || selected === 'tozero' || selected === 'tozero_inv') {
        return 'GRAYSCALE';
      }
      return 'MASK';
    }

    const override = STEP_IO_OVERRIDES[defn.id];
    if (override && Object.prototype.hasOwnProperty.call(override, 'outputType')) {
      return override.outputType ?? null;
    }
    return defn.output_type ?? null;
  }

  private areTypesCompatible(outputType: DataType | null, inputType: DataType | null): boolean {
    if (!outputType || !inputType) return false;

    if (outputType === inputType) return true;

    // IMAGE input accepts any raster-like image output.
    if (inputType === 'IMAGE') {
      return outputType === 'IMAGE' || outputType === 'GRAYSCALE' || outputType === 'MASK';
    }

    // GRAYSCALE input accepts grayscale-like outputs and auto-converts from IMAGE.
    // Also accepts HISTOGRAM output (e.g., from calculate_histograms for histogram_pca).
    if (inputType === 'GRAYSCALE') {
      return outputType === 'GRAYSCALE' || outputType === 'MASK' || outputType === 'IMAGE' || outputType === 'HISTOGRAM';
    }

    // MASK input accepts mask outputs.
    if (inputType === 'MASK') {
      return outputType === 'MASK' || outputType === 'GRAYSCALE';
    }

    return false;
  }

  /**
   * Validate only the primary chain (secondaries are modeled as branch inputs).
   * Rules:
   * - first primary node must be load_image
   * - source nodes have no primary input (can only appear at first primary position)
   * - sink nodes have no primary output (must be last primary node)
   * - adjacent primary nodes must have compatible output/input data types
   */
  private isPrimaryChainCompatible(steps: StepInstance[]): boolean {
    const main = this.getMainChain(steps);
    if (main.length === 0) return true;

    const firstDef = main[0].definition;
    if (!firstDef || firstDef.id !== 'load_image') {
      return false;
    }

    for (let i = 0; i < main.length; i++) {
      const currDef = main[i].definition;
      if (!currDef) return false;

      const currDir = this.getDirection(currDef);
      const currInput = this.getInputType(currDef);
      const currOutput = this.getOutputType(currDef, main[i].step);
      const isLast = i === main.length - 1;

      if (i === 0) {
        if (currDir !== 'source') return false;
      } else {
        if (currDir === 'source') return false;

        if (currDir === 'sink') {
          if (!isLast) return false;
          continue;
        }

        const prevDef = main[i - 1].definition;
        if (!prevDef) return false;
        const prevDir = this.getDirection(prevDef);
        if (prevDir === 'sink') return false;

        const prevOutput = this.getOutputType(prevDef, main[i - 1].step);

        // Histogram equalization requires grayscale-like input.
        // Prevent dropping it directly after generic IMAGE output steps.
        if (currDef.id === 'histogram_equalization' && prevOutput === 'IMAGE') {
          return false;
        }

        // Range mask should only be used on grayscale-like outputs
        // (e.g. selected grayscale channel), not directly on generic IMAGE.
        if (currDef.id === 'apply_range_mask' && prevOutput === 'IMAGE') {
          return false;
        }

        // Flat-field and advanced illumination correction operate on
        // grayscale-like images, so block direct generic IMAGE input.
        if ((currDef.id === 'flat_field_correction' || currDef.id === 'advanced_illumin_corr')
            && prevOutput === 'IMAGE') {
          return false;
        }

        // Data-array save can only follow numeric outputs.
        if (currDef.id === 'save_array') {
          if (!(prevOutput === 'SCALAR' || prevOutput === 'HISTOGRAM' || prevOutput === 'CONTOURS')) {
            return false;
          }
        } else if (!this.areTypesCompatible(prevOutput, currInput)) {
          return false;
        }
      }

      if (!isLast && !currOutput) return false;
      if (i > 0 && currDir === 'transform' && !currInput) return false;
    }

    return true;
  }
}
