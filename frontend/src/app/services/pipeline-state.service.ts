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

  /** Side outputs from the currently selected step. */
  private sideOutputsSubject = new BehaviorSubject<Record<string, Record<string, any>>>({});
  sideOutputs$ = this.sideOutputsSubject.asObservable();

  /** Emitted when pipeline changes (debounced for preview). */
  private pipelineChangedSubject = new Subject<void>();

  /** Recipe dirty flag. */
  private dirtySubject = new BehaviorSubject<boolean>(false);
  dirty$ = this.dirtySubject.asObservable();

  /** Current recipe name. */
  private recipeNameSubject = new BehaviorSubject<string>('');
  recipeName$ = this.recipeNameSubject.asObservable();

  constructor(private recipeService: RecipeService) {
    // Auto-preview on pipeline change (debounced)
    this.pipelineChangedSubject.pipe(debounceTime(400)).subscribe(() => {
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
    const inst = createStepInstance(stepDefId, idx, defaults);

    const steps = [...pipeline.steps];
    steps.splice(idx, 0, inst);
    this.updateSteps(steps);
    this.selectStep(idx);
  }

  removeStep(index: number): void {
    const pipeline = this.getPipeline();
    if (index < 0 || index >= pipeline.steps.length) return;

    const steps = [...pipeline.steps];
    steps.splice(index, 1);
    this.updateSteps(steps);

    // Adjust selection
    const selected = this.selectedStepIndexSubject.value;
    if (selected >= steps.length) {
      this.selectStep(steps.length - 1);
    } else if (selected === index) {
      this.selectStep(Math.min(index, steps.length - 1));
    }
  }

  moveStep(fromIndex: number, toIndex: number): void {
    const pipeline = this.getPipeline();
    if (fromIndex < 0 || fromIndex >= pipeline.steps.length) return;
    if (toIndex < 0 || toIndex >= pipeline.steps.length) return;

    const steps = [...pipeline.steps];
    const [moved] = steps.splice(fromIndex, 1);
    steps.splice(toIndex, 0, moved);
    this.updateSteps(steps);
    this.selectStep(toIndex);
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
    // Trigger preview for the newly selected step
    this.pipelineChangedSubject.next();
  }

  // --- Pipeline state helpers ---

  getPipeline(): PipelineDocument {
    return this.pipelineSubject.value;
  }

  getSelectedStepIndex(): number {
    return this.selectedStepIndexSubject.value;
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

  requestPreview(): void {
    const pipeline = this.getPipeline();
    const stepIndex = this.selectedStepIndexSubject.value;

    if (pipeline.steps.length === 0 || stepIndex < 0) {
      this.previewImageSubject.next(null);
      this.sideOutputsSubject.next({});
      return;
    }

    this.previewLoadingSubject.next(true);
    this.recipeService.previewStep(pipeline, stepIndex).subscribe({
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
        } else {
          this.validationErrorsSubject.next(res.errors || []);
          this.previewImageSubject.next(null);
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
}
