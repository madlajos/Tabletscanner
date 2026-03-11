import {
  Component,
  OnInit,
  OnDestroy,
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { MatIconModule } from '@angular/material/icon';
import { Subscription, combineLatest } from 'rxjs';
import { PipelineStateService } from '../../services/pipeline-state.service';
import { RecipeService } from '../../services/recipe.service';
import {
  StepDefinition,
  StepInstance,
  StepError,
  ParamSchema,
} from '../../models/pipeline.models';

@Component({
  selector: 'app-step-inspector',
  standalone: true,
  imports: [CommonModule, FormsModule, MatIconModule],
  template: `
    <div class="inspector-wrapper">
      @if (!definition) {
        <div class="no-selection">Válasszon egy lépést a szerkesztéshez</div>
      } @else {
        <div class="step-header">
          <span class="step-icon">{{ definition.icon }}</span>
          <span class="step-name">{{ definition.name }}</span>
        </div>
        <p class="step-desc">{{ definition.description }}</p>

        @if (stepErrors.length > 0) {
          <div class="error-list">
            @for (err of stepErrors; track err.error_code) {
              <div class="error-item">⚠ {{ err.message }}</div>
            }
          </div>
        }

        <div class="params-section">
          @for (param of definition.params; track param.name) {
            @if (param.name !== 'file_order') {
            <div class="param-row">
              <label class="param-label" [attr.for]="'param-' + param.name">
                {{ param.label }}
              </label>

              @switch (param.type) {
                @case ('int') {
                  <div class="param-control slider-control">
                    <input
                      type="range"
                      [id]="'param-' + param.name"
                      [min]="param.min ?? 0"
                      [max]="param.max ?? 100"
                      [step]="param.odd_only ? 2 : (param.step ?? 1)"
                      [ngModel]="getParamValue(param.name)"
                      (ngModelChange)="onParamChange(param.name, $event)"
                    />
                    <span class="slider-value">{{ getParamValue(param.name) }}</span>
                  </div>
                }
                @case ('float') {
                  <div class="param-control slider-control">
                    <input
                      type="range"
                      [id]="'param-' + param.name"
                      [min]="param.min ?? 0"
                      [max]="param.max ?? 1"
                      [step]="param.step ?? 0.01"
                      [ngModel]="getParamValue(param.name)"
                      (ngModelChange)="onParamChange(param.name, +$event)"
                    />
                    <span class="slider-value">{{ formatFloat(getParamValue(param.name)) }}</span>
                  </div>
                }
                @case ('bool') {
                  <div class="param-control">
                    <label class="toggle-wrap">
                      <input
                        type="checkbox"
                        [id]="'param-' + param.name"
                        [ngModel]="getParamValue(param.name)"
                        (ngModelChange)="onParamChange(param.name, $event)"
                      />
                      <span class="toggle-label">{{ getParamValue(param.name) ? 'Be' : 'Ki' }}</span>
                    </label>
                  </div>
                }
                @case ('enum') {
                  <div class="param-control">
                    <select
                      [id]="'param-' + param.name"
                      [ngModel]="getParamValue(param.name)"
                      (ngModelChange)="onParamChange(param.name, $event)"
                    >
                      @for (opt of getFilteredOptions(param); track opt) {
                        <option [value]="opt">{{ opt }}</option>
                      }
                    </select>
                  </div>
                }
                @case ('string') {
                  <div class="param-control">
                    <input
                      type="text"
                      [id]="'param-' + param.name"
                      [ngModel]="getParamValue(param.name)"
                      (ngModelChange)="onParamChange(param.name, $event)"
                    />
                  </div>
                }
                @case ('file_path') {
                  <div class="param-control file-path-control">
                    <input
                      type="text"
                      [id]="'param-' + param.name"
                      [ngModel]="getParamValue(param.name)"
                      (ngModelChange)="onParamChange(param.name, $event)"
                      placeholder="Fájl elérési útja..."
                    />
                    <button class="browse-btn" (click)="browseFile(param.name)" title="Fájl tallózása"><mat-icon>image</mat-icon></button>
                    <button class="browse-btn" (click)="browseFolder(param.name)" title="Mappa tallózása"><mat-icon>folder_open</mat-icon></button>
                  </div>
                }
              }

              @if (param.description) {
                <div class="param-hint">{{ param.description }}</div>
              }
            </div>
            }
          }
        </div>

        @if (isLoadImageStep()) {
          <div class="image-manager-section">
            <button class="image-manager-btn" (click)="showImageManager = true">
              <mat-icon>photo_library</mat-icon> Képek kezelése
            </button>
          </div>
        }

        @if (showImageManager) {
          <div class="img-manager-overlay" (click)="showImageManager = false">
            <div class="img-manager-dialog" (click)="$event.stopPropagation()">
              <div class="img-manager-header">
                <span class="img-manager-title">Képek sorrendje</span>
                <button class="img-manager-close" (click)="showImageManager = false"><mat-icon>close</mat-icon></button>
              </div>
              @if (loadedImageNames.length === 0) {
                <p class="img-manager-empty">Nincs betöltött kép. Először válasszon forrás útvonalat.</p>
              } @else {
                <div class="img-manager-list">
                  @for (name of loadedImageNames; track $index) {
                    <div
                      class="img-manager-item"
                      [class.selected]="selectedImageIdx === $index"
                      (click)="selectedImageIdx = $index"
                    >
                      <span class="img-idx">{{ $index + 1 }}.</span>
                      <span class="img-name">{{ name }}</span>
                    </div>
                  }
                </div>
                <div class="img-manager-actions">
                  <button (click)="moveImage('top')" [disabled]="selectedImageIdx <= 0" title="Legelejére"><mat-icon>vertical_align_top</mat-icon></button>
                  <button (click)="moveImage('up')" [disabled]="selectedImageIdx <= 0" title="Fel"><mat-icon>arrow_upward</mat-icon></button>
                  <button (click)="moveImage('down')" [disabled]="selectedImageIdx >= loadedImageNames.length - 1" title="Le"><mat-icon>arrow_downward</mat-icon></button>
                  <button (click)="moveImage('bottom')" [disabled]="selectedImageIdx >= loadedImageNames.length - 1" title="Legvégére"><mat-icon>vertical_align_bottom</mat-icon></button>
                </div>
                <div class="img-manager-footer">
                  <button class="img-manager-apply" (click)="applyImageOrder()">Alkalmaz</button>
                </div>
              }
            </div>
          </div>
        }

        @if (hasSideOutputs()) {
          <div class="side-outputs-section">
            <div class="section-label">Eredmények</div>
            @for (key of sideOutputKeys(); track key) {
              <div class="side-output-item">
                <span class="side-key">{{ key }}:</span>
                <span class="side-value">{{ formatSideOutput(sideOutputs[key]) }}</span>
              </div>
            }
          </div>
        }
      }
    </div>
  `,
  styles: [`
    :host {
      display: block;
      height: 100%;
      overflow-y: auto;
    }

    .inspector-wrapper {
      padding: 12px;
    }

    .no-selection {
      color: #666;
      font-size: 12px;
      text-align: center;
      padding: 40px 16px;
    }

    .step-header {
      display: flex;
      align-items: center;
      gap: 8px;
      margin-bottom: 4px;
    }

    .step-icon {
      font-size: 18px;
    }

    .step-name {
      font-size: 14px;
      font-weight: 600;
      color: #e0e0e0;
    }

    .step-desc {
      font-size: 11px;
      color: #888;
      margin: 0 0 12px;
    }

    .error-list {
      margin-bottom: 12px;
    }

    .error-item {
      font-size: 11px;
      color: #ef4444;
      padding: 4px 8px;
      background: rgba(239, 68, 68, 0.1);
      border-radius: 4px;
      margin-bottom: 4px;
    }

    .params-section {
      display: flex;
      flex-direction: column;
      gap: 12px;
    }

    .param-row {
      display: flex;
      flex-direction: column;
      gap: 4px;
    }

    .param-label {
      font-size: 11px;
      font-weight: 600;
      color: #aaa;
      text-transform: uppercase;
      letter-spacing: 0.03em;
    }

    .param-control {
      width: 100%;
    }

    .slider-control {
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .slider-control input[type="range"] {
      flex: 1;
      accent-color: #3b82f6;
    }

    .slider-value {
      font-size: 12px;
      color: #ccc;
      min-width: 40px;
      text-align: right;
      font-variant-numeric: tabular-nums;
    }

    .param-control input[type="text"],
    .param-control select {
      width: 100%;
      padding: 4px 8px;
      background: #2a2a2a;
      border: 1px solid #444;
      border-radius: 4px;
      color: #e0e0e0;
      font-size: 12px;
      box-sizing: border-box;
    }

    .param-control select {
      cursor: pointer;
    }

    .toggle-wrap {
      display: flex;
      align-items: center;
      gap: 6px;
      cursor: pointer;
    }

    .toggle-label {
      font-size: 12px;
      color: #ccc;
    }

    .param-hint {
      font-size: 10px;
      color: #666;
    }

    .side-outputs-section {
      margin-top: 20px;
      padding-top: 12px;
      border-top: 1px solid #333;
    }

    .section-label {
      font-size: 11px;
      font-weight: 600;
      color: #999;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      margin-bottom: 8px;
    }

    .side-output-item {
      display: flex;
      gap: 6px;
      font-size: 12px;
      margin-bottom: 4px;
    }

    .side-key {
      color: #888;
    }

    .side-value {
      color: #e0e0e0;
      font-variant-numeric: tabular-nums;
    }

    /* File path browse buttons */
    .file-path-control {
      display: flex;
      gap: 4px;
      align-items: center;
    }

    .file-path-control input[type="text"] {
      flex: 1;
      padding: 4px 8px;
      background: #2a2a2a;
      border: 1px solid #444;
      border-radius: 4px;
      color: #e0e0e0;
      font-size: 12px;
      box-sizing: border-box;
    }

    .browse-btn {
      background: #333;
      border: 1px solid #555;
      border-radius: 4px;
      color: #e0e0e0;
      cursor: pointer;
      padding: 3px 6px;
      font-size: 14px;
      line-height: 1;
      flex-shrink: 0;
      display: flex;
      align-items: center;
      justify-content: center;
    }

    .browse-btn mat-icon {
      font-size: 16px;
      width: 16px;
      height: 16px;
    }

    .browse-btn:hover {
      background: #444;
      border-color: #3b82f6;
    }

    /* Image manager button */
    .image-manager-section {
      margin-top: 12px;
    }

    .image-manager-btn {
      width: 100%;
      padding: 8px 12px;
      background: #333;
      border: 1px solid #555;
      border-radius: 6px;
      color: #e0e0e0;
      cursor: pointer;
      font-size: 12px;
      font-weight: 600;
      display: flex;
      align-items: center;
      gap: 6px;
      justify-content: center;
    }

    .image-manager-btn:hover {
      background: #3b82f6;
      border-color: #3b82f6;
    }

    /* Image manager dialog */
    .img-manager-overlay {
      position: fixed;
      inset: 0;
      background: rgba(0, 0, 0, 0.6);
      display: flex;
      align-items: center;
      justify-content: center;
      z-index: 1000;
    }

    .img-manager-dialog {
      background: #2a2a2a;
      border: 1px solid #555;
      border-radius: 8px;
      padding: 16px;
      min-width: 340px;
      max-width: 440px;
      max-height: 70vh;
      display: flex;
      flex-direction: column;
    }

    .img-manager-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 12px;
    }

    .img-manager-title {
      font-size: 15px;
      font-weight: 600;
      color: #e0e0e0;
    }

    .img-manager-close {
      background: none;
      border: none;
      color: #888;
      cursor: pointer;
      font-size: 16px;
      padding: 2px 6px;
      display: flex;
      align-items: center;
    }

    .img-manager-close:hover {
      color: #fff;
    }

    .img-manager-empty {
      color: #888;
      font-size: 12px;
      text-align: center;
      padding: 20px;
    }

    .img-manager-list {
      flex: 1;
      overflow-y: auto;
      max-height: 40vh;
      border: 1px solid #444;
      border-radius: 4px;
      background: #1e1e1e;
    }

    .img-manager-item {
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 6px 10px;
      cursor: pointer;
      font-size: 12px;
      color: #ccc;
      border-bottom: 1px solid #333;
    }

    .img-manager-item:last-child {
      border-bottom: none;
    }

    .img-manager-item:hover {
      background: #333;
    }

    .img-manager-item.selected {
      background: #224477;
      color: #fff;
    }

    .img-idx {
      color: #888;
      min-width: 24px;
      font-variant-numeric: tabular-nums;
    }

    .img-manager-item.selected .img-idx {
      color: #bfdbfe;
    }

    .img-name {
      flex: 1;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .img-manager-actions {
      display: flex;
      justify-content: center;
      gap: 6px;
      padding: 10px 0;
    }

    .img-manager-actions button {
      background: #333;
      border: 1px solid #555;
      border-radius: 4px;
      color: #e0e0e0;
      cursor: pointer;
      padding: 6px 12px;
      font-size: 16px;
      display: flex;
      align-items: center;
      justify-content: center;
    }

    .img-manager-actions button mat-icon {
      font-size: 20px;
      width: 20px;
      height: 20px;
    }

    .img-manager-actions button:hover:not(:disabled) {
      background: #444;
      border-color: #3b82f6;
    }

    .img-manager-actions button:disabled {
      opacity: 0.3;
      cursor: default;
    }

    .img-manager-footer {
      display: flex;
      justify-content: flex-end;
      gap: 8px;
      padding-top: 8px;
      border-top: 1px solid #444;
    }

    .img-manager-apply {
      padding: 6px 16px;
      border: 1px solid #224477;
      border-radius: 4px;
      cursor: pointer;
      font-size: 12px;
      font-weight: 600;
      background: #224477;
      color: #fff;
    }

    .img-manager-apply:hover {
      background: #1f4b8f;
    }
  `],
})
export class StepInspectorComponent implements OnInit, OnDestroy {
  definition: StepDefinition | undefined;
  step: StepInstance | undefined;
  stepErrors: StepError[] = [];
  sideOutputs: Record<string, any> = {};

  // Image manager state
  showImageManager = false;
  loadedImageNames: string[] = [];
  /** Tracks the original index for each position so reordering can be applied. */
  imageOrderIndices: number[] = [];
  selectedImageIdx = 0;

  private selectedIndex = -1;
  private subs: Subscription[] = [];

  constructor(
    private pipelineState: PipelineStateService,
    private recipeService: RecipeService,
  ) {}

  ngOnInit(): void {
    this.subs.push(
      combineLatest([
        this.pipelineState.pipeline$,
        this.pipelineState.selectedStepIndex$,
        this.pipelineState.validationErrors$,
        this.pipelineState.sideOutputs$,
      ]).subscribe(([pipeline, idx, errors, sideOutputs]) => {
        this.selectedIndex = idx;
        if (idx >= 0 && idx < pipeline.steps.length) {
          this.step = pipeline.steps[idx];
          this.definition = this.pipelineState.getStepDefinition(this.step.step_def_id);
          this.stepErrors = errors.filter((e) => e.step_index === idx);
          this.sideOutputs = sideOutputs ?? {};
          // Populate loaded image names from side outputs
          const paths: string[] = sideOutputs?.['loaded_paths'] ?? [];
          if (paths.length > 0 && paths.length !== this.loadedImageNames.length) {
            this.loadedImageNames = [...paths];
            this.imageOrderIndices = paths.map((_, i) => i);
            this.selectedImageIdx = 0;
          }
        } else {
          this.step = undefined;
          this.definition = undefined;
          this.stepErrors = [];
          this.sideOutputs = {};
          this.loadedImageNames = [];
          this.imageOrderIndices = [];
        }
      })
    );
  }

  ngOnDestroy(): void {
    this.subs.forEach((s) => s.unsubscribe());
  }

  getParamValue(paramName: string): any {
    return this.step?.param_values?.[paramName];
  }

  onParamChange(paramName: string, value: any): void {
    if (!this.step) return;
    const updated = { ...this.step.param_values, [paramName]: value };

    // When color space changes, auto-select first valid channel
    if (this.step.step_def_id === 'select_channel' && paramName === 'space') {
      const validChannels = this.CHANNEL_MAP[value] ?? ['GRAY'];
      if (!validChannels.includes(updated['channel'])) {
        updated['channel'] = validChannels[0];
      }
    }

    this.pipelineState.updateParams(this.selectedIndex, updated);
  }

  formatFloat(val: any): string {
    if (val == null) return '-';
    return Number(val).toFixed(2);
  }

  hasSideOutputs(): boolean {
    return Object.keys(this.sideOutputs).length > 0;
  }

  sideOutputKeys(): string[] {
    return Object.keys(this.sideOutputs);
  }

  formatSideOutput(value: any): string {
    if (value == null) return '-';
    if (typeof value === 'number') return Number(value).toFixed(4);
    if (typeof value === 'object') return JSON.stringify(value);
    return String(value);
  }

  // --- File/folder browsing ---

  browseFile(paramName: string): void {
    this.recipeService.browseFile().subscribe({
      next: (res) => {
        if (res.path) {
          this.onParamChange(paramName, res.path);
        }
      },
    });
  }

  browseFolder(paramName: string): void {
    this.recipeService.browseFolder().subscribe({
      next: (res) => {
        if (res.path) {
          this.onParamChange(paramName, res.path);
        }
      },
    });
  }

  // --- Image manager ---

  isLoadImageStep(): boolean {
    return this.step?.step_def_id === 'load_image';
  }

  moveImage(direction: 'up' | 'down' | 'top' | 'bottom'): void {
    const idx = this.selectedImageIdx;
    const arr = this.loadedImageNames;
    const orderArr = this.imageOrderIndices;
    if (arr.length < 2) return;

    let newIdx = idx;
    switch (direction) {
      case 'top':
        newIdx = 0;
        break;
      case 'up':
        newIdx = idx - 1;
        break;
      case 'down':
        newIdx = idx + 1;
        break;
      case 'bottom':
        newIdx = arr.length - 1;
        break;
    }
    if (newIdx < 0 || newIdx >= arr.length || newIdx === idx) return;

    // Swap in names
    const tmpName = arr[idx];
    arr[idx] = arr[newIdx];
    arr[newIdx] = tmpName;

    // Swap in order indices
    const tmpOrder = orderArr[idx];
    orderArr[idx] = orderArr[newIdx];
    orderArr[newIdx] = tmpOrder;

    this.selectedImageIdx = newIdx;
  }

  applyImageOrder(): void {
    this.pipelineState.reorderLoadedImages(this.imageOrderIndices);
    this.showImageManager = false;
  }

  // --- Dynamic enum filtering ---

  private readonly CHANNEL_MAP: Record<string, string[]> = {
    BGR: ['B', 'G', 'R'],
    HSV: ['H', 'S', 'V'],
    LAB: ['L', 'A', 'B'],
    GRAY: ['GRAY'],
  };

  getFilteredOptions(param: ParamSchema): string[] {
    if (this.step?.step_def_id === 'select_channel' && param.name === 'channel') {
      const space = this.getParamValue('space') ?? 'GRAY';
      return this.CHANNEL_MAP[space] ?? param.options ?? [];
    }
    return param.options ?? [];
  }
}
