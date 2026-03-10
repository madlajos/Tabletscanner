import {
  Component,
  OnInit,
  OnDestroy,
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Subscription, combineLatest } from 'rxjs';
import { PipelineStateService } from '../../services/pipeline-state.service';
import {
  StepDefinition,
  StepInstance,
  StepError,
  ParamSchema,
} from '../../models/pipeline.models';

@Component({
  selector: 'app-step-inspector',
  standalone: true,
  imports: [CommonModule, FormsModule],
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
                      @for (opt of param.options; track opt) {
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
                  <div class="param-control">
                    <input
                      type="text"
                      [id]="'param-' + param.name"
                      [ngModel]="getParamValue(param.name)"
                      (ngModelChange)="onParamChange(param.name, $event)"
                      placeholder="Fájl elérési útja..."
                    />
                  </div>
                }
              }

              @if (param.description) {
                <div class="param-hint">{{ param.description }}</div>
              }
            </div>
          }
        </div>

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
  `],
})
export class StepInspectorComponent implements OnInit, OnDestroy {
  definition: StepDefinition | undefined;
  step: StepInstance | undefined;
  stepErrors: StepError[] = [];
  sideOutputs: Record<string, any> = {};

  private selectedIndex = -1;
  private subs: Subscription[] = [];

  constructor(private pipelineState: PipelineStateService) {}

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
        } else {
          this.step = undefined;
          this.definition = undefined;
          this.stepErrors = [];
          this.sideOutputs = {};
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
}
