import { Component, EventEmitter, Input, Output } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { ParamSchema, StepDefinition, StepInstance } from '../../models/pipeline.models';

const OUTPUT_PARAMETERS = new Set([
  'display_mode', 'distribution_mode', 'chart_enabled', 'chart_metric', 'percentiles',
  'output_mode', 'output_unit', 'size_metric', 'bin_count', 'smooth_distribution',
  'use_filtered', 'include_excluded', 'selected_columns', 'group_labels',
  'draw', 'draw_label', 'draw_only_filtered', 'draw_label_key', 'contour_thickness',
  'show_image', 'show_reference', 'show_references', 'show_reference_montage',
  'reference_opacity', 'overlay_opacity', 'background_color', 'invert_mask',
]);

@Component({
  selector: 'app-inline-output-settings', standalone: true, imports: [FormsModule],
  template: `
    @if (definition && step) {
      <h3>{{ definition.name }}</h3>
      @if (step.step_def_id === 'load_image') {
        <p>A képek az automata mérésből érkeznek, a tabletta teljes képsorozatának elkészülte után.</p>
      }
      @for (param of parameters; track param.name) {
        <label class="parameter" [for]="'inline-param-' + param.name">
          <span>{{ param.label }}</span>
          @if (param.type === 'bool') {
            <input type="checkbox" [id]="'inline-param-' + param.name" [ngModel]="value(param)" (ngModelChange)="change(param, $event)" />
          } @else if (param.type === 'enum' || param.name === 'chart_metric') {
            <select [id]="'inline-param-' + param.name" [ngModel]="value(param)" (ngModelChange)="change(param, $event)">
              @for (option of options(param); track option) { <option [value]="option">{{ optionLabel(option) }}</option> }
            </select>
          } @else {
            <input [id]="'inline-param-' + param.name" [type]="param.type === 'int' || param.type === 'float' ? 'number' : 'text'"
              [min]="param.min ?? null" [max]="param.max ?? null" [step]="param.step ?? 1"
              [ngModel]="value(param)" (ngModelChange)="change(param, $event)" />
          }
        </label>
      }
      @if (!parameters.length && step.step_def_id !== 'load_image') { <p>Ehhez a kimenethez nincs külön megjelenítési beállítás.</p> }
      <p>A módosítások a mérés elemzésére vonatkoznak. Az eredmények a háttérben újraszámítódnak.</p>
    }
  `,
  styles: [`
    :host { display: block; font-size: 12px; }
    h3 { font-size: 14px; margin: 0 0 14px; color: #e6edf7; }
    p { color: #a1afc0; font-size: 11px; line-height: 1.5; }
    .parameter { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 8px; margin: 12px 0; }
    select, input:not([type=checkbox]) { min-width: 0; width: 100%; box-sizing: border-box; color: #e6edf7; background: #202730; border: 1px solid #485767; border-radius: 5px; padding: 7px; }
    input[type=checkbox] { accent-color: #69aef1; }
    input:focus-visible, select:focus-visible { outline: 2px solid #78b8f5; outline-offset: 2px; }
  `],
})
export class InlineOutputSettingsComponent {
  @Input() definition: StepDefinition | undefined;
  @Input() step: StepInstance | undefined;
  @Output() parameterChange = new EventEmitter<{ name: string; value: string | number | boolean }>();
  get parameters(): ParamSchema[] { return (this.definition?.params ?? []).filter(param => OUTPUT_PARAMETERS.has(param.name)); }
  value(param: ParamSchema): string | number | boolean {
    const value: unknown = this.step?.param_values[param.name] ?? param.default;
    return typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean' ? value : '';
  }
  options(param: ParamSchema): string[] {
    if (param.name !== 'chart_metric') return param.options ?? [];
    const percentiles = String(this.step?.param_values['percentiles'] ?? '5,25,50,75,95').split(',')
      .map(value => Number(value.trim())).filter(value => Number.isFinite(value) && value >= 0 && value <= 100);
    return [...new Set(['mean', 'median', 'min', 'max', 'std', 'pixel_count', 'dynamic_range', ...percentiles.map(value => `p${value}`), String(this.value(param))])];
  }
  optionLabel(option: string): string {
    const labels: Record<string, string> = {
      mean: 'Átlag', median: 'Medián', min: 'Minimum', max: 'Maximum', std: 'Szórás',
      pixel_count: 'Pixelszám', dynamic_range: 'Dinamikus tartomány', per_image: 'Képenként',
      pooled: 'Együtt számítva', grouped: 'Csoportonként', overlay: 'Közös diagram',
      crop: 'Kivágott kép', mask: 'Maszk', px: 'Pixel', mm: 'Milliméter', um: 'Mikrométer',
    };
    return labels[option] ?? option;
  }
  change(param: ParamSchema, value: unknown): void {
    if (typeof value !== 'string' && typeof value !== 'number' && typeof value !== 'boolean') return;
    if (param.type === 'int' || param.type === 'float') {
      const numeric = Number(value);
      if (!Number.isFinite(numeric) || (param.min !== undefined && numeric < param.min)
        || (param.max !== undefined && numeric > param.max) || (param.type === 'int' && !Number.isInteger(numeric))) return;
      value = numeric;
    }
    this.parameterChange.emit({ name: param.name, value: value as string | number | boolean });
  }
}
