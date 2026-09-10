import { Component, EventEmitter, Input, OnChanges, OnDestroy, Output, SimpleChanges } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { IntensityMode } from '../../models/intensity.models';
import { parseGroupCsv, selectGroupLabels } from './intensity-csv';

@Component({
  selector: 'app-intensity-settings', standalone: true, imports: [FormsModule],
  template: `
    <div class="field">
      <label for="intensity-display-mode">Megjelenítés</label>
      <select id="intensity-display-mode" [ngModel]="mode" (ngModelChange)="modeChange.emit($event)">
        <option value="per_image">Egy minta</option><option value="pooled">Összes minta összevonva</option>
        <option value="grouped">CSV szerinti csoportok</option>
      </select>
      <p class="hint">{{ kind === 'histogram' ? 'Az összevonás a képenkénti gyakoriságokat adja össze. A PCA képenkénti hisztogramjai megmaradnak.' : 'Az összevonás a maszkon belüli pixelekből számol. A görbeillesztés mintánkénti értékei megmaradnak.' }}</p>
    </div>

    <div class="option-card">
      <label class="toggle-row">
        <input type="checkbox" [ngModel]="chartEnabled" (ngModelChange)="chartEnabledChange.emit($event)" />
        <span><strong>Ábrázolás diagramon</strong><small>{{ kind === 'histogram' ? 'Hisztogram megjelenítése' : 'Diagram megjelenítése' }} az osztott preview jobb oldalán</small></span>
      </label>
    @if (chartEnabled && kind === 'intensity') {
      <div class="field nested">
        <label for="intensity-chart-metric">Y tengely paramétere</label>
        <select id="intensity-chart-metric" [ngModel]="chartMetric" (ngModelChange)="chartMetricChange.emit($event)">
          @for (metric of metricOptions; track metric.value) {
            <option [value]="metric.value">{{ metric.label }}</option>
          }
        </select>
        <p class="hint">X tengely: {{ mode === 'grouped' ? 'CSV-csoportok' : 'képek sorszáma' }}</p>
      </div>
    }
    </div>

    @if (mode === 'grouped') {
      <div class="csv-card">
        <div class="card-heading"><span>CSV-csoportok</span><span class="count-badge">{{ savedCount }} mentve</span></div>
        <label class="file-picker">
          <input type="file" accept=".csv,.txt,text/csv" (change)="readFile($event)" />
          <span class="file-button">CSV kiválasztása</span>
          <span class="file-name" [title]="fileName">{{ fileName || 'Nincs kiválasztott fájl' }}</span>
        </label>
      @if (rows.length) {
        <div class="csv-grid">
          <div class="field"><label for="intensity-csv-orientation">Elrendezés</label><select id="intensity-csv-orientation" [(ngModel)]="orientation" (ngModelChange)="refresh()"><option value="column">Oszlop</option><option value="row">Sor</option></select></div>
          <div class="field"><label for="intensity-csv-position">{{ orientation === 'row' ? 'Sor száma' : 'Oszlop száma' }}</label><input id="intensity-csv-position" type="number" min="1" [(ngModel)]="position" (ngModelChange)="refresh()" /></div>
        </div>
        <label class="toggle-row compact"><input type="checkbox" [(ngModel)]="skipFirst" (ngModelChange)="refresh()" /><span>Első érték kihagyása <small>Fejléc esetén kapcsolja be</small></span></label>
        @if (!error) { <div class="import-summary"><strong>{{ candidate.length }} címke felismerve</strong><span [title]="candidate.join(', ')">{{ candidatePreview }}</span></div> }
        <button class="apply-button" type="button" [disabled]="!candidate.length || !!error" (click)="labelsChange.emit(candidate)">Csoportok alkalmazása</button>
      }
      @if (error) { <p role="alert" class="error">{{ error }}</p> }
      <p class="hint footer-hint">Képenként egy címke szükséges, a képek betöltési sorrendjében.</p>
      </div>
    }
  `,
  styles: [`
    :host { display: flex; flex-direction: column; gap: 12px; padding: 4px 0 2px; color: #e4e9ef; font-size: 12px; }
    .field { display: flex; flex-direction: column; gap: 5px; min-width: 0; }
    .field > label { color: #c9d1db; font-size: 11px; font-weight: 600; }
    select, input[type="number"] { width: 100%; height: 34px; box-sizing: border-box; padding: 0 10px; border: 1px solid #474f59; border-radius: 7px; background: #25292e; color: #eef2f7; font: inherit; }
    select:hover, input[type="number"]:hover { border-color: #606b78; }
    select:focus-visible, input:focus-visible, button:focus-visible, .file-picker:focus-within { outline: 2px solid #3b82f6; outline-offset: 1px; }
    .hint { margin: 1px 0 0; color: #8f9aa7; font-size: 10.5px; line-height: 1.35; }
    .option-card, .csv-card { border: 1px solid #39414a; border-radius: 9px; background: #22262b; }
    .option-card { padding: 9px 10px; }
    .nested { margin: 10px 0 1px 25px; padding-top: 9px; border-top: 1px solid #363d45; }
    .toggle-row { display: flex; align-items: flex-start; gap: 9px; cursor: pointer; }
    .toggle-row input { width: 15px; height: 15px; margin: 2px 0 0; accent-color: #3b82f6; flex: 0 0 auto; }
    .toggle-row span { display: flex; flex-direction: column; min-width: 0; color: #e6ebf1; line-height: 1.25; }
    .toggle-row strong { font-size: 11.5px; font-weight: 600; }
    .toggle-row small { margin-top: 2px; color: #8995a2; font-size: 10px; }
    .toggle-row.compact { margin: 2px 0; }
    .csv-card { padding: 10px; display: flex; flex-direction: column; gap: 10px; }
    .card-heading { display: flex; align-items: center; justify-content: space-between; color: #dbe3ec; font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: .04em; }
    .count-badge { padding: 3px 7px; border-radius: 999px; background: #303944; color: #9eb5cc; font-size: 9px; font-weight: 600; text-transform: none; letter-spacing: 0; }
    .file-picker { height: 36px; display: flex; align-items: center; overflow: hidden; border: 1px solid #474f59; border-radius: 7px; background: #1d2024; cursor: pointer; }
    .file-picker input { position: absolute; width: 1px; height: 1px; opacity: 0; pointer-events: none; }
    .file-button { align-self: stretch; display: flex; align-items: center; padding: 0 10px; background: #343a42; border-right: 1px solid #474f59; color: #e8edf3; font-size: 10.5px; font-weight: 600; white-space: nowrap; }
    .file-picker:hover .file-button { background: #3d4651; }
    .file-name { min-width: 0; padding: 0 9px; color: #909ba7; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .csv-grid { display: grid; grid-template-columns: minmax(0, 1fr) 82px; gap: 8px; }
    .import-summary { display: flex; flex-direction: column; gap: 2px; padding: 7px 9px; border-radius: 6px; background: #1c2521; border: 1px solid #30463a; color: #92c9a4; }
    .import-summary strong { font-size: 10.5px; }
    .import-summary span { color: #7f9b88; font-size: 9.5px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .apply-button { min-height: 34px; padding: 7px 10px; border: 1px solid #3c6ea8; border-radius: 7px; background: #285f9e; color: #fff; font: inherit; font-weight: 600; cursor: pointer; }
    .apply-button:hover:not(:disabled) { background: #3272bb; border-color: #5591d3; }
    .apply-button:disabled { opacity: .42; cursor: default; }
    .error { margin: 0; padding: 7px 9px; border: 1px solid #713d43; border-radius: 6px; background: #342326; color: #ff9da7; font-size: 10.5px; }
    .footer-hint { padding-top: 1px; }
  `],
})
export class IntensitySettingsComponent implements OnChanges, OnDestroy {
  @Input() instanceId = '';
  @Input() kind: 'intensity' | 'histogram' = 'intensity';
  @Input() mode: IntensityMode = 'per_image';
  @Input() labelsJson = '[]';
  @Input() chartEnabled = false;
  @Input() chartMetric = 'mean';
  @Input() percentiles = '5,25,50,75,95';
  @Output() modeChange = new EventEmitter<IntensityMode>();
  @Output() labelsChange = new EventEmitter<string[]>();
  @Output() chartEnabledChange = new EventEmitter<boolean>();
  @Output() chartMetricChange = new EventEmitter<string>();
  rows: string[][] = []; orientation: 'row' | 'column' = 'column'; position = 1; skipFirst = false;
  candidate: string[] = []; error = ''; private requestId = 0;
  fileName = '';
  get savedCount(): number { try { const labels = JSON.parse(this.labelsJson); return Array.isArray(labels) ? labels.length : 0; } catch { return 0; } }
  get candidatePreview(): string {
    if (!this.candidate.length) return '';
    const preview = this.candidate.slice(0, 6).join(', ');
    return this.candidate.length > 6 ? `${preview}, …` : preview;
  }
  get metricOptions(): { value: string; label: string }[] {
    const base = [
      { value: 'min', label: 'Minimum' }, { value: 'max', label: 'Maximum' },
      { value: 'mean', label: 'Átlag' }, { value: 'median', label: 'Medián' },
      { value: 'std', label: 'Szórás' }, { value: 'pixel_count', label: 'Pixelszám' },
      { value: 'dynamic_range', label: 'Dinamikus tartomány' },
    ];
    const percentiles = String(this.percentiles).split(',').map(value => Number(value.trim()))
      .filter(value => Number.isFinite(value) && value >= 0 && value <= 100)
      .map(value => ({ value: `p${value}`, label: `P${value}` }));
    const options = [...base, ...percentiles];
    if (!options.some(option => option.value === this.chartMetric)) {
      options.push({ value: this.chartMetric, label: this.chartMetric.toUpperCase() });
    }
    return options.filter((option, index) => options.findIndex(other => other.value === option.value) === index);
  }
  async readFile(event: Event): Promise<void> {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0]; input.value = '';
    if (!file) return;
    this.fileName = file.name;
    const request = ++this.requestId;
    this.error = ''; this.rows = []; this.candidate = [];
    try {
      if (file.size > 2_000_000) throw new Error('Legfeljebb 2 MB-os CSV tölthető be.');
      const text = await file.text();
      if (request !== this.requestId) return;
      this.rows = parseGroupCsv(text);
      this.orientation = this.rows.length === 1 ? 'row' : 'column'; this.position = 1; this.skipFirst = false;
      this.refresh();
    } catch (error) { if (request === this.requestId) this.error = error instanceof Error ? error.message : 'A CSV nem olvasható.'; }
  }
  refresh(): void {
    this.error = ''; this.candidate = [];
    try { this.candidate = selectGroupLabels(this.rows, this.orientation, this.position - 1, this.skipFirst); }
    catch (error) { this.error = error instanceof Error ? error.message : 'Érvénytelen csoportok.'; }
  }
  ngOnDestroy(): void { this.requestId++; }
  ngOnChanges(changes: SimpleChanges): void {
    if (changes['instanceId']) {
      this.requestId++; this.rows = []; this.candidate = []; this.error = '';
      this.fileName = '';
    }
  }
}
