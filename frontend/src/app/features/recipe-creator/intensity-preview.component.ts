import { Component, Input } from '@angular/core';
import { DecimalPipe } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { IntensitySummary } from '../../models/intensity.models';
import { PreviewMediaZoomDirective } from './preview-media-zoom.directive';

@Component({
  selector: 'app-intensity-preview', standalone: true, imports: [DecimalPipe, FormsModule, PreviewMediaZoomDirective],
  template: `
    <div class="split" (wheel)="$event.stopPropagation()" (mousedown)="$event.stopPropagation()" (dblclick)="$event.stopPropagation()">
      <section class="image" style="overflow: auto"><h3>Aktuális minta: {{ imageName }}</h3>
        @if (imageSrc) { <img appPreviewMediaZoom [src]="imageSrc" alt="Az intenzitásmérés aktuális mintája" /> }
        @else { <p>Nincs megjeleníthető kép.</p> }
      </section>
      <section class="results" tabindex="0" aria-label="Intenzitásstatisztikák">
        <h3>{{ summary?.mode === 'pooled' ? 'Összes minta összevonva' : summary?.mode === 'grouped' ? 'Csoportosított intenzitásstatisztikák' : 'A minta intenzitásstatisztikái' }}</h3>
        @if (chartEnabled) {
          @if (chartHasValues) {
            <div class="chart-title">{{ label(chartMetric) }}</div>
            <svg appPreviewMediaZoom class="chart" [attr.viewBox]="'0 0 ' + chartWidth + ' 360'" [style.min-width.px]="chartWidth" role="img" [attr.aria-label]="label(chartMetric) + ' diagram'">
              <line class="axis" x1="72" y1="20" x2="72" y2="290" />
              <line class="axis" x1="72" y1="290" [attr.x2]="chartRight" y2="290" />
              @for (tick of yTicks; track tick) {
                <line class="grid" x1="72" [attr.y1]="chartY(tick)" [attr.x2]="chartRight" [attr.y2]="chartY(tick)" />
                <text class="tick" x="65" [attr.y]="chartY(tick) + 4" text-anchor="end">{{ formatNumber(tick) }}</text>
              }
              @for (series of chartSeries; track series.channel; let seriesIndex = $index) {
                <polyline class="series" [attr.stroke]="seriesColor(seriesIndex)" [attr.points]="series.points" />
                @for (point of series.values; track point.index) {
                  <circle [attr.cx]="chartX(point.index)" [attr.cy]="chartY(point.value)" r="4" [attr.fill]="seriesColor(seriesIndex)">
                    <title>{{ chartRows[point.index].label }}: {{ formatNumber(point.value) }}</title>
                  </circle>
                }
              }
              @for (row of chartRows; track $index; let i = $index) {
                <text class="x-label" [attr.x]="chartX(i)" y="310" text-anchor="end" [attr.transform]="'rotate(-35 ' + chartX(i) + ' 310)'">{{ row.label }}</text>
              }
              @if (editingAxis === 'x') {
                <foreignObject [attr.x]="(72 + chartRight) / 2 - 120" y="332" width="240" height="28">
                  <input style="width:100%;box-sizing:border-box;background:#202020;color:#fff;text-align:center" [(ngModel)]="customXAxisLabel" (blur)="finishAxisEdit()" (keydown.enter)="finishAxisEdit()" autofocus />
                </foreignObject>
              } @else {
                <text class="axis-label editable-axis-label" [attr.x]="(72 + chartRight) / 2" y="354" text-anchor="middle" tabindex="0" (click)="startAxisEdit('x')" (keydown.enter)="startAxisEdit('x')">{{ xAxisLabel }}</text>
              }
              @if (editingAxis === 'y') {
                <foreignObject x="-104" y="142" width="240" height="28" transform="rotate(-90 16 155)">
                  <input style="width:100%;box-sizing:border-box;background:#202020;color:#fff;text-align:center" [(ngModel)]="customYAxisLabel" (blur)="finishAxisEdit()" (keydown.enter)="finishAxisEdit()" autofocus />
                </foreignObject>
              } @else {
                <text class="axis-label editable-axis-label" x="16" y="155" text-anchor="middle" transform="rotate(-90 16 155)" tabindex="0" (click)="startAxisEdit('y')" (keydown.enter)="startAxisEdit('y')">{{ yAxisLabel }}</text>
              }
            </svg>
            @if (chartSeries.length > 1) {
              <div class="legend">
                @for (series of chartSeries; track series.channel; let i = $index) {
                  <span><i [style.background]="seriesColor(i)"></i>{{ channelName(series.channel, chartSeries.length) }}</span>
                }
              </div>
            }
          } @else { <p>A kiválasztott paraméterhez nincs ábrázolható érték.</p> }
        } @else {
          @for (group of visibleGroups; track $index) {
          <h4>{{ group.label }} · {{ group.sample_count }} minta</h4>
          @for (stat of group.channels; track $index; let channel = $index) {
            <h4>{{ group.channels.length === 1 ? 'Intenzitás' : channelName(channel, group.channels.length) }}</h4>
            @if (stat) {
              <table><thead><tr><th>Jellemző</th><th>Érték</th></tr></thead><tbody>
                @for (key of keys(stat); track key) {
                  <tr><th scope="row">{{ label(key) }}</th><td>{{ stat[key] | number:'1.0-4' }}</td></tr>
                }
              </tbody></table>
            } @else { <p>A maszk nem tartalmaz mérhető pixelt.</p> }
          }
          } @empty { <p>Nincs statisztika. Frissítse az előnézetet.</p> }
        }
      </section>
    </div>
  `,
  styles: [`:host{display:block;width:100%;height:100%;min-height:0}.split{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:12px;padding:12px;box-sizing:border-box;height:100%;color:#eee}section{min-width:0;min-height:0;background:#202020;border:1px solid #444;border-radius:8px;padding:12px}.image{display:flex;flex-direction:column}.image img{width:100%;flex:1;min-height:0;object-fit:contain}.results{overflow:auto}h3{font-size:15px;margin:0 0 12px}h4{font-size:13px;overflow-wrap:anywhere}table{width:100%;border-collapse:collapse;font-size:13px}th,td{padding:6px;border-bottom:1px solid #444;text-align:left}td{text-align:right;font-variant-numeric:tabular-nums}p{color:#bbb}.chart{display:block;width:100%;min-width:460px;max-height:calc(100% - 55px);overflow:visible}.axis{stroke:#aaa;stroke-width:1.5}.grid{stroke:#444;stroke-width:1}.series{fill:none;stroke-width:2}.tick,.x-label{fill:#bbb;font-size:11px}.axis-label,.chart-title{fill:#eee;color:#eee;font-size:13px;font-weight:600}.legend{display:flex;gap:12px;flex-wrap:wrap;font-size:12px}.legend span{display:flex;align-items:center;gap:5px}.legend i{width:10px;height:10px;border-radius:50%}:focus-visible{outline:2px solid #82b1ff}@media(max-width:700px){.split{grid-template-columns:1fr;grid-template-rows:minmax(150px,1fr) 1fr}}`],
})
export class IntensityPreviewComponent {
  editingAxis: 'x' | 'y' | null = null;
  customXAxisLabel: string | null = null;
  customYAxisLabel: string | null = null;
  @Input() imageSrc: string | null = null;
  @Input() imageName = '';
  @Input() imageIndex = 0;
  @Input() summary: IntensitySummary | null = null;
  @Input() chartEnabled = false;
  @Input() chartMetric = 'mean';
  get xAxisLabel(): string { return this.customXAxisLabel ?? (this.summary?.mode === 'grouped' ? 'Csoport' : 'Kép sorszáma'); }
  get yAxisLabel(): string { return this.customYAxisLabel ?? this.label(this.chartMetric); }
  get visibleGroups() {
    const groups = this.summary?.groups ?? [];
    return this.summary?.mode === 'per_image' ? groups.filter(group => group.image_indices.includes(this.imageIndex)) : groups;
  }
  channelName(index: number, count: number): string { return (count === 3 || count === 4) ? ['B', 'G', 'R', 'A'][index] : `Csatorna ${index + 1}`; }
  keys(stat: Record<string, number | null>): string[] { return Object.keys(stat).filter(key => stat[key] !== null); }
  label(key: string): string { return ({ min: 'Minimum', max: 'Maximum', mean: 'Átlag', median: 'Medián', std: 'Szórás', pixel_count: 'Pixelszám', dynamic_range: 'Dinamikus tartomány' } as Record<string, string>)[key] ?? key.toUpperCase(); }
  get chartRows(): { label: string; channels: (Record<string, number | null> | null)[] }[] {
    if (!this.summary) return [];
    return this.summary.mode === 'grouped' ? this.summary.groups : (this.summary.samples ?? []);
  }
  get chartSeries(): { channel: number; values: { index: number; value: number }[]; points: string }[] {
    const channelCount = Math.max(0, ...this.chartRows.map(row => row.channels.length));
    return Array.from({ length: channelCount }, (_, channel) => {
      const values = this.chartRows.map((row, index) => ({ index, value: Number(row.channels[channel]?.[this.chartMetric]) }))
        .filter(point => Number.isFinite(point.value));
      return { channel, values, points: values.map(point => `${this.chartX(point.index)},${this.chartY(point.value)}`).join(' ') };
    }).filter(series => series.values.length);
  }
  get chartHasValues(): boolean { return this.chartSeries.length > 0; }
  private get chartBounds(): { min: number; max: number } {
    const values = this.chartRows.flatMap(row => row.channels.map(channel => Number(channel?.[this.chartMetric])))
      .filter(value => Number.isFinite(value));
    if (!values.length) return { min: 0, max: 1 };
    let min = Math.min(...values), max = Math.max(...values);
    if (min === max) { const padding = Math.abs(min) * .1 || 1; min -= padding; max += padding; }
    else { const padding = (max - min) * .08; min -= padding; max += padding; }
    return { min, max };
  }
  get yTicks(): number[] { const { min, max } = this.chartBounds; return Array.from({ length: 5 }, (_, i) => min + (max - min) * i / 4); }
  get chartWidth(): number { return Math.max(640, 92 + Math.max(0, this.chartRows.length - 1) * 60); }
  get chartRight(): number { return this.chartWidth - 20; }
  chartX(index: number): number { return this.chartRows.length <= 1 ? (72 + this.chartRight) / 2 : 72 + index * (this.chartRight - 72) / (this.chartRows.length - 1); }
  chartY(value: number): number { const { min, max } = this.chartBounds; return 290 - (value - min) * 270 / (max - min); }
  seriesColor(index: number): string { return ['#60a5fa', '#34d399', '#f87171', '#fbbf24', '#c084fc'][index % 5]; }
  formatNumber(value: number): string { return Number.isInteger(value) ? String(value) : value.toFixed(Math.abs(value) < 10 ? 3 : 2); }
  startAxisEdit(axis: 'x' | 'y'): void {
    if (axis === 'x' && this.customXAxisLabel === null) this.customXAxisLabel = this.xAxisLabel;
    if (axis === 'y' && this.customYAxisLabel === null) this.customYAxisLabel = this.yAxisLabel;
    this.editingAxis = axis;
  }
  finishAxisEdit(): void {
    if (!this.customXAxisLabel?.trim()) this.customXAxisLabel = null;
    if (!this.customYAxisLabel?.trim()) this.customYAxisLabel = null;
    this.editingAxis = null;
  }
}
