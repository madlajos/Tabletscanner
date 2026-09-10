import { Component, Input } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { HistogramSummary } from '../../models/histogram.models';
import { PreviewMediaZoomDirective } from './preview-media-zoom.directive';

@Component({
  selector: 'app-histogram-preview', standalone: true, imports: [FormsModule, PreviewMediaZoomDirective],
  template: `
    <div class="split" (wheel)="$event.stopPropagation()" (mousedown)="$event.stopPropagation()" (dblclick)="$event.stopPropagation()">
      <section class="image" style="overflow: auto"><h3>Aktuális minta: {{ imageName }}</h3>
        @if (imageSrc) { <img appPreviewMediaZoom [src]="imageSrc" alt="A hisztogramszámítás aktuális mintája" /> }
        @else { <p>Nincs megjeleníthető kép.</p> }
      </section>
      <section class="results" tabindex="0" aria-label="Hisztogramok">
        <h3>{{ title }}</h3>
        @if (chartEnabled) {
          @if (visibleRows.length && chartMax > 0) {
            <div class="chart-frame">
              <svg appPreviewMediaZoom class="chart" viewBox="0 0 720 390" role="img" aria-label="Intenzitáshisztogram">
                @for (tick of yTicks; track tick.value) {
                  <line class="grid" x1="64" [attr.y1]="tick.y" x2="704" [attr.y2]="tick.y" />
                  <text class="tick" x="57" [attr.y]="tick.y + 4" text-anchor="end">{{ formatCount(tick.value) }}</text>
                }
                <line class="axis" x1="64" y1="18" x2="64" y2="320" />
                <line class="axis" x1="64" y1="320" x2="704" y2="320" />
                @for (tick of xTicks; track tick.value) {
                  <line class="x-tick" [attr.x1]="tick.x" y1="320" [attr.x2]="tick.x" y2="325" />
                  <text class="tick" [attr.x]="tick.x" y="340" text-anchor="middle">{{ tick.value }}</text>
                }
                @for (row of visibleRows; track row.label; let i = $index) {
                  <polyline class="hist-line" [attr.stroke]="seriesColor(i)" [attr.points]="histogramPoints(row.histogram)">
                    <title>{{ row.label }}</title>
                  </polyline>
                }
                @if (editingAxis === 'x') {
                  <foreignObject x="264" y="350" width="240" height="28"><input style="width:100%;box-sizing:border-box;background:#202020;color:#fff;text-align:center" [(ngModel)]="xAxisLabel" (blur)="finishAxisEdit()" (keydown.enter)="finishAxisEdit()" autofocus /></foreignObject>
                } @else {
                  <text class="axis-label editable-axis-label" x="384" y="372" text-anchor="middle" tabindex="0" (click)="startAxisEdit('x')" (keydown.enter)="startAxisEdit('x')">{{ xAxisLabel }}</text>
                }
                @if (editingAxis === 'y') {
                  <foreignObject x="-105" y="155" width="240" height="28" transform="rotate(-90 15 169)"><input style="width:100%;box-sizing:border-box;background:#202020;color:#fff;text-align:center" [(ngModel)]="yAxisLabel" (blur)="finishAxisEdit()" (keydown.enter)="finishAxisEdit()" autofocus /></foreignObject>
                } @else {
                  <text class="axis-label editable-axis-label" x="15" y="169" text-anchor="middle" transform="rotate(-90 15 169)" tabindex="0" (click)="startAxisEdit('y')" (keydown.enter)="startAxisEdit('y')">{{ yAxisLabel }}</text>
                }
              </svg>
            </div>
            <div class="legend">
              @for (row of visibleRows; track row.label; let i = $index) {
                <span [title]="row.label"><i [style.background]="seriesColor(i)"></i>{{ row.label }}</span>
              }
            </div>
          } @else { <p>Nincs megjeleníthető hisztogram.</p> }
        } @else { <p>A diagram megjelenítése ki van kapcsolva.</p> }
      </section>
    </div>
  `,
  styles: [`:host{display:block;width:100%;height:100%;min-height:0}.split{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:12px;padding:12px;box-sizing:border-box;height:100%;color:#eee}section{min-width:0;min-height:0;background:#202020;border:1px solid #444;border-radius:8px;padding:12px}.image{display:flex;flex-direction:column}.image img{width:100%;flex:1;min-height:0;object-fit:contain}.results{display:flex;flex-direction:column;overflow:auto}h3{font-size:15px;margin:0 0 12px}.chart-frame{min-height:260px;flex:1;overflow:auto;border:1px solid #39414a;border-radius:8px;background:#181b1f}.chart{display:block;width:100%;min-width:560px;height:100%;min-height:330px}.grid{stroke:#353b43;stroke-width:1}.axis,.x-tick{stroke:#78828e;stroke-width:1.2}.hist-line{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}.tick{fill:#929daa;font-size:11px}.axis-label{fill:#cfd7e1;font-size:12px;font-weight:600}.legend{display:flex;gap:7px 12px;flex-wrap:wrap;margin-top:10px;max-height:82px;overflow:auto}.legend span{display:flex;align-items:center;gap:5px;max-width:145px;color:#b7c0cb;font-size:10.5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.legend i{width:10px;height:10px;border-radius:50%;flex:0 0 auto}p{color:#9aa4af}@media(max-width:700px){.split{grid-template-columns:1fr;grid-template-rows:minmax(150px,1fr) 1fr}}`],
})
export class HistogramPreviewComponent {
  editingAxis: 'x' | 'y' | null = null;
  xAxisLabel = 'Intenzitás';
  yAxisLabel = 'Gyakoriság';
  @Input() imageSrc: string | null = null;
  @Input() imageName = '';
  @Input() imageIndex = 0;
  @Input() summary: HistogramSummary | null = null;
  @Input() chartEnabled = true;
  @Input() rangeMin = 0;
  @Input() rangeMax = 256;
  get title(): string {
    return this.summary?.mode === 'grouped' ? 'Csoportosított hisztogramok'
      : this.summary?.mode === 'pooled' ? 'Összes kép hisztogramja' : 'A minta hisztogramja';
  }
  get visibleRows(): { label: string; sampleCount: number; histogram: number[] }[] {
    if (!this.summary) return [];
    if (this.summary.mode === 'per_image') {
      return this.summary.samples.filter(row => row.image_index === this.imageIndex)
        .map(row => ({ label: `Kép ${row.label}`, sampleCount: 1, histogram: row.histogram }));
    }
    if (this.summary.mode === 'pooled') {
      return this.summary.samples.map(row => ({ label: `Kép ${row.label}`, sampleCount: 1, histogram: row.histogram }));
    }
    return this.summary.groups.map(row => ({ label: row.label, sampleCount: row.sample_count, histogram: row.histogram }));
  }
  get chartMax(): number { return Math.max(0, ...this.visibleRows.flatMap(row => row.histogram)); }
  get yTicks(): { value: number; y: number }[] {
    return Array.from({ length: 5 }, (_, index) => {
      const value = this.chartMax * (4 - index) / 4;
      return { value, y: 18 + index * 302 / 4 };
    });
  }
  get xTicks(): { value: string; x: number }[] {
    return Array.from({ length: 5 }, (_, index) => ({
      value: this.formatIntensity(this.rangeMin + (this.rangeMax - this.rangeMin) * index / 4),
      x: 64 + index * 640 / 4,
    }));
  }
  histogramPoints(histogram: number[]): string {
    if (!histogram.length || this.chartMax <= 0) return '';
    return histogram.map((value, index) => {
      const x = histogram.length === 1 ? 384 : 64 + index * 640 / (histogram.length - 1);
      const y = 320 - Math.max(0, Number(value) || 0) * 302 / this.chartMax;
      return `${x.toFixed(2)},${y.toFixed(2)}`;
    }).join(' ');
  }
  seriesColor(index: number): string {
    const palette = ['#60a5fa', '#34d399', '#f87171', '#fbbf24', '#c084fc', '#22d3ee', '#fb7185', '#a3e635'];
    return palette[index % palette.length];
  }
  formatCount(value: number): string { return value >= 1000000 ? `${(value / 1000000).toFixed(1)}M` : value >= 1000 ? `${(value / 1000).toFixed(1)}k` : String(Math.round(value)); }
  formatIntensity(value: number): string { return Number.isInteger(value) ? String(value) : value.toFixed(1); }
  startAxisEdit(axis: 'x' | 'y'): void { this.editingAxis = axis; }
  finishAxisEdit(): void { this.editingAxis = null; }
}
