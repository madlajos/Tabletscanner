import { CommonModule, DecimalPipe } from '@angular/common';
import { Component, EventEmitter, Input, Output } from '@angular/core';
import { PreviewMediaZoomDirective } from './preview-media-zoom.directive';

export interface CharacterizedParticle {
  particle_id: string;
  label: number | null;
  image_index: number;
  value: number;
  polygon: number[][];
  excluded: boolean;
}

interface DistributionGroup {
  label: string;
  particle_count: number;
  particle_values: { particle_id: string | null; label: number | null; value: number }[];
  unit: string;
  bin_edges: number[];
  number_percent: number[];
  volume_percent: number[];
  min: number;
  max: number;
  mean: number;
  median: number;
  std: number;
  dv10: number | null;
  dv50: number | null;
  dv90: number | null;
}

@Component({
  selector: 'app-particle-size-distribution-preview',
  standalone: true,
  imports: [CommonModule, DecimalPipe, PreviewMediaZoomDirective],
  template: `
    <div class="distribution-preview">
      <section class="image-panel">
        <h3>{{ montage ? 'Szemcsedetektálás – összes kép' : 'Szemcsedetektálás' }}</h3>
        @if (imageSrc) {
          <svg appPreviewMediaZoom class="particle-image" [attr.viewBox]="'0 0 ' + imageWidth + ' ' + imageHeight" role="img" aria-label="Detektált szemcsék előnézete">
            <image [attr.href]="imageSrc" [attr.width]="imageWidth" [attr.height]="imageHeight" />
            @if (selectedParticle; as particle) {
              <polygon [attr.points]="polygonPoints(particle)" fill="#ffff0040" stroke="#ffff00" stroke-width="3" vector-effect="non-scaling-stroke" />
            }
          </svg>
        } @else {
          <div class="empty">Nincs megjeleníthető kép.</div>
        }
      </section>

      <section class="charts-panel">
        <div class="heading">
          <h3>{{ statisticsOnly ? 'Szemcsénkénti értékek' : 'Szemcseméret-eloszlás' }}</h3>
          <span>{{ distribution?.size_label || '' }}</span>
        </div>
        @if (groups.length || statisticsOnly) {
          @if (statisticsOnly) {
            <div class="statistics-only">
                <div class="statistics-title">{{ particles.length }} szemcse</div>
                <div class="particle-table-scroll" tabindex="0" aria-label="Szemcsénkénti értékek">
                  <table>
                    <thead><tr>
                      <th scope="col" [attr.aria-sort]="sortColumn === 'label' ? sortDirection : 'none'"><button type="button" class="sort-button" (click)="sortBy('label')">Szemcse {{ sortColumn === 'label' ? (sortDirection === 'ascending' ? '↑' : '↓') : '↕' }}</button></th>
                      <th scope="col" [attr.aria-sort]="sortColumn === 'value' ? sortDirection : 'none'"><button type="button" class="sort-button" (click)="sortBy('value')">{{ metricLabel }} ({{ distribution?.unit || unit }}) {{ sortColumn === 'value' ? (sortDirection === 'ascending' ? '↑' : '↓') : '↕' }}</button></th>
                      <th scope="col">Elemzés</th>
                    </tr></thead>
                    <tbody>
                      @for (particle of sortedParticles; track particle.particle_id) {
                        <tr tabindex="0" [class.selected]="selectedId === particle.particle_id" [class.excluded]="particle.excluded"
                            (click)="selectedId = particle.particle_id" (keydown.enter)="selectedId = particle.particle_id"
                            (keydown.space)="$event.preventDefault(); selectedId = particle.particle_id">
                          <td>{{ particle.label ?? particle.particle_id }}</td><td>{{ particle.value | number:'1.0-4' }}</td>
                          <td><button type="button" (click)="$event.stopPropagation(); toggleExcluded.emit(particle.particle_id)">{{ particle.excluded ? 'Visszavétel' : 'Kizárás' }}</button></td>
                        </tr>
                      }
                    </tbody>
                  </table>
                </div>
                @if (!particles.length) {
                  <div class="statistics-title">{{ distribution?.particles == null ? 'A szemcseadatok hiányoznak. Indítsa újra a backendet, majd frissítse az előnézetet.' : 'Nincs megjeleníthető szemcse.' }}</div>
                }
              @if (resizeCorrection?.applied) {
                <div class="correction-note">✓ Átméretezés korrigálva az eredeti méretre (X: {{ resizeCorrection.scale_x | number:'1.0-4' }}×, Y: {{ resizeCorrection.scale_y | number:'1.0-4' }}×)</div>
              }
            </div>
          } @else {
          <div class="chart-card">
            <div class="chart-title">Szám szerinti eloszlás (%)</div>
            <svg appPreviewMediaZoom viewBox="0 0 620 220" role="img" aria-label="Szám szerinti szemcseméret-eloszlás">
              <path class="grid" d="M50 15V185H600 M50 142.5H600 M50 100H600 M50 57.5H600" />
              @for (group of groups; track group.label; let i = $index) {
                <polyline [attr.points]="seriesPoints(group.number_percent)" [attr.stroke]="color(i)" />
              }
              <text x="50" y="205">{{ rangeMin | number:'1.0-2' }}</text>
              <text x="600" y="205" text-anchor="end">{{ rangeMax | number:'1.0-2' }} {{ unit }}</text>
              <text x="44" y="20" text-anchor="end">{{ numberMax | number:'1.0-1' }}%</text>
              <text x="44" y="188" text-anchor="end">0%</text>
              <text class="editable-axis-label" x="325" y="218" text-anchor="middle" tabindex="0" (click)="editAxisLabel('x')" (keydown.enter)="editAxisLabel('x')">{{ xAxisLabel }}</text>
              <text class="editable-axis-label" x="12" y="100" text-anchor="middle" transform="rotate(-90 12 100)" tabindex="0" (click)="editAxisLabel('numberY')" (keydown.enter)="editAxisLabel('numberY')">{{ numberYAxisLabel }}</text>
            </svg>
          </div>

          <div class="chart-card">
            <div class="chart-title">Térfogat szerinti eloszlás (%)</div>
            <svg appPreviewMediaZoom viewBox="0 0 620 220" role="img" aria-label="Térfogat szerinti szemcseméret-eloszlás">
              <path class="grid" d="M50 15V185H600 M50 142.5H600 M50 100H600 M50 57.5H600" />
              @for (group of groups; track group.label; let i = $index) {
                <polyline [attr.points]="seriesPoints(group.volume_percent, true)" [attr.stroke]="color(i)" />
              }
              <text x="50" y="205">{{ rangeMin | number:'1.0-2' }}</text>
              <text x="600" y="205" text-anchor="end">{{ rangeMax | number:'1.0-2' }} {{ unit }}</text>
              <text x="44" y="20" text-anchor="end">{{ volumeMax | number:'1.0-1' }}%</text>
              <text x="44" y="188" text-anchor="end">0%</text>
              <text class="editable-axis-label" x="325" y="218" text-anchor="middle" tabindex="0" (click)="editAxisLabel('x')" (keydown.enter)="editAxisLabel('x')">{{ xAxisLabel }}</text>
              <text class="editable-axis-label" x="12" y="100" text-anchor="middle" transform="rotate(-90 12 100)" tabindex="0" (click)="editAxisLabel('volumeY')" (keydown.enter)="editAxisLabel('volumeY')">{{ volumeYAxisLabel }}</text>
            </svg>
          </div>

          <div class="legend">
            @for (group of groups; track group.label; let i = $index) {
              <div class="legend-row">
                <span class="swatch" [style.background]="color(i)"></span>
                <strong>{{ group.label }}</strong>
                <span>n={{ group.particle_count }}</span>
                <span>Dv10: {{ group.dv10 | number:'1.0-2' }} {{ group.unit }}</span>
                <span>Dv50: {{ group.dv50 | number:'1.0-2' }} {{ group.unit }}</span>
                <span>Dv90: {{ group.dv90 | number:'1.0-2' }} {{ group.unit }}</span>
              </div>
            }
          </div>
          @if (resizeCorrection?.applied) {
            <div class="correction-note compact">✓ Átméretezés korrigálva az eredeti méretre (X: {{ resizeCorrection.scale_x | number:'1.0-4' }}×, Y: {{ resizeCorrection.scale_y | number:'1.0-4' }}×)</div>
          }
          }
        } @else {
          <div class="empty">Nincs eloszlás számítására alkalmas szemcse.</div>
        }
      </section>
    </div>
  `,
  styles: [`
    :host { display: block; width: 100%; height: 100%; min-height: 0; }
    .distribution-preview { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 12px; width: 100%; height: 100%; padding: 12px; box-sizing: border-box; }
    section { min-width: 0; min-height: 0; border: 1px solid #3b3b3b; border-radius: 8px; background: #202020; overflow: hidden; }
    h3 { margin: 0; color: #ddd; font-size: 13px; font-weight: 600; }
    .image-panel { display: flex; flex-direction: column; }
    .image-panel h3, .heading { padding: 10px 12px; border-bottom: 1px solid #383838; }
    .image-panel img { width: 100%; height: calc(100% - 38px); object-fit: contain; min-height: 0; }
    .particle-image { flex: 1; min-height: 0; }
    tr.selected { background: #555000; }
    tr.excluded td:first-child { text-decoration: line-through; opacity: .6; }
    tr:focus-visible, button:focus-visible { outline: 2px solid #ffff00; outline-offset: -2px; }
    tbody tr, button { cursor: pointer; }
    button { background: #333; color: #eee; border: 1px solid #666; border-radius: 4px; padding: 5px 8px; }
    .charts-panel { display: grid; grid-template-rows: auto minmax(0, 1fr) minmax(0, 1fr) auto auto; overflow: hidden; padding-bottom: 6px; }
    .heading { display: flex; justify-content: space-between; gap: 8px; color: #999; font-size: 11px; }
    .chart-card { display: flex; flex-direction: column; min-height: 0; margin: 6px 8px 0; padding: 5px 7px; background: #191919; border-radius: 6px; }
    .chart-title { color: #bbb; font-size: 11px; margin-bottom: 4px; }
    svg { display: block; width: 100%; height: 100%; min-height: 0; }
    svg polyline { fill: none; stroke-width: 2; vector-effect: non-scaling-stroke; }
    svg text { fill: #888; font: 10px sans-serif; }
    .grid { fill: none; stroke: #3a3a3a; stroke-width: 1; vector-effect: non-scaling-stroke; }
    .legend { display: flex; flex-wrap: wrap; gap: 3px 10px; max-height: 52px; overflow: hidden; padding: 5px 9px 0; }
    .legend-row { display: flex; flex-wrap: nowrap; align-items: center; gap: 5px; color: #aaa; font-size: 9px; white-space: nowrap; }
    .legend-row strong { color: #ddd; }
    .swatch { width: 10px; height: 10px; border-radius: 2px; }
    .empty { display: grid; place-items: center; min-height: 180px; color: #888; }
    .statistics-only { grid-row: 2 / -1; display: flex; flex-direction: column; min-height: 0; padding: 18px; overflow: auto; }
    .statistics-title { margin-bottom: 14px; color: #ccc; font-size: 13px; }
    .particle-table-scroll { overflow: auto; flex-shrink: 0; margin-bottom: 12px; }
    .statistics-only, .particle-table-scroll { scrollbar-width: thin; scrollbar-color: #444 #1a1a1a; color-scheme: dark; }
    .statistics-only::-webkit-scrollbar, .particle-table-scroll::-webkit-scrollbar { width: 10px; height: 10px; }
    .statistics-only::-webkit-scrollbar-track, .particle-table-scroll::-webkit-scrollbar-track { background: #1a1a1a; border-radius: 8px; }
    .statistics-only::-webkit-scrollbar-thumb, .particle-table-scroll::-webkit-scrollbar-thumb { background: #444; border: 2px solid #1a1a1a; border-radius: 8px; }
    .statistics-only::-webkit-scrollbar-thumb:hover, .particle-table-scroll::-webkit-scrollbar-thumb:hover { background: #5a5a5a; }
    .statistics-only::-webkit-scrollbar-corner, .particle-table-scroll::-webkit-scrollbar-corner { background: #1a1a1a; }
    .particle-table-scroll:focus-visible { outline: 2px solid #42a5f5; outline-offset: 2px; }
    table { width: 100%; border-collapse: collapse; color: #ddd; font-size: 12px; }
    th, td { padding: 8px; border-bottom: 1px solid #3a3a3a; text-align: right; }
    th:first-child, td:first-child { text-align: left; }
    th { background: #191919; }
    .sort-button { background: transparent; border: 0; padding: 0; font: inherit; color: inherit; text-align: inherit; }
    .correction-note { margin-top: 12px; color: #81c784; font-size: 10px; }
    .correction-note.compact { margin: 3px 9px 0; }
  `],
})
export class ParticleSizeDistributionPreviewComponent {
  xAxisLabel = 'Szemcseméret';
  numberYAxisLabel = 'Szám szerinti eloszlás (%)';
  volumeYAxisLabel = 'Térfogat szerinti eloszlás (%)';
  @Input() imageSrc: string | null = null;
  @Input() distribution: any = null;
  @Input() montage = false;
  @Input() imageWidth = 100;
  @Input() imageHeight = 100;
  @Output() toggleExcluded = new EventEmitter<string>();
  selectedId: string | null = null;
  sortColumn: 'label' | 'value' = 'label';
  sortDirection: 'ascending' | 'descending' = 'ascending';

  sortBy(column: 'label' | 'value'): void {
    this.sortDirection = this.sortColumn === column && this.sortDirection === 'ascending' ? 'descending' : 'ascending';
    this.sortColumn = column;
  }

  get sortedParticles(): CharacterizedParticle[] {
    const direction = this.sortDirection === 'ascending' ? 1 : -1;
    return [...this.particles].sort((a, b) => {
      const comparison = this.sortColumn === 'value'
        ? a.value - b.value
        : String(a.label ?? a.particle_id).localeCompare(String(b.label ?? b.particle_id), 'hu', { numeric: true });
      return direction * comparison;
    });
  }

  get particles(): CharacterizedParticle[] { return this.distribution?.particles ?? []; }
  get selectedParticle(): CharacterizedParticle | undefined { return this.particles.find(p => p.particle_id === this.selectedId); }
  get metricLabel(): string { return (this.distribution?.size_label ?? '').replace(/\s*\([^)]*\)$/, ''); }
  polygonPoints(particle: CharacterizedParticle): string { return particle.polygon.map(p => p.join(',')).join(' '); }

  private readonly colors = ['#42a5f5', '#ef5350', '#66bb6a', '#ffa726', '#ab47bc', '#26c6da', '#ec407a', '#d4e157'];

  get groups(): DistributionGroup[] { return Array.isArray(this.distribution?.groups) ? this.distribution.groups : []; }
  get statisticsOnly(): boolean { return this.distribution?.mode === 'per_image'; }
  get resizeCorrection(): any { return this.distribution?.resize_correction ?? null; }
  get rangeMin(): number { return Number(this.groups[0]?.bin_edges?.[0] ?? 0); }
  get rangeMax(): number { const edges = this.groups[0]?.bin_edges ?? []; return Number(edges[edges.length - 1] ?? 1); }
  get unit(): string { return this.groups[0]?.unit ?? ''; }
  get numberMax(): number { return this.maxFor('number_percent'); }
  get volumeMax(): number { return this.maxFor('volume_percent'); }

  color(index: number): string { return this.colors[index % this.colors.length]; }

  seriesPoints(values: number[], volume = false): string {
    if (!Array.isArray(values) || !values.length) return '';
    const max = volume ? this.volumeMax : this.numberMax;
    return values.map((value, index) => {
      const x = 50 + ((index + 0.5) / values.length) * 550;
      const y = 185 - (Math.max(0, Number(value)) / Math.max(max, 1e-9)) * 170;
      return `${x.toFixed(2)},${y.toFixed(2)}`;
    }).join(' ');
  }

  private maxFor(key: 'number_percent' | 'volume_percent'): number {
    const values = this.groups.flatMap((group) => Array.isArray(group[key]) ? group[key] : []);
    return Math.max(...values.map(Number).filter(Number.isFinite), 1);
  }

  editAxisLabel(axis: 'x' | 'numberY' | 'volumeY'): void {
    const current = axis === 'x' ? this.xAxisLabel : axis === 'numberY' ? this.numberYAxisLabel : this.volumeYAxisLabel;
    const edited = window.prompt('Tengely címe:', current)?.trim();
    if (!edited) return;
    if (axis === 'x') this.xAxisLabel = edited;
    else if (axis === 'numberY') this.numberYAxisLabel = edited;
    else this.volumeYAxisLabel = edited;
  }
}
