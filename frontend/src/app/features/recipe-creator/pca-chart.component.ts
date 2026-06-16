import { Component, Input, Output, EventEmitter, OnChanges, SimpleChanges, ViewChild, ElementRef, AfterViewInit, ChangeDetectionStrategy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { MatIconModule } from '@angular/material/icon';

export interface PCAData {
  scores: number[][];
  explained_ratio: number[];
  cumulative_ratio: number[];
  n_samples?: number;
  n_features?: number;
}

@Component({
  selector: 'app-pca-chart',
  standalone: true,
  imports: [CommonModule, MatIconModule],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <div class="pca-chart-container">
      @if (data && data.scores.length > 0) {
        <div class="controls-section">
          <div class="control-group">
            <label>X tengely:</label>
            <select (change)="onPCChange('x', $event)" [value]="pcX">
              @for (i of getComponentOptions(); track i) {
                <option [value]="i">PC{{ i + 1 }} ({{ (data.explained_ratio[i] * 100).toFixed(1) }}%)</option>
              }
            </select>
          </div>
          <div class="control-group">
            <label>Y tengely:</label>
            <select (change)="onPCChange('y', $event)" [value]="pcY">
              @for (i of getComponentOptions(); track i) {
                <option [value]="i">PC{{ i + 1 }} ({{ (data.explained_ratio[i] * 100).toFixed(1) }}%)</option>
              }
            </select>
          </div>
        </div>
        <div class="chart-section">
          <canvas #canvas [width]="canvasWidth" [height]="canvasHeight"></canvas>
        </div>
        <div class="pca-info">
          <div class="pca-stats-row">
            <span class="pca-stat-label">PC{{ pcX + 1 }}:</span>
            <span class="pca-stat-value">{{ (data.explained_ratio[pcX] * 100).toFixed(1) }}%</span>
            <span class="pca-stat-label">PC{{ pcY + 1 }}:</span>
            <span class="pca-stat-value">{{ (data.explained_ratio[pcY] * 100).toFixed(1) }}%</span>
            <span class="pca-stat-label">Kumulatív:</span>
            <span class="pca-stat-value">{{ ((data.cumulative_ratio[pcX] + (data.cumulative_ratio[pcY] - (pcX > 0 ? data.cumulative_ratio[pcX - 1] : 0))) * 100).toFixed(1) }}%</span>
            <span class="pca-stat-label">n:</span>
            <span class="pca-stat-value">{{ data.scores.length }}</span>
          </div>
        </div>
      } @else {
        <div class="no-data">Nincs PCA adat. Futtasd a PCA-t!</div>
      }
    </div>
  `,
  styles: [`
    .pca-chart-container {
      display: flex;
      flex-direction: column;
      gap: 8px;
      width: 100%;
    }

    .controls-section {
      display: flex;
      gap: 16px;
      padding: 8px 12px;
      background: #2a2a2a;
      border-radius: 4px;
      align-items: center;
    }

    .control-group {
      display: flex;
      align-items: center;
      gap: 6px;
    }

    .control-group label {
      font-size: 11px;
      font-weight: 600;
      color: #888;
      min-width: 50px;
      text-transform: uppercase;
      letter-spacing: 0.02em;
    }

    .control-group select {
      padding: 4px 6px;
      border: 1px solid #444;
      border-radius: 3px;
      font-size: 11px;
      background: #1e1e1e;
      color: #bbb;
      cursor: pointer;
    }

    .control-group select:focus {
      outline: none;
      border-color: #6b8fad;
    }

    .chart-section {
      position: relative;
      width: 100%;
      height: 250px;
      background: #1e1e1e;
      border-radius: 4px;
      overflow: hidden;
    }

    canvas {
      width: 100%;
      height: 100%;
      display: block;
    }

    .pca-info {
      margin-top: 4px;
      padding: 6px 8px;
      background: #1e1e1e;
      border-radius: 4px;
      font-size: 10px;
      font-family: monospace;
    }

    .pca-stats-row {
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      align-items: center;
    }

    .pca-stat-label {
      color: #7f8792;
      font-weight: 600;
    }

    .pca-stat-value {
      color: #d2d9e3;
      font-weight: 500;
    }

    .no-data {
      padding: 40px 20px;
      text-align: center;
      color: #666;
      background: #1e1e1e;
      border-radius: 4px;
      font-size: 12px;
    }
  `],
})
export class PCAChartComponent implements OnChanges, AfterViewInit {
  @Input() data: PCAData | null = null;
  @Output() componentChanged = new EventEmitter<{ pcX: number; pcY: number }>();
  @ViewChild('canvas', { read: ElementRef }) canvasRef?: ElementRef<HTMLCanvasElement>;

  pcX = 0;
  pcY = 1;
  canvasWidth = 400;
  canvasHeight = 350;

  getComponentOptions(): number[] {
    if (!this.data || !this.data.scores[0]) return [];
    const nComponents = this.data.scores[0].length;
    return Array.from({ length: Math.min(nComponents, 10) }, (_, i) => i);
  }

  onPCChange(axis: 'x' | 'y', event: Event): void {
    const value = parseInt((event.target as HTMLSelectElement).value);
    if (axis === 'x') {
      this.pcX = value;
    } else {
      this.pcY = value;
    }
    this.componentChanged.emit({ pcX: this.pcX, pcY: this.pcY });
    this.redraw();
  }

  ngAfterViewInit(): void {
    this.redraw();
  }

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['data']) {
      this.redraw();
    }
  }

  private redraw(): void {
    if (!this.canvasRef) return;

    const canvas = this.canvasRef.nativeElement;
    const ctx = canvas.getContext('2d');
    if (!ctx || !this.data) return;

    this.drawChart(ctx, canvas);
  }

  private drawChart(ctx: CanvasRenderingContext2D, canvas: HTMLCanvasElement): void {
    if (!this.data || this.data.scores.length === 0) {
      return;
    }

    const w = canvas.width;
    const h = canvas.height;
    const pad = { top: 12, right: 12, bottom: 26, left: 48 };
    const plotW = w - pad.left - pad.right;
    const plotH = h - pad.top - pad.bottom;

    // Clear and fill background
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = '#1e1e1e';
    ctx.fillRect(0, 0, w, h);

    // Extract scores
    const pc_x_scores = this.data.scores.map(row => row[this.pcX] ?? 0);
    const pc_y_scores = this.data.scores.map(row => row[this.pcY] ?? 0);

    // Calculate bounds
    const xMin = Math.min(...pc_x_scores);
    const xMax = Math.max(...pc_x_scores);
    const yMin = Math.min(...pc_y_scores);
    const yMax = Math.max(...pc_y_scores);
    const xRange = xMax - xMin || 1;
    const yRange = yMax - yMin || 1;
    const xPad = xRange * 0.05;
    const yPad = yRange * 0.05;

    const toX = (v: number) => pad.left + ((v - xMin + xPad) / (xRange + 2 * xPad)) * plotW;
    const toY = (v: number) => pad.top + plotH - ((v - yMin + yPad) / (yRange + 2 * yPad)) * plotH;

    // Grid
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= 4; i++) {
      const y = pad.top + (plotH / 4) * i;
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(pad.left + plotW, y);
      ctx.stroke();
    }

    // Axes
    ctx.strokeStyle = '#555';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad.left, pad.top);
    ctx.lineTo(pad.left, pad.top + plotH);
    ctx.lineTo(pad.left + plotW, pad.top + plotH);
    ctx.stroke();

    // Data points (dots)
    ctx.fillStyle = '#6b8fad';
    for (let i = 0; i < this.data.scores.length; i++) {
      const px = toX(pc_x_scores[i]);
      const py = toY(pc_y_scores[i]);
      ctx.beginPath();
      ctx.arc(px, py, 3.5, 0, Math.PI * 2);
      ctx.fill();
    }

    // Axis labels
    ctx.fillStyle = '#777';
    ctx.font = '9px sans-serif';
    ctx.textAlign = 'center';
    const xLabels = 5;
    for (let i = 0; i <= xLabels; i++) {
      const val = xMin + (xRange / xLabels) * i;
      const x = toX(val);
      ctx.fillText(val.toFixed(1), x, pad.top + plotH + 14);
    }

    ctx.textAlign = 'right';
    for (let i = 0; i <= 4; i++) {
      const val = yMax - ((yMax - yMin) / 4) * i;
      const y = pad.top + (plotH / 4) * i;
      ctx.fillText(val.toFixed(1), pad.left - 4, y + 3);
    }
  }
}
