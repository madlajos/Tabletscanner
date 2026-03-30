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
          <canvas #canvas class="pca-canvas"></canvas>
        </div>
        <div class="pca-stats">
          <div class="stat-item">
            <span class="stat-label">PC{{ pcX + 1 }} variancia:</span>
            <span class="stat-value">{{ (data.explained_ratio[pcX] * 100).toFixed(1) }}%</span>
          </div>
          <div class="stat-item">
            <span class="stat-label">PC{{ pcY + 1 }} variancia:</span>
            <span class="stat-value">{{ (data.explained_ratio[pcY] * 100).toFixed(1) }}%</span>
          </div>
          <div class="stat-item">
            <span class="stat-label">Kumulatív (PC{{ pcX + 1 }}+PC{{ pcY + 1 }}):</span>
            <span class="stat-value">{{ ((data.cumulative_ratio[pcX] + (data.cumulative_ratio[pcY] - (pcX > 0 ? data.cumulative_ratio[pcX - 1] : 0))) * 100).toFixed(1) }}%</span>
          </div>
          <div class="stat-item">
            <span class="stat-label">Minták száma:</span>
            <span class="stat-value">{{ data.scores.length }}</span>
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
      gap: 12px;
      padding: 12px;
      background: #f5f5f5;
      border-radius: 8px;
    }

    .controls-section {
      display: flex;
      gap: 16px;
      padding: 8px 12px;
      background: white;
      border-radius: 4px;
      align-items: center;
    }

    .control-group {
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .control-group label {
      font-size: 12px;
      font-weight: 600;
      color: #333;
      min-width: 60px;
    }

    .control-group select {
      padding: 6px 8px;
      border: 1px solid #ccc;
      border-radius: 4px;
      font-size: 12px;
      background: white;
      cursor: pointer;
    }

    .control-group select:focus {
      outline: none;
      border-color: #3b82f6;
      box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.1);
    }

    .chart-section {
      position: relative;
      width: 100%;
      height: 400px;
      background: white;
      border-radius: 6px;
      border: 1px solid #ddd;
      overflow: hidden;
    }

    .pca-canvas {
      width: 100%;
      height: 100%;
      display: block;
    }

    .pca-stats {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 12px;
      padding: 8px;
    }

    .stat-item {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 8px 12px;
      background: white;
      border-radius: 4px;
      border-left: 3px solid #6b8fad;
    }

    .stat-label {
      font-size: 12px;
      color: #666;
      font-weight: 500;
    }

    .stat-value {
      font-size: 14px;
      font-weight: bold;
      color: #333;
    }

    .no-data {
      padding: 40px;
      text-align: center;
      color: #999;
      background: white;
      border-radius: 4px;
    }
  `],
})
export class PCAChartComponent implements OnChanges, AfterViewInit {
  @Input() data: PCAData | null = null;
  @Output() componentChanged = new EventEmitter<{ pcX: number; pcY: number }>();
  @ViewChild('canvas', { read: ElementRef }) canvasRef?: ElementRef<HTMLCanvasElement>;

  pcX = 0;
  pcY = 1;

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

    const rect = canvas.parentElement?.getBoundingClientRect();
    if (!rect) return;

    canvas.width = rect.width;
    canvas.height = rect.height;

    this.drawChart(ctx, canvas);
  }

  private drawChart(ctx: CanvasRenderingContext2D, canvas: HTMLCanvasElement): void {
    if (!this.data || this.data.scores.length === 0) {
      return;
    }

    const width = canvas.width;
    const height = canvas.height;
    const padding = 50;

    // Clear canvas
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, width, height);

    // Extract selected PC scores
    const pc_x_scores = this.data.scores.map(row => row[this.pcX] ?? 0);
    const pc_y_scores = this.data.scores.map(row => row[this.pcY] ?? 0);

    // Calculate bounds
    const min_x = Math.min(...pc_x_scores);
    const max_x = Math.max(...pc_x_scores);
    const min_y = Math.min(...pc_y_scores);
    const max_y = Math.max(...pc_y_scores);

    const range_x = max_x - min_x || 1;
    const range_y = max_y - min_y || 1;

    const margin_x = range_x * 0.1;
    const margin_y = range_y * 0.1;

    const chart_width = width - 2 * padding;
    const chart_height = height - 2 * padding;

    // Draw grid
    ctx.strokeStyle = '#e0e0e0';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 5; i++) {
      const x = padding + (i / 5) * chart_width;
      const y = padding + (i / 5) * chart_height;
      ctx.beginPath();
      ctx.moveTo(x, padding);
      ctx.lineTo(x, height - padding);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(padding, y);
      ctx.lineTo(width - padding, y);
      ctx.stroke();
    }

    // Draw axes
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(padding, height - padding);
    ctx.lineTo(width - padding, height - padding);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(padding, padding);
    ctx.lineTo(padding, height - padding);
    ctx.stroke();

    // Draw axis labels
    ctx.fillStyle = '#333';
    ctx.font = '12px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';

    const pc_x_var = (this.data.explained_ratio[this.pcX] * 100).toFixed(1);
    ctx.fillText(`PC${this.pcX + 1} (${pc_x_var}%)`, width / 2, height - 20);

    ctx.save();
    ctx.translate(20, height / 2);
    ctx.rotate(-Math.PI / 2);
    const pc_y_var = (this.data.explained_ratio[this.pcY] * 100).toFixed(1);
    ctx.fillText(`PC${this.pcY + 1} (${pc_y_var}%)`, 0, 0);
    ctx.restore();

    // Draw data points
    ctx.fillStyle = '#6b8fad';
    for (let i = 0; i < this.data.scores.length; i++) {
      const x = padding + ((pc_x_scores[i] - min_x + margin_x) / (range_x + 2 * margin_x)) * chart_width;
      const y = height - padding - ((pc_y_scores[i] - min_y + margin_y) / (range_y + 2 * margin_y)) * chart_height;

      ctx.beginPath();
      ctx.arc(x, y, 3.5, 0, 2 * Math.PI);
      ctx.fill();
    }
  }
}
