import {
  Component,
  Input,
  ElementRef,
  ViewChild,
  AfterViewInit,
  OnChanges,
  SimpleChanges,
} from '@angular/core';

export interface CurveFitData {
  x_values: number[];
  y_values: number[];
  fitted_y: number[];
  coefficients: number[];
  model: string;
  degree: number;
  r2: number;
  x_name: string;
  y_name: string;
  point_colors?: (string | null)[];
}

@Component({
  selector: 'app-scatter-chart',
  standalone: true,
  template: `
    <div class="chart-container">
      @if (label) {
        <div class="chart-label">{{ label }}</div>
      }
      <canvas #canvas [width]="width" [height]="height"></canvas>
      @if (data) {
        <div class="fit-info">
          <div class="fit-formula">{{ getFormulaText() }}</div>
          <div class="fit-r2">R² = {{ data.r2.toFixed(6) }}</div>
        </div>
      }
    </div>
  `,
  styles: [`
    .chart-container {
      width: 100%;
    }

    .chart-label {
      font-size: 10px;
      color: #888;
      margin-bottom: 4px;
      text-transform: uppercase;
      letter-spacing: 0.03em;
    }

    canvas {
      width: 100%;
      height: auto;
      display: block;
      border-radius: 4px;
    }

    .fit-info {
      margin-top: 6px;
      padding: 6px 8px;
      background: #1e1e1e;
      border-radius: 4px;
      font-size: 11px;
      font-family: monospace;
    }

    .fit-formula {
      color: #ccc;
      word-break: break-all;
    }

    .fit-r2 {
      color: #888;
      margin-top: 2px;
    }
  `],
})
export class ScatterChartComponent implements AfterViewInit, OnChanges {
  @Input() data: CurveFitData | null = null;
  @Input() label = '';
  @Input() width = 400;
  @Input() height = 200;
  @Input() omittedIndices: Set<number> = new Set();

  @ViewChild('canvas') canvasRef!: ElementRef<HTMLCanvasElement>;

  private drawn = false;

  ngAfterViewInit(): void {
    this.draw();
    this.drawn = true;
  }

  ngOnChanges(changes: SimpleChanges): void {
    if (this.drawn) {
      this.draw();
    }
  }

  getFormulaText(): string {
    if (!this.data) return '';
    const c = this.data.coefficients;
    if (this.data.model === 'linear' || (this.data.model === 'poly' && this.data.degree === 1)) {
      return `y = ${c[0].toFixed(4)}x + ${c[1].toFixed(4)}`;
    }
    // Polynomial
    return c
      .map((coeff, i) => {
        const power = c.length - 1 - i;
        const val = coeff.toFixed(4);
        if (power === 0) return val;
        if (power === 1) return `${val}x`;
        return `${val}x^${power}`;
      })
      .join(' + ');
  }

  private draw(): void {
    const canvas = this.canvasRef?.nativeElement;
    if (!canvas || !this.data) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const d = this.data;
    const w = canvas.width;
    const h = canvas.height;
    const pad = { top: 12, right: 12, bottom: 26, left: 48 };
    const plotW = w - pad.left - pad.right;
    const plotH = h - pad.top - pad.bottom;

    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = '#1e1e1e';
    ctx.fillRect(0, 0, w, h);

    if (d.x_values.length === 0) return;

    const allY = [...d.y_values, ...d.fitted_y];
    const xMin = Math.min(...d.x_values);
    const xMax = Math.max(...d.x_values);
    const yMin = Math.min(...allY);
    const yMax = Math.max(...allY);
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

    // Fit line — sort by x for smooth drawing
    const sortedIndices = d.x_values
      .map((_, i) => i)
      .sort((a, b) => d.x_values[a] - d.x_values[b]);

    ctx.strokeStyle = '#6b8fad';
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < sortedIndices.length; i++) {
      const si = sortedIndices[i];
      const px = toX(d.x_values[si]);
      const py = toY(d.fitted_y[si]);
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.stroke();

    // Data points
    for (let i = 0; i < d.x_values.length; i++) {
      const px = toX(d.x_values[i]);
      const py = toY(d.y_values[i]);
      ctx.beginPath();
      ctx.arc(px, py, 3.5, 0, Math.PI * 2);
      const base = d.point_colors?.[i] || '#a0c4e8';
      if (this.omittedIndices.has(i)) {
        ctx.fillStyle = this.toAlpha(base, 0.2);
      } else {
        ctx.fillStyle = base;
      }
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

  private toAlpha(color: string, alpha: number): string {
    const m = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(color || '');
    if (!m) return `rgba(160,196,232,${alpha})`;
    const r = parseInt(m[1], 16);
    const g = parseInt(m[2], 16);
    const b = parseInt(m[3], 16);
    return `rgba(${r},${g},${b},${alpha})`;
  }
}
