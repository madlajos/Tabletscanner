import {
  Component,
  Input,
  ElementRef,
  ViewChild,
  AfterViewInit,
  OnChanges,
  SimpleChanges,
} from '@angular/core';

@Component({
  selector: 'app-histogram-chart',
  standalone: true,
  template: `
    <div class="chart-container">
      @if (label) {
        <div class="chart-label">{{ label }}</div>
      }
      <canvas #canvas [width]="width" [height]="height"></canvas>
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
  `],
})
export class HistogramChartComponent implements AfterViewInit, OnChanges {
  @Input() data: number[] = [];
  @Input() label = '';
  @Input() rangeMin = 0;
  @Input() rangeMax = 256;
  @Input() markerLines: Array<{ value: number; label?: string; color?: string }> = [];
  @Input() width = 400;
  @Input() height = 120;

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

  private draw(): void {
    const canvas = this.canvasRef?.nativeElement;
    if (!canvas || !this.data?.length) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const w = canvas.width;
    const h = canvas.height;
    const pad = { top: 8, right: 8, bottom: 22, left: 40 };
    const plotW = w - pad.left - pad.right;
    const plotH = h - pad.top - pad.bottom;

    ctx.clearRect(0, 0, w, h);

    // Background
    ctx.fillStyle = '#1e1e1e';
    ctx.fillRect(0, 0, w, h);

    const maxVal = Math.max(...this.data, 1);
    const bins = this.data.length;
    const barW = plotW / bins;

    // Grid lines
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= 4; i++) {
      const y = pad.top + (plotH / 4) * i;
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(pad.left + plotW, y);
      ctx.stroke();
    }

    // Bars
    ctx.fillStyle = '#6b7b8d';
    for (let i = 0; i < bins; i++) {
      const barH = (this.data[i] / maxVal) * plotH;
      const x = pad.left + i * barW;
      const y = pad.top + plotH - barH;
      ctx.fillRect(x, y, Math.max(barW, 1), barH);
    }

    // Marker lines (for thresholds/ranges)
    if (Array.isArray(this.markerLines) && this.markerLines.length > 0) {
      for (const marker of this.markerLines) {
        const v = Number(marker?.value);
        if (!Number.isFinite(v)) continue;
        const normalized = (v - this.rangeMin) / Math.max(this.rangeMax - this.rangeMin, 1e-9);
        const x = pad.left + Math.max(0, Math.min(1, normalized)) * plotW;
        const color = marker?.color || '#f59e0b';

        ctx.strokeStyle = color;
        ctx.lineWidth = 1.25;
        ctx.beginPath();
        ctx.moveTo(x, pad.top);
        ctx.lineTo(x, pad.top + plotH);
        ctx.stroke();

        if (marker?.label) {
          ctx.fillStyle = color;
          ctx.font = '9px sans-serif';
          ctx.textAlign = 'left';
          const labelX = Math.min(x + 3, pad.left + plotW - 26);
          ctx.fillText(marker.label, labelX, pad.top + 9);
        }
      }
    }

    // Axes
    ctx.strokeStyle = '#555';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad.left, pad.top);
    ctx.lineTo(pad.left, pad.top + plotH);
    ctx.lineTo(pad.left + plotW, pad.top + plotH);
    ctx.stroke();

    // X-axis labels
    ctx.fillStyle = '#777';
    ctx.font = '9px sans-serif';
    ctx.textAlign = 'center';
    const xLabels = 5;
    for (let i = 0; i <= xLabels; i++) {
      const val = this.rangeMin + ((this.rangeMax - this.rangeMin) / xLabels) * i;
      const x = pad.left + (plotW / xLabels) * i;
      ctx.fillText(Math.round(val).toString(), x, pad.top + plotH + 14);
    }

    // Y-axis labels
    ctx.textAlign = 'right';
    for (let i = 0; i <= 4; i++) {
      const val = maxVal - (maxVal / 4) * i;
      const y = pad.top + (plotH / 4) * i;
      const label = val >= 1000 ? (val / 1000).toFixed(0) + 'k' : Math.round(val).toString();
      ctx.fillText(label, pad.left - 4, y + 3);
    }
  }
}
