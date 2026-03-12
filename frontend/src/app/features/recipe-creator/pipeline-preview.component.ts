import { Component, OnInit, OnDestroy, ViewChild, ElementRef, AfterViewInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Subscription } from 'rxjs';
import { PipelineStateService } from '../../services/pipeline-state.service';

@Component({
  selector: 'app-pipeline-preview',
  standalone: true,
  imports: [CommonModule, FormsModule],
  template: `
    <div class="preview-wrapper" #previewContainer
         (wheel)="onWheel($event)"
         (mousedown)="onMouseDown($event)"
         (mousemove)="onMouseMove($event)"
         (mouseup)="onMouseUp()"
         (mouseleave)="onMouseUp()"
         (dblclick)="resetZoom()"
         [style.cursor]="getCursor()">
      @if (loading) {
        <div class="loading-overlay">
          <div class="spinner"></div>
          <span>Előnézet betöltése...</span>
        </div>
      }
      @if (showGraphViewer) {
        <div class="graph-viewer-overlay">
          <div class="graph-toolbar">
            @if (graphSelectedPoint >= 0) {
              @if (isSelectedPointOmitted()) {
                <button class="graph-tool-btn" (click)="restoreSelectedPoint()" title="Adatpont tartalmazása">
                  Adatpont tartalmazása
                </button>
              } @else {
                <button class="graph-tool-btn" (click)="omitSelectedPoint()" title="Adatpont kihagyása">
                  Adatpont kihagyása
                </button>
              }
              <button class="graph-tool-btn" (click)="viewSelectedImage()" title="Kép megtekintése" [disabled]="isViewImageDisabled()" [class.disabled]="isViewImageDisabled()">
                Kép megtekintése
              </button>
            } @else {
              <button class="graph-tool-btn disabled" disabled>
                Adatpont kihagyása
              </button>
              <button class="graph-tool-btn disabled" disabled>
                Kép megtekintése
              </button>
            }
            <div class="graph-toolbar-spacer"></div>
            <button class="graph-close-btn" (click)="closeGraphViewer()" title="Bezárás">✕</button>
          </div>
          <canvas #graphCanvas class="graph-canvas"
                  (click)="onGraphClick($event)"
                  (contextmenu)="onGraphContextMenu($event)"
                  (wheel)="onGraphWheel($event)"
                  (mousedown)="onGraphMouseDown($event)"
                  (mousemove)="onGraphMouseMove($event)"
                  (mouseup)="onGraphMouseUp()"
                  (mouseleave)="onGraphMouseLeave()"
                  (dblclick)="resetGraphTransform()">
          </canvas>
          @if (showGraphContextMenu) {
            <div class="graph-context-menu"
                 [style.left.px]="graphContextMenuX"
                 [style.top.px]="graphContextMenuY">
              @if (isContextPointOmitted()) {
                <button (click)="restoreContextPoint()">Adatpont tartalmazása</button>
              } @else {
                <button (click)="omitContextPoint()">Adatpont kihagyása</button>
              }
              <button (click)="viewContextImage()" [disabled]="isViewImageDisabled()">Kép megtekintése</button>
            </div>
          }
        </div>
      }
      <div class="preview-scroll-area" #scrollArea
           [class.zoomed]="zoomLevel > 1">
        @if (imageSrc && !showGraphViewer) {
          <img #previewImg
            [src]="imageSrc"
            alt="Pipeline előnézet"
            class="preview-image"
            draggable="false"
          />
        } @else if (!loading && !showGraphViewer) {
          <div class="no-preview">
            <div class="no-preview-icon">🖼</div>
            <span>Nincs előnézet</span>
            <span class="no-preview-hint">Adjon hozzá lépéseket és válasszon képet</span>
          </div>
        }
      </div>

      @if (imageCount > 1 && !showGraphViewer) {
        <div class="pagination-bar">
          <button
            class="page-btn"
            (click)="prevImage()"
            [disabled]="currentIndex <= 0"
            title="Előző kép"
          >◀</button>
          <div class="page-indicator">
            <input
              type="number"
              class="page-input"
              [ngModel]="currentIndex + 1"
              (ngModelChange)="goToImage($event)"
              [min]="1"
              [max]="imageCount"
            />
            <span class="page-total">/ {{ imageCount }}</span>
          </div>
          <button
            class="page-btn"
            (click)="nextImage()"
            [disabled]="currentIndex >= imageCount - 1"
            title="Következő kép"
          >▶</button>
        </div>
      }
    </div>
  `,
  styles: [`
    :host {
      display: block;
      height: 100%;
      overflow: hidden;
    }

    .preview-wrapper {
      position: relative;
      width: 100%;
      height: 100%;
      display: flex;
      flex-direction: column;
      background: #1a1a1a;
      border-radius: 4px;
      overflow: hidden;
    }

    .preview-scroll-area {
      flex: 1;
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
      min-height: 0;
    }

    .preview-scroll-area.zoomed {
      overflow: auto;
      align-items: flex-start;
      justify-content: flex-start;
      scrollbar-width: thin;
      scrollbar-color: #444 #1a1a1a;
    }

    .preview-scroll-area.zoomed::-webkit-scrollbar {
      width: 10px;
      height: 10px;
    }

    .preview-scroll-area.zoomed::-webkit-scrollbar-track {
      background: #1a1a1a;
      border-radius: 8px;
    }

    .preview-scroll-area.zoomed::-webkit-scrollbar-thumb {
      background: #444;
      border-radius: 8px;
      border: 2px solid #1a1a1a;
    }

    .preview-scroll-area.zoomed::-webkit-scrollbar-thumb:hover {
      background: #5a5a5a;
    }

    .preview-scroll-area.zoomed .preview-image {
      max-width: none;
      max-height: none;
    }

    .loading-overlay {
      position: absolute;
      inset: 0;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 12px;
      background: rgba(26, 26, 26, 0.85);
      z-index: 1;
      color: #999;
      font-size: 12px;
    }

    .spinner {
      width: 28px;
      height: 28px;
      border: 3px solid #333;
      border-top-color: #3b82f6;
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
    }

    .progress-counter {
      font-size: 11px;
      color: #777;
      font-variant-numeric: tabular-nums;
    }

    @keyframes spin {
      to { transform: rotate(360deg); }
    }

    .preview-image {
      max-width: 100%;
      max-height: calc(100% - 40px);
      object-fit: contain;
      user-select: none;
      transform-origin: center center;
    }

    .no-preview {
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 8px;
      color: #666;
      font-size: 13px;
    }

    .no-preview-icon {
      font-size: 36px;
      opacity: 0.4;
    }

    .no-preview-hint {
      font-size: 11px;
      color: #555;
    }

    /* Graph viewer overlay */
    .graph-viewer-overlay {
      position: absolute;
      inset: 0;
      background: #1a1a1a;
      z-index: 2;
      display: flex;
      flex-direction: column;
    }

    .graph-toolbar {
      display: flex;
      align-items: center;
      gap: 6px;
      padding: 6px 10px;
      background: #2a2a2a;
      border-bottom: 1px solid #444;
      flex-shrink: 0;
    }

    .graph-tool-btn {
      display: flex;
      align-items: center;
      gap: 4px;
      padding: 4px 10px;
      background: #333;
      border: 1px solid #555;
      border-radius: 4px;
      color: #e0e0e0;
      font-size: 11px;
      cursor: pointer;
      white-space: nowrap;
    }

    .graph-tool-btn:hover:not(.disabled) {
      background: #3b82f6;
      border-color: #3b82f6;
    }

    .graph-tool-btn.disabled {
      opacity: 0.35;
      cursor: default;
    }

    .graph-tool-icon {
      font-size: 13px;
    }

    .graph-toolbar-spacer {
      flex: 1;
    }

    .graph-close-btn {
      background: none;
      border: 1px solid #555;
      border-radius: 4px;
      color: #e0e0e0;
      font-size: 14px;
      cursor: pointer;
      padding: 4px 10px;
      line-height: 1;
    }

    .graph-close-btn:hover {
      background: #ef4444;
      border-color: #ef4444;
    }

    .graph-canvas {
      flex: 1;
      width: 100%;
      cursor: crosshair;
    }

    .graph-context-menu {
      position: absolute;
      background: #2a2a2a;
      border: 1px solid #555;
      border-radius: 6px;
      overflow: hidden;
      z-index: 10;
      box-shadow: 0 4px 12px rgba(0,0,0,0.5);
    }

    .graph-context-menu button {
      display: block;
      width: 100%;
      padding: 8px 16px;
      background: none;
      border: none;
      color: #e0e0e0;
      font-size: 12px;
      cursor: pointer;
      text-align: left;
      white-space: nowrap;
    }

    .graph-context-menu button:hover {
      background: #3b82f6;
    }

    /* Pagination bar */
    .pagination-bar {
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      padding: 6px 12px;
      background: rgba(30, 30, 30, 0.9);
      border-top: 1px solid #444;
      flex-shrink: 0;
    }

    .page-btn {
      background: #333;
      border: 1px solid #555;
      border-radius: 4px;
      color: #e0e0e0;
      cursor: pointer;
      padding: 4px 10px;
      font-size: 12px;
      line-height: 1;
    }

    .page-btn:hover:not(:disabled) {
      background: #3b82f6;
      border-color: #3b82f6;
    }

    .page-btn:disabled {
      opacity: 0.3;
      cursor: default;
    }

    .page-indicator {
      display: flex;
      align-items: center;
      gap: 4px;
    }

    .page-input {
      width: 44px;
      padding: 3px 4px;
      background: #2a2a2a;
      border: 1px solid #555;
      border-radius: 4px;
      color: #e0e0e0;
      font-size: 12px;
      text-align: center;
      -moz-appearance: textfield;
    }

    .page-input::-webkit-outer-spin-button,
    .page-input::-webkit-inner-spin-button {
      -webkit-appearance: none;
      margin: 0;
    }

    .page-total {
      color: #888;
      font-size: 12px;
      font-variant-numeric: tabular-nums;
    }
  `],
})
export class PipelinePreviewComponent implements OnInit, OnDestroy {
  @ViewChild('previewContainer') previewContainer!: ElementRef<HTMLDivElement>;
  @ViewChild('scrollArea') scrollArea!: ElementRef<HTMLDivElement>;
  @ViewChild('previewImg') previewImg!: ElementRef<HTMLImageElement>;
  @ViewChild('graphCanvas') graphCanvasRef!: ElementRef<HTMLCanvasElement>;

  imageSrc: string | null = null;
  loading = false;
  imageCount = 0;
  currentIndex = 0;

  // Zoom/pan for gallery image
  zoomLevel = 1.0;
  private isDragging = false;
  private dragStart = { x: 0, y: 0, scrollLeft: 0, scrollTop: 0 };

  // Graph viewer state
  showGraphViewer = false;
  graphSelectedPoint = -1;
  private graphData: any = null;
  private graphOmittedIndices: Set<number> = new Set();
  private imageNames: string[] = [];

  // Graph context menu
  showGraphContextMenu = false;
  graphContextMenuX = 0;
  graphContextMenuY = 0;
  private contextMenuPointIndex = -1;

  // Graph pan/zoom
  private graphZoom = 1.0;
  private graphPanX = 0;
  private graphPanY = 0;
  private graphDragging = false;
  private graphDragStart = { x: 0, y: 0, panX: 0, panY: 0 };

  // Graph coordinate mapping (needs caching for hit-testing)
  private pointCoords: { px: number; py: number }[] = [];
  private graphPad = { top: 30, right: 20, bottom: 36, left: 56 };

  private subs: Subscription[] = [];

  constructor(private pipelineState: PipelineStateService) {}

  ngOnInit(): void {
    this.subs.push(
      this.pipelineState.previewImage$.subscribe((img) => (this.imageSrc = img)),
      this.pipelineState.previewLoading$.subscribe((l) => (this.loading = l)),
      this.pipelineState.imageCount$.subscribe((c) => (this.imageCount = c)),
      this.pipelineState.previewImageIndex$.subscribe((i) => (this.currentIndex = i)),
      this.pipelineState.sideOutputs$.subscribe((so) => {
        this.imageNames = so?.['loaded_paths'] ?? [];
        // Auto-update the maximized chart when a new curve fit arrives
        if (this.showGraphViewer && so?.['curve_fits']) {
          const fits = so['curve_fits'];
          if (Array.isArray(fits) && fits.length > 0) {
            this.graphData = fits[fits.length - 1];
            this.drawGraph();
          }
        }
      }),
      this.pipelineState.maximizeGraph$.subscribe(({ data, omittedIndices }) => {
        this.openGraphViewer(data, omittedIndices);
      }),
    );
  }

  ngOnDestroy(): void {
    this.subs.forEach((s) => s.unsubscribe());
  }

  // --- Gallery zoom/pan ---

  onWheel(event: WheelEvent): void {
    if (this.showGraphViewer) return;
    if (!event.ctrlKey) return;
    event.preventDefault();
    const factor = event.deltaY > 0 ? 0.9 : 1.1;
    this.zoomLevel = Math.max(1.0, Math.min(5.0, this.zoomLevel * factor));
    this.applyImageTransform();
  }

  onMouseDown(event: MouseEvent): void {
    if (this.showGraphViewer) return;
    if (this.zoomLevel <= 1.0) return;
    event.preventDefault();
    const container = this.scrollArea?.nativeElement;
    if (!container) return;
    this.isDragging = true;
    this.dragStart = {
      x: event.clientX,
      y: event.clientY,
      scrollLeft: container.scrollLeft,
      scrollTop: container.scrollTop,
    };
  }

  onMouseMove(event: MouseEvent): void {
    if (this.showGraphViewer || !this.isDragging || this.zoomLevel <= 1.0) return;
    const container = this.scrollArea?.nativeElement;
    if (!container) return;
    container.scrollLeft = this.dragStart.scrollLeft - (event.clientX - this.dragStart.x);
    container.scrollTop = this.dragStart.scrollTop - (event.clientY - this.dragStart.y);
  }

  onMouseUp(): void {
    this.isDragging = false;
  }

  resetZoom(): void {
    if (this.showGraphViewer) return;
    this.zoomLevel = 1.0;
    this.applyImageTransform();
  }

  getCursor(): string {
    if (this.showGraphViewer) return 'default';
    if (this.zoomLevel > 1.0) return this.isDragging ? 'grabbing' : 'grab';
    return 'default';
  }

  private applyImageTransform(): void {
    const img = this.previewImg?.nativeElement;
    const container = this.scrollArea?.nativeElement;
    if (!img || !container) return;
    const pct = this.zoomLevel * 100;
    img.style.width = `${pct}%`;
    img.style.height = `${pct}%`;
    if (this.zoomLevel === 1.0) {
      container.scrollLeft = 0;
      container.scrollTop = 0;
    }
  }

  // --- Pagination ---

  prevImage(): void {
    this.pipelineState.setPreviewImageIndex(this.currentIndex - 1);
  }

  nextImage(): void {
    this.pipelineState.setPreviewImageIndex(this.currentIndex + 1);
  }

  goToImage(oneBasedIndex: number): void {
    const idx = Math.round(oneBasedIndex) - 1;
    this.pipelineState.setPreviewImageIndex(idx);
  }

  // --- Graph viewer ---

  openGraphViewer(data: any, omittedIndices: Set<number>): void {
    this.graphData = data;
    this.graphOmittedIndices = new Set(omittedIndices);
    this.graphSelectedPoint = -1;
    this.graphZoom = 1.0;
    this.graphPanX = 0;
    this.graphPanY = 0;
    this.showGraphViewer = true;
    this.showGraphContextMenu = false;
    setTimeout(() => this.drawGraph(), 0);
  }

  closeGraphViewer(): void {
    this.showGraphViewer = false;
    this.showGraphContextMenu = false;
  }

  getOmittedIndices(): Set<number> {
    return this.graphOmittedIndices;
  }

  /** Draw the interactive graph on the full-size canvas */
  private drawGraph(): void {
    const canvasEl = this.graphCanvasRef?.nativeElement;
    if (!canvasEl || !this.graphData) return;
    const d = this.graphData;

    // Size canvas to container
    const rect = canvasEl.parentElement!.getBoundingClientRect();
    const toolbarH = canvasEl.parentElement!.querySelector('.graph-toolbar')?.getBoundingClientRect().height ?? 0;
    const cw = Math.floor(rect.width);
    const ch = Math.floor(rect.height - toolbarH);
    canvasEl.width = cw;
    canvasEl.height = ch;

    const ctx = canvasEl.getContext('2d');
    if (!ctx) return;

    const pad = this.graphPad;
    const plotW = cw - pad.left - pad.right;
    const plotH = ch - pad.top - pad.bottom;

    // Clear
    ctx.clearRect(0, 0, cw, ch);
    ctx.fillStyle = '#1e1e1e';
    ctx.fillRect(0, 0, cw, ch);

    if (!d.x_values?.length) return;

    const allY = [...d.y_values, ...d.fitted_y];
    const xMin = Math.min(...d.x_values);
    const xMax = Math.max(...d.x_values);
    const yMin = Math.min(...allY);
    const yMax = Math.max(...allY);
    const xRange = xMax - xMin || 1;
    const yRange = yMax - yMin || 1;
    const xPad2 = xRange * 0.05;
    const yPad2 = yRange * 0.05;

    // Apply zoom/pan transform
    const z = this.graphZoom;
    const px = this.graphPanX;
    const py = this.graphPanY;

    const toX = (v: number) => (pad.left + ((v - xMin + xPad2) / (xRange + 2 * xPad2)) * plotW) * z + px;
    const toY = (v: number) => (pad.top + plotH - ((v - yMin + yPad2) / (yRange + 2 * yPad2)) * plotH) * z + py;

    // Grid lines
    ctx.save();
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= 4; i++) {
      const yg = toY(yMin + (yRange / 4) * i);
      ctx.beginPath();
      ctx.moveTo(toX(xMin - xPad2), yg);
      ctx.lineTo(toX(xMax + xPad2), yg);
      ctx.stroke();
    }
    ctx.restore();

    // Axes
    ctx.strokeStyle = '#555';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(toX(xMin - xPad2), toY(yMax + yPad2));
    ctx.lineTo(toX(xMin - xPad2), toY(yMin - yPad2));
    ctx.lineTo(toX(xMax + xPad2), toY(yMin - yPad2));
    ctx.stroke();

    // Fitted line (sort by x)
    const sortedIdx = d.x_values
      .map((_: number, i: number) => i)
      .sort((a: number, b: number) => d.x_values[a] - d.x_values[b]);
    ctx.strokeStyle = '#6b8fad';
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < sortedIdx.length; i++) {
      const si = sortedIdx[i];
      if (this.graphOmittedIndices.has(si)) continue;
      const ppx = toX(d.x_values[si]);
      const ppy = toY(d.fitted_y[si]);
      if (i === 0) ctx.moveTo(ppx, ppy);
      else ctx.lineTo(ppx, ppy);
    }
    ctx.stroke();

    // Data points + build pointCoords for hit-testing
    this.pointCoords = [];
    for (let i = 0; i < d.x_values.length; i++) {
      const ptx = toX(d.x_values[i]);
      const pty = toY(d.y_values[i]);
      this.pointCoords.push({ px: ptx, py: pty });

      const isOmitted = this.graphOmittedIndices.has(i);
      const isSelected = this.graphSelectedPoint === i;
      const baseColor = d.point_colors?.[i] || '#a0c4e8';

      ctx.beginPath();
      ctx.arc(ptx, pty, isSelected ? 6 : 4.5, 0, Math.PI * 2);

      if (isOmitted) {
        ctx.fillStyle = this.toAlpha(baseColor, 0.2);
        ctx.strokeStyle = this.toAlpha(baseColor, 0.3);
        ctx.lineWidth = 1;
        ctx.fill();
        ctx.stroke();
      } else if (isSelected) {
        ctx.fillStyle = '#ff9800';
        ctx.fill();
      } else {
        ctx.fillStyle = baseColor;
        ctx.fill();
      }
    }

    // Axis labels
    ctx.fillStyle = '#777';
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'center';
    const xlCount = 6;
    for (let i = 0; i <= xlCount; i++) {
      const val = xMin + (xRange / xlCount) * i;
      ctx.fillText(val.toFixed(1), toX(val), ch - 6);
    }
    ctx.textAlign = 'right';
    for (let i = 0; i <= 4; i++) {
      const val = yMin + (yRange / 4) * i;
      ctx.fillText(val.toFixed(1), toX(xMin - xPad2) - 4, toY(val) + 3);
    }

    // Axis name labels
    if (d.x_name) {
      ctx.fillStyle = '#999';
      ctx.font = '11px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(d.x_name, cw / 2, ch - 2);
    }
    if (d.y_name) {
      ctx.save();
      ctx.fillStyle = '#999';
      ctx.font = '11px sans-serif';
      ctx.translate(12, ch / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.textAlign = 'center';
      ctx.fillText(d.y_name, 0, 0);
      ctx.restore();
    }

    // Formula + R²
    ctx.fillStyle = '#aaa';
    ctx.font = '11px monospace';
    ctx.textAlign = 'left';
    const formula = this.getFormulaText(d);
    ctx.fillText(formula, pad.left + 8, pad.top - 10);
    ctx.fillStyle = '#888';
    ctx.fillText(`R² = ${d.r2.toFixed(6)}`, pad.left + 8 + ctx.measureText(formula).width + 20, pad.top - 10);

    // Omitted count
    if (this.graphOmittedIndices.size > 0) {
      ctx.fillStyle = '#ef4444';
      ctx.font = '10px sans-serif';
      ctx.textAlign = 'right';
      ctx.fillText(`${this.graphOmittedIndices.size} kihagyott adatpont`, cw - pad.right, pad.top - 10);
    }

    // Selected point tooltip
    if (this.graphSelectedPoint >= 0 && this.graphSelectedPoint < d.x_values.length) {
      const si = this.graphSelectedPoint;
      const sx = this.pointCoords[si].px;
      const sy = this.pointCoords[si].py;
      const name = this.imageNames[si] ?? `Kép ${si + 1}`;
      const tipText = `#${si + 1} ${name}  (${d.x_values[si].toFixed(2)}, ${d.y_values[si].toFixed(2)})`;
      ctx.font = '11px sans-serif';
      const tw = ctx.measureText(tipText).width;
      const tipX = Math.min(sx - tw / 2, cw - tw - 10);
      const tipY = sy - 18;
      ctx.fillStyle = 'rgba(40,40,40,0.9)';
      ctx.fillRect(tipX - 4, tipY - 12, tw + 8, 16);
      ctx.fillStyle = '#fff';
      ctx.textAlign = 'left';
      ctx.fillText(tipText, tipX, tipY);
    }
  }

  private getFormulaText(d: any): string {
    const c = d.coefficients;
    if (d.model === 'linear' || (d.model === 'poly' && d.degree === 1)) {
      return `y = ${c[0].toFixed(4)}x + ${c[1].toFixed(4)}`;
    }
    return c
      .map((coeff: number, i: number) => {
        const power = c.length - 1 - i;
        const val = coeff.toFixed(4);
        if (power === 0) return val;
        if (power === 1) return `${val}x`;
        return `${val}x^${power}`;
      })
      .join(' + ');
  }

  private hitTestPoint(event: MouseEvent): number {
    const canvas = this.graphCanvasRef?.nativeElement;
    if (!canvas) return -1;
    const rect = canvas.getBoundingClientRect();
    const mx = event.clientX - rect.left;
    const my = event.clientY - rect.top;
    const threshold = 10;
    let closest = -1;
    let closestDist = Infinity;
    for (let i = 0; i < this.pointCoords.length; i++) {
      const dx = mx - this.pointCoords[i].px;
      const dy = my - this.pointCoords[i].py;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist < threshold && dist < closestDist) {
        closestDist = dist;
        closest = i;
      }
    }
    return closest;
  }

  onGraphClick(event: MouseEvent): void {
    this.showGraphContextMenu = false;
    const pt = this.hitTestPoint(event);
    this.graphSelectedPoint = pt;
    this.drawGraph();
  }

  onGraphContextMenu(event: MouseEvent): void {
    event.preventDefault();
    const pt = this.hitTestPoint(event);
    if (pt < 0) {
      this.showGraphContextMenu = false;
      return;
    }
    this.contextMenuPointIndex = pt;
    this.graphSelectedPoint = pt;
    const canvas = this.graphCanvasRef?.nativeElement;
    if (!canvas) return;
    const wrapperRect = this.previewContainer.nativeElement.getBoundingClientRect();
    this.graphContextMenuX = event.clientX - wrapperRect.left;
    this.graphContextMenuY = event.clientY - wrapperRect.top;
    this.showGraphContextMenu = true;
    this.drawGraph();
  }

  isSelectedPointOmitted(): boolean {
    return this.graphSelectedPoint >= 0 && this.graphOmittedIndices.has(this.graphSelectedPoint);
  }

  omitSelectedPoint(): void {
    if (this.graphSelectedPoint < 0) return;
    this.graphOmittedIndices.add(this.graphSelectedPoint);
    this.graphSelectedPoint = -1;
    this.drawGraph();
    this.pipelineState.notifyOmittedPoints(this.graphOmittedIndices, this.imageNames);
  }

  restoreSelectedPoint(): void {
    if (this.graphSelectedPoint < 0) return;
    this.graphOmittedIndices.delete(this.graphSelectedPoint);
    this.graphSelectedPoint = -1;
    this.drawGraph();
    this.pipelineState.notifyOmittedPoints(this.graphOmittedIndices, this.imageNames);
  }

  viewSelectedImage(): void {
    if (this.graphSelectedPoint < 0 || this.isViewImageDisabled()) return;
    this.showGraphViewer = false;
    this.showGraphContextMenu = false;
    this.pipelineState.setPreviewImageIndex(this.graphSelectedPoint);
  }

  isContextPointOmitted(): boolean {
    return this.contextMenuPointIndex >= 0 && this.graphOmittedIndices.has(this.contextMenuPointIndex);
  }

  omitContextPoint(): void {
    this.showGraphContextMenu = false;
    if (this.contextMenuPointIndex >= 0) {
      this.graphOmittedIndices.add(this.contextMenuPointIndex);
      this.graphSelectedPoint = -1;
      this.drawGraph();
      this.pipelineState.notifyOmittedPoints(this.graphOmittedIndices, this.imageNames);
    }
  }

  restoreContextPoint(): void {
    this.showGraphContextMenu = false;
    if (this.contextMenuPointIndex >= 0) {
      this.graphOmittedIndices.delete(this.contextMenuPointIndex);
      this.graphSelectedPoint = -1;
      this.drawGraph();
      this.pipelineState.notifyOmittedPoints(this.graphOmittedIndices, this.imageNames);
    }
  }

  viewContextImage(): void {
    this.showGraphContextMenu = false;
    if (this.contextMenuPointIndex >= 0 && !this.isViewImageDisabled()) {
      this.showGraphViewer = false;
      this.pipelineState.setPreviewImageIndex(this.contextMenuPointIndex);
    }
  }

  isViewImageDisabled(): boolean {
    return !!(this.graphData?.aggregation?.enabled);
  }

  private toAlpha(color: string, alpha: number): string {
    const m = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(color || '');
    if (!m) return `rgba(160,196,232,${alpha})`;
    const r = parseInt(m[1], 16);
    const g = parseInt(m[2], 16);
    const b = parseInt(m[3], 16);
    return `rgba(${r},${g},${b},${alpha})`;
  }

  // --- Graph pan/zoom ---

  onGraphWheel(event: WheelEvent): void {
    event.preventDefault();
    const factor = event.deltaY > 0 ? 0.9 : 1.1;
    const canvas = this.graphCanvasRef?.nativeElement;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const mx = event.clientX - rect.left;
    const my = event.clientY - rect.top;

    const newZoom = Math.max(0.5, Math.min(10, this.graphZoom * factor));
    // Zoom toward cursor
    this.graphPanX = mx - (mx - this.graphPanX) * (newZoom / this.graphZoom);
    this.graphPanY = my - (my - this.graphPanY) * (newZoom / this.graphZoom);
    this.graphZoom = newZoom;
    this.drawGraph();
  }

  onGraphMouseDown(event: MouseEvent): void {
    if (event.button !== 0) return;
    // Only start drag if not clicking on a point
    if (this.graphZoom > 1.0 || event.shiftKey) {
      this.graphDragging = true;
      this.graphDragStart = {
        x: event.clientX,
        y: event.clientY,
        panX: this.graphPanX,
        panY: this.graphPanY,
      };
    }
  }

  onGraphMouseMove(event: MouseEvent): void {
    if (!this.graphDragging) return;
    this.graphPanX = this.graphDragStart.panX + (event.clientX - this.graphDragStart.x);
    this.graphPanY = this.graphDragStart.panY + (event.clientY - this.graphDragStart.y);
    this.drawGraph();
  }

  onGraphMouseUp(): void {
    this.graphDragging = false;
  }

  onGraphMouseLeave(): void {
    this.graphDragging = false;
    this.showGraphContextMenu = false;
  }

  resetGraphTransform(): void {
    this.graphZoom = 1.0;
    this.graphPanX = 0;
    this.graphPanY = 0;
    this.drawGraph();
  }
}
