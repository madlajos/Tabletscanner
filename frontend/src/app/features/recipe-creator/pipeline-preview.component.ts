import { Component, OnInit, OnDestroy, ViewChild, ElementRef, AfterViewInit, ChangeDetectorRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Subscription, combineLatest } from 'rxjs';
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
         (auxclick)="onAuxClick($event)"
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
          <div class="image-roi-container" #imageRoiContainer>
            <img #previewImg
              [src]="imageSrc"
              alt="Pipeline előnézet"
              class="preview-image"
              draggable="false"
              (load)="onImageLoad()"
            />
            @if (roiActive) {
              <svg class="roi-overlay"
                   [attr.viewBox]="'0 0 ' + roiImgW + ' ' + roiImgH"
                   (mousedown)="onRoiMouseDown($event)"
                   (mousemove)="onRoiMouseMove($event)"
                   (mouseup)="onRoiMouseUp($event)"
                   (mouseleave)="onRoiMouseUp($event)"
                   (dblclick)="onRoiDblClick($event)"
                   (contextmenu)="onRoiContextMenu($event)">
                <!-- Rectangle ROI -->
                @if (roiType === 'rect' && (hasRoiShape || roiDragMode === 'draw-rect')) {
                  <rect [attr.x]="roiRect.x" [attr.y]="roiRect.y"
                        [attr.width]="roiRect.w" [attr.height]="roiRect.h"
                        fill="rgba(59,130,246,0.15)" stroke="#3b82f6"
                        [attr.stroke-width]="2 * roiScale" [attr.stroke-dasharray]="(6 * roiScale) + ' ' + (3 * roiScale)" />
                  <!-- Corner handles -->
                  <circle [attr.cx]="roiRect.x" [attr.cy]="roiRect.y" [attr.r]="5 * roiScale"
                          class="roi-handle" data-handle="tl" />
                  <circle [attr.cx]="roiRect.x + roiRect.w" [attr.cy]="roiRect.y" [attr.r]="5 * roiScale"
                          class="roi-handle" data-handle="tr" />
                  <circle [attr.cx]="roiRect.x" [attr.cy]="roiRect.y + roiRect.h" [attr.r]="5 * roiScale"
                          class="roi-handle" data-handle="bl" />
                  <circle [attr.cx]="roiRect.x + roiRect.w" [attr.cy]="roiRect.y + roiRect.h" [attr.r]="5 * roiScale"
                          class="roi-handle" data-handle="br" />
                  <!-- Edge handles -->
                  <circle [attr.cx]="roiRect.x + roiRect.w / 2" [attr.cy]="roiRect.y" [attr.r]="4 * roiScale"
                          class="roi-handle edge" data-handle="t" />
                  <circle [attr.cx]="roiRect.x + roiRect.w / 2" [attr.cy]="roiRect.y + roiRect.h" [attr.r]="4 * roiScale"
                          class="roi-handle edge" data-handle="b" />
                  <circle [attr.cx]="roiRect.x" [attr.cy]="roiRect.y + roiRect.h / 2" [attr.r]="4 * roiScale"
                          class="roi-handle edge" data-handle="l" />
                  <circle [attr.cx]="roiRect.x + roiRect.w" [attr.cy]="roiRect.y + roiRect.h / 2" [attr.r]="4 * roiScale"
                          class="roi-handle edge" data-handle="r" />
                }
                <!-- Ellipse ROI -->
                @if (roiType === 'ellipse' && hasRoiShape) {
                  <ellipse [attr.cx]="roiEllipse.cx" [attr.cy]="roiEllipse.cy"
                           [attr.rx]="roiEllipse.rx" [attr.ry]="roiEllipse.ry"
                           fill="rgba(59,130,246,0.15)" stroke="#3b82f6"
                           [attr.stroke-width]="2 * roiScale" [attr.stroke-dasharray]="(6 * roiScale) + ' ' + (3 * roiScale)" />
                  <!-- Bounding rect handles -->
                  <circle [attr.cx]="roiEllipse.cx" [attr.cy]="roiEllipse.cy - roiEllipse.ry" [attr.r]="5 * roiScale"
                          class="roi-handle" data-handle="t" />
                  <circle [attr.cx]="roiEllipse.cx" [attr.cy]="roiEllipse.cy + roiEllipse.ry" [attr.r]="5 * roiScale"
                          class="roi-handle" data-handle="b" />
                  <circle [attr.cx]="roiEllipse.cx - roiEllipse.rx" [attr.cy]="roiEllipse.cy" [attr.r]="5 * roiScale"
                          class="roi-handle" data-handle="l" />
                  <circle [attr.cx]="roiEllipse.cx + roiEllipse.rx" [attr.cy]="roiEllipse.cy" [attr.r]="5 * roiScale"
                          class="roi-handle" data-handle="r" />
                  <!-- Center handle -->
                  <circle [attr.cx]="roiEllipse.cx" [attr.cy]="roiEllipse.cy" [attr.r]="4 * roiScale"
                          class="roi-handle center" data-handle="c" />
                }
                <!-- Ellipse guide points during 4-point drawing -->
                @if (roiType === 'ellipse' && roiEllipseDrawing) {
                  <!-- Constraint guide line: vertical after 1st point, horizontal after 3rd -->
                  @if (roiEllipseGuidePoints.length === 1) {
                    <line [attr.x1]="roiEllipseGuidePoints[0].x" [attr.y1]="0"
                          [attr.x2]="roiEllipseGuidePoints[0].x" [attr.y2]="roiImgH"
                          stroke="#3b82f6" stroke-opacity="0.3"
                          [attr.stroke-width]="1 * roiScale" [attr.stroke-dasharray]="(4 * roiScale) + ' ' + (4 * roiScale)" />
                  }
                  @if (roiEllipseGuidePoints.length === 3) {
                    <line [attr.x1]="0" [attr.y1]="roiEllipseGuidePoints[2].y"
                          [attr.x2]="roiImgW" [attr.y2]="roiEllipseGuidePoints[2].y"
                          stroke="#3b82f6" stroke-opacity="0.3"
                          [attr.stroke-width]="1 * roiScale" [attr.stroke-dasharray]="(4 * roiScale) + ' ' + (4 * roiScale)" />
                  }
                  @if (roiEllipseGuidePoints.length >= 2) {
                    <!-- Show preview ellipse once top+bottom are placed -->
                    <ellipse [attr.cx]="ellipsePreview().cx" [attr.cy]="ellipsePreview().cy"
                             [attr.rx]="ellipsePreview().rx" [attr.ry]="ellipsePreview().ry"
                             fill="rgba(59,130,246,0.08)" stroke="#3b82f6" stroke-opacity="0.4"
                             [attr.stroke-width]="2 * roiScale" [attr.stroke-dasharray]="(6 * roiScale) + ' ' + (3 * roiScale)" />
                  }
                  @for (pt of roiEllipseGuidePoints; track $index) {
                    <circle [attr.cx]="pt.x" [attr.cy]="pt.y" [attr.r]="6 * roiScale"
                            class="roi-handle" [class.first-point]="$index === 0" />
                  }
                  <!-- Guide label -->
                  <text [attr.x]="10 * roiScale" [attr.y]="24 * roiScale"
                        fill="#3b82f6" [attr.font-size]="14 * roiScale" font-family="sans-serif">
                    {{ ellipseDrawingHint() }}
                  </text>
                }
                <!-- Polygon ROI -->
                @if (roiType === 'polygon' && roiPolygon.length > 0) {
                  @if (roiPolygon.length > 2) {
                    <polygon [attr.points]="polygonPointsStr()"
                             fill="rgba(59,130,246,0.15)" stroke="#3b82f6"
                             [attr.stroke-width]="2 * roiScale" [attr.stroke-dasharray]="(6 * roiScale) + ' ' + (3 * roiScale)" />
                  } @else {
                    <polyline [attr.points]="polygonPointsStr()"
                              fill="none" stroke="#3b82f6"
                              [attr.stroke-width]="2 * roiScale" [attr.stroke-dasharray]="(6 * roiScale) + ' ' + (3 * roiScale)" />
                  }
                  @for (pt of roiPolygon; track $index) {
                    <circle [attr.cx]="pt.x" [attr.cy]="pt.y" [attr.r]="($index === 0 && roiPolygonDrawing) ? 7 * roiScale : 5 * roiScale"
                            class="roi-handle" [attr.data-handle]="'p' + $index"
                            [class.first-point]="$index === 0 && roiPolygonDrawing" />
                  }
                }
              </svg>
            }
            @if (showRoiContextMenu) {
              <div class="roi-context-menu-overlay"
                   [style.left.px]="roiContextMenuScreenX"
                   [style.top.px]="roiContextMenuScreenY"
                   (mousedown)="$event.stopPropagation()">
                <button (click)="clearRoiSelection()">Kijelölés törlése</button>
              </div>
            }
          </div>
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
      min-width: 0;
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
      max-height: 100%;
      object-fit: contain;
      user-select: none;
      transform-origin: center center;
    }

    .image-roi-container {
      position: relative;
      overflow: visible;
    }

    .image-roi-container .preview-image {
      display: block;
      max-width: 100%;
      max-height: 100%;
      object-fit: contain;
    }

    .roi-overlay {
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      cursor: crosshair;
      overflow: visible;
    }

    .roi-handle {
      fill: #fff;
      stroke: #3b82f6;
      stroke-width: 1.5;
      cursor: pointer;
    }

    .roi-handle:hover { fill: #3b82f6; }

    .roi-handle.edge { fill: #bfdbfe; }

    .roi-handle.center { fill: #93c5fd; cursor: move; }

    .roi-handle.first-point {
      fill: #22c55e;
      stroke: #16a34a;
      cursor: pointer;
    }

    .roi-context-menu-overlay {
      position: absolute;
      background: #2a2a2a;
      border: 1px solid #555;
      border-radius: 6px;
      overflow: hidden;
      box-shadow: 0 4px 12px rgba(0,0,0,0.5);
      z-index: 10;
    }

    .roi-context-menu-overlay button {
      display: block;
      width: 100%;
      padding: 8px 16px;
      background: none;
      border: none;
      color: #e0e0e0;
      font-size: 13px;
      cursor: pointer;
      text-align: left;
      white-space: nowrap;
    }

    .roi-context-menu-overlay button:hover {
      background: #3b82f6;
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
  @ViewChild('imageRoiContainer') imageRoiContainer!: ElementRef<HTMLDivElement>;

  imageSrc: string | null = null;
  loading = false;
  imageCount = 0;
  currentIndex = 0;

  // Zoom/pan for gallery image
  zoomLevel = 1.0;
  private baseFitScale = 1;
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

  // ROI editing state
  roiActive = false;
  roiType: 'rect' | 'ellipse' | 'polygon' = 'rect';
  roiImgW = 100;
  roiImgH = 100;
  roiRect = { x: 0, y: 0, w: 100, h: 100 };
  roiEllipse = { cx: 50, cy: 50, rx: 50, ry: 50 };
  roiPolygon: { x: number; y: number }[] = [];
  roiPolygonDrawing = false;
  roiEllipseDrawing = false;
  roiEllipseGuidePoints: { x: number; y: number }[] = [];
  hasRoiShape = false;

  // ROI right-click context menu
  showRoiContextMenu = false;
  roiContextMenuX = 0;
  roiContextMenuY = 0;
  roiContextMenuScreenX = 0;
  roiContextMenuScreenY = 0;

  /** Scale factor for ROI handles/strokes based on image resolution */
  get roiScale(): number {
    return Math.max(1, Math.max(this.roiImgW, this.roiImgH) / 1000);
  }

  roiDragMode: string | null = null;
  private roiDragStart = { mx: 0, my: 0, ox: 0, oy: 0, ow: 0, oh: 0 };
  private roiSelectedStepIndex = -1;

  private subs: Subscription[] = [];

  constructor(
    private pipelineState: PipelineStateService,
    private cdr: ChangeDetectorRef,
  ) {}

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
      // Track ROI editing when an ROI step is selected
      combineLatest([
        this.pipelineState.pipeline$,
        this.pipelineState.selectedStepIndex$,
        this.pipelineState.imageDims$,
      ]).subscribe(([pipeline, idx, dims]) => {
        if (idx >= 0 && idx < pipeline.steps.length &&
            pipeline.steps[idx].step_def_id === 'mask_rect_roi') {
          const step = pipeline.steps[idx];
          this.roiSelectedStepIndex = idx;
          this.roiImgW = dims.w || 100;
          this.roiImgH = dims.h || 100;
          const newType = step.param_values?.['roi_type'] ?? 'rect';
          if (newType !== this.roiType) {
            // Reset drawing state when ROI type changes
            this.roiEllipseDrawing = false;
            this.roiEllipseGuidePoints = [];
            this.roiPolygonDrawing = false;
          }
          this.roiType = newType;
          this.syncRoiFromParams(step.param_values);
          this.roiActive = true;
        } else {
          this.roiActive = false;
          this.roiSelectedStepIndex = -1;
          this.roiPolygonDrawing = false;
          this.roiEllipseDrawing = false;
          this.roiEllipseGuidePoints = [];
        }
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
    if (this.showRoiContextMenu) this.showRoiContextMenu = false;
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

  onAuxClick(event: MouseEvent): void {
    // Middle mouse button double-click resets zoom
    if (event.button === 1) {
      event.preventDefault();
      this.resetZoom();
    }
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

    const nw = img.naturalWidth;
    const nh = img.naturalHeight;
    if (nw === 0 || nh === 0) return;

    if (this.zoomLevel <= 1.0) {
      // Calculate fit scale based on container dimensions
      const cw = container.clientWidth;
      const ch = container.clientHeight;
      if (cw === 0 || ch === 0) return;
      this.baseFitScale = Math.min(cw / nw, ch / nh, 1);
      img.style.width = `${Math.floor(nw * this.baseFitScale)}px`;
      img.style.height = `${Math.floor(nh * this.baseFitScale)}px`;
      container.scrollLeft = 0;
      container.scrollTop = 0;
    } else {
      const fitW = Math.floor(nw * this.baseFitScale * this.zoomLevel);
      const fitH = Math.floor(nh * this.baseFitScale * this.zoomLevel);
      img.style.width = `${fitW}px`;
      img.style.height = `${fitH}px`;
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

  // --- ROI interaction ---

  onImageLoad(): void {
    const img = this.previewImg?.nativeElement;
    if (img) {
      this.roiImgW = img.naturalWidth;
      this.roiImgH = img.naturalHeight;
      // Reset zoom and fit image into the container
      this.zoomLevel = 1.0;
      this.applyImageTransform();
    }
  }

  private syncRoiFromParams(params: Record<string, any>): void {
    const t = params?.['roi_type'] ?? 'rect';
    if (t === 'rect') {
      const rx = params?.['roi_x'] ?? 0;
      const ry = params?.['roi_y'] ?? 0;
      const rw = params?.['roi_width'] ?? 0;
      const rh = params?.['roi_height'] ?? 0;
      this.roiRect = { x: rx, y: ry, w: rw, h: rh };
      this.hasRoiShape = rw > 0 && rh > 0;
    } else if (t === 'ellipse') {
      const ecx = params?.['roi_cx'] ?? 0;
      const ecy = params?.['roi_cy'] ?? 0;
      const erx = params?.['roi_rx'] ?? 0;
      const ery = params?.['roi_ry'] ?? 0;
      this.roiEllipse = { cx: ecx, cy: ecy, rx: erx, ry: ery };
      this.hasRoiShape = erx > 0 && ery > 0;
    } else if (t === 'polygon') {
      const raw = params?.['roi_points'] ?? '[]';
      try {
        const pts = typeof raw === 'string' ? JSON.parse(raw) : raw;
        this.roiPolygon = Array.isArray(pts) ? pts : [];
      } catch {
        this.roiPolygon = [];
      }
      this.hasRoiShape = this.roiPolygon.length > 0;
    }
  }

  private svgCoords(event: MouseEvent): { x: number; y: number } {
    const svg = (event.currentTarget ?? event.target) as SVGSVGElement;
    const rect = svg.getBoundingClientRect();
    const sx = this.roiImgW / rect.width;
    const sy = this.roiImgH / rect.height;
    return {
      x: Math.round(Math.max(0, Math.min(this.roiImgW, (event.clientX - rect.left) * sx))),
      y: Math.round(Math.max(0, Math.min(this.roiImgH, (event.clientY - rect.top) * sy))),
    };
  }

  onRoiMouseDown(event: MouseEvent): void {
    if (event.button !== 0) return;
    this.showRoiContextMenu = false;
    const { x, y } = this.svgCoords(event);
    const target = event.target as SVGElement;
    const handle = target.getAttribute?.('data-handle');

    if (this.roiType === 'polygon') {
      if (handle?.startsWith('p')) {
        const idx = parseInt(handle.substring(1), 10);
        // Click on first point while drawing → close polygon
        if (this.roiPolygonDrawing && idx === 0 && this.roiPolygon.length >= 3) {
          this.roiPolygonDrawing = false;
          this.hasRoiShape = true;
          this.commitRoi();
          return;
        }
        // Drag existing point
        this.roiDragMode = handle;
        this.roiDragStart = { mx: x, my: y, ox: this.roiPolygon[idx].x, oy: this.roiPolygon[idx].y, ow: 0, oh: 0 };
        event.preventDefault();
        event.stopPropagation();
        return;
      }
      if (this.roiPolygonDrawing) {
        // Add new polygon point
        this.roiPolygon = [...this.roiPolygon, { x, y }];
        event.preventDefault();
        event.stopPropagation();
        return;
      }
      if (!handle && this.roiPolygon.length === 0) {
        // Start new polygon
        this.roiPolygonDrawing = true;
        this.roiPolygon = [{ x, y }];
        event.preventDefault();
        event.stopPropagation();
        return;
      }
      // Click inside polygon → drag entire polygon; outside → add point
      if (!handle && this.roiPolygon.length >= 3) {
        if (this.isInsidePolygon(x, y)) {
          this.roiDragMode = 'move-poly';
          this.roiDragStart = { mx: x, my: y, ox: 0, oy: 0, ow: 0, oh: 0 };
        } else {
          // Add point to the existing polygon at the nearest edge
          const insertIdx = this.findNearestEdgeIndex(x, y);
          const pts = [...this.roiPolygon];
          pts.splice(insertIdx, 0, { x, y });
          this.roiPolygon = pts;
          this.commitRoi();
        }
        event.preventDefault();
        event.stopPropagation();
        return;
      }
      return;
    }

    event.preventDefault();
    event.stopPropagation();

    // Handle 4-point ellipse drawing mode
    if (this.roiType === 'ellipse' && this.roiEllipseDrawing) {
      // Constrain: 2nd point (bottom) must share X with 1st (top)
      //            4th point (right) must share Y with 3rd (left)
      let cx = x, cy = y;
      if (this.roiEllipseGuidePoints.length === 1) {
        cx = this.roiEllipseGuidePoints[0].x;
      } else if (this.roiEllipseGuidePoints.length === 3) {
        cy = this.roiEllipseGuidePoints[2].y;
      }
      this.roiEllipseGuidePoints = [...this.roiEllipseGuidePoints, { x: cx, y: cy }];
      if (this.roiEllipseGuidePoints.length >= 4) {
        // All 4 points placed: top, bottom, left, right
        const top = this.roiEllipseGuidePoints[0];
        const bottom = this.roiEllipseGuidePoints[1];
        const left = this.roiEllipseGuidePoints[2];
        const right = this.roiEllipseGuidePoints[3];
        this.roiEllipse = {
          cy: Math.round((top.y + bottom.y) / 2),
          ry: Math.round(Math.abs(bottom.y - top.y) / 2),
          cx: Math.round((left.x + right.x) / 2),
          rx: Math.round(Math.abs(right.x - left.x) / 2),
        };
        this.roiEllipseDrawing = false;
        this.roiEllipseGuidePoints = [];
        this.hasRoiShape = true;
        this.commitRoi();
      }
      return;
    }

    if (handle) {
      this.roiDragMode = handle;
      if (this.roiType === 'rect') {
        this.roiDragStart = { mx: x, my: y, ox: this.roiRect.x, oy: this.roiRect.y, ow: this.roiRect.w, oh: this.roiRect.h };
      } else {
        this.roiDragStart = { mx: x, my: y, ox: this.roiEllipse.cx, oy: this.roiEllipse.cy, ow: this.roiEllipse.rx, oh: this.roiEllipse.ry };
      }
    } else {
      // Check if click is inside existing shape → move it
      if (this.roiType === 'rect' && this.hasRoiShape && this.isInsideRect(x, y)) {
        this.roiDragMode = 'move-rect';
        this.roiDragStart = { mx: x, my: y, ox: this.roiRect.x, oy: this.roiRect.y, ow: this.roiRect.w, oh: this.roiRect.h };
      } else if (this.roiType === 'ellipse' && this.hasRoiShape && this.isInsideEllipse(x, y)) {
        this.roiDragMode = 'move-ellipse';
        this.roiDragStart = { mx: x, my: y, ox: this.roiEllipse.cx, oy: this.roiEllipse.cy, ow: this.roiEllipse.rx, oh: this.roiEllipse.ry };
      } else if (this.roiType === 'rect' && !this.hasRoiShape) {
        // Start drawing new rect only if none exists
        this.roiDragMode = 'draw-rect';
        this.roiRect = { x, y, w: 0, h: 0 };
        this.roiDragStart = { mx: x, my: y, ox: x, oy: y, ow: 0, oh: 0 };
      } else if (this.roiType === 'ellipse' && !this.hasRoiShape) {
        // Start 4-point ellipse drawing
        this.roiEllipseDrawing = true;
        this.roiEllipseGuidePoints = [{ x, y }];
      }
      // If shape exists and click is outside, do nothing
    }
  }

  onRoiMouseMove(event: MouseEvent): void {
    if (!this.roiDragMode) return;
    event.preventDefault();
    const { x, y } = this.svgCoords(event);
    const dx = x - this.roiDragStart.mx;
    const dy = y - this.roiDragStart.my;
    const s = this.roiDragStart;

    if (this.roiType === 'rect') {
      if (this.roiDragMode === 'draw-rect') {
        const nx = Math.min(s.ox, x);
        const ny = Math.min(s.oy, y);
        this.roiRect = { x: nx, y: ny, w: Math.abs(x - s.ox), h: Math.abs(y - s.oy) };
      } else if (this.roiDragMode === 'move-rect') {
        const r = { x: s.ox + dx, y: s.oy + dy, w: s.ow, h: s.oh };
        r.x = Math.max(0, Math.min(this.roiImgW - r.w, r.x));
        r.y = Math.max(0, Math.min(this.roiImgH - r.h, r.y));
        this.roiRect = r;
      } else {
        this.applyRectHandle(this.roiDragMode, dx, dy);
      }
    } else if (this.roiType === 'ellipse') {
      if (this.roiDragMode === 'move-ellipse') {
        const e = { cx: s.ox + dx, cy: s.oy + dy, rx: s.ow, ry: s.oh };
        e.cx = Math.max(e.rx, Math.min(this.roiImgW - e.rx, e.cx));
        e.cy = Math.max(e.ry, Math.min(this.roiImgH - e.ry, e.cy));
        this.roiEllipse = e;
      } else {
        this.applyEllipseHandle(this.roiDragMode, dx, dy);
      }
    } else if (this.roiType === 'polygon') {
      if (this.roiDragMode.startsWith('p')) {
        const idx = parseInt(this.roiDragMode.substring(1), 10);
        const pts = [...this.roiPolygon];
        pts[idx] = {
          x: Math.max(0, Math.min(this.roiImgW, s.ox + dx)),
          y: Math.max(0, Math.min(this.roiImgH, s.oy + dy)),
        };
        this.roiPolygon = pts;
      } else if (this.roiDragMode === 'move-poly') {
        const lastX = (this.roiDragStart as any).lastX ?? s.mx;
        const lastY = (this.roiDragStart as any).lastY ?? s.my;
        const ddx = x - lastX;
        const ddy = y - lastY;
        this.roiPolygon = this.roiPolygon.map((pt) => ({
          x: Math.max(0, Math.min(this.roiImgW, pt.x + ddx)),
          y: Math.max(0, Math.min(this.roiImgH, pt.y + ddy)),
        }));
        (this.roiDragStart as any).lastX = x;
        (this.roiDragStart as any).lastY = y;
      }
    }
  }

  onRoiMouseUp(event: MouseEvent): void {
    if (this.roiDragMode) {
      const wasDraw = this.roiDragMode === 'draw-rect';
      this.roiDragMode = null;
      if (wasDraw && this.roiRect.w > 0 && this.roiRect.h > 0) {
        this.hasRoiShape = true;
      }
      this.commitRoi();
    }
  }

  onRoiDblClick(event: MouseEvent): void {
    // Double-click inside rect/ellipse ROI → clear it
    if (this.roiType === 'rect' || this.roiType === 'ellipse') {
      const { x, y } = this.svgCoords(event);
      const inside =
        (this.roiType === 'rect' && this.hasRoiShape && this.isInsideRect(x, y)) ||
        (this.roiType === 'ellipse' && this.hasRoiShape && this.isInsideEllipse(x, y));
      if (inside) {
        event.preventDefault();
        event.stopPropagation();
        this.clearRoiSelection();
        return;
      }
    }
    if (this.roiType === 'polygon' && this.roiPolygonDrawing && this.roiPolygon.length >= 3) {
      this.roiPolygonDrawing = false;
      this.hasRoiShape = true;
      this.commitRoi();
      return;
    }
    // Double-click inside polygon ROI → clear it
    if (this.roiType === 'polygon' && !this.roiPolygonDrawing && this.roiPolygon.length >= 3) {
      const { x, y } = this.svgCoords(event);
      const target = event.target as SVGElement;
      const handle = target.getAttribute?.('data-handle');
      if (!handle && this.isInsidePolygon(x, y)) {
        event.preventDefault();
        event.stopPropagation();
        this.clearRoiSelection();
        return;
      }
    }
    // Double-click on a polygon point → remove it
    if (this.roiType === 'polygon' && !this.roiPolygonDrawing) {
      const target = event.target as SVGElement;
      const handle = target.getAttribute?.('data-handle');
      if (handle?.startsWith('p')) {
        const idx = parseInt(handle.substring(1), 10);
        if (this.roiPolygon.length > 3) {
          this.roiPolygon = this.roiPolygon.filter((_, i) => i !== idx);
          this.commitRoi();
        } else if (this.roiPolygon.length === 3) {
          // Removing a point from a 3-point polygon clears the selection
          this.roiPolygon = [];
          this.hasRoiShape = false;
          this.commitRoi();
        }
        event.preventDefault();
        event.stopPropagation();
      }
    }
  }

  private applyRectHandle(handle: string, dx: number, dy: number): void {
    const s = this.roiDragStart;
    const r = { x: s.ox, y: s.oy, w: s.ow, h: s.oh };
    switch (handle) {
      case 'tl': r.x = s.ox + dx; r.y = s.oy + dy; r.w = s.ow - dx; r.h = s.oh - dy; break;
      case 'tr': r.y = s.oy + dy; r.w = s.ow + dx; r.h = s.oh - dy; break;
      case 'bl': r.x = s.ox + dx; r.w = s.ow - dx; r.h = s.oh + dy; break;
      case 'br': r.w = s.ow + dx; r.h = s.oh + dy; break;
      case 't': r.y = s.oy + dy; r.h = s.oh - dy; break;
      case 'b': r.h = s.oh + dy; break;
      case 'l': r.x = s.ox + dx; r.w = s.ow - dx; break;
      case 'r': r.w = s.ow + dx; break;
      default:
        // Move entire rect
        r.x = s.ox + dx;
        r.y = s.oy + dy;
    }
    if (r.w < 1) r.w = 1;
    if (r.h < 1) r.h = 1;
    // Clamp to image bounds
    r.x = Math.max(0, Math.min(this.roiImgW - r.w, r.x));
    r.y = Math.max(0, Math.min(this.roiImgH - r.h, r.y));
    r.w = Math.min(r.w, this.roiImgW - r.x);
    r.h = Math.min(r.h, this.roiImgH - r.y);
    this.roiRect = r;
  }

  private applyEllipseHandle(handle: string, dx: number, dy: number): void {
    const s = this.roiDragStart;
    const e = { ...this.roiEllipse };
    switch (handle) {
      // Vertical: both t and b move symmetrically, center stays fixed
      case 't': e.ry = Math.max(1, s.oh - dy); break;
      case 'b': e.ry = Math.max(1, s.oh + dy); break;
      // Horizontal: both l and r move symmetrically, center stays fixed
      case 'l': e.rx = Math.max(1, s.ow - dx); break;
      case 'r': e.rx = Math.max(1, s.ow + dx); break;
      case 'c': e.cx = s.ox + dx; e.cy = s.oy + dy; break;
    }
    // Clamp ellipse to image bounds
    e.rx = Math.min(e.rx, Math.min(e.cx, this.roiImgW - e.cx));
    e.ry = Math.min(e.ry, Math.min(e.cy, this.roiImgH - e.cy));
    e.cx = Math.max(e.rx, Math.min(this.roiImgW - e.rx, e.cx));
    e.cy = Math.max(e.ry, Math.min(this.roiImgH - e.ry, e.cy));
    this.roiEllipse = e;
  }

  private isInsideRect(x: number, y: number): boolean {
    const r = this.roiRect;
    return r.w > 0 && r.h > 0 && x >= r.x && x <= r.x + r.w && y >= r.y && y <= r.y + r.h;
  }

  private isInsideEllipse(x: number, y: number): boolean {
    const e = this.roiEllipse;
    if (e.rx < 1 || e.ry < 1) return false;
    const dx = (x - e.cx) / e.rx;
    const dy = (y - e.cy) / e.ry;
    return (dx * dx + dy * dy) <= 1;
  }

  private commitRoi(): void {
    if (this.roiSelectedStepIndex < 0) return;
    const pipeline = this.pipelineState.getPipeline();
    const step = pipeline.steps[this.roiSelectedStepIndex];
    if (!step) return;

    const updated = { ...step.param_values };
    if (this.roiType === 'rect') {
      updated['roi_x'] = Math.round(this.roiRect.x);
      updated['roi_y'] = Math.round(this.roiRect.y);
      updated['roi_width'] = Math.round(this.roiRect.w);
      updated['roi_height'] = Math.round(this.roiRect.h);
    } else if (this.roiType === 'ellipse') {
      updated['roi_cx'] = Math.round(this.roiEllipse.cx);
      updated['roi_cy'] = Math.round(this.roiEllipse.cy);
      updated['roi_rx'] = Math.round(this.roiEllipse.rx);
      updated['roi_ry'] = Math.round(this.roiEllipse.ry);
    } else if (this.roiType === 'polygon') {
      updated['roi_points'] = JSON.stringify(
        this.roiPolygon.map((p) => ({ x: Math.round(p.x), y: Math.round(p.y) }))
      );
    }
    this.pipelineState.updateParams(this.roiSelectedStepIndex, updated);
  }

  onRoiContextMenu(event: MouseEvent): void {
    event.preventDefault();
    const { x, y } = this.svgCoords(event);
    const inside =
      (this.roiType === 'rect' && this.hasRoiShape && this.isInsideRect(x, y)) ||
      (this.roiType === 'ellipse' && this.hasRoiShape && this.isInsideEllipse(x, y)) ||
      (this.roiType === 'polygon' && this.roiPolygon.length >= 3 && this.isInsidePolygon(x, y));
    if (inside) {
      this.roiContextMenuX = x;
      this.roiContextMenuY = y;
      // Calculate screen-space position relative to the image-roi-container
      const container = this.imageRoiContainer?.nativeElement;
      if (container) {
        const containerRect = container.getBoundingClientRect();
        this.roiContextMenuScreenX = event.clientX - containerRect.left;
        this.roiContextMenuScreenY = event.clientY - containerRect.top;
      }
      this.showRoiContextMenu = true;
    } else {
      this.showRoiContextMenu = false;
    }
  }

  clearRoiSelection(): void {
    this.showRoiContextMenu = false;
    if (this.roiType === 'rect') {
      this.roiRect = { x: 0, y: 0, w: 0, h: 0 };
    } else if (this.roiType === 'ellipse') {
      this.roiEllipse = { cx: 0, cy: 0, rx: 0, ry: 0 };
      this.roiEllipseDrawing = false;
      this.roiEllipseGuidePoints = [];
    } else if (this.roiType === 'polygon') {
      this.roiPolygon = [];
      this.roiPolygonDrawing = false;
    }
    this.hasRoiShape = false;
    this.commitRoi();
  }

  private isInsidePolygon(px: number, py: number): boolean {
    const pts = this.roiPolygon;
    let inside = false;
    for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
      const xi = pts[i].x, yi = pts[i].y;
      const xj = pts[j].x, yj = pts[j].y;
      if (((yi > py) !== (yj > py)) && (px < (xj - xi) * (py - yi) / (yj - yi) + xi)) {
        inside = !inside;
      }
    }
    return inside;
  }

  polygonPointsStr(): string {
    return this.roiPolygon.map((p) => `${p.x},${p.y}`).join(' ');
  }

  /** Preview ellipse while placing guide points */
  ellipsePreview(): { cx: number; cy: number; rx: number; ry: number } {
    const pts = this.roiEllipseGuidePoints;
    if (pts.length < 2) return { cx: 0, cy: 0, rx: 0, ry: 0 };
    const top = pts[0];
    const bottom = pts[1];
    const cy = Math.round((top.y + bottom.y) / 2);
    const ry = Math.round(Math.abs(bottom.y - top.y) / 2);
    if (pts.length < 4) {
      // Only top+bottom placed: show circle-ish preview using ry as rx too
      const cx = Math.round((top.x + bottom.x) / 2);
      return { cx, cy, rx: ry, ry };
    }
    const left = pts[2];
    const right = pts[3];
    const cx = Math.round((left.x + right.x) / 2);
    const rx = Math.round(Math.abs(right.x - left.x) / 2);
    return { cx, cy, rx, ry };
  }

  /** Hint text during ellipse drawing */
  ellipseDrawingHint(): string {
    const n = this.roiEllipseGuidePoints.length;
    if (n === 0) return 'Kattintson a felső pontra';
    if (n === 1) return 'Kattintson az alsó pontra';
    if (n === 2) return 'Kattintson a bal pontra';
    if (n === 3) return 'Kattintson a jobb pontra';
    return '';
  }

  /** Find the polygon edge nearest to (px,py) and return the index to insert a new point */
  private findNearestEdgeIndex(px: number, py: number): number {
    const pts = this.roiPolygon;
    if (pts.length < 2) return pts.length;
    let bestDist = Infinity;
    let bestIdx = pts.length;
    for (let i = 0; i < pts.length; i++) {
      const j = (i + 1) % pts.length;
      const dist = this.pointToSegmentDist(px, py, pts[i].x, pts[i].y, pts[j].x, pts[j].y);
      if (dist < bestDist) {
        bestDist = dist;
        bestIdx = j;
      }
    }
    return bestIdx;
  }

  private pointToSegmentDist(px: number, py: number, ax: number, ay: number, bx: number, by: number): number {
    const dx = bx - ax;
    const dy = by - ay;
    const lenSq = dx * dx + dy * dy;
    if (lenSq === 0) return Math.hypot(px - ax, py - ay);
    const t = Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / lenSq));
    return Math.hypot(px - (ax + t * dx), py - (ay + t * dy));
  }
}
