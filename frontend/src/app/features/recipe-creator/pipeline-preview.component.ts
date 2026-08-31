import { Component, OnInit, OnDestroy, ViewChild, ElementRef, AfterViewInit, ChangeDetectorRef } from '@angular/core';
import { CommonModule, DecimalPipe } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Subscription, combineLatest, forkJoin } from 'rxjs';
import { DataType } from '../../models/pipeline.models';
import { PipelineStateService } from '../../services/pipeline-state.service';
import { RecipeService } from '../../services/recipe.service';
import { ScatterChartComponent } from './scatter-chart.component';
import { PCAChartComponent } from './pca-chart.component';

interface ScaleBarOverlayState {
  x: number;
  y: number;
  width: number;
  height: number;
  barStartX: number;
  barEndX: number;
  barY: number;
  labelX: number;
  labelY: number;
  label: string;
  fontFamily: string;
  fontSize: number;
  barThickness: number;
  fontThickness: number;
  padding: number;
  textGap: number;
  backgroundOpacity: number;
  backgroundColor: string;
  textColor: string;
  barColor: string;
}

interface BranchMergePanel {
  label: string;
  imageSrc: string;
  sourceName: string;
  imageWidth: number;
  imageHeight: number;
  imageCount: number;
  isGrayscale: boolean;
}

interface ReferenceColorPreviewState {
  sourceSrc: string;
  alignedSrc: string;
  cropSrcs: string[];
  histograms: Record<'source' | 'reference' | 'aligned', number[][]>;
}

@Component({
  selector: 'app-pipeline-preview',
  standalone: true,
  imports: [CommonModule, FormsModule, DecimalPipe, ScatterChartComponent, PCAChartComponent],
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

      <!-- Image Tools Toolbar -->
      <div class="image-toolbar">
        <div class="toolbar-content">
          <div class="toolbar-tools">
            <button class="tool-btn icon-tool-btn" [class.active]="rulerActive"
                    (click)="toggleRuler()" title="Vonalz\u00f3 (t\u00e1vols\u00e1gm\u00e9r\u00e9s)">
              \uD83D\uDCCF
            </button>
                <button class="tool-btn icon-tool-btn" [class.active]="scaleActive"
                  (click)="toggleScale()" [disabled]="showScaleBar" title="Sk\u00e1la eszk\u00f6z">
              \uD83D\uDCD0
            </button>
            <button class="tool-btn icon-btn icon-tool-btn" (click)="saveAnnotatedImage()" title="Preview ment\u00e9se"
                    [disabled]="!hasSavablePreview()">
              \uD83D\uDCBE
            </button>
            <button class="tool-btn icon-tool-btn" [class.active]="pixelActive"
                    (click)="togglePixelTool()" title="Pixel m\u00e9r\u00e9s">
              \uD83D\uDD0D
            </button>
            <button class="tool-btn icon-tool-btn split-preview-btn"
                    [class.active]="splitPreviewActive"
                    (click)="toggleSplitPreview()"
                    [disabled]="!canShowSplitPreview()"
                    [attr.aria-pressed]="splitPreviewActive"
                    title="Osztott preview (dupla kattint\u00e1s egy node-on)">
              <span class="split-preview-icon" aria-hidden="true">
                <span></span><span></span>
              </span>
            </button>
          </div>
          @if (pixelActive) {
            <span class="pixel-color-space">{{ pixelColorSpace }}</span>
            <div class="pixel-values-display">
              @for (val of pixelGridValues; track $index) {
                }
              </div>
              <button class="tool-btn icon-btn" (click)="copyPixelValues()"
                      title="Értékek másolása" [disabled]="!pixelCurrentPos && !pixelFrozenPos">
                ⎘
              </button>
            }
            @if (rulerActive) {
              @for (i of rulerSlots; track i) {
                <input type="text" class="ruler-measurement-box" readonly
                       [value]="getRulerMeasurement(i)"
                       [class.used]="i < rulerLines.length || (i === rulerLines.length && rulerDrawingStart)">
              }
              <button class="tool-btn icon-btn" (click)="copyRulerMeasurements()"
                      title="Mérések másolása" [disabled]="rulerLines.length === 0">
                ⎘
              </button>
              <button class="tool-btn icon-btn" (click)="clearAllRulerLines()"
                      title="Vonalak törlése" [disabled]="rulerLines.length === 0">
                ✕
              </button>
            }
            @if (scaleActive || showScaleBar) {
              @if (pxPerMm > 0) {
                  <span class="scale-ratio-display">{{ getScaleResolutionDisplay() }}</span>
              }
              <span class="scale-label">Mértékegység:</span>
              <select class="scale-unit-select" [ngModel]="scaleMeasureUnit" (ngModelChange)="onScaleMeasureUnitChange($event)">
                <option value="mm">mm</option>
                <option value="cm">cm</option>
                <option value="um">um</option>
              </select>
              <span class="scale-label">Valós távolság:</span>
              <input type="number" class="scale-mm-input" [(ngModel)]="scaleMm"
                [placeholder]="scaleMeasureUnit" min="0" step="0.1"
                     (ngModelChange)="onScaleMmChange()">
              <span class="scale-unit">{{ scaleMeasureUnit }}</span>
              <button class="tool-btn icon-btn" (click)="clearScaleLine()"
                      title="Kalibráló vonal törlése" [disabled]="scaleStart === null && scaleEnd === null">
                ✕
              </button>
              <span class="scale-label">Skála hossza:</span>
              <input type="number" class="scale-mm-input" [(ngModel)]="scaleBarLengthMm"
                [placeholder]="scaleBarUnit" min="0" step="1"
                (ngModelChange)="onScaleBarLengthChange()">
              <span class="scale-label">Betűméret:</span>
              <input type="number" class="scale-mm-input" [(ngModel)]="scaleBarFontSize"
                     min="8" step="1" (ngModelChange)="onScaleBarStyleChange()">
              <span class="scale-label">Vonalvastagság:</span>
              <input type="number" class="scale-mm-input" [(ngModel)]="scaleBarBarThickness"
                     min="1" step="1" (ngModelChange)="onScaleBarStyleChange()">
              <span class="scale-label">Betűvastagság:</span>
              <input type="number" class="scale-mm-input" [(ngModel)]="scaleBarFontThickness"
                     min="1" step="1" (ngModelChange)="onScaleBarStyleChange()">
              <span class="scale-label">Vonal színe:</span>
              <select class="scale-unit-select" [ngModel]="scaleBarBarColor" (ngModelChange)="onScaleBarStyleChange($event, 'barColor')">
                <option value="white">fehér</option>
                <option value="black">fekete</option>
                <option value="yellow">sárga</option>
              </select>
              <span class="scale-label">Szöveg színe:</span>
              <select class="scale-unit-select" [ngModel]="scaleBarTextColor" (ngModelChange)="onScaleBarStyleChange($event, 'textColor')">
                <option value="white">fehér</option>
                <option value="black">fekete</option>
                <option value="yellow">sárga</option>
              </select>
              <label class="scale-checkbox-label">
                <input type="checkbox" [(ngModel)]="showScaleBar"
                       (ngModelChange)="onShowScaleBarChange()">
                <span>Skála mutatása</span>
              </label>
            }
          <div class="toolbar-spacer"></div>
          <!-- Montage button - shows if more than 1 image -->
          @if (imageCount > 1) {
            <button class="tool-btn icon-tool-btn"
                    (click)="generateAndShowMontage()" 
                    [disabled]="generatingMontage"
                    title="Mont\u00e1zs n\u00e9zet (összes k\u00e9p egy r\u00e1csban)">
              @if (!generatingMontage) {
                <span>\uD83C\uDF9E\uFE0F</span>
              } @else {
                <span>\u23F3</span>
              }
            </button>
          }
        </div>
      </div>
      @if (loading || generatingMontage) {
        <div class="loading-overlay">
          <div class="spinner"></div>
          <span>{{ generatingMontage ? 'Montázs betöltése...' : 'Előnézet betöltése...' }}</span>
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
            <button class="graph-tool-btn" (click)="copyGraphEquation()" [disabled]="!hasGraphData()" [class.disabled]="!hasGraphData()" title="Egyenlet másolása">
              Egyenlet másolása
            </button>
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
                 [style.top.px]="graphContextMenuY"
                 (mousedown)="$event.stopPropagation()"
                 (click)="$event.stopPropagation()"
                 (contextmenu)="$event.stopPropagation()">
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
      @if (showExpandedChart && expandedChartData) {
        <div class="expanded-chart-viewer">
          <div class="expanded-chart-toolbar">
            <span class="expanded-chart-title">{{ expandedChartTitle }}</span>
            <button class="graph-close-btn" (click)="closeExpandedChart()" title="Bezárás">✕</button>
          </div>
          <div class="expanded-chart-container">
            @if (expandedChartType === 'scatter') {
              <app-scatter-chart
                [data]="expandedChartData"
                [label]="expandedChartTitle"
                [width]="1400"
                [height]="800"
              />
            } @else if (expandedChartType === 'pca') {
              <div class="pca-chart-wrapper">
                <app-pca-chart
                  [data]="expandedChartData"
                  (componentChanged)="onPCAComponentChanged($event)"
                />
              </div>
            }
          </div>
        </div>
      }
      <div class="preview-scroll-area" #scrollArea
           [class.zoomed]="zoomLevel > 1">
        @if (showingMontage && montagePreview && !showGraphViewer) {
          <!-- Montage gallery - replaces normal preview -->
          <div class="montage-gallery-container">
            <div class="montage-gallery-wrapper">
              <img [src]="montagePreview" 
                   alt="Montázs nézet" 
                   class="montage-gallery-image"
                   (click)="onMontageImageClick($event)"
                   (load)="onMontageImageLoaded()"
                   style="cursor: pointer;" />
            </div>
          </div>
        } @else if (referenceColorPreview && !showGraphViewer) {
          <div class="reference-color-preview">
            <div class="reference-color-panels">
              <div class="reference-color-panel">
                <div class="branch-merge-title">Eredeti, illesztendő kép</div>
                <img [src]="referenceColorPreview.sourceSrc" class="reference-color-main-image" alt="Eredeti kép" draggable="false" />
              </div>
              <div class="reference-color-panel">
                <div class="branch-merge-title">Referencia cropok</div>
                <div class="reference-color-crops">
                  @for (cropSrc of referenceColorPreview.cropSrcs; track $index) {
                    <img [src]="cropSrc" class="reference-color-crop" [alt]="'Referencia ' + ($index + 1)" draggable="false" />
                  }
                </div>
              </div>
              <div class="reference-color-panel">
                <div class="branch-merge-title">Illesztett eredmény</div>
                <img [src]="referenceColorPreview.alignedSrc" class="reference-color-main-image" alt="Illesztett kép" draggable="false" />
              </div>
            </div>
            <div class="reference-color-histograms">
              @for (kind of referenceHistogramKinds; track kind.key) {
                <div class="reference-histogram-card">
                  <div class="reference-histogram-title">{{ kind.label }} – LAB hisztogram{{ kind.key === 'reference' ? ' (összes crop együtt)' : '' }}</div>
                  <svg viewBox="0 0 255 100" preserveAspectRatio="none" class="reference-histogram-svg" role="img">
                    @for (channel of [0, 1, 2]; track channel) {
                      <polyline [attr.points]="getReferenceHistogramPoints(kind.key, channel)" [attr.class]="'hist-line hist-line-' + channel" />
                      @if (kind.key === 'aligned') {
                        <polyline [attr.points]="getReferenceHistogramPoints('reference', channel)" [attr.class]="'hist-line hist-reference-overlay hist-line-' + channel" />
                      }
                    }
                  </svg>
                  <div class="reference-histogram-legend"><span class="lab-l">L</span><span class="lab-a">A</span><span class="lab-b">B</span>@if (kind.key === 'aligned') { <span>szaggatott: referencia</span> }</div>
                </div>
              }
            </div>
          </div>
        } @else if (showBranchMergeView() && !showGraphViewer) {
          <div class="branch-merge-compare-container">
            @for (panel of branchMergePanels; track $index) {
              <div class="branch-merge-panel">
                <div class="branch-merge-title">{{ panel.label }}</div>
                <img
                  [src]="panel.imageSrc"
                  [alt]="panel.label"
                  class="branch-merge-image"
                  [class.grayscale]="panel.isGrayscale"
                  draggable="false"
                />
                <div class="branch-merge-meta">
                  <span>{{ panel.sourceName || 'Kijelolt kep' }}</span>
                  <span>{{ panel.imageWidth }} x {{ panel.imageHeight }}</span>
                  <span>{{ panel.imageCount }} kep</span>
                </div>
              </div>
            }
          </div>
        } @else if (showKmeansComparison() && kmeansSourceSrc && kmeansOverlaySrc && !showGraphViewer) {
          <div class="gray-map-compare-container">
            <div class="gray-map-compare-panel">
              <div class="gray-map-compare-title">Eredeti kép</div>
              <img [src]="kmeansSourceSrc" alt="Eredeti kép" class="gray-map-compare-image" draggable="false" />
            </div>
            <div class="gray-map-compare-panel">
              <div class="gray-map-compare-title">Klaszter overlay</div>
              <img [src]="kmeansOverlaySrc" alt="Klaszterek az eredeti képen" class="gray-map-compare-image" draggable="false" />
              <div class="cluster-legend" aria-label="Klaszter jelmagyarázat">
                @for (item of kmeansLegend; track item.label) {
                  <label class="cluster-legend-item cluster-legend-item--editable" title="Kattints a klaszter szinenek modositasahoz">
                    <span class="cluster-legend-swatch" [style.background]="item.color"></span>
                    <span>Label {{ item.label }}</span>
                    <input
                      type="color"
                      class="cluster-legend-color-input"
                      [value]="normalizeLegendColor(item.color)"
                      (change)="onKmeansLegendColorChange(item.label, $event)"
                    />
                  </label>
                }
              </div>
            </div>
          </div>
        } @else if (showClusterReferenceMap() && kmeansOverlaySrc && clusterMapSrc && !showGraphViewer) {
          <div class="gray-map-compare-container">
            <div class="gray-map-compare-panel">
              <div class="gray-map-compare-title">K-közép overlay</div>
              <img [src]="kmeansOverlaySrc" alt="K-közép klaszterek az eredeti képen" class="gray-map-compare-image" draggable="false" />
              <div class="cluster-legend" aria-label="Klaszter jelmagyarázat">
                @for (item of kmeansLegend; track item.label) {
                  <label class="cluster-legend-item cluster-legend-item--editable" title="Kattints a klaszter szinenek modositasahoz">
                    <span class="cluster-legend-swatch" [style.background]="item.color"></span>
                    <span>Label {{ item.label }}</span>
                    <input
                      type="color"
                      class="cluster-legend-color-input"
                      [value]="normalizeLegendColor(item.color)"
                      (change)="onKmeansLegendColorChange(item.label, $event)"
                    />
                  </label>
                }
              </div>
              @if (clusterMapLabelValues.length > 0) {
                <div class="cluster-value-chart">
                  @for (component of clusterMapValueComponents; track component) {
                    <div class="reference-sequence-histogram-title">{{ component }}</div>
                    @for (item of clusterMapLabelValues; track item.label) {
                      <div class="reference-sequence-bar-row">
                        <span class="reference-sequence-bar-label">Label {{ item.label }}</span>
                        <div class="reference-sequence-bar-track">
                          <div
                            class="reference-sequence-bar-fill"
                            [style.width.%]="getClusterMapValueWidth(item, component)"
                            [style.background]="getKmeansLegendColor(item.label)"
                          ></div>
                        </div>
                        <span class="reference-sequence-bar-value">
                          {{ getClusterMapValueLabel(item, component) }}
                        </span>
                      </div>
                    }
                  }
                </div>
              }
            </div>
            <div class="gray-map-compare-panel">
              <div class="gray-map-compare-title cluster-map-title">
                <span>Referencia map – aktuális maradék</span>
                <button
                  type="button"
                  class="cluster-map-accept"
                  (click)="acceptClusterMap()"
                  [disabled]="clusterMapRemainderIsFinal"
                  title="Aktuális térkép eltárolása"
                >Kész</button>
              </div>
              <img [src]="clusterMapSrc" alt="Referencia map" class="gray-map-compare-image" draggable="false" />
            </div>
          </div>
        } @else if (showGrayMapComparison() && grayMapBaseSrc && grayMapOverlaySrc && !showGraphViewer) {
          <div class="gray-map-compare-container">
            <div class="gray-map-compare-panel">
              <div class="gray-map-compare-title">Eredeti betöltött kép</div>
              <img
                [src]="grayMapBaseSrc"
                alt="Eredeti kép"
                class="gray-map-compare-image"
                [class.grayscale]="isGrayscale"
                draggable="false"
              />
            </div>
            <div class="gray-map-compare-panel">
              <div class="gray-map-compare-title">Kiválasztott eredmény</div>
              <img
                [src]="grayMapOverlaySrc"
                alt="Gray map eredmény"
                class="gray-map-compare-image"
                draggable="false"
              />
            </div>
          </div>
        } @else if (showDualMapView() && dualMapState && !showGraphViewer) {
          <!-- dual_map: 2-3 rows (gray / RGB / sub), columns = original + N component results -->
          <div class="dual-map-multipanel"
               [style.grid-template-columns]="'repeat(' + (dualMapMaxCols + 1) + ', minmax(0, 1fr))'"
               [style.grid-template-rows]="dualMapState.subOverlays.length > 0 ? '1fr 1fr 1fr' : '1fr 1fr'">
            <!-- Row 1: Gray original + gray components -->
            <div class="dual-map-cell">
              <div class="dual-map-cell-title">Szürke — eredeti</div>
              @if (dualMapState.grayBase) {
                <img [src]="dualMapState.grayBase" alt="Szürke eredeti" class="dual-map-cell-image" draggable="false" />
              } @else {
                <div class="dual-map-cell-empty">–</div>
              }
            </div>
            @for (overlay of dualMapState.grayOverlays; track $index) {
              <div class="dual-map-cell">
                <div class="dual-map-cell-title">Szürke — {{ $index + 1 }}. komponens</div>
                <img [src]="overlay" alt="Szürke {{ $index + 1 }}. komponens" class="dual-map-cell-image" draggable="false" />
              </div>
            }
            @if (!dualMapState.grayOverlays.length) {
              <div class="dual-map-cell"><div class="dual-map-cell-empty">–</div></div>
            }
            <!-- Row 2: RGB original + RGB components -->
            <div class="dual-map-cell">
              <div class="dual-map-cell-title">RGB — eredeti</div>
              @if (dualMapState.rgbBase) {
                <img [src]="dualMapState.rgbBase" alt="RGB eredeti" class="dual-map-cell-image" draggable="false" />
              } @else {
                <div class="dual-map-cell-empty">–</div>
              }
            </div>
            @for (overlay of dualMapState.rgbOverlays; track $index) {
              <div class="dual-map-cell">
                <div class="dual-map-cell-title">RGB — {{ $index + 1 }}. komponens</div>
                <img [src]="overlay" alt="RGB {{ $index + 1 }}. komponens" class="dual-map-cell-image" draggable="false" />
              </div>
            }
            @if (!dualMapState.rgbOverlays.length) {
              <div class="dual-map-cell"><div class="dual-map-cell-empty">–</div></div>
            }
            <!-- Row 3 (optional): Sub-classification original + sub components -->
            @if (dualMapState.subOverlays.length > 0) {
              <div class="dual-map-cell dual-map-cell--sub">
                <div class="dual-map-cell-title">{{ dualMapState.subLabel }} — eredeti</div>
                @if (dualMapState.subBase) {
                  <img [src]="dualMapState.subBase" alt="Sub eredeti" class="dual-map-cell-image" draggable="false" />
                } @else {
                  <div class="dual-map-cell-empty">–</div>
                }
              </div>
              @for (overlay of dualMapState.subOverlays; track $index) {
                <div class="dual-map-cell dual-map-cell--sub">
                  <div class="dual-map-cell-title">{{ dualMapState.subLabel }} — {{ $index + 1 }}. komponens</div>
                  <img [src]="overlay" alt="{{ dualMapState.subLabel }} {{ $index + 1 }}. komponens" class="dual-map-cell-image" draggable="false" />
                </div>
              }
            }
          </div>
        } @else if (referenceCropStripActive && referenceCropImages.length > 0 && !showGraphViewer) {
          <div class="reference-sequence-view" [class.reference-sequence-view--with-histogram]="referenceCropScores.length > 0">
            <div class="reference-crop-strip">
              @for (src of referenceCropImages; track $index) {
                <div class="reference-crop-tile">
                  <img [src]="src" [alt]="getReferenceCropLabel($index)" draggable="false" />
                  <span class="reference-crop-name-badge">{{ getReferenceCropLabel($index) }}</span>
                  @if (referenceCropScores.length > $index) {
                    <span class="reference-crop-score-badge">{{ getReferenceCropScoreLabel($index) }}</span>
                  }
                </div>
              }
            </div>
            @if (referenceSequenceComponents.length > 0) {
              <div class="reference-sequence-histogram">
                @for (component of referenceSequenceComponents; track component) {
                  <div class="reference-sequence-histogram-title">{{ component }}</div>
                  <div class="reference-sequence-bars">
                    @for (score of getReferenceComponentScores(component); track $index) {
                      <div class="reference-sequence-bar-row">
                        <span class="reference-sequence-bar-label">{{ getReferenceCropLabel($index) }}</span>
                        <div class="reference-sequence-bar-track">
                          <div
                            class="reference-sequence-bar-fill"
                            [style.width.%]="getReferenceComponentBarWidth(component, score)"
                            [style.background]="getReferenceSequenceColor(component)"
                          ></div>
                        </div>
                        <span class="reference-sequence-bar-value">
                          {{ getReferenceComponentScoreLabel(component, $index) }}
                          @if (getReferenceComponentDiffLabel(component, $index)) {
                            <small>{{ getReferenceComponentDiffLabel(component, $index) }}</small>
                          }
                        </span>
                      </div>
                    }
                  </div>
                }
              </div>
            }
          </div>
        } @else if (imageSrc && !showGraphViewer) {
          <div class="image-roi-container" #imageRoiContainer>
            <img #previewImg
              [src]="imageSrc"
              alt="Pipeline előnézet"
              class="preview-image"
              [class.grayscale]="isGrayscale"
              draggable="false"
              (load)="onImageLoad()"
            />
            @if (referenceCropActive) {
              <svg class="reference-crop-overlay"
                   [attr.viewBox]="'0 0 ' + referenceCropImgW + ' ' + referenceCropImgH"
                   (mousedown)="onReferenceCropMouseDown($event)"
                   (mousemove)="onReferenceCropMouseMove($event)"
                   (mouseup)="onReferenceCropMouseUp($event)"
                   (mouseleave)="onReferenceCropMouseUp($event)"
                   (contextmenu)="onReferenceCropContextMenu($event)">
                @for (sq of referenceCropSquares; track $index) {
                  <g>
                    <rect [attr.x]="sq.x"
                          [attr.y]="sq.y"
                          [attr.width]="sq.size"
                          [attr.height]="sq.size"
                          class="reference-crop-square"
                          [class.dragging]="referenceCropDragIndex === $index"
                          [attr.data-ref-index]="$index" />
                      <text [attr.x]="sq.x + 6 * referenceCropScale"
                          [attr.y]="sq.y + 18 * referenceCropScale"
                          class="reference-crop-label"
                          [attr.font-size]="14 * referenceCropScale"
                          [attr.data-ref-index]="$index">
                      {{ getReferenceCropOverlayLabel($index) }}
                    </text>
                  </g>
                }
              </svg>
            }
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
                  <g [attr.transform]="roiRectTransform()">
                    <rect [attr.x]="roiRect.x" [attr.y]="roiRect.y"
                          [attr.width]="roiRect.w" [attr.height]="roiRect.h"
                          [attr.fill]="roiAllSelected ? 'rgba(249,115,22,0.25)' : 'rgba(59,130,246,0.15)'"
                          [attr.stroke]="roiAllSelected ? '#f97316' : '#3b82f6'"
                          [attr.stroke-width]="2 * roiScale" [attr.stroke-dasharray]="(6 * roiScale) + ' ' + (3 * roiScale)"
                          class="roi-shape-body" />
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
                    <!-- Rotation handle (inside top-right corner) -->
                    <circle [attr.cx]="roiRect.x + roiRect.w - 14 * roiScale" [attr.cy]="roiRect.y + 14 * roiScale"
                            [attr.r]="10 * roiScale" [attr.fill]="'#f97316'" [attr.stroke]="'#ea580c'" [attr.stroke-width]="2 * roiScale"
                            class="roi-rot-btn" data-handle="rot" />
                    <text [attr.x]="roiRect.x + roiRect.w - 14 * roiScale" [attr.y]="roiRect.y + 14 * roiScale + 5 * roiScale"
                          text-anchor="middle" [attr.fill]="'white'" [attr.font-size]="13 * roiScale" font-family="Arial, sans-serif"
                          class="roi-rot-icon" data-handle="rot">&#x21bb;</text>
                  </g>
                }
                <!-- Ellipse ROI -->
                @if (roiType === 'ellipse' && hasRoiShape) {
                  <g [attr.transform]="roiEllipseTransform()">
                    <ellipse [attr.cx]="roiEllipse.cx" [attr.cy]="roiEllipse.cy"
                             [attr.rx]="roiEllipse.rx" [attr.ry]="roiEllipse.ry"
                             [attr.fill]="roiAllSelected ? 'rgba(249,115,22,0.25)' : 'rgba(59,130,246,0.15)'"
                             [attr.stroke]="roiAllSelected ? '#f97316' : '#3b82f6'"
                             [attr.stroke-width]="2 * roiScale" [attr.stroke-dasharray]="(6 * roiScale) + ' ' + (3 * roiScale)"
                             class="roi-shape-body" />
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
                    <!-- Rotation handle (near top of ellipse, offset right) -->
                    <circle [attr.cx]="roiEllipse.cx + roiEllipse.rx * 0.7" [attr.cy]="roiEllipse.cy - roiEllipse.ry * 0.7"
                            [attr.r]="10 * roiScale" [attr.fill]="'#f97316'" [attr.stroke]="'#ea580c'" [attr.stroke-width]="2 * roiScale"
                            class="roi-rot-btn" data-handle="rot" />
                    <text [attr.x]="roiEllipse.cx + roiEllipse.rx * 0.7" [attr.y]="roiEllipse.cy - roiEllipse.ry * 0.7 + 5 * roiScale"
                          text-anchor="middle" [attr.fill]="'white'" [attr.font-size]="13 * roiScale" font-family="Arial, sans-serif"
                          class="roi-rot-icon" data-handle="rot">&#x21bb;</text>
                  </g>
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
                  <g [attr.transform]="roiPolygonTransform()">
                    @if (roiPolygon.length > 2) {
                      <polygon [attr.points]="polygonPointsStr()"
                               [attr.fill]="roiAllSelected ? 'rgba(249,115,22,0.25)' : 'rgba(59,130,246,0.15)'"
                               [attr.stroke]="roiAllSelected ? '#f97316' : '#3b82f6'"
                               [attr.stroke-width]="2 * roiScale" [attr.stroke-dasharray]="(6 * roiScale) + ' ' + (3 * roiScale)"
                               class="roi-shape-body" />
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
                    <!-- Rotation handle (at topmost point of polygon) -->
                    @if (roiPolygon.length >= 3 && !roiPolygonDrawing) {
                      <circle [attr.cx]="roiPolygonTopRight().x" [attr.cy]="roiPolygonTopRight().y"
                              [attr.r]="10 * roiScale" [attr.fill]="'#f97316'" [attr.stroke]="'#ea580c'" [attr.stroke-width]="2 * roiScale"
                              class="roi-rot-btn" data-handle="rot" />
                      <text [attr.x]="roiPolygonTopRight().x" [attr.y]="roiPolygonTopRight().y + 5 * roiScale"
                            text-anchor="middle" [attr.fill]="'white'" [attr.font-size]="13 * roiScale" font-family="Arial, sans-serif"
                            class="roi-rot-icon" data-handle="rot">&#x21bb;</text>
                    }
                  </g>
                }
                <!-- Selection hint (Ctrl+A active) -->
                @if (roiAllSelected) {
                  <rect [attr.x]="0" [attr.y]="roiImgH - 28 * roiScale" [attr.width]="roiImgW" [attr.height]="28 * roiScale"
                        fill="rgba(0,0,0,0.55)" />
                  <text [attr.x]="10 * roiScale" [attr.y]="roiImgH - 9 * roiScale"
                        fill="#f97316" [attr.font-size]="13 * roiScale" font-family="sans-serif">
                    ROI kijelölve — Delete a törléshez, Esc a kijelölés megszüntetéséhez
                  </text>
                }
                <!-- Per-image ROI hint -->
                @if (hasRoiShape && imageCount > 1 && !roiHasOverride) {
                  <rect [attr.x]="0" [attr.y]="0" [attr.width]="roiImgW" [attr.height]="28 * roiScale"
                        fill="rgba(0,0,0,0.55)" />
                  <text [attr.x]="10 * roiScale" [attr.y]="19 * roiScale"
                        fill="#fbbf24" [attr.font-size]="13 * roiScale" font-family="sans-serif">
                    ⚠ Örökölt ROI — húzza a kívánt pozícióba
                  </text>
                }
              </svg>
            }
            @if (particleOverlayActive && particlesForOverlay.length > 0) {
              <svg class="particle-overlay"
                   [attr.viewBox]="'0 0 ' + particleImgW + ' ' + particleImgH">
                @for (p of particlesForOverlay; track p.particle_id) {
                  <polygon [attr.points]="particlePolygonStr(p)"
                           [class.excluded]="isParticleExcluded(p.particle_id)"
                           [class.filtered-out]="!p.passed_filters && !p.excluded"
                           (click)="onParticleClick(p, $event)"
                           class="particle-hitarea" />
                }
              </svg>
            }
            <!-- Circle detection overlay -->
            @if (circleOverlayActive && circlesForOverlay.length > 0) {
              <svg class="circle-overlay"
                   [attr.viewBox]="'0 0 ' + circleImgW + ' ' + circleImgH">
                @for (c of circlesForOverlay; track $index) {
                  <circle [attr.cx]="c.center_x"
                          [attr.cy]="c.center_y"
                          [attr.r]="c.radius"
                          class="detection-circle"
                          [title]="'Radius: ' + c.radius + 'px'" />
                  <circle [attr.cx]="c.center_x"
                          [attr.cy]="c.center_y"
                          [attr.r]="2"
                          class="circle-center-point" />
                }
              </svg>
            }
            <!-- Measurement & annotation overlay -->
            @if (rulerActive || scaleActive || pixelActive || hasAnnotations) {
              <svg class="ruler-overlay"
                   [attr.viewBox]="'0 0 ' + rulerImgW + ' ' + rulerImgH"
                   [style.pointer-events]="(rulerActive || scaleActive || pixelActive || showScaleBar) ? 'auto' : 'none'"
                   [style.cursor]="(rulerActive || scaleActive || pixelActive) ? 'crosshair' : (showScaleBar ? 'grab' : 'default')"
                   (mousedown)="onToolMouseDown($event)"
                   (mousemove)="onToolMouseMove($event)"
                   (mouseleave)="onToolMouseLeave()"
                   (click)="onToolClick($event)">
                <!-- Completed ruler lines -->
                @for (line of rulerLines; track $index) {
                  <g (contextmenu)="onRulerLineContextMenu($event, $index)">
                    <line [attr.x1]="line.start.x" [attr.y1]="line.start.y"
                          [attr.x2]="line.end.x" [attr.y2]="line.end.y"
                          stroke="transparent" [attr.stroke-width]="12 * rulerScale"
                          style="pointer-events:stroke"/>
                    <line [attr.x1]="line.start.x" [attr.y1]="line.start.y"
                          [attr.x2]="line.end.x" [attr.y2]="line.end.y"
                          stroke="#1a5fb4" [attr.stroke-width]="2 * rulerScale"/>
                    <circle [attr.cx]="line.start.x" [attr.cy]="line.start.y" [attr.r]="6 * rulerScale"
                            fill="#1a5fb4" stroke="#fff" [attr.stroke-width]="1.5 * rulerScale"/>
                    <circle [attr.cx]="line.end.x" [attr.cy]="line.end.y" [attr.r]="6 * rulerScale"
                            fill="#1a5fb4" stroke="#fff" [attr.stroke-width]="1.5 * rulerScale"/>
                    <text [attr.x]="(line.start.x + line.end.x) / 2 + 12 * rulerScale"
                          [attr.y]="(line.start.y + line.end.y) / 2 - 10 * rulerScale"
                          fill="#fff" [attr.font-size]="13 * rulerScale" font-family="monospace"
                          stroke="#000" [attr.stroke-width]="3 * rulerScale" paint-order="stroke">
                      {{ line.distance | number:'1.1-1' }} px
                    </text>
                  </g>
                }
                <!-- Line being drawn (ruler) -->
                @if (rulerDrawingStart && rulerActive) {
                  <circle [attr.cx]="rulerDrawingStart.x" [attr.cy]="rulerDrawingStart.y" [attr.r]="6 * rulerScale"
                          fill="#1a5fb4" stroke="#fff" [attr.stroke-width]="1.5 * rulerScale"/>
                  @if (rulerDrawingCurrent) {
                    <line [attr.x1]="rulerDrawingStart.x" [attr.y1]="rulerDrawingStart.y"
                          [attr.x2]="rulerDrawingCurrent.x" [attr.y2]="rulerDrawingCurrent.y"
                          stroke="#1a5fb4" [attr.stroke-width]="2 * rulerScale"
                          [attr.stroke-dasharray]="(8 * rulerScale) + ' ' + (5 * rulerScale)"/>
                    <text [attr.x]="rulerDrawingCurrent.x + 16 * rulerScale"
                          [attr.y]="rulerDrawingCurrent.y - 12 * rulerScale"
                          fill="#fff" [attr.font-size]="14 * rulerScale" font-family="monospace"
                          stroke="#000" [attr.stroke-width]="3 * rulerScale" paint-order="stroke">
                      {{ rulerDrawingDistance | number:'1.1-1' }} px
                    </text>
                  }
                }
                <!-- Scale line (only show while drawing, hide once calibrated) -->
                @if (scaleStart) {
                  <circle [attr.cx]="scaleStart.x" [attr.cy]="scaleStart.y" [attr.r]="6 * rulerScale"
                          fill="#e67e22" stroke="#fff" [attr.stroke-width]="1.5 * rulerScale"/>
                  @if (scaleEnd) {
                    <line [attr.x1]="scaleStart.x" [attr.y1]="scaleStart.y"
                          [attr.x2]="scaleEnd.x" [attr.y2]="scaleEnd.y"
                          stroke="#e67e22" [attr.stroke-width]="2 * rulerScale"/>
                    <circle [attr.cx]="scaleEnd.x" [attr.cy]="scaleEnd.y" [attr.r]="6 * rulerScale"
                            fill="#e67e22" stroke="#fff" [attr.stroke-width]="1.5 * rulerScale"/>
                    <text [attr.x]="(scaleStart.x + scaleEnd.x) / 2 + 16 * rulerScale"
                          [attr.y]="(scaleStart.y + scaleEnd.y) / 2 - 12 * rulerScale"
                          fill="#fff" [attr.font-size]="14 * rulerScale" font-family="monospace"
                          stroke="#000" [attr.stroke-width]="3 * rulerScale" paint-order="stroke">
                      {{ scaleLinePx | number:'1.1-1' }} px
                    </text>
                  } @else if (scaleCurrentPos) {
                    <line [attr.x1]="scaleStart.x" [attr.y1]="scaleStart.y"
                          [attr.x2]="scaleCurrentPos.x" [attr.y2]="scaleCurrentPos.y"
                          stroke="#e67e22" [attr.stroke-width]="2 * rulerScale"
                          [attr.stroke-dasharray]="(8 * rulerScale) + ' ' + (5 * rulerScale)"/>
                    <text [attr.x]="scaleCurrentPos.x + 16 * rulerScale"
                          [attr.y]="scaleCurrentPos.y - 12 * rulerScale"
                          fill="#fff" [attr.font-size]="14 * rulerScale" font-family="monospace"
                          stroke="#000" [attr.stroke-width]="3 * rulerScale" paint-order="stroke">
                      {{ scaleLinePx | number:'1.1-1' }} px
                    </text>
                  }
                }
                <!-- Pixel measurement 3x3 grid -->
                @if (pixelActive && (pixelCurrentPos !== null || pixelFrozenPos !== null)) {
                  @let pos = pixelFrozenPos || pixelCurrentPos;
                  @if (pos) {
                    @for (val of pixelGridValues; track $index) {
                      @let row = Math.floor($index / 3) - 1;
                      @let col = ($index % 3) - 1;
                      @let cx = pos.x + col * pixelGridSpacing + pixelGridOffsetX;
                      @let cy = pos.y + row * pixelGridSpacing + pixelGridOffsetY;
                      @let lines = getPixelDisplayLines(val);
                      <g>
                        <rect [attr.x]="cx - pixelGridHalfSize" [attr.y]="cy - pixelGridHalfSize"
                              [attr.width]="pixelGridCellSize" [attr.height]="pixelGridCellSize"
                              [attr.fill]="pixelGridColors[$index]"
                              stroke="#fff" [attr.stroke-width]="pixelGridStrokeWidth"/>
                        <text [attr.x]="cx" [attr.y]="getPixelTextStartY(cy, lines.length)"
                              [attr.fill]="isColorBright(pixelGridColors[$index]) ? '#000' : '#fff'"
                              [attr.font-size]="pixelGridFontSize" font-family="monospace"
                              text-anchor="middle">
                          @for (line of lines; track $index) {
                            <tspan [attr.x]="cx" [attr.dy]="$index === 0 ? 0 : pixelGridLineHeight">{{ line }}</tspan>
                          }
                        </text>
                      </g>
                    }
                  }
                }
                <!-- Scale bar overlay -->
                  @if (showScaleBar && pxPerMm > 0 && scaleBarPx > 0 && !scaleBarOverlay) {
                    <g [style.cursor]="scaleBarDragging ? 'grabbing' : 'grab'"
                           (mousedown)="onScaleBarMouseDown($event)">
                      <rect [attr.x]="rulerImgW - scaleBarPx - 35 * rulerScale"
                        [attr.y]="rulerImgH - 58 * rulerScale"
                        [attr.width]="scaleBarPx + 30 * rulerScale"
                        [attr.height]="44 * rulerScale"
                        fill="rgba(0,0,0,0.55)" [attr.rx]="4 * rulerScale"/>
                      <line [attr.x1]="rulerImgW - scaleBarPx - 20 * rulerScale"
                        [attr.y1]="rulerImgH - 25 * rulerScale"
                        [attr.x2]="rulerImgW - 20 * rulerScale"
                        [attr.y2]="rulerImgH - 25 * rulerScale"
                        stroke="#fff" [attr.stroke-width]="3 * rulerScale" stroke-linecap="round"/>
                      <circle [attr.cx]="rulerImgW - scaleBarPx - 20 * rulerScale"
                          [attr.cy]="rulerImgH - 25 * rulerScale"
                          [attr.r]="Math.max(1.5, 1.4 * rulerScale)"
                          fill="#fff"/>
                      <circle [attr.cx]="rulerImgW - 20 * rulerScale"
                          [attr.cy]="rulerImgH - 25 * rulerScale"
                          [attr.r]="Math.max(1.5, 1.4 * rulerScale)"
                          fill="#fff"/>
                      <text [attr.x]="rulerImgW - scaleBarPx / 2 - 20 * rulerScale"
                        [attr.y]="rulerImgH - 42 * rulerScale"
                        fill="#fff" [attr.font-size]="13 * rulerScale" font-family="sans-serif"
                        text-anchor="middle">
                      {{ formatScaleBarLabel() }}
                      </text>
                    </g>
                }
                @if (scaleBarOverlay) {
                  <g [style.cursor]="scaleBarDragging ? 'grabbing' : 'grab'"
                         (mousedown)="onScaleBarMouseDown($event)">
                    <rect [attr.x]="scaleBarOverlay.x"
                      [attr.y]="scaleBarOverlay.y"
                      [attr.width]="scaleBarOverlay.width"
                      [attr.height]="scaleBarOverlay.height"
                      [attr.fill]="'rgba(0,0,0,' + scaleBarOverlay.backgroundOpacity + ')'"
                      [attr.rx]="4 * rulerScale"
                      [attr.stroke]="scaleBarDragging ? '#ffd166' : 'rgba(255, 209, 102, 0.9)'"
                      [attr.stroke-width]="2 * rulerScale"
                      [attr.stroke-dasharray]="scaleBarDragging ? '8 4' : '6 4'"/>
                    <line [attr.x1]="scaleBarOverlay.barStartX"
                      [attr.y1]="scaleBarOverlay.barY"
                      [attr.x2]="scaleBarOverlay.barEndX"
                      [attr.y2]="scaleBarOverlay.barY"
                      [attr.stroke]="scaleBarOverlay.barColor"
                      [attr.stroke-width]="scaleBarOverlay.barThickness"
                      stroke-linecap="round"/>
                    <circle [attr.cx]="scaleBarOverlay.barStartX"
                      [attr.cy]="scaleBarOverlay.barY"
                      [attr.r]="Math.max(1.5, scaleBarOverlay.barThickness * 0.55)"
                      [attr.fill]="scaleBarOverlay.barColor"/>
                    <circle [attr.cx]="scaleBarOverlay.barEndX"
                      [attr.cy]="scaleBarOverlay.barY"
                      [attr.r]="Math.max(1.5, scaleBarOverlay.barThickness * 0.55)"
                      [attr.fill]="scaleBarOverlay.barColor"/>
                    <text [attr.x]="scaleBarOverlay.labelX"
                      [attr.y]="scaleBarOverlay.labelY"
                      [attr.fill]="scaleBarOverlay.textColor"
                      [attr.font-size]="scaleBarOverlay.fontSize"
                      [attr.font-family]="scaleBarOverlay.fontFamily"
                      [attr.stroke]="scaleBarOverlay.backgroundColor"
                      [attr.stroke-width]="scaleBarOverlay.fontThickness + 2"
                      paint-order="stroke"
                      text-anchor="middle">
                      {{ scaleBarOverlay.label }}
                    </text>
                  </g>
                }
              </svg>
            }
            @if (showRulerContextMenu) {
              <div class="ruler-context-menu"
                   [style.left.px]="rulerContextMenuScreenX"
                   [style.top.px]="rulerContextMenuScreenY"
                   (mousedown)="$event.stopPropagation()">
                <button (click)="deleteRulerLineFromContext()">Kijel\u00f6l\u00e9s t\u00f6rl\u00e9se</button>
              </div>
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

      @if (imageCount > 1 && !showGraphViewer && !showingMontage) {
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
            @if (roiActive && roiHasOverride) {
              <span class="roi-override-dot" title="Egyedi ROI ezen a képen">●</span>
            }
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

    .gray-map-compare-container {
      width: 100%;
      height: 100%;
      display: flex;
      gap: 16px;
      padding: 16px;
      box-sizing: border-box;
      overflow: auto;
      align-items: stretch;
      justify-content: center;
    }

    .gray-map-compare-panel {
      flex: 1 1 0;
      min-width: 0;
      min-height: 0;
      display: flex;
      flex-direction: column;
      gap: 10px;
      padding: 12px;
      background: rgba(255, 255, 255, 0.03);
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 10px;
      overflow: hidden;
    }

    .gray-map-compare-title {
      font-size: 12px;
      color: #c9d4e5;
      letter-spacing: 0.02em;
      text-transform: uppercase;
      font-weight: 600;
      flex-shrink: 0;
    }

    .cluster-map-title {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
    }

    .cluster-map-accept {
      min-width: 54px;
      height: 30px;
      padding: 0 12px;
      border: 0;
      border-radius: 15px;
      color: #fff;
      background: #2e7d32;
      font-size: 12px;
      font-weight: 700;
      cursor: pointer;
    }

    .cluster-map-accept:disabled {
      opacity: .45;
      cursor: default;
    }

    .gray-map-compare-image {
      flex: 1;
      min-height: 0;
      max-width: 100%;
      object-fit: contain;
      background: #111;
      border-radius: 6px;
    }

    .cluster-legend {
      display: flex;
      flex-wrap: wrap;
      gap: 6px 12px;
      flex-shrink: 0;
      color: #d7deea;
      font-size: 12px;
    }

    .cluster-legend-item {
      position: relative;
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }

    .cluster-legend-item--editable {
      padding: 3px 6px;
      border-radius: 5px;
      cursor: pointer;
    }

    .cluster-legend-item--editable:hover {
      background: rgba(255, 255, 255, 0.1);
    }

    .cluster-legend-color-input {
      position: absolute;
      width: 100%;
      height: 100%;
      inset: 0;
      opacity: 0;
      cursor: pointer;
    }

    .cluster-legend-swatch {
      width: 12px;
      height: 12px;
      border: 1px solid rgba(255, 255, 255, 0.55);
      border-radius: 3px;
      box-shadow: 0 0 0 1px rgba(0, 0, 0, 0.3);
    }

    .cluster-value-chart {
      flex-shrink: 0;
      padding: 10px;
      background: #111;
      border: 1px solid rgba(255, 255, 255, 0.16);
      border-radius: 6px;
    }

    /* multi-panel dual_map layout — 2-3 rows (gray / RGB / sub), columns = original + N components */
    .branch-merge-compare-container {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
      width: 100%;
      height: 100%;
      padding: 12px;
      box-sizing: border-box;
      overflow: auto;
    }

    .reference-color-preview { width: 100%; height: 100%; padding: 12px; box-sizing: border-box; overflow: auto; }
    .reference-color-panels, .reference-color-histograms { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }
    .reference-color-panel, .reference-histogram-card { padding: 10px; background: rgba(255,255,255,.03); border: 1px solid rgba(255,255,255,.1); border-radius: 6px; min-width: 0; }
    .reference-color-panel { display: flex; flex-direction: column; gap: 8px; min-height: 220px; }
    .reference-color-main-image { width: 100%; height: 100%; min-height: 180px; object-fit: contain; }
    .reference-color-crops { display: flex; flex-wrap: wrap; align-items: center; justify-content: center; gap: 8px; flex: 1; }
    .reference-color-crop { width: 96px; height: 96px; object-fit: contain; image-rendering: pixelated; border: 1px solid rgba(255,255,255,.15); }
    .reference-color-histograms { margin-top: 12px; }
    .reference-histogram-title { color: #d7deea; font-size: 12px; font-weight: 600; margin-bottom: 6px; }
    .reference-histogram-svg { display: block; width: 100%; height: 120px; background: #111; border-radius: 4px; }
    .hist-line { fill: none; stroke-width: 1.5; vector-effect: non-scaling-stroke; }
    .hist-reference-overlay { stroke-dasharray: 5 3; opacity: .75; }
    .hist-line-0 { stroke: #f1f5f9; } .hist-line-1 { stroke: #ef4444; } .hist-line-2 { stroke: #3b82f6; }
    .reference-histogram-legend { display: flex; justify-content: center; gap: 12px; margin-top: 5px; font-size: 11px; }
    .lab-l { color: #f1f5f9; } .lab-a { color: #ef4444; } .lab-b { color: #60a5fa; }

    .branch-merge-panel {
      display: grid;
      grid-template-rows: auto minmax(0, 1fr) auto;
      gap: 8px;
      padding: 10px;
      background: rgba(255, 255, 255, 0.03);
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 6px;
      overflow: hidden;
    }

    .branch-merge-title {
      color: #e5e7eb;
      font-size: 13px;
      font-weight: 600;
    }

    .branch-merge-image {
      width: 100%;
      height: 100%;
      min-height: 0;
      object-fit: contain;
      border-radius: 4px;
      background: #111;
    }

    .branch-merge-meta {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      color: #b6bcc7;
      font-size: 11px;
    }

    .branch-merge-meta span {
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .split-preview-btn {
      display: inline-flex;
      align-items: center;
      justify-content: center;
    }

    .split-preview-icon {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 2px;
      width: 18px;
      height: 14px;
      padding: 2px;
      box-sizing: border-box;
      border: 1.5px solid currentColor;
      border-radius: 2px;
    }

    .split-preview-icon span {
      display: block;
      background: currentColor;
      border-radius: 1px;
      opacity: 0.8;
    }

    .dual-map-multipanel {
      display: grid;
      grid-auto-flow: row;
      gap: 12px;
      padding: 16px;
      box-sizing: border-box;
      width: 100%;
      height: 100%;
      overflow: auto;
    }

    .dual-map-cell {
      display: flex;
      flex-direction: column;
      gap: 8px;
      padding: 10px 12px 12px;
      background: rgba(255, 255, 255, 0.03);
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 10px;
      min-width: 0;
      min-height: 0;
      overflow: hidden;
    }

    .dual-map-cell--sub {
      background: rgba(100, 180, 80, 0.04);
      border-color: rgba(100, 180, 80, 0.18);
    }

    .dual-map-cell-title {
      font-size: 11px;
      font-weight: 600;
      color: #c9d4e5;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      flex-shrink: 0;
    }

    .dual-map-cell-image {
      flex: 1;
      width: 100%;
      min-height: 0;
      min-width: 0;
      max-width: 100%;
      object-fit: contain;
      background: #111;
      border-radius: 6px;
    }

    .dual-map-cell-empty {
      flex: 1;
      display: flex;
      align-items: center;
      justify-content: center;
      color: #555;
      font-size: 18px;
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

    .preview-image.grayscale {
      filter: grayscale(100%);
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

    .reference-sequence-view {
      width: 100%;
      height: 100%;
      display: grid;
      grid-template-columns: minmax(0, 1fr);
      gap: 12px;
      padding: 18px;
      box-sizing: border-box;
      overflow: auto;
    }

    .reference-sequence-view--with-histogram {
      grid-template-columns: minmax(0, 1fr) minmax(220px, 280px);
    }

    .reference-crop-strip {
      min-width: 0;
      min-height: 0;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      grid-auto-rows: minmax(160px, 1fr);
      gap: 10px;
    }

    .reference-crop-tile {
      position: relative;
      display: grid;
      grid-template-rows: minmax(0, 1fr) auto;
      min-width: 0;
      min-height: 0;
      color: #d1d5db;
      font-size: 12px;
      background: #111;
      border: 1px solid rgba(255, 255, 255, 0.16);
      border-radius: 6px;
      overflow: hidden;
    }

    .reference-crop-tile img {
      width: 100%;
      height: 100%;
      object-fit: contain;
      display: block;
    }

    .reference-crop-name-badge,
    .reference-crop-score-badge {
      max-width: calc(100% - 12px);
      padding: 4px 7px;
      color: #f8fafc;
      font-size: 12px;
      font-weight: 700;
      line-height: 1;
      background: rgba(15, 23, 42, 0.78);
      border-radius: 999px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      pointer-events: none;
    }

    .reference-crop-name-badge {
      position: absolute;
      top: 6px;
      left: 6px;
    }

    .reference-crop-score-badge {
      position: static;
      max-width: none;
      width: 100%;
      box-sizing: border-box;
      background: rgba(2, 6, 23, 0.82);
      border-radius: 0;
      font-weight: 600;
      text-align: center;
    }

    .reference-sequence-histogram {
      min-width: 0;
      min-height: 0;
      padding: 12px;
      color: #d1d5db;
      background: #111;
      border: 1px solid rgba(255, 255, 255, 0.16);
      border-radius: 6px;
      overflow: auto;
    }

    .reference-sequence-histogram-title {
      color: #f8fafc;
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      margin-bottom: 10px;
    }

    .reference-sequence-bar-row {
      display: grid;
      grid-template-columns: minmax(34px, 52px) minmax(0, 1fr) minmax(44px, auto);
      align-items: center;
      gap: 7px;
      min-height: 20px;
      font-size: 11px;
      margin-bottom: 8px;
    }

    .reference-sequence-bar-label,
    .reference-sequence-bar-value {
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .reference-sequence-bar-value {
      text-align: right;
      color: #f8fafc;
      font-variant-numeric: tabular-nums;
    }

    .reference-sequence-bar-value small {
      display: block;
      color: #94a3b8;
      font-size: 10px;
    }

    .reference-sequence-bar-track {
      height: 9px;
      overflow: hidden;
      background: rgba(255, 255, 255, 0.08);
      border-radius: 999px;
    }

    .reference-sequence-bar-fill {
      height: 100%;
      border-radius: inherit;
    }

    .reference-crop-overlay {
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      cursor: crosshair;
      overflow: visible;
    }

    .reference-crop-square {
      fill: rgba(34, 197, 94, 0.16);
      stroke: #22c55e;
      stroke-width: 2;
      stroke-dasharray: 8 4;
      cursor: move;
    }

    .reference-crop-square.dragging {
      fill: rgba(250, 204, 21, 0.22);
      stroke: #facc15;
    }

    .reference-crop-label {
      fill: #fff;
      stroke: #111;
      stroke-width: 3;
      paint-order: stroke;
      pointer-events: none;
      font-family: sans-serif;
      font-weight: 700;
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

    .roi-shape-body { cursor: move; }

    .roi-handle.rotate {
      fill: #f97316;
      stroke: #ea580c;
      cursor: grab;
    }
    .roi-handle.rotate:hover { fill: #fb923c; }

    .roi-rot-btn {
      fill: #f97316 !important;
      stroke: #ea580c !important;
      cursor: grab;
      pointer-events: all;
    }
    .roi-rot-btn:hover { fill: #fb923c !important; }

    .roi-rot-icon {
      fill: white !important;
      pointer-events: none;
      user-select: none;
    }

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

    .particle-overlay {
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      pointer-events: none;
      overflow: visible;
    }

    .particle-hitarea {
      fill: transparent;
      stroke: transparent;
      stroke-width: 0;
      pointer-events: all;
      cursor: pointer;
    }

    .particle-hitarea:hover {
      fill: rgba(255, 255, 0, 0.18);
    }

    .particle-hitarea.excluded {
      fill: rgba(255, 200, 0, 0.15);
    }

    .particle-hitarea.filtered-out {
      pointer-events: none;
      cursor: default;
    }

    .circle-overlay {
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      pointer-events: none;
      overflow: visible;
    }

    .detection-circle {
      fill: transparent;
      stroke: #ff6b6b;
      stroke-width: 2;
      pointer-events: none;
      cursor: default;
    }

    .detection-circle:hover {
      stroke: #ffd700;
      stroke-width: 3;
      filter: drop-shadow(0 0 4px #ffd700);
    }

    .circle-center-point {
      fill: #00ff00;
      pointer-events: none;
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

    /* === Image Tools Toolbar === */

    .image-toolbar {
      display: flex;
      align-items: center;
      gap: 6px;
      padding: 3px 10px;
      background: #2a2a2a;
      border-bottom: 1px solid #333;
      flex-shrink: 0;
      min-height: 28px;
    }

    .toolbar-content {
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
    }

    .toolbar-tools {
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .toolbar-spacer {
      flex: 1;
      min-width: 8px;
    }

    .tool-btn {
      background-color: #333;
      color: #ccc;
      border: 1px solid #555;
      border-radius: 4px;
      padding: 3px 8px;
      cursor: pointer;
      font-size: 14px;
      line-height: 1;
      display: flex;
      align-items: center;
      justify-content: center;
      transition: background-color 0.2s, color 0.2s, border-color 0.2s;
    }

    .tool-btn:hover:not(:disabled) {
      background-color: #444;
      color: #fff;
    }

    .tool-btn:disabled {
      opacity: 0.4;
      cursor: default;
    }

    .tool-btn.active {
      background-color: #1e3a5f;
      color: #fff;
      border-color: #1e3a5f;
    }

    .ruler-measurement-box {
      width: 72px;
      padding: 2px 4px;
      background: #222;
      border: 1px solid #444;
      border-radius: 3px;
      color: #666;
      font-size: 11px;
      font-family: monospace;
      text-align: center;
    }

    .ruler-measurement-box.used {
      color: #e0e0e0;
      border-color: #1a5fb4;
      background: #1a2a3a;
    }

    .icon-btn {
      font-size: 14px;
      filter: grayscale(1);
    }

    .icon-tool-btn {
      width: 32px;
      min-width: 32px;
      height: 24px;
      padding: 0;
      flex-shrink: 0;
      filter: grayscale(1);
    }

    .scale-label {
      color: #999;
      font-size: 11px;
      white-space: nowrap;
    }

    .scale-mm-input {
      width: 44px;
      padding: 2px 4px;
      background: #222;
      border: 1px solid #555;
      border-radius: 3px;
      color: #e0e0e0;
      font-size: 11px;
      text-align: center;
      -moz-appearance: textfield;
    }

    .scale-mm-input::-webkit-outer-spin-button,
    .scale-mm-input::-webkit-inner-spin-button {
      -webkit-appearance: none;
      margin: 0;
    }

    .scale-unit {
      color: #666;
      font-size: 11px;
    }

    .scale-ratio-display {
      color: #4fc3f7;
      font-size: 11px;
      font-family: monospace;
      padding: 2px 6px;
      background: #1a2a3a;
      border: 1px solid #1a5fb4;
      border-radius: 3px;
      white-space: nowrap;
    }

    .scale-checkbox-label {
      display: flex;
      align-items: center;
      gap: 4px;
      color: #ccc;
      font-size: 11px;
      cursor: pointer;
      white-space: nowrap;
    }

    .scale-checkbox-label input[type="checkbox"] {
      accent-color: #3b82f6;
    }

    .ruler-context-menu {
      position: absolute;
      background: #2a2a2a;
      border: 1px solid #555;
      border-radius: 6px;
      overflow: hidden;
      box-shadow: 0 4px 12px rgba(0,0,0,0.5);
      z-index: 10;
    }

    .ruler-context-menu button {
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

    .ruler-context-menu button:hover {
      background: #3b82f6;
    }

    /* === Pixel measurement tool === */

    .pixel-color-space {
      color: #999;
      font-size: 11px;
      white-space: nowrap;
      border-right: 1px solid #444;
      padding-right: 8px;
      margin-right: 4px;
    }

    .pixel-values-display {
      display: flex;
      gap: 4px;
      align-items: center;
    }

    .pixel-value-item {
      color: #e0e0e0;
      font-size: 11px;
      font-family: monospace;
      background: #1a2a3a;
      border: 1px solid #444;
      border-radius: 3px;
      padding: 2px 4px;
      min-width: 32px;
      text-align: center;
    }

    /* === Ruler SVG Overlay === */

    .ruler-overlay {
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      overflow: visible;
      z-index: 5;
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

    .roi-override-dot {
      color: #3b82f6;
      font-size: 10px;
      margin-left: 2px;
      line-height: 1;
    }

    .montage-gallery-container {
      display: flex;
      flex-direction: column;
      width: 100%;
      height: 100%;
      overflow: auto;
      flex: 1;
    }

    .montage-gallery-wrapper {
      flex: 1;
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: visible;
      padding: 0;
      background: #1a1a1a;
      min-width: min-content;
      min-height: min-content;
    }

    .montage-gallery-image {
      max-width: none;
      max-height: none;
      object-fit: contain;
      display: block;
    }

    /* Expanded chart viewer */
    .expanded-chart-viewer {
      position: absolute;
      inset: 0;
      display: flex;
      flex-direction: column;
      background: #1a1a1a;
      z-index: 10;
    }

    .expanded-chart-toolbar {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 12px 16px;
      border-bottom: 1px solid #333;
      background: #1e1e1e;
    }

    .expanded-chart-title {
      font-size: 14px;
      font-weight: 600;
      color: #e0e0e0;
    }

    .expanded-chart-container {
      flex: 1;
      overflow: auto;
      padding: 0;
      display: flex;
      align-items: center;
      justify-content: center;
      width: 100%;
      height: calc(100% - 45px);
      background: #1a1a1a;
    }

    .expanded-chart-container app-scatter-chart {
      display: block;
      width: auto;
      height: auto;
      max-width: 100%;
      max-height: 100%;
    }

    .expanded-chart-container app-pca-chart {
      display: block;
      width: 100%;
      height: 100%;
    }

    .pca-chart-wrapper {
      width: 100%;
      height: 100%;
      display: flex;
      flex-direction: column;
      overflow: auto;
    }

    .pca-chart-wrapper app-pca-chart {
      flex: 1;
      width: 100%;
      min-height: 600px;
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
  grayMapOverlaySrc: string | null = null;
  grayMapBaseSrc: string | null = null;
  branchMergePanels: BranchMergePanel[] = [];
  splitPreviewActive = false;
  referenceColorPreview: ReferenceColorPreviewState | null = null;
  readonly referenceHistogramKinds: Array<{ key: 'source' | 'reference' | 'aligned'; label: string }> = [
    { key: 'source', label: 'Eredeti' },
    { key: 'reference', label: 'Referenciák' },
    { key: 'aligned', label: 'Illesztett' },
  ];
  kmeansSourceSrc: string | null = null;
  kmeansOverlaySrc: string | null = null;
  clusterMapLabelSrc: string | null = null;
  clusterMapSrc: string | null = null;
  kmeansLegend: Array<{ label: number; color: string }> = [];
  clusterMapLabelValues: Array<{ label: number; pixelCount: number; values: Record<string, number> }> = [];
  clusterMapValueComponents: string[] = [];
  clusterMapRemainderIsFinal = false;
  dualMapState: {
    grayBase: string | null;
    grayOverlays: string[];
    rgbBase: string | null;
    rgbOverlays: string[];
    subBase: string | null;
    subOverlays: string[];
    subLabel: string;
  } | null = null;
  isGrayscale = false;
  loading = false;
  imageCount = 0;
  currentIndex = 0;

  // Zoom/pan for gallery image
  zoomLevel = 1.0;
  private baseFitScale = 1;
  private isDragging = false;
  private dragStart = { x: 0, y: 0, scrollLeft: 0, scrollTop: 0 };
  private zoomAnchorFrame: number | null = null;

  // Zoom for montage
  montageZoomLevel = 1.0;
  private montageMontageBaseFitScale = 1;

  // Graph viewer state
  showGraphViewer = false;
  graphSelectedPoint = -1;
  private graphData: any = null;
  private graphOmittedIndices: Set<number> = new Set();
  private graphViewerStepIndex = -1;
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

  // Expanded chart viewer state
  showExpandedChart = false;
  expandedChartData: any = null;
  expandedChartTitle = '';
  expandedChartType: 'scatter' | 'pca' = 'scatter';

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
  roiAngle = 0; // degrees, for rect and ellipse
  roiAllSelected = false;

  /** Whether the current image has its own per-image ROI override */
  roiHasOverride = false;

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
  private roiCurrentImageIndex = 0;
  private boundOnKeyDown: ((e: KeyboardEvent) => void) | null = null;

  // Particle contour click overlay
  particleOverlayActive = false;
  particlesForOverlay: any[] = [];
  particleExcludedIds: Set<string> = new Set();
  particleImgW = 100;
  particleImgH = 100;
  private particleStepIndex = -1;
  private particleClickPending = false;

  // Circle detection overlay
  circleOverlayActive = false;
  circlesForOverlay: any[] = [];
  circleImgW = 100;
  circleImgH = 100;
  private circleStepIndex = -1;

  // Reference crop overlay
  referenceCropActive = false;
  referenceCropStripActive = false;
  referenceCropImages: string[] = [];
  referenceCropSquares: Array<{ x: number; y: number; size: number; name?: string }> = [];
  referenceCropLabels: string[] = [];
  referenceCropScores: number[] = [];
  referenceSequenceComponents: string[] = [];
  referenceSequenceScores: Record<string, number[]> = {};
  referenceSequenceDiffs: Record<string, Array<number | null>> = {};
  referenceSequenceComponent = '';
  referenceSequenceColor = '#94a3b8';
  referenceSequenceMaxScore = 1;
  referenceCropImgW = 100;
  referenceCropImgH = 100;
  referenceCropDragIndex = -1;
  private referenceCropStepIndex = -1;
  private referenceCropCurrentImageIndex = 0;
  private referenceCropGlobalOffset = 0;
  private referenceCropSize = 64;
  private referenceCropDragOffset = { x: 0, y: 0 };

  get referenceCropScale(): number {
    return Math.max(1, Math.max(this.referenceCropImgW, this.referenceCropImgH) / 1000);
  }

  getReferenceCropLabel(index: number): string {
    const sortedLabel = this.referenceCropLabels[index]?.trim();
    if (sortedLabel) return sortedLabel;
    const name = this.referenceCropSquares[index]?.name?.trim();
    return name || String(index + 1);
  }

  getReferenceCropOverlayLabel(index: number): string {
    return String(this.referenceCropGlobalOffset + index + 1);
  }

  getReferenceCropScoreLabel(index: number): string {
    const value = this.referenceCropScores[index];
    return Number.isFinite(value) ? value.toFixed(2) : '';
  }

  getReferenceSequenceBarWidth(value: number): number {
    return Number.isFinite(value) ? Math.max(2, Math.min(100, (Math.abs(value) / this.referenceSequenceMaxScore) * 100)) : 0;
  }

  getReferenceComponentScores(component: string): number[] {
    return this.referenceSequenceScores[component] || [];
  }

  getReferenceComponentScoreLabel(component: string, index: number): string {
    const value = this.referenceSequenceScores[component]?.[index];
    return Number.isFinite(value) ? Number(value).toFixed(2) : '';
  }

  getReferenceComponentDiffLabel(component: string, index: number): string {
    const value = this.referenceSequenceDiffs[component]?.[index];
    if (!Number.isFinite(value)) return '';
    const num = Number(value);
    return `${num >= 0 ? '+' : ''}${num.toFixed(2)}`;
  }

  getReferenceComponentBarWidth(component: string, value: number): number {
    const scores = this.referenceSequenceScores[component] || [];
    const finiteScores = scores.filter((score) => Number.isFinite(score));
    const max = finiteScores.length ? Math.max(...finiteScores.map((score) => Math.abs(score)), 1) : 1;
    return Number.isFinite(value) ? Math.max(2, Math.min(100, (Math.abs(value) / max) * 100)) : 0;
  }

  hasSavablePreview(): boolean {
    return !!(
      (this.showGraphViewer && this.graphCanvasRef?.nativeElement) ||
      (this.showingMontage && this.montagePreview) ||
      (this.referenceCropStripActive && this.referenceCropImages.length > 0) ||
      this.imageSrc
    );
  }

  // Ruler tool state (multi-ruler: up to 5 lines)
  rulerActive = false;
  rulerLines: Array<{start: {x: number; y: number}, end: {x: number; y: number}, distance: number}> = [];
  rulerDrawingStart: {x: number; y: number} | null = null;
  rulerDrawingCurrent: {x: number; y: number} | null = null;
  rulerDrawingDistance = 0;
  readonly RULER_MAX_LINES = 5;
  readonly rulerSlots = [0, 1, 2, 3, 4];
  rulerImgW = 100;
  rulerImgH = 100;

  // Scale tool state
  scaleActive = false;
  scaleStart: {x: number; y: number} | null = null;
  scaleEnd: {x: number; y: number} | null = null;
  scaleCurrentPos: {x: number; y: number} | null = null;
  scaleLinePx = 0;
  scaleMm = 0;
  scaleMeasureUnit = 'mm';
  showScaleBar = false;
  scaleBarPx = 0;
  scaleBarMm = 0;
  scaleBarLengthMm = 0;
  scaleBarUnit = 'mm';
  scaleBarFontSize = 24;
  scaleBarFontThickness = 1;
  scaleBarBarThickness = 3;
  scaleBarBarColor = 'white';
  scaleBarTextColor = 'white';
  scaleBarBackgroundColor = 'black';
  scaleBarPositionX = -1;
  scaleBarPositionY = -1;
  scaleBarOverlayActive = false;
  scaleBarOverlay: ScaleBarOverlayState | null = null;
  scaleBarDragging = false;
  private scaleBarDragOffset = { x: 0, y: 0 };
  private scaleBarSelectedStepIndex = -1;
  private scaleBarSelectedParams: Record<string, any> | null = null;

  // Pixel measurement tool state
  pixelActive = false;
  pixelCurrentPos: {x: number; y: number} | null = null;
  pixelFrozenPos: {x: number; y: number} | null = null;
  pixelGridValues: string[] = ['', '', '', '', '', '', '', '', ''];
  pixelGridColors: string[] = ['#000', '#000', '#000', '#000', '#000', '#000', '#000', '#000', '#000'];
  pixelColorSpace = 'RGB';
  pixelOutputType: DataType | null = null;
  pixelImgW = 100;
  pixelImgH = 100;
  private pixelCanvasCache: HTMLCanvasElement | null = null;
  private pixelImageDataCache: ImageData | null = null;

  // Montage feature state
  generatingMontage = false;
  showingMontage = false;
  montageImagePaths: string[] = [];
  montagePreview: string | null = null;
  private currentPipeline: any = null;
  private selectedStepIndex = -1;
  private montageGridCols = 0;
  private montageGridRows = 0;
  private montageCellWidth = 0;
  private montageCellHeight = 0;
  private montageImageCount = 0;
  private montageCache = new Map<string, any>();
  private montageCacheKey = '';

  readonly Math = Math;

  get pxPerMm(): number {
    const measuredMm = this.getScaleMeasurementMm();
    if (this.scaleLinePx > 0 && measuredMm > 0) {
      return this.scaleLinePx / measuredMm;
    }
    return 0;
  }

  getScaleResolutionDisplay(): string {
    const measuredMm = this.getScaleMeasurementMm();
    if (this.scaleLinePx <= 0 || measuredMm <= 0) {
      return '';
    }

    const valueInUnitPerPx = this.scaleMm > 0 ? this.scaleMm / this.scaleLinePx : 0;
    const decimals = valueInUnitPerPx < 0.1 ? 3 : valueInUnitPerPx < 1 ? 2 : valueInUnitPerPx < 10 ? 1 : 0;
    return `${valueInUnitPerPx.toFixed(decimals)} ${this.scaleMeasureUnit}/px`;
  }

  get rulerScale(): number {
    return Math.max(1, Math.max(this.rulerImgW, this.rulerImgH) / 1000);
  }

  get pixelDisplayScale(): number {
    const img = this.previewImg?.nativeElement;
    if (img?.clientWidth && img.naturalWidth) {
      return Math.max(img.clientWidth / img.naturalWidth, 0.01);
    }
    return Math.max(this.baseFitScale * this.zoomLevel, 0.01);
  }

  get pixelGridCoordScale(): number {
    return 1 / this.pixelDisplayScale;
  }

  get pixelGridCellSize(): number {
    return 30 * this.pixelGridCoordScale;
  }

  get pixelGridHalfSize(): number {
    return this.pixelGridCellSize / 2;
  }

  get pixelGridSpacing(): number {
    return 34 * this.pixelGridCoordScale;
  }

  get pixelGridOffsetX(): number {
    return 96 * this.pixelGridCoordScale;
  }

  get pixelGridOffsetY(): number {
    return -82 * this.pixelGridCoordScale;
  }

  get pixelGridFontSize(): number {
    return 10 * this.pixelGridCoordScale;
  }

  get pixelGridLineHeight(): number {
    return 9.5 * this.pixelGridCoordScale;
  }

  get pixelGridStrokeWidth(): number {
    return Math.max(1.25 * this.pixelGridCoordScale, 1 / this.pixelDisplayScale);
  }

  // Ruler line context menu
  showRulerContextMenu = false;
  rulerContextMenuScreenX = 0;
  rulerContextMenuScreenY = 0;
  private rulerContextLineIndex = -1;

  get hasAnnotations(): boolean {
    return this.rulerLines.length > 0 ||
           this.rulerDrawingStart !== null ||
           (this.scaleStart !== null && !(this.scaleEnd && this.getScaleMeasurementMm() > 0)) ||
           this.scaleBarOverlayActive ||
           (this.showScaleBar && this.pxPerMm > 0) ||
           (this.pixelActive && (this.pixelCurrentPos !== null || this.pixelFrozenPos !== null));
  }

  showGrayMapComparison(): boolean {
    if (!this.grayMapOverlaySrc || !this.currentPipeline) return false;
    const step = this.currentPipeline.steps[this.selectedStepIndex];
    return step?.step_def_id === 'gray_map';
  }

  showDualMapView(): boolean {
    if (!this.dualMapState || !this.currentPipeline) return false;
    const step = this.currentPipeline.steps[this.selectedStepIndex];
    return step?.step_def_id === 'dual_map';
  }

  showClusterReferenceMap(): boolean {
    const step = this.currentPipeline?.steps[this.selectedStepIndex];
    return step?.step_def_id === 'cluster_reference_map';
  }

  acceptClusterMap(): void {
    if (!this.currentPipeline || this.selectedStepIndex < 0) return;
    const step = this.currentPipeline.steps[this.selectedStepIndex];
    if (step?.step_def_id !== 'cluster_reference_map' || step.param_values?.['remainder_as_last']) return;

    let accepted: any[] = [];
    try {
      const parsed = JSON.parse(String(step.param_values?.['accepted_components'] ?? '[]'));
      accepted = Array.isArray(parsed) ? parsed : [];
    } catch {
      accepted = [];
    }
    accepted.push({
      name: `Komponens ${accepted.length + 1}`,
      selected_labels: String(step.param_values?.['selected_labels'] ?? '1'),
      reference_label: String(step.param_values?.['reference_label'] ?? '1'),
      center_mode: String(step.param_values?.['center_mode'] ?? 'cluster_median'),
      map_multiplier: Number(step.param_values?.['map_multiplier'] ?? 1),
      invert: !!step.param_values?.['invert'],
    });
    this.pipelineState.updateParams(this.selectedStepIndex, {
      ...step.param_values,
      accepted_components: JSON.stringify(accepted),
    });
  }

  showKmeansComparison(): boolean {
    const step = this.currentPipeline?.steps[this.selectedStepIndex];
    return step?.step_def_id === 'kmeans_cluster';
  }

  private getFallbackKmeansLegend(stepIndex: number, pipeline: any): Array<{ label: number; color: string }> {
    const colors = [
      '#ff0000', '#00ff00', '#0000ff', '#ffff00',
      '#ff00ff', '#00ffff', '#ff0080', '#0080ff',
      '#ff8000', '#80ff00', '#8000ff', '#00ff80',
    ];
    let count = 3;
    for (let index = stepIndex; index >= 0; index--) {
      const step = pipeline?.steps?.[index];
      if (step?.step_def_id === 'kmeans_cluster') {
        count = Math.max(2, Number(step.param_values?.['k'] ?? 3));
        break;
      }
    }
    return Array.from({ length: count }, (_, index) => ({
      label: index + 1,
      color: colors[index % colors.length],
    }));
  }

  getKmeansLegendColor(label: number): string {
    return this.kmeansLegend.find((item) => item.label === label)?.color || '#94a3b8';
  }

  normalizeLegendColor(color: string): string {
    if (/^#[0-9a-f]{6}$/i.test(color)) return color;
    const channels = color.match(/\d+/g)?.slice(0, 3).map(Number);
    if (!channels || channels.length !== 3) return '#ffffff';
    return `#${channels.map((value) =>
      Math.max(0, Math.min(255, value)).toString(16).padStart(2, '0')
    ).join('')}`;
  }

  onKmeansLegendColorChange(label: number, event: Event): void {
    const color = (event.target as HTMLInputElement).value;
    const pipeline = this.currentPipeline;
    if (!pipeline || !color) return;

    let kmeansIndex = -1;
    for (let index = this.selectedStepIndex; index >= 0; index--) {
      if (pipeline.steps[index]?.step_def_id === 'kmeans_cluster') {
        kmeansIndex = index;
        break;
      }
    }
    if (kmeansIndex < 0) return;

    const step = pipeline.steps[kmeansIndex];
    let colors: Record<string, string> = {};
    try {
      const raw = step.param_values?.['cluster_colors'] ?? '{}';
      const parsed = typeof raw === 'string' ? JSON.parse(raw) : raw;
      if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) colors = { ...parsed };
    } catch {
      colors = {};
    }
    colors[String(label)] = color;
    this.kmeansLegend = this.kmeansLegend.map((item) =>
      item.label === label ? { ...item, color } : item
    );
    this.pipelineState.updateParams(kmeansIndex, {
      ...step.param_values,
      cluster_colors: JSON.stringify(colors),
    });
  }

  getClusterMapValueWidth(
    item: { values: Record<string, number> },
    component: string,
  ): number {
    const value = Number(item.values[component] ?? 0);
    const maximum = Math.max(
      1,
      ...this.clusterMapLabelValues.map((entry) => Math.abs(Number(entry.values[component] ?? 0))),
    );
    return Math.max(2, Math.min(100, (Math.abs(value) / maximum) * 100));
  }

  getClusterMapValueLabel(
    item: { pixelCount: number; values: Record<string, number> },
    component: string,
  ): string {
    const value = Number(item.values[component]);
    return `${Number.isFinite(value) ? value.toFixed(2) : '–'} · ${item.pixelCount} px`;
  }

  showBranchMergeView(): boolean {
    return this.splitPreviewActive && this.branchMergePanels.length >= 2;
  }

  getReferenceHistogramPoints(kind: 'source' | 'reference' | 'aligned', channel: number): string {
    const values = this.referenceColorPreview?.histograms?.[kind]?.[channel];
    if (!Array.isArray(values)) return '';
    return values.map((value, index) => `${index},${100 - Math.max(0, Math.min(1, Number(value))) * 96}`).join(' ');
  }

  canShowSplitPreview(): boolean {
    return this.branchMergePanels.length >= 2;
  }

  toggleSplitPreview(): void {
    if (!this.canShowSplitPreview()) return;
    this.splitPreviewActive = !this.splitPreviewActive;
    this.resetZoom();
  }

  get dualMapMaxCols(): number {
    if (!this.dualMapState) return 1;
    return Math.max(
      this.dualMapState.grayOverlays.length || 1,
      this.dualMapState.rgbOverlays.length  || 1,
      this.dualMapState.subOverlays.length  || 0,
    );
  }

  private buildBranchMergePanels(sideOutputs: Record<string, any> | null | undefined): BranchMergePanel[] {
    const preview = sideOutputs?.['branch_merge_preview'];
    const panels = preview?.['panels'];
    if (!Array.isArray(panels)) return [];

    return panels
      .filter((panel) => panel && typeof panel['image_base64'] === 'string' && panel['image_base64'])
      .map((panel, index) => ({
        label: String(panel['label'] || (index === 0 ? 'Elso ag' : 'Masodik ag')),
        imageSrc: `data:image/jpeg;base64,${panel['image_base64']}`,
        sourceName: String(panel['source_name'] || ''),
        imageWidth: Number(panel['image_width'] || 0),
        imageHeight: Number(panel['image_height'] || 0),
        imageCount: Number(panel['image_count'] || 0),
        isGrayscale: !!panel['is_grayscale'],
      }));
  }

  private splitPreviewRequestId = 0;
  private splitPreviewNodeIndex = -1;

  private loadNodeSplitPreview(stepIndex: number): void {
    if (!this.currentPipeline || stepIndex <= 0 || stepIndex >= this.currentPipeline.steps.length) {
      this.branchMergePanels = [];
      this.splitPreviewActive = false;
      return;
    }

    const requestId = ++this.splitPreviewRequestId;
    this.splitPreviewNodeIndex = stepIndex;
    const indices = [stepIndex - 1, stepIndex];
    const imageIndex = this.currentIndex;
    const requests = indices.map((index) => {
      const context = this.pipelineState.getPreviewContext(index);
      return this.recipeService.previewStep(
        context.pipeline,
        context.stepIndex,
        imageIndex,
        true,
      );
    });

    this.loading = true;
    forkJoin(requests).subscribe({
      next: (responses) => {
        if (requestId !== this.splitPreviewRequestId) return;
        this.loading = false;

        const selectedName =
          this.pipelineState.getStepDefinition(this.currentPipeline!.steps[stepIndex].step_def_id)?.name ||
          this.currentPipeline!.steps[stepIndex].step_def_id;
        const labels = [`Bemenet – ${selectedName}`, `Kimenet – ${selectedName}`];

        this.branchMergePanels = responses
          .map((response, index): BranchMergePanel | null => {
            if (!response.success || !response.image_base64) return null;
            const sourceStep = this.currentPipeline!.steps[indices[index]];
            const sourceName =
              this.pipelineState.getStepDefinition(sourceStep.step_def_id)?.name ||
              sourceStep.step_def_id;
            return {
              label: labels[index],
              imageSrc: `data:image/jpeg;base64,${response.image_base64}`,
              sourceName,
              imageWidth: Number(response.image_width || 0),
              imageHeight: Number(response.image_height || 0),
              imageCount: Number(response.image_count || 0),
              isGrayscale: !!response.is_grayscale,
            };
          })
          .filter((panel): panel is BranchMergePanel => panel !== null);

        this.splitPreviewActive = this.branchMergePanels.length === 2;
        if (this.splitPreviewActive) this.resetZoom();
      },
      error: () => {
        if (requestId !== this.splitPreviewRequestId) return;
        this.loading = false;
        this.branchMergePanels = [];
        this.splitPreviewActive = false;
      },
    });
  }

  private subs: Subscription[] = [];

  constructor(
    private pipelineState: PipelineStateService,
    private cdr: ChangeDetectorRef,
    private recipeService: RecipeService,
  ) {}

  ngOnInit(): void {
    this.boundOnKeyDown = this.onRoiKeyDown.bind(this);
    window.addEventListener('keydown', this.boundOnKeyDown);
    this.subs.push(
      this.pipelineState.previewImage$.subscribe((img) => {
        this.imageSrc = img;
        this.clearAllRulerLines();
        this.clearScaleLine();
        this.pixelCurrentPos = null;
        this.pixelFrozenPos = null;
        this.pixelGridValues = ['', '', '', '', '', '', '', '', ''];
        this.pixelGridColors = ['#000', '#000', '#000', '#000', '#000', '#000', '#000', '#000', '#000'];
        this.pixelCanvasCache = null;
        this.pixelImageDataCache = null;
      }),
      this.pipelineState.previewImageOverride$.subscribe((img) => {
        this.grayMapOverlaySrc = img;
        if (img) {
          this.grayMapBaseSrc = this.imageSrc;
        } else {
          this.grayMapBaseSrc = null;
        }
      }),
      this.pipelineState.dualMapPreview$.subscribe((state) => {
        this.dualMapState = state;
      }),
      this.pipelineState.previewImageIsGrayscale$.subscribe((isGray) => {
        this.isGrayscale = isGray;
      }),
      this.pipelineState.previewLoading$.subscribe((l) => {
        if (this.particleClickPending) return;
        this.loading = l;
      }),
      this.pipelineState.imageCount$.subscribe((c) => (this.imageCount = c)),
      this.pipelineState.previewImageIndex$.subscribe((i) => (this.currentIndex = i)),
      this.pipelineState.splitPreviewRequest$.subscribe((stepIndex) => {
        this.loadNodeSplitPreview(stepIndex);
      }),
      combineLatest([
        this.pipelineState.sideOutputs$,
        this.pipelineState.selectedStepIndex$,
        this.pipelineState.pipeline$,
        this.pipelineState.previewImageIndex$
      ]).subscribe(([so, stepIdx, pipeline, imgIdx]) => {
        this.imageNames = so?.['loaded_paths'] ?? [];
        if (this.splitPreviewActive && stepIdx !== this.splitPreviewNodeIndex) {
          this.splitPreviewActive = false;
          this.splitPreviewNodeIndex = -1;
        }
        this.selectedStepIndex = stepIdx;
        this.currentPipeline = pipeline;
        if (stepIdx >= 0 && pipeline.steps[stepIdx]?.step_def_id === 'reference_color_align') {
          const sourceRows = so?.['reference_color_align_source_images_base64'];
          const alignedRows = so?.['reference_color_align_aligned_images_base64'];
          const cropRows = so?.['reference_crops_base64'];
          const histograms = so?.['reference_color_align_histograms'];
          const cropSrcs = Array.isArray(cropRows)
            ? cropRows.flat().filter((value: unknown) => typeof value === 'string' && value)
                .map((value: string) => `data:image/jpeg;base64,${value}`)
            : [];
          this.referenceColorPreview = Array.isArray(sourceRows) && sourceRows[0]
            && Array.isArray(alignedRows) && alignedRows[0]
            && cropSrcs.length && histograms
            ? {
                sourceSrc: `data:image/jpeg;base64,${sourceRows[0]}`,
                alignedSrc: `data:image/jpeg;base64,${alignedRows[0]}`,
                cropSrcs,
                histograms,
              }
            : null;
        } else {
          this.referenceColorPreview = null;
        }
        if (!this.splitPreviewActive) {
          this.branchMergePanels = stepIdx >= 0 && pipeline.steps[stepIdx]?.step_def_id === 'branch_merge'
            ? this.buildBranchMergePanels(so)
            : [];
        }
        const sourceRows = so?.['kmeans_source_images_base64'];
        const labelRows = so?.['kmeans_labeled_images_base64'];
        const overlayRows = so?.['kmeans_overlay_images_base64'];
        const mapRows = so?.['cluster_map_images_base64'];
        const safeIndex = (rows: any[]) => Math.min(Math.max(imgIdx, 0), rows.length - 1);
        this.kmeansSourceSrc = Array.isArray(sourceRows) && sourceRows.length && sourceRows[safeIndex(sourceRows)]
          ? `data:image/jpeg;base64,${sourceRows[safeIndex(sourceRows)]}` : null;
        this.clusterMapLabelSrc = Array.isArray(labelRows) && labelRows.length && labelRows[safeIndex(labelRows)]
          ? `data:image/jpeg;base64,${labelRows[safeIndex(labelRows)]}` : null;
        this.kmeansOverlaySrc = Array.isArray(overlayRows) && overlayRows.length && overlayRows[safeIndex(overlayRows)]
          ? `data:image/jpeg;base64,${overlayRows[safeIndex(overlayRows)]}` : this.clusterMapLabelSrc;
        this.clusterMapSrc = Array.isArray(mapRows) && mapRows.length && mapRows[safeIndex(mapRows)]
          ? `data:image/png;base64,${mapRows[safeIndex(mapRows)]}` : null;
        const legendRows = so?.['kmeans_legend'];
        const legend = Array.isArray(legendRows) && legendRows.length ? legendRows[safeIndex(legendRows)] : [];
        this.kmeansLegend = Array.isArray(legend) && legend.length ? legend.map((item: any) => {
          const rgb = Array.isArray(item?.color) ? item.color : [255, 255, 255];
          return { label: Number(item?.label), color: `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})` };
        }) : this.getFallbackKmeansLegend(stepIdx, pipeline);
        const labelValueRows = so?.['cluster_map_label_values'];
        const labelValues = Array.isArray(labelValueRows) && labelValueRows.length
          ? labelValueRows[safeIndex(labelValueRows)]
          : [];
        this.clusterMapLabelValues = Array.isArray(labelValues) ? labelValues.map((item: any) => ({
          label: Number(item?.label),
          pixelCount: Number(item?.pixel_count ?? 0),
          values: item?.values && typeof item.values === 'object' ? item.values : {},
        })) : [];
        this.clusterMapValueComponents = Array.from(new Set(
          this.clusterMapLabelValues.flatMap((item) => Object.keys(item.values)),
        ));
        this.clusterMapRemainderIsFinal =
          !!pipeline.steps[stepIdx]?.param_values?.['remainder_as_last'];
        // Invalidate montage cache when pipeline or step changes
        const newCacheKey = `${stepIdx}:${JSON.stringify(pipeline)}`;
        if (newCacheKey !== this.montageCacheKey) {
          this.montageCache.clear();
          this.montageCacheKey = newCacheKey;
        }
        
        // For color_thresh steps, show the mask overlay on top of the original image
        if (stepIdx >= 0 && pipeline.steps[stepIdx]?.step_def_id === 'color_thresh') {
          const maskOverlays = so?.['color_thresh_mask_overlays'];
          if (Array.isArray(maskOverlays) && maskOverlays.length > 0) {
            const idx = Math.min(imgIdx, maskOverlays.length - 1);
            this.imageSrc = maskOverlays[idx] || this.imageSrc;
          }
        }
        
        // Auto-update the maximized chart when a new curve fit arrives
        if (this.showGraphViewer && so?.['curve_fits']) {
          const fits = so['curve_fits'];
          if (Array.isArray(fits) && fits.length > 0) {
            this.graphData = fits[fits.length - 1];
            this.drawGraph();
          }
        }
      }),
      this.pipelineState.maximizeGraph$.subscribe(({ data, omittedIndices, sourceStepIndex }) => {
        this.openGraphViewer(data, omittedIndices, sourceStepIndex);
      }),
      this.pipelineState.omittedPoints$.subscribe(({ indices, imageNames }) => {
        this.graphOmittedIndices = new Set(indices);
        if (imageNames.length > 0) {
          this.imageNames = [...imageNames];
        }
        if (this.showGraphViewer) {
          this.drawGraph();
        }
      }),
      this.pipelineState.expandedChart$.subscribe(({ data, type, title }) => {
        this.expandedChartData = data;
        this.expandedChartType = type;
        this.expandedChartTitle = title;
        this.showExpandedChart = true;
      }),
      this.pipelineState.selectedStepIndex$.subscribe((idx) => {
        if (this.showGraphViewer && this.graphViewerStepIndex >= 0 && idx !== this.graphViewerStepIndex) {
          this.closeGraphViewer();
        }
      }),
      this.pipelineState.pipeline$.subscribe((pipeline) => {
        if (pipeline.steps.length === 0 && this.showGraphViewer) {
          this.closeGraphViewer();
        }
      }),
      combineLatest([
        this.pipelineState.pipeline$,
        this.pipelineState.selectedStepIndex$,
        this.pipelineState.stepCatalog$,
      ]).subscribe(([pipeline, idx]) => {
        this.pixelOutputType = idx >= 0 && idx < pipeline.steps.length
          ? this.pipelineState.getStepOutputType(idx)
          : null;
        this.detectColorSpace();
      }),
      // Track ROI editing when an ROI step is selected
      combineLatest([
        this.pipelineState.pipeline$,
        this.pipelineState.selectedStepIndex$,
        this.pipelineState.imageDims$,
        this.pipelineState.previewImageIndex$,
      ]).subscribe(([pipeline, idx, dims, imgIdx]) => {
        if (idx >= 0 && idx < pipeline.steps.length &&
            pipeline.steps[idx].step_def_id === 'mask_rect_roi') {
          this.deactivateMeasurementTools();
          const step = pipeline.steps[idx];
          const isCropMode = step.param_values?.['output_mode'] === 'crop';
          this.roiSelectedStepIndex = idx;
          this.roiImgW = dims.w || 100;
          this.roiImgH = dims.h || 100;
          const newType = this.normalizeRoiType(step.param_values?.['roi_type']);
          if (newType !== this.roiType) {
            // Reset drawing state when ROI type changes
            this.roiEllipseDrawing = false;
            this.roiEllipseGuidePoints = [];
            this.roiPolygonDrawing = false;
          }
          this.roiType = newType;
          this.roiCurrentImageIndex = imgIdx;
          this.syncRoiFromParams(step.param_values, imgIdx);
          this.roiActive = !isCropMode;
          if (isCropMode) {
            this.roiSelectedStepIndex = -1;
          }
        } else {
          this.roiActive = false;
          this.roiSelectedStepIndex = -1;
          this.roiPolygonDrawing = false;
          this.roiEllipseDrawing = false;
          this.roiEllipseGuidePoints = [];
        }
      }),
      // Track particle overlay when detect_particles step is selected
      combineLatest([
        this.pipelineState.pipeline$,
        this.pipelineState.selectedStepIndex$,
        this.pipelineState.sideOutputs$,
        this.pipelineState.previewImageIndex$,
        this.pipelineState.imageDims$,
      ]).subscribe(([pipeline, idx, sideOutputs, imgIdx, dims]) => {
        if (idx >= 0 && idx < pipeline.steps.length &&
            pipeline.steps[idx].step_def_id === 'detect_particles') {
          this.particleStepIndex = idx;
          this.particleImgW = dims.w || 100;
          this.particleImgH = dims.h || 100;
          const particles = sideOutputs?.['meta']?.['particles'];
          if (Array.isArray(particles) && particles[imgIdx]) {
            this.particlesForOverlay = particles[imgIdx];
          } else {
            this.particlesForOverlay = [];
          }
          const step = pipeline.steps[idx];
          const excludedArr: string[] = step.param_values?.['excluded_ids'] ?? [];
          this.particleExcludedIds = new Set(excludedArr);
          this.particleOverlayActive = true;
        } else {
          this.particleOverlayActive = false;
          this.particlesForOverlay = [];
          this.particleExcludedIds = new Set();
          this.particleStepIndex = -1;
        }
      }),
      // Track circle overlay when detect_circles step is selected
      combineLatest([
        this.pipelineState.pipeline$,
        this.pipelineState.selectedStepIndex$,
        this.pipelineState.sideOutputs$,
        this.pipelineState.previewImageIndex$,
        this.pipelineState.imageDims$,
      ]).subscribe(([pipeline, idx, sideOutputs, imgIdx, dims]) => {
        if (idx >= 0 && idx < pipeline.steps.length &&
            pipeline.steps[idx].step_def_id === 'detect_circles') {
          this.circleStepIndex = idx;
          this.circleImgW = dims.w || 100;
          this.circleImgH = dims.h || 100;
          const circles = sideOutputs?.['circles'];
          // In single-image preview mode, circles array has only 1 element (index 0)
          const cIdx = Array.isArray(circles) && circles.length === 1 ? 0 : imgIdx;
          if (Array.isArray(circles) && Array.isArray(circles[cIdx])) {
            this.circlesForOverlay = circles[cIdx];
          } else {
            this.circlesForOverlay = [];
          }
          // Only show overlay if apply_mask is not enabled
          const applyMask = pipeline.steps[idx].param_values?.['apply_mask'] ?? false;
          this.circleOverlayActive = this.circlesForOverlay.length > 0 && !applyMask;
        } else {
          this.circleOverlayActive = false;
          this.circlesForOverlay = [];
          this.circleStepIndex = -1;
        }
      }),
      // Track reference crop overlay and sorted crop strip.
      combineLatest([
        this.pipelineState.pipeline$,
        this.pipelineState.selectedStepIndex$,
        this.pipelineState.sideOutputs$,
        this.pipelineState.previewImageIndex$,
        this.pipelineState.imageDims$,
      ]).subscribe(([pipeline, idx, sideOutputs, imgIdx, dims]) => {
        const stepDefId = idx >= 0 && idx < pipeline.steps.length ? pipeline.steps[idx].step_def_id : '';
        if (stepDefId === 'reference_crop' || stepDefId === 'reference_sequence') {
          this.deactivateMeasurementTools();
          const step = pipeline.steps[idx];
          this.referenceCropStepIndex = idx;
          this.referenceCropImgW = dims.w || this.rulerImgW || 100;
          this.referenceCropImgH = dims.h || this.rulerImgH || 100;
          if (stepDefId === 'reference_crop') {
            this.referenceCropSize = Math.max(1, Number(step.param_values?.['crop_size'] ?? 64) || 64);
            this.referenceCropCurrentImageIndex = imgIdx;
            this.referenceCropSquares = this.getReferenceCropSquaresForImage(step.param_values, imgIdx);
            this.referenceCropGlobalOffset = this.getReferenceCropGlobalOffset(step.param_values, imgIdx);
            this.referenceCropLabels = [];
            this.referenceCropScores = [];
            this.referenceSequenceComponents = [];
            this.referenceSequenceScores = {};
            this.referenceSequenceDiffs = {};
            this.referenceSequenceComponent = '';
            this.referenceSequenceColor = '#94a3b8';
            this.referenceSequenceMaxScore = 1;
            this.referenceCropStripActive = !!step.param_values?.['show_references'];
            this.referenceCropActive = !this.referenceCropStripActive;
          } else {
            this.referenceCropSquares = [];
            const sequence = this.getReferenceSequenceState(sideOutputs, imgIdx, pipeline, idx);
            this.referenceCropLabels = sequence.labels;
            this.referenceCropScores = sequence.scores;
            this.referenceSequenceComponents = sequence.components;
            this.referenceSequenceScores = sequence.scoresByComponent;
            this.referenceSequenceDiffs = sequence.diffsByComponent;
            this.referenceSequenceComponent = sequence.component;
            this.referenceSequenceColor = this.getReferenceSequenceColor(sequence.component);
            this.referenceSequenceMaxScore = this.getReferenceSequenceMaxScore(sequence.scores);
            this.referenceCropStripActive = true;
            this.referenceCropActive = false;
          }

          const cropRows = sideOutputs?.['reference_crops_base64'];
          const rowIdx = Array.isArray(cropRows) && cropRows.length === 1 ? 0 : imgIdx;
          const row = Array.isArray(cropRows) ? cropRows[rowIdx] : [];
          this.referenceCropImages = Array.isArray(row)
            ? row.filter((src: string | null) => !!src).map((src: string) => `data:image/jpeg;base64,${src}`)
            : [];
        } else {
          this.referenceCropActive = false;
          this.referenceCropStripActive = false;
          this.referenceCropImages = [];
          this.referenceCropSquares = [];
          this.referenceCropLabels = [];
          this.referenceCropScores = [];
          this.referenceSequenceComponents = [];
          this.referenceSequenceScores = {};
          this.referenceSequenceDiffs = {};
          this.referenceSequenceComponent = '';
          this.referenceSequenceColor = '#94a3b8';
          this.referenceSequenceMaxScore = 1;
          this.referenceCropDragIndex = -1;
          this.referenceCropStepIndex = -1;
          this.referenceCropGlobalOffset = 0;
        }
      }),
      // Track scale bar overlay when scale_bar_overlay step is selected
      combineLatest([
        this.pipelineState.pipeline$,
        this.pipelineState.selectedStepIndex$,
        this.pipelineState.imageDims$,
        this.pipelineState.previewImageIndex$,
      ]).subscribe(([pipeline, idx, dims, imgIdx]) => {
        if (idx >= 0 && idx < pipeline.steps.length &&
            pipeline.steps[idx].step_def_id === 'scale_bar_overlay') {
          this.scaleBarOverlayActive = true;
          this.scaleBarSelectedStepIndex = idx;
          this.scaleBarSelectedParams = pipeline.steps[idx].param_values ?? {};
          this.showScaleBar = true;
          this.syncScaleBarEditorFromParams(this.scaleBarSelectedParams);
          this.scaleBarUnit = String(this.scaleBarSelectedParams?.['label_unit'] ?? this.scaleBarSelectedParams?.['bar_length_unit'] ?? this.scaleBarUnit ?? 'mm');
          const selectedBarLengthMm = Number(this.scaleBarSelectedParams?.['bar_length_mm'] ?? 0) || 0;
          this.scaleBarLengthMm = selectedBarLengthMm > 0 ? this.fromMm(selectedBarLengthMm, this.scaleBarUnit) : 0;
          this.scaleBarPositionX = Number(this.scaleBarSelectedParams?.['position_x'] ?? -1) || -1;
          this.scaleBarPositionY = Number(this.scaleBarSelectedParams?.['position_y'] ?? -1) || -1;
          this.syncScaleBarCalibration(this.pxPerMm);
          this.pipelineState.setScaleBarExportParams(this.scaleBarSelectedParams);
          this.rebuildScaleBarOverlay(this.scaleBarSelectedParams, dims.w || 100, dims.h || 100, imgIdx);
        } else {
          this.scaleBarOverlayActive = false;
          this.scaleBarDragging = false;
          this.refreshScaleBarOverlay();
        }
      }),
    );
  }

  ngOnDestroy(): void {
    this.subs.forEach((s) => s.unsubscribe());
    if (this.boundOnKeyDown) {
      window.removeEventListener('keydown', this.boundOnKeyDown);
    }
    if (this.zoomAnchorFrame !== null) {
      cancelAnimationFrame(this.zoomAnchorFrame);
    }
  }

  // --- Gallery zoom/pan ---

  onWheel(event: WheelEvent): void {
    if (this.showGraphViewer) return;
    if (!event.ctrlKey && !event.shiftKey) return;
    event.preventDefault();

    const factor = event.deltaY > 0 ? 0.9 : 1.1;

    // Determine if we're in montage view or regular image view
    if (this.showingMontage && this.montagePreview) {
      const montageContainer = this.previewContainer?.nativeElement
        .querySelector('.montage-gallery-container') as HTMLElement | null;
      const montageImg = montageContainer
        ?.querySelector('.montage-gallery-image') as HTMLImageElement | null;

      this.montageZoomLevel = Math.max(1.0, Math.min(5.0, this.montageZoomLevel * factor));
      this.zoomAtPointer(event, montageContainer, montageImg, () => this.applyMontageTransform());
    } else {
      this.zoomLevel = Math.max(1.0, Math.min(5.0, this.zoomLevel * factor));
      this.zoomAtPointer(
        event,
        this.scrollArea?.nativeElement,
        this.previewImg?.nativeElement,
        () => this.applyImageTransform(),
      );
    }
  }

  private zoomAtPointer(
    event: WheelEvent,
    container: HTMLElement | null | undefined,
    image: HTMLImageElement | null | undefined,
    applyTransform: () => void,
  ): void {
    if (!container || !image) {
      applyTransform();
      return;
    }

    const oldRect = image.getBoundingClientRect();
    if (oldRect.width <= 0 || oldRect.height <= 0) {
      applyTransform();
      return;
    }

    // Remember the exact image point under the cursor before changing its size.
    const imageX = (event.clientX - oldRect.left) / oldRect.width;
    const imageY = (event.clientY - oldRect.top) / oldRect.height;
    const pointerX = event.clientX;
    const pointerY = event.clientY;

    applyTransform();

    // The first zoom also changes flex alignment, so adjust after Angular/CSS layout.
    if (this.zoomAnchorFrame !== null) {
      cancelAnimationFrame(this.zoomAnchorFrame);
    }
    this.zoomAnchorFrame = requestAnimationFrame(() => {
      this.zoomAnchorFrame = null;
      const newRect = image.getBoundingClientRect();
      container.scrollLeft += newRect.left + imageX * newRect.width - pointerX;
      container.scrollTop += newRect.top + imageY * newRect.height - pointerY;
    });
  }

  onMouseDown(event: MouseEvent): void {
    if (this.showRoiContextMenu) this.showRoiContextMenu = false;
    if (this.showGraphViewer) return;
    
    // Check zoom level based on which view is active
    const isZoomed = this.showingMontage ? this.montageZoomLevel > 1.0 : this.zoomLevel > 1.0;
    if (!isZoomed) return;
    
    event.preventDefault();
    
    // Get the appropriate scrollable container
    let container: HTMLElement | null = null;
    if (this.showingMontage) {
      const previewWrapper = this.previewContainer?.nativeElement;
      container = previewWrapper?.querySelector('.montage-gallery-container');
    } else {
      container = this.scrollArea?.nativeElement;
    }
    
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
    if (this.scaleBarDragging) {
      this.onScaleBarMouseMove(event);
      return;
    }
    const isZoomed = this.showingMontage ? this.montageZoomLevel > 1.0 : this.zoomLevel > 1.0;
    if (this.showGraphViewer || !this.isDragging || !isZoomed) return;
    
    // Get the appropriate scrollable container
    let container: HTMLElement | null = null;
    if (this.showingMontage) {
      const previewWrapper = this.previewContainer?.nativeElement;
      container = previewWrapper?.querySelector('.montage-gallery-container');
    } else {
      container = this.scrollArea?.nativeElement;
    }
    
    if (!container) return;
    container.scrollLeft = this.dragStart.scrollLeft - (event.clientX - this.dragStart.x);
    container.scrollTop = this.dragStart.scrollTop - (event.clientY - this.dragStart.y);
  }

  onMouseUp(): void {
    if (this.scaleBarDragging) {
      this.onScaleBarMouseUp();
    }
    this.isDragging = false;
  }

  resetZoom(): void {
    if (this.showGraphViewer) return;
    if (this.showingMontage && this.montagePreview) {
      this.montageZoomLevel = 1.0;
      this.applyMontageTransform();
    } else {
      this.zoomLevel = 1.0;
      this.applyImageTransform();
    }
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
    if (this.pixelActive) return 'crosshair';
    
    const isZoomed = this.showingMontage ? this.montageZoomLevel > 1.0 : this.zoomLevel > 1.0;
    if (isZoomed && !this.rulerActive && !this.scaleActive) {
      return this.isDragging ? 'grabbing' : 'grab';
    }
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

  private applyMontageTransform(): void {
    // Find the montage gallery container (which is scrollable)
    const previewWrapper = this.previewContainer?.nativeElement;
    if (!previewWrapper) return;

    const montageContainer = previewWrapper.querySelector('.montage-gallery-container') as HTMLDivElement;
    if (!montageContainer) return;

    // Find the montage image element
    const montageImg = montageContainer.querySelector('.montage-gallery-image') as HTMLImageElement;
    if (!montageImg) return;

    const nw = montageImg.naturalWidth;
    const nh = montageImg.naturalHeight;
    if (nw === 0 || nh === 0) return;

    if (this.montageZoomLevel <= 1.0) {
      // Calculate fit scale based on container dimensions
      const cw = montageContainer.clientWidth;
      const ch = montageContainer.clientHeight;
      if (cw === 0 || ch === 0) return;
      this.montageMontageBaseFitScale = Math.min(cw / nw, ch / nh, 1);
      montageImg.style.width = `${Math.floor(nw * this.montageMontageBaseFitScale)}px`;
      montageImg.style.height = `${Math.floor(nh * this.montageMontageBaseFitScale)}px`;
      montageContainer.scrollLeft = 0;
      montageContainer.scrollTop = 0;
    } else {
      const fitW = Math.floor(nw * this.montageMontageBaseFitScale * this.montageZoomLevel);
      const fitH = Math.floor(nh * this.montageMontageBaseFitScale * this.montageZoomLevel);
      montageImg.style.width = `${fitW}px`;
      montageImg.style.height = `${fitH}px`;
    }
  }

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

  // --- Particle contour click ---

  particlePolygonStr(particle: any): string {
    const pts: number[][] = particle.polygon ?? particle.contour;
    if (!Array.isArray(pts)) return '';
    return pts.map((p: number[]) => `${p[0]},${p[1]}`).join(' ');
  }

  isParticleExcluded(particleId: string): boolean {
    return this.particleExcludedIds.has(particleId);
  }

  onParticleClick(particle: any, event: MouseEvent): void {
    event.stopPropagation();
    if (!particle.passed_filters && !particle.excluded) return;
    const pipeline = this.pipelineState.getPipeline();
    if (this.particleStepIndex < 0 || this.particleStepIndex >= pipeline.steps.length) return;
    const step = pipeline.steps[this.particleStepIndex];
    const particleId: string = particle.particle_id;
    const currentExcluded: string[] = [...(step.param_values?.['excluded_ids'] ?? [])];
    const idx = currentExcluded.indexOf(particleId);
    if (idx >= 0) {
      currentExcluded.splice(idx, 1);
    } else {
      currentExcluded.push(particleId);
    }
    this.particleClickPending = true;
    const updated = { ...step.param_values, excluded_ids: currentExcluded };
    this.pipelineState.updateParams(this.particleStepIndex, updated);
  }

  // --- Graph viewer ---

  openGraphViewer(data: any, omittedIndices: Set<number>, sourceStepIndex: number): void {
    this.graphData = data;
    this.graphOmittedIndices = new Set(omittedIndices);
    this.graphViewerStepIndex = sourceStepIndex;
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
    this.graphViewerStepIndex = -1;
  }

  // --- Expanded chart viewer ---

  getOmittedIndices(): Set<number> {
    return this.graphOmittedIndices;
  }

  onPCAComponentChanged(event: { pcX: number; pcY: number }): void {
    // Pass through from PCA chart component
  }

  closeExpandedChart(): void {
    this.showExpandedChart = false;
    this.expandedChartData = null;
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

  copyGraphEquation(): void {
    if (!this.graphData) return;
    const formula = this.getFormulaText(this.graphData);
    if (!formula) return;
    navigator.clipboard.writeText(formula).catch(() => undefined);
  }

  hasGraphData(): boolean {
    return !!this.graphData;
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
  }

  resetGraphTransform(): void {
    this.graphZoom = 1.0;
    this.graphPanX = 0;
    this.graphPanY = 0;
    this.drawGraph();
  }

  // --- Reference crop interaction ---

  private parseReferenceCropSquares(raw: any): Array<{ x: number; y: number; size: number; name?: string }> {
    try {
      const parsed = typeof raw === 'string' ? JSON.parse(raw || '[]') : (raw || []);
      if (!Array.isArray(parsed)) return [];
      return parsed
        .map((sq) => ({
          x: Number(sq?.x ?? 0) || 0,
          y: Number(sq?.y ?? 0) || 0,
          size: this.referenceCropSize,
          name: typeof sq?.name === 'string' ? sq.name : '',
        }))
        .filter((sq) => sq.size > 0);
    } catch {
      return [];
    }
  }

  private parseReferenceCropSquareOverrides(raw: any): Record<string, any[]> {
    try {
      const parsed = typeof raw === 'string' ? JSON.parse(raw || '{}') : (raw || {});
      return parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? parsed : {};
    } catch {
      return {};
    }
  }

  private getReferenceCropSquaresForImage(
    params: Record<string, any> | undefined,
    imageIndex: number
  ): Array<{ x: number; y: number; size: number; name?: string }> {
    const overrides = this.parseReferenceCropSquareOverrides(params?.['reference_square_overrides']);
    const current = overrides[String(imageIndex)];
    if (Array.isArray(current)) {
      return this.parseReferenceCropSquares(current);
    }
    if (Object.keys(overrides).length > 0) {
      return [];
    }
    return this.parseReferenceCropSquares(params?.['reference_squares']);
  }

  private getReferenceCropGlobalOffset(params: Record<string, any> | undefined, imageIndex: number): number {
    const overrides = this.parseReferenceCropSquareOverrides(params?.['reference_square_overrides']);
    const keys = Object.keys(overrides);
    if (!keys.length) return 0;
    return keys
      .filter((key) => Number(key) < imageIndex)
      .reduce((sum, key) => sum + (Array.isArray(overrides[key]) ? overrides[key].length : 0), 0);
  }

  private getAllReferenceCropSquares(params: Record<string, any> | undefined): Array<{ x: number; y: number; size: number; name?: string }> {
    const overrides = this.parseReferenceCropSquareOverrides(params?.['reference_square_overrides']);
    const keys = Object.keys(overrides).sort((a, b) => Number(a) - Number(b));
    if (!keys.length) return this.parseReferenceCropSquares(params?.['reference_squares']);
    return keys.flatMap((key) => this.parseReferenceCropSquares(overrides[key]));
  }

  private getReferenceSequenceState(
    sideOutputs: any,
    imgIdx: number,
    pipeline: any,
    stepIndex: number
  ): {
    labels: string[];
    scores: number[];
    component: string;
    components: string[];
    scoresByComponent: Record<string, number[]>;
    diffsByComponent: Record<string, Array<number | null>>;
  } {
    const sequenceMeta = sideOutputs?.meta?.reference_sequence;
    const rows = sequenceMeta?.rows;
    const rowIdx = Array.isArray(rows) && rows.length === 1 ? 0 : imgIdx;
    const row = Array.isArray(rows) ? rows[rowIdx] : null;
    const items = row?.items;
    const component = String(sequenceMeta?.component ?? '');
    const components = Array.isArray(sequenceMeta?.components) && sequenceMeta.components.length
      ? sequenceMeta.components.map((value: any) => String(value))
      : [component].filter(Boolean);
    if (!Array.isArray(items)) {
      return { labels: [], scores: [], component, components, scoresByComponent: {}, diffsByComponent: {} };
    }
    const cropNames = this.getPreviousReferenceCropNames(pipeline, stepIndex);
    const labels = items.map((item: any, index: number) => {
      const label = String(item?.label ?? '').trim();
      const originalIndex = Number(item?.original_index);
      const cropName = Number.isInteger(originalIndex) ? cropNames[originalIndex]?.trim() : '';
      return cropName || label || String(index + 1);
    });
    const scores = items.map((item: any) => Number(item?.score));
    const scoresByComponent: Record<string, number[]> = {};
    const diffsByComponent: Record<string, Array<number | null>> = {};
    for (const comp of components) {
      scoresByComponent[comp] = items.map((item: any) => Number(item?.scores?.[comp] ?? item?.score));
      diffsByComponent[comp] = items.map((item: any) => {
        const value = item?.diffs?.[comp];
        return value === null || value === undefined ? null : Number(value);
      });
    }
    return {
      labels,
      scores,
      component,
      components,
      scoresByComponent,
      diffsByComponent,
    };
  }

  private getPreviousReferenceCropNames(pipeline: any, stepIndex: number): string[] {
    const steps = Array.isArray(pipeline?.steps) ? pipeline.steps : [];
    for (let i = stepIndex - 1; i >= 0; i--) {
      if (steps[i]?.step_def_id !== 'reference_crop') continue;
      const squares = this.getAllReferenceCropSquares(steps[i]?.param_values);
      return squares.map((sq) => String(sq.name ?? ''));
    }
    return [];
  }

  private getReferenceSequenceMaxScore(scores: number[]): number {
    const finiteScores = scores.filter((value) => Number.isFinite(value));
    return finiteScores.length ? Math.max(...finiteScores.map((value) => Math.abs(value)), 1) : 1;
  }

  getReferenceSequenceColor(component: string): string {
    const colors: Record<string, string> = {
      R: '#ef4444',
      G: '#22c55e',
      B: '#3b82f6',
      H: '#f97316',
      S: '#a855f7',
      V: '#facc15',
      L: '#e5e7eb',
      A: '#ec4899',
      LAB_B: '#06b6d4',
      GRAY: '#94a3b8',
    };
    return colors[String(component || 'GRAY').toUpperCase()] || colors['GRAY'];
  }

  private referenceCropSvgCoords(event: MouseEvent): { x: number; y: number } {
    const svg = (event.currentTarget ?? event.target) as SVGSVGElement;
    const rect = svg.getBoundingClientRect();
    const sx = this.referenceCropImgW / rect.width;
    const sy = this.referenceCropImgH / rect.height;
    return {
      x: Math.round(Math.max(0, Math.min(this.referenceCropImgW, (event.clientX - rect.left) * sx))),
      y: Math.round(Math.max(0, Math.min(this.referenceCropImgH, (event.clientY - rect.top) * sy))),
    };
  }

  private commitReferenceCropSquares(): void {
    if (this.referenceCropStepIndex < 0) return;
    const pipeline = this.pipelineState.getPipeline();
    const step = pipeline.steps[this.referenceCropStepIndex];
    if (!step) return;
    const squares = this.referenceCropSquares.map((sq) => ({
      x: Math.round(sq.x),
      y: Math.round(sq.y),
      size: Math.round(sq.size),
      name: sq.name ?? '',
    }));
    const updated = { ...step.param_values };
    const overrides = this.parseReferenceCropSquareOverrides(updated['reference_square_overrides']);
    overrides[String(this.referenceCropCurrentImageIndex)] = squares;
    updated['reference_square_overrides'] = JSON.stringify(overrides);
    updated['reference_squares'] = JSON.stringify(this.getAllReferenceCropSquares(updated));
    this.pipelineState.updateParams(this.referenceCropStepIndex, updated);
  }

  private referenceCropHitIndex(x: number, y: number): number {
    for (let i = this.referenceCropSquares.length - 1; i >= 0; i--) {
      const sq = this.referenceCropSquares[i];
      if (x >= sq.x && x <= sq.x + sq.size && y >= sq.y && y <= sq.y + sq.size) {
        return i;
      }
    }
    return -1;
  }

  onReferenceCropMouseDown(event: MouseEvent): void {
    if (event.button !== 0 || !this.referenceCropActive) return;
    event.preventDefault();
    event.stopPropagation();

    const { x, y } = this.referenceCropSvgCoords(event);
    const hit = this.referenceCropHitIndex(x, y);
    if (hit >= 0) {
      const sq = this.referenceCropSquares[hit];
      this.referenceCropDragIndex = hit;
      this.referenceCropDragOffset = { x: x - sq.x, y: y - sq.y };
      return;
    }

    const size = this.referenceCropSize;
    const nx = Math.max(0, Math.min(this.referenceCropImgW - size, x - size / 2));
    const ny = Math.max(0, Math.min(this.referenceCropImgH - size, y - size / 2));
    this.referenceCropSquares = [...this.referenceCropSquares, { x: nx, y: ny, size }];
    this.commitReferenceCropSquares();
  }

  onReferenceCropMouseMove(event: MouseEvent): void {
    if (this.referenceCropDragIndex < 0) return;
    event.preventDefault();
    event.stopPropagation();

    const { x, y } = this.referenceCropSvgCoords(event);
    const squares = [...this.referenceCropSquares];
    const sq = { ...squares[this.referenceCropDragIndex] };
    sq.x = Math.max(0, Math.min(this.referenceCropImgW - sq.size, x - this.referenceCropDragOffset.x));
    sq.y = Math.max(0, Math.min(this.referenceCropImgH - sq.size, y - this.referenceCropDragOffset.y));
    squares[this.referenceCropDragIndex] = sq;
    this.referenceCropSquares = squares;
  }

  onReferenceCropMouseUp(event: MouseEvent): void {
    if (this.referenceCropDragIndex < 0) return;
    event.preventDefault();
    event.stopPropagation();
    this.referenceCropDragIndex = -1;
    this.commitReferenceCropSquares();
  }

  onReferenceCropContextMenu(event: MouseEvent): void {
    if (!this.referenceCropActive) return;
    event.preventDefault();
    event.stopPropagation();
    const { x, y } = this.referenceCropSvgCoords(event);
    const hit = this.referenceCropHitIndex(x, y);
    if (hit >= 0) {
      this.referenceCropSquares = this.referenceCropSquares.filter((_, i) => i !== hit);
      this.commitReferenceCropSquares();
    }
  }

  // --- ROI interaction ---

  onImageLoad(): void {
    const img = this.previewImg?.nativeElement;
    if (img) {
      this.roiImgW = img.naturalWidth;
      this.roiImgH = img.naturalHeight;
      this.referenceCropImgW = img.naturalWidth;
      this.referenceCropImgH = img.naturalHeight;
      this.rulerImgW = img.naturalWidth;
      this.rulerImgH = img.naturalHeight;
      if (this.particleClickPending) {
        this.particleClickPending = false;
      } else {
        this.zoomLevel = 1.0;
      }
      this.applyImageTransform();
    }
  }

  private syncRoiFromParams(params: Record<string, any>, imageIndex?: number): void {
    // Check for per-image override first
    let effectiveParams = params;
    const imgIdx = imageIndex ?? this.roiCurrentImageIndex;
    const overridesRaw = params?.['roi_overrides'] ?? '{}';
    let hasOwnOverride = false;
    try {
      const overrides = typeof overridesRaw === 'string' ? JSON.parse(overridesRaw) : (overridesRaw || {});
      const imgKey = String(imgIdx);
      if (overrides[imgKey]) {
        // Merge: per-image override values take precedence over global params
        effectiveParams = { ...params, ...overrides[imgKey] };
        hasOwnOverride = true;
      }
    } catch { /* ignore parse errors, use global params */ }
    this.roiHasOverride = hasOwnOverride;

    const t = this.normalizeRoiType(effectiveParams?.['roi_type'] ?? 'rect');
    this.roiAngle = effectiveParams?.['roi_angle'] ?? 0;
    if (t === 'rect') {
      const rx = effectiveParams?.['roi_x'] ?? 0;
      const ry = effectiveParams?.['roi_y'] ?? 0;
      const rw = effectiveParams?.['roi_width'] ?? 0;
      const rh = effectiveParams?.['roi_height'] ?? 0;
      this.roiRect = { x: rx, y: ry, w: rw, h: rh };
      this.hasRoiShape = rw > 0 && rh > 0;
    } else if (t === 'ellipse') {
      const ecx = effectiveParams?.['roi_cx'] ?? 0;
      const ecy = effectiveParams?.['roi_cy'] ?? 0;
      const erx = effectiveParams?.['roi_rx'] ?? 0;
      const ery = effectiveParams?.['roi_ry'] ?? 0;
      this.roiEllipse = { cx: ecx, cy: ecy, rx: erx, ry: ery };
      this.hasRoiShape = erx > 0 && ery > 0;
    } else if (t === 'polygon') {
      const raw = effectiveParams?.['roi_points'] ?? '[]';
      try {
        const pts = typeof raw === 'string' ? JSON.parse(raw) : raw;
        this.roiPolygon = Array.isArray(pts) ? pts : [];
      } catch {
        this.roiPolygon = [];
      }
      this.hasRoiShape = this.roiPolygon.length > 0;
    }
  }

  private normalizeRoiType(value: any): 'rect' | 'ellipse' | 'polygon' {
    if (value === 'circle') return 'ellipse';
    if (value === 'ellipse' || value === 'polygon' || value === 'rect') return value;
    return 'rect';
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
    this.roiAllSelected = false;
    const { x, y } = this.svgCoords(event);
    const target = event.target as SVGElement;
    const handle = target.getAttribute?.('data-handle');

    if (this.roiType === 'polygon') {
      // Rotation handle for polygon
      if (handle === 'rot' && this.roiPolygon.length >= 3 && !this.roiPolygonDrawing) {
        const c = this.roiPolygonCentroid();
        this.roiDragMode = 'rot';
        this.roiDragStart = { mx: x, my: y, ox: c.x, oy: c.y, ow: this.roiAngle, oh: 0 };
        event.preventDefault();
        event.stopPropagation();
        return;
      }
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
      if (handle === 'rot') {
        // Store center for angle calculation
        if (this.roiType === 'rect') {
          this.roiDragStart = { mx: x, my: y, ox: this.roiRect.x + this.roiRect.w / 2, oy: this.roiRect.y + this.roiRect.h / 2, ow: this.roiAngle, oh: 0 };
        } else {
          this.roiDragStart = { mx: x, my: y, ox: this.roiEllipse.cx, oy: this.roiEllipse.cy, ow: this.roiAngle, oh: 0 };
        }
      } else if (this.roiType === 'rect') {
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

    // Handle rotation for rect/ellipse
    if (this.roiDragMode === 'rot') {
      const cx = s.ox; // center x
      const cy = s.oy; // center y
      const startAngle = Math.atan2(s.mx - cx, -(s.my - cy)) * 180 / Math.PI;
      const currAngle = Math.atan2(x - cx, -(y - cy)) * 180 / Math.PI;
      let newAngle = s.ow + (currAngle - startAngle);
      // Normalize to -180..180
      while (newAngle > 180) newAngle -= 360;
      while (newAngle < -180) newAngle += 360;
      this.roiAngle = Math.round(newAngle * 10) / 10;
      return;
    }

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
    if (r.w <= 0 || r.h <= 0) return false;
    // Rotate point back to axis-aligned space around rect center
    const cx = r.x + r.w / 2;
    const cy = r.y + r.h / 2;
    const { x: ux, y: uy } = this.unrotatePoint(x, y, cx, cy);
    return ux >= r.x && ux <= r.x + r.w && uy >= r.y && uy <= r.y + r.h;
  }

  private isInsideEllipse(x: number, y: number): boolean {
    const e = this.roiEllipse;
    if (e.rx < 1 || e.ry < 1) return false;
    // Rotate point back to axis-aligned space around ellipse center
    const { x: ux, y: uy } = this.unrotatePoint(x, y, e.cx, e.cy);
    const dx = (ux - e.cx) / e.rx;
    const dy = (uy - e.cy) / e.ry;
    return (dx * dx + dy * dy) <= 1;
  }

  /** Rotate a point (px, py) by -roiAngle around (cx, cy) to get unrotated coordinates */
  private unrotatePoint(px: number, py: number, cx: number, cy: number): { x: number; y: number } {
    if (Math.abs(this.roiAngle) < 0.01) return { x: px, y: py };
    const rad = -this.roiAngle * Math.PI / 180;
    const cos = Math.cos(rad);
    const sin = Math.sin(rad);
    const dx = px - cx;
    const dy = py - cy;
    return { x: cx + dx * cos - dy * sin, y: cy + dx * sin + dy * cos };
  }

  private commitRoi(): void {
    if (this.roiSelectedStepIndex < 0) return;
    const pipeline = this.pipelineState.getPipeline();
    const step = pipeline.steps[this.roiSelectedStepIndex];
    if (!step) return;

    const updated = { ...step.param_values };

    const roiValues: Record<string, any> = {};
    roiValues['roi_angle'] = this.roiAngle;
    roiValues['roi_type'] = this.roiType;
    if (this.roiType === 'rect') {
      roiValues['roi_x'] = Math.round(this.roiRect.x);
      roiValues['roi_y'] = Math.round(this.roiRect.y);
      roiValues['roi_width'] = Math.round(this.roiRect.w);
      roiValues['roi_height'] = Math.round(this.roiRect.h);
    } else if (this.roiType === 'ellipse') {
      roiValues['roi_cx'] = Math.round(this.roiEllipse.cx);
      roiValues['roi_cy'] = Math.round(this.roiEllipse.cy);
      roiValues['roi_rx'] = Math.round(this.roiEllipse.rx);
      roiValues['roi_ry'] = Math.round(this.roiEllipse.ry);
    } else if (this.roiType === 'polygon') {
      roiValues['roi_points'] = JSON.stringify(
        this.roiPolygon.map((p) => ({ x: Math.round(p.x), y: Math.round(p.y) }))
      );
    }

    // Update per-image override in roi_overrides
    let overrides: Record<string, any> = {};
    try {
      const raw = updated['roi_overrides'] ?? '{}';
      overrides = typeof raw === 'string' ? JSON.parse(raw) : (raw || {});
    } catch { overrides = {}; }

    const imgKey = String(this.roiCurrentImageIndex);
    overrides[imgKey] = roiValues;
    updated['roi_overrides'] = JSON.stringify(overrides);

    // Sync shape to global fallback (images without overrides inherit this),
    // but keep roi_angle per-image only so rotating one image doesn't affect others.
    const { roi_angle: _ignoreAngle, ...shapeValues } = roiValues;
    Object.assign(updated, shapeValues);

    this.roiHasOverride = true;
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
    this.roiAngle = 0;
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
    this.roiHasOverride = false;
    // Clear all per-image overrides (full reset)
    if (this.roiSelectedStepIndex >= 0) {
      const pipeline = this.pipelineState.getPipeline();
      const step = pipeline.steps[this.roiSelectedStepIndex];
      if (step) {
        const updated = { ...step.param_values };
        updated['roi_overrides'] = '{}';
        updated['roi_angle'] = 0;
        // Zero out global ROI params
        updated['roi_x'] = 0; updated['roi_y'] = 0;
        updated['roi_width'] = 0; updated['roi_height'] = 0;
        updated['roi_cx'] = 0; updated['roi_cy'] = 0;
        updated['roi_rx'] = 0; updated['roi_ry'] = 0;
        updated['roi_points'] = '[]';
        this.pipelineState.updateParams(this.roiSelectedStepIndex, updated);
      }
    }
  }

  private onRoiKeyDown(event: KeyboardEvent): void {
    if (!this.roiActive || !this.hasRoiShape) return;
    // Ctrl+A → select all (visual feedback, marks shape for deletion)
    if ((event.ctrlKey || event.metaKey) && event.key === 'a') {
      event.preventDefault();
      this.roiAllSelected = true;
      return;
    }
    // Delete or Backspace → clear ROI if selected
    if (this.roiAllSelected && (event.key === 'Delete' || event.key === 'Backspace')) {
      event.preventDefault();
      this.roiAllSelected = false;
      this.clearRoiSelection();
      return;
    }
    // Escape → deselect
    if (event.key === 'Escape') {
      this.roiAllSelected = false;
    }
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

  /** Centroid of the polygon */
  roiPolygonCentroid(): { x: number; y: number } {
    if (this.roiPolygon.length === 0) return { x: 0, y: 0 };
    const sx = this.roiPolygon.reduce((a, p) => a + p.x, 0);
    const sy = this.roiPolygon.reduce((a, p) => a + p.y, 0);
    return { x: sx / this.roiPolygon.length, y: sy / this.roiPolygon.length };
  }

  /** Position for the polygon rotation handle (topmost-rightmost vertex, offset outward) */
  roiPolygonTopRight(): { x: number; y: number } {
    if (this.roiPolygon.length < 3) return { x: 0, y: 0 };
    // Find the point with the smallest y (topmost), break ties with largest x
    let best = this.roiPolygon[0];
    for (const p of this.roiPolygon) {
      if (p.y < best.y || (p.y === best.y && p.x > best.x)) best = p;
    }
    return { x: best.x, y: best.y - 18 * this.roiScale };
  }

  /** SVG transform string for rotated polygon group */
  roiPolygonTransform(): string {
    if (Math.abs(this.roiAngle) < 0.01 || this.roiPolygon.length < 3) return '';
    const c = this.roiPolygonCentroid();
    return `rotate(${this.roiAngle} ${c.x} ${c.y})`;
  }

  /** SVG transform string for rotated rectangle group */
  roiRectTransform(): string {
    if (Math.abs(this.roiAngle) < 0.01) return '';
    const cx = this.roiRect.x + this.roiRect.w / 2;
    const cy = this.roiRect.y + this.roiRect.h / 2;
    return `rotate(${this.roiAngle} ${cx} ${cy})`;
  }

  /** SVG transform string for rotated ellipse group */
  roiEllipseTransform(): string {
    if (Math.abs(this.roiAngle) < 0.01) return '';
    return `rotate(${this.roiAngle} ${this.roiEllipse.cx} ${this.roiEllipse.cy})`;
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

  // === Ruler tool methods (multi-ruler, up to 5 lines) ===

  toggleRuler(): void {
    this.rulerActive = !this.rulerActive;
    if (this.rulerActive) {
      this.deactivateMeasurementTools('ruler');
    }
    this.rulerDrawingStart = null;
    this.rulerDrawingCurrent = null;
    this.rulerDrawingDistance = 0;
  }

  clearAllRulerLines(): void {
    this.rulerLines = [];
    this.rulerDrawingStart = null;
    this.rulerDrawingCurrent = null;
    this.rulerDrawingDistance = 0;
  }

  getRulerMeasurement(index: number): string {
    if (index < this.rulerLines.length) {
      return this.rulerLines[index].distance.toFixed(1) + ' px';
    }
    if (index === this.rulerLines.length && this.rulerDrawingStart && this.rulerDrawingDistance > 0) {
      return this.rulerDrawingDistance.toFixed(1) + ' px';
    }
    return '';
  }

  copyRulerMeasurements(): void {
    if (this.rulerLines.length === 0) return;
    const values = this.rulerLines.map(l => l.distance.toFixed(1)).join('\t');
    navigator.clipboard.writeText(values).catch(() => undefined);
  }

  // === Scale tool methods ===

  toggleScale(): void {
    if (this.showScaleBar) {
      return;
    }
    this.scaleActive = !this.scaleActive;
    if (this.scaleActive) {
      this.deactivateMeasurementTools('scale');
    }
  }

  clearScaleLine(): void {
    this.scaleStart = null;
    this.scaleEnd = null;
    this.scaleCurrentPos = null;
    this.scaleLinePx = 0;
  }

  onScaleMmChange(): void {
    this.computeScaleBar();
  }

  onScaleMeasureUnitChange(nextUnit: string): void {
    const previousUnit = this.scaleMeasureUnit || 'mm';
    const currentValue = Number(this.scaleMm) || 0;
    const currentMm = currentValue > 0 ? this.toMm(currentValue, previousUnit) : 0;
    this.scaleMeasureUnit = nextUnit || 'mm';
    this.scaleMm = currentMm > 0 ? this.fromMm(currentMm, this.scaleMeasureUnit) : 0;
    this.computeScaleBar();
  }

  onShowScaleBarChange(): void {
    if (this.showScaleBar) {
      this.scaleActive = false;
    }
    this.computeScaleBar();
  }

  onScaleBarLengthChange(): void {
    this.scaleBarLengthMm = this.normalizeScaleBarLength(this.scaleBarLengthMm);
    this.syncScaleBarEditorToSelectedStep();
    this.computeScaleBar();
  }

  onScaleBarStyleChange(nextValue?: string, field?: 'barColor' | 'textColor'): void {
    if (field === 'barColor' && nextValue) {
      this.scaleBarBarColor = nextValue;
    }
    if (field === 'textColor' && nextValue) {
      this.scaleBarTextColor = nextValue;
    }
    this.syncScaleBarEditorToSelectedStep();
    this.computeScaleBar();
  }

  onScaleBarUnitChange(nextUnit: string): void {
    const hadManualLength = this.scaleBarLengthMm > 0;
    const currentMm = hadManualLength ? this.toMm(this.scaleBarLengthMm, this.scaleBarUnit) : 0;
    this.scaleBarUnit = nextUnit || 'mm';
    this.scaleBarLengthMm = hadManualLength && currentMm > 0 ? this.normalizeScaleBarLength(this.fromMm(currentMm, this.scaleBarUnit)) : 0;
    this.syncScaleBarEditorToSelectedStep();
    this.computeScaleBar();
  }

  private computeScaleBar(): void {
    const pm = this.pxPerMm;
    if (pm <= 0) {
      this.scaleBarPx = 0;
      this.scaleBarMm = 0;
      this.scaleBarOverlay = null;
      return;
    }
    this.syncScaleBarCalibration(pm);
    const autoTargetPx = this.rulerImgW / 5;
    const autoTargetMm = autoTargetPx / pm;
    const niceValues = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000];
    let best = niceValues[0];
    let bestDiff = Math.abs(autoTargetMm - best);
    for (const v of niceValues) {
      const diff = Math.abs(autoTargetMm - v);
      if (diff < bestDiff) {
        bestDiff = diff;
        best = v;
      }
    }

    const barLengthMm = this.scaleBarLengthMm > 0 ? this.getScaleBarLengthMm() : best;
    this.scaleBarMm = barLengthMm;
    this.scaleBarPx = Math.max(1, Math.round(barLengthMm * pm));

    this.refreshScaleBarOverlay();
  }

  private refreshScaleBarOverlay(): void {
    const imgW = this.rulerImgW || 100;
    const imgH = this.rulerImgH || 100;
    if (!this.showScaleBar || this.pxPerMm <= 0) {
      this.scaleBarOverlay = null;
      this.pipelineState.setScaleBarExportParams(null);
      return;
    }

    if (this.scaleBarOverlayActive && this.scaleBarSelectedStepIndex >= 0 && this.scaleBarSelectedParams) {
      this.pipelineState.setScaleBarExportParams(this.scaleBarSelectedParams);
      this.rebuildScaleBarOverlay(this.scaleBarSelectedParams, imgW, imgH, this.currentIndex);
      return;
    }

    const toolbarParams: Record<string, any> = {
      pixels_per_mm: this.pxPerMm,
      bar_length_mm: this.getScaleBarLengthMm(),
      label_unit: this.scaleBarUnit,
      position_x: this.scaleBarPositionX,
      position_y: this.scaleBarPositionY,
      font_family: 'sans',
      font_size: this.scaleBarFontSize,
      font_thickness: this.scaleBarFontThickness,
      bar_thickness: this.scaleBarBarThickness,
      box_padding: 14,
      text_gap: 16,
      background_opacity: 0,
      background_color: this.scaleBarBackgroundColor,
      show_background: false,
      text_color: this.scaleBarTextColor,
      bar_color: this.scaleBarBarColor,
      label_text: this.formatScaleBarLabel(),
    };
    this.pipelineState.setScaleBarExportParams(toolbarParams);
    this.rebuildScaleBarOverlay(toolbarParams, imgW, imgH, this.currentIndex);
  }

  formatScaleBarLabel(): string {
    const unit = this.scaleBarUnit || 'mm';
    const value = this.scaleBarLengthMm > 0 ? this.scaleBarLengthMm : this.fromMm(this.scaleBarMm, unit);
    return `${Math.round(value)} ${unit}`;
  }

  private getScaleMeasurementMm(): number {
    return this.scaleMm > 0 ? this.toMm(this.scaleMm, this.scaleMeasureUnit) : 0;
  }

  private getScaleBarLengthMm(): number {
    if (this.scaleBarLengthMm <= 0) {
      return this.scaleBarMm;
    }
    return this.toMm(this.scaleBarLengthMm, this.scaleBarUnit);
  }

  private normalizeScaleBarLength(value: number): number {
    if (!Number.isFinite(value) || value <= 0) {
      return 0;
    }
    return Math.round(value);
  }

  private syncScaleBarEditorFromParams(params: Record<string, any>): void {
    this.scaleBarFontSize = Math.max(8, Number(params?.['font_size'] ?? 24) || 24);
    this.scaleBarFontThickness = Math.max(1, Number(params?.['font_thickness'] ?? 1) || 1);
    this.scaleBarBarThickness = Math.max(1, Number(params?.['bar_thickness'] ?? 3) || 3);
    this.scaleBarBackgroundColor = String(params?.['background_color'] ?? 'black');
    this.scaleBarTextColor = String(params?.['text_color'] ?? 'white');
    this.scaleBarBarColor = String(params?.['bar_color'] ?? 'white');
  }

  private syncScaleBarEditorToSelectedStep(): void {
    if (this.scaleBarSelectedStepIndex < 0) {
      return;
    }

    const pipeline = this.pipelineState.getPipeline();
    if (this.scaleBarSelectedStepIndex >= pipeline.steps.length) {
      return;
    }

    const step = pipeline.steps[this.scaleBarSelectedStepIndex];
    if (!step || step.step_def_id !== 'scale_bar_overlay') {
      return;
    }

    const updated = {
      ...step.param_values,
      pixels_per_mm: this.pxPerMm,
      bar_length_mm: this.normalizeScaleBarLength(this.getScaleBarLengthMm()),
      label_unit: this.scaleBarUnit,
      position_x: this.scaleBarPositionX,
      position_y: this.scaleBarPositionY,
      font_family: 'sans',
      font_size: this.scaleBarFontSize,
      font_thickness: this.scaleBarFontThickness,
      bar_thickness: this.scaleBarBarThickness,
      box_padding: 14,
      text_gap: 16,
      background_opacity: 0,
      background_color: this.scaleBarBackgroundColor,
      show_background: false,
      text_color: this.scaleBarTextColor,
      bar_color: this.scaleBarBarColor,
      label_text: this.formatScaleBarLabel(),
    };

    this.scaleBarSelectedParams = updated;
    this.pipelineState.setScaleBarExportParams(updated);
    this.pipelineState.updateParams(this.scaleBarSelectedStepIndex, updated);
  }

  private toMm(value: number, unit: string): number {
    switch ((unit || 'mm').trim().toLowerCase()) {
      case 'cm':
        return value * 10;
      case 'um':
        return value / 1000;
      default:
        return value;
    }
  }

  private fromMm(valueMm: number, unit: string): number {
    switch ((unit || 'mm').trim().toLowerCase()) {
      case 'cm':
        return valueMm / 10;
      case 'um':
        return valueMm * 1000;
      default:
        return valueMm;
    }
  }

  private syncScaleBarCalibration(pm: number): void {
    if (this.scaleBarSelectedStepIndex < 0 || !this.scaleBarSelectedParams) {
      return;
    }

    const pipeline = this.pipelineState.getPipeline();
    if (this.scaleBarSelectedStepIndex >= pipeline.steps.length) {
      return;
    }

    const step = pipeline.steps[this.scaleBarSelectedStepIndex];
    if (!step || step.step_def_id !== 'scale_bar_overlay') {
      return;
    }

    const currentPixelsPerMm = Number(step.param_values?.['pixels_per_mm'] ?? step.param_values?.['px_per_mm'] ?? 0) || 0;
    if (Math.abs(currentPixelsPerMm - pm) < 0.0001) {
      return;
    }

    const { px_per_mm: _unusedAlias, ...rest } = step.param_values ?? {};
    const updated = { ...rest, pixels_per_mm: pm };
    this.scaleBarSelectedParams = updated;
    this.pipelineState.setScaleBarExportParams(updated);
    this.pipelineState.updateParams(this.scaleBarSelectedStepIndex, updated);
  }

  private rebuildScaleBarOverlay(params: Record<string, any>, imgW: number, imgH: number, imageIndex: number): void {
    void imageIndex;
    const rawPixelsPerMm = Number(params?.['pixels_per_mm'] ?? params?.['px_per_mm'] ?? this.pxPerMm ?? 0) || 0;
    if (rawPixelsPerMm <= 0) {
      this.scaleBarOverlay = null;
      return;
    }

    const rawLengthMm = Number(params?.['bar_length_mm'] ?? 0) || 0;
    const barLengthMm = rawLengthMm > 0 ? rawLengthMm : this.getAutoScaleBarMm(rawPixelsPerMm, imgW);
    const barLengthPx = Math.max(1, Math.round(barLengthMm * rawPixelsPerMm));
    const fontSize = Math.max(8, Number(params?.['font_size'] ?? 24) || 24);
    const fontThickness = Math.max(1, Number(params?.['font_thickness'] ?? 1) || 1);
    const barThickness = Math.max(1, Number(params?.['bar_thickness'] ?? 3) || 3);
    const padding = Math.max(0, Number(params?.['box_padding'] ?? 14) || 14);
    const textGap = Math.max(0, Number(params?.['text_gap'] ?? 16) || 16);
    const backgroundOpacity = Math.max(0, Math.min(1, Number(params?.['background_opacity'] ?? 0.55) || 0.55));
    const fontFamily = this.mapFontFamily(String(params?.['font_family'] ?? 'sans'));
    const labelUnit = String(params?.['label_unit'] ?? this.scaleBarUnit ?? 'mm');
    const labelValue = this.fromMm(barLengthMm, labelUnit);
    const label = `${Math.round(labelValue)} ${labelUnit}`;
    const estimatedTextWidth = Math.max(1, Math.round(label.length * fontSize * 0.62));
    const boxWidth = Math.max(barLengthPx, estimatedTextWidth) + 2 * padding;
    const boxHeight = padding + fontSize + textGap + Math.max(barThickness * 2, 12) + padding;

    let boxX = Number(params?.['position_x'] ?? -1);
    let boxY = Number(params?.['position_y'] ?? -1);
    if (!Number.isFinite(boxX) || boxX < 0 || !Number.isFinite(boxY) || boxY < 0) {
      boxX = Math.max(0, imgW - boxWidth - 20);
      boxY = Math.max(0, imgH - boxHeight - 20);
    } else {
      boxX = Math.max(0, Math.min(imgW - boxWidth, boxX));
      boxY = Math.max(0, Math.min(imgH - boxHeight, boxY));
    }

    const overlay: ScaleBarOverlayState = {
      x: boxX,
      y: boxY,
      width: boxWidth,
      height: boxHeight,
      barStartX: boxX + Math.round((boxWidth - barLengthPx) / 2),
      barEndX: boxX + Math.round((boxWidth - barLengthPx) / 2) + barLengthPx,
      barY: boxY + padding + barThickness,
      labelX: boxX + boxWidth / 2,
      labelY: boxY + padding + barThickness + textGap + fontSize,
      label,
      fontFamily,
      fontSize,
      barThickness,
      fontThickness,
      padding,
      textGap,
      backgroundOpacity,
      backgroundColor: this.mapSimpleColor(String(params?.['background_color'] ?? 'black')),
      textColor: this.mapSimpleColor(String(params?.['text_color'] ?? 'white')),
      barColor: this.mapSimpleColor(String(params?.['bar_color'] ?? 'white')),
    };

    if (this.scaleBarDragging && this.scaleBarOverlay) {
      overlay.x = this.scaleBarOverlay.x;
      overlay.y = this.scaleBarOverlay.y;
    }

    this.scaleBarOverlay = overlay;
  }

  private getAutoScaleBarMm(pxPerMm: number, imgW: number): number {
    if (pxPerMm <= 0) return 0;
    const targetPx = imgW / 5;
    const targetMm = targetPx / pxPerMm;
    const niceValues = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000];
    let best = niceValues[0];
    let bestDiff = Math.abs(targetMm - best);
    for (const value of niceValues) {
      const diff = Math.abs(targetMm - value);
      if (diff < bestDiff) {
        bestDiff = diff;
        best = value;
      }
    }
    return best;
  }

  private mapFontFamily(fontFamily: string): string {
    switch (fontFamily) {
      case 'serif': return 'serif';
      case 'mono': return 'monospace';
      case 'complex': return 'fantasy';
      case 'script': return 'cursive';
      default: return 'sans-serif';
    }
  }

  private mapSimpleColor(colorName: string): string {
    switch (colorName.trim().toLowerCase()) {
      case 'black': return '#000000';
      case 'yellow': return '#ffd166';
      default: return '#ffffff';
    }
  }

  onScaleBarMouseDown(event: MouseEvent): void {
    if (!this.scaleBarOverlay || event.button !== 0) return;
    event.preventDefault();
    event.stopPropagation();
    const pos = this.scaleBarSvgCoords(event);
    if (!pos) return;
    this.scaleBarDragging = true;
    this.scaleBarDragOffset = {
      x: pos.x - this.scaleBarOverlay.x,
      y: pos.y - this.scaleBarOverlay.y,
    };
  }

  onScaleBarMouseMove(event: MouseEvent): void {
    if (!this.scaleBarDragging || !this.scaleBarOverlay) return;
    event.preventDefault();
    const pos = this.scaleBarSvgCoords(event);
    if (!pos) return;
    const nextX = Math.max(0, Math.min(this.rulerImgW - this.scaleBarOverlay.width, pos.x - this.scaleBarDragOffset.x));
    const nextY = Math.max(0, Math.min(this.rulerImgH - this.scaleBarOverlay.height, pos.y - this.scaleBarDragOffset.y));
    const barLengthPx = this.scaleBarOverlay.barEndX - this.scaleBarOverlay.barStartX;
    if (this.scaleBarOverlay) {
      this.scaleBarOverlay = {
        ...this.scaleBarOverlay,
        x: nextX,
        y: nextY,
        barStartX: nextX + this.scaleBarOverlay.padding,
        barEndX: nextX + this.scaleBarOverlay.padding + barLengthPx,
        barY: nextY + this.scaleBarOverlay.padding + this.scaleBarOverlay.barThickness,
        labelX: nextX + this.scaleBarOverlay.width / 2,
        labelY: nextY + this.scaleBarOverlay.padding + this.scaleBarOverlay.barThickness + this.scaleBarOverlay.textGap + this.scaleBarOverlay.fontSize,
      };
    }
    this.cdr.markForCheck();
  }

  onScaleBarMouseUp(): void {
    if (!this.scaleBarDragging || !this.scaleBarOverlay || this.scaleBarSelectedStepIndex < 0) {
      if (this.scaleBarDragging && this.scaleBarOverlay) {
        this.scaleBarPositionX = Math.round(this.scaleBarOverlay.x);
        this.scaleBarPositionY = Math.round(this.scaleBarOverlay.y);
        this.refreshScaleBarOverlay();
      }
      this.scaleBarDragging = false;
      return;
    }

    this.scaleBarDragging = false;
    const pipeline = this.pipelineState.getPipeline();
    const step = pipeline.steps[this.scaleBarSelectedStepIndex];
    if (!step) return;

    const updated = {
      ...step.param_values,
      position_x: Math.round(this.scaleBarOverlay.x),
      position_y: Math.round(this.scaleBarOverlay.y),
    };
    this.scaleBarSelectedParams = updated;
    this.pipelineState.updateParams(this.scaleBarSelectedStepIndex, updated);
    this.scaleBarPositionX = Math.round(this.scaleBarOverlay.x);
    this.scaleBarPositionY = Math.round(this.scaleBarOverlay.y);
  }

  private scaleBarSvgCoords(event: MouseEvent): { x: number; y: number } | null {
    const img = this.previewImg?.nativeElement;
    if (!img) return null;
    const rect = img.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) return null;
    const sx = this.rulerImgW / rect.width;
    const sy = this.rulerImgH / rect.height;
    return {
      x: Math.max(0, Math.min(this.rulerImgW, (event.clientX - rect.left) * sx)),
      y: Math.max(0, Math.min(this.rulerImgH, (event.clientY - rect.top) * sy)),
    };
  }

  private drawScaleBarOverlayToCanvas(ctx: CanvasRenderingContext2D, overlay: ScaleBarOverlayState, scale: number): void {
    const bgColor = overlay.backgroundColor === '#000000' ? '0,0,0' : '255,255,255';
    ctx.fillStyle = `rgba(${bgColor},${overlay.backgroundOpacity})`;
    ctx.fillRect(overlay.x, overlay.y, overlay.width, overlay.height);

    ctx.beginPath();
    ctx.moveTo(overlay.barStartX, overlay.barY);
    ctx.lineTo(overlay.barEndX, overlay.barY);
    ctx.strokeStyle = overlay.barColor;
    ctx.lineWidth = overlay.barThickness * scale;
    ctx.stroke();

    for (const capX of [overlay.barStartX, overlay.barEndX]) {
      ctx.beginPath();
      ctx.moveTo(capX, overlay.barY - 10 * scale);
      ctx.lineTo(capX, overlay.barY + 10 * scale);
      ctx.strokeStyle = overlay.barColor;
      ctx.lineWidth = Math.max(1, overlay.barThickness - 1) * scale;
      ctx.stroke();
    }

    ctx.font = `${overlay.fontSize * scale}px ${overlay.fontFamily}`;
    ctx.textAlign = 'center';
    ctx.fillStyle = overlay.textColor;
    ctx.strokeStyle = overlay.backgroundColor;
    ctx.lineWidth = (overlay.fontThickness + 2) * scale;
    ctx.strokeText(overlay.label, overlay.labelX, overlay.labelY);
    ctx.fillText(overlay.label, overlay.labelX, overlay.labelY);
    ctx.textAlign = 'start';
  }

  // === Montage feature ===

  async generateAndShowMontage(): Promise<void> {
    if (this.generatingMontage || this.imageCount < 2 || !this.currentPipeline || this.selectedStepIndex < 0) {
      console.warn('Montage generation skipped:', {
        generatingMontage: this.generatingMontage,
        imageCount: this.imageCount,
        hasPipeline: !!this.currentPipeline,
        selectedStepIndex: this.selectedStepIndex
      });
      return;
    }

    // Check cache first
    const cacheKey = `${this.selectedStepIndex}:${JSON.stringify(this.currentPipeline)}`;
    const cached = this.montageCache.get(cacheKey);
    if (cached) {
      this.montagePreview = cached.preview;
      this.montageImageCount = cached.imageCount;
      this.montageGridRows = cached.gridRows;
      this.montageGridCols = cached.gridCols;
      this.montageCellWidth = cached.cellWidth;
      this.montageCellHeight = cached.cellHeight;
      this.montageZoomLevel = 1.0;
      this.showingMontage = true;
      this.cdr.markForCheck();
      return;
    }

    this.generatingMontage = true;
    console.log('Starting montage generation for step:', this.selectedStepIndex);

    try {
      const response = await this.recipeService.getStepImagesMontage(this.currentPipeline, this.selectedStepIndex).toPromise();
      console.log('Montage response:', response);
      
      if (response?.montage_base64) {
        this.montagePreview = `data:image/jpeg;base64,${response.montage_base64}`;
        this.montageImageCount = response.image_count || 0;
        this.montageGridRows = response.grid_rows || 0;
        this.montageGridCols = response.grid_cols || 0;
        this.montageCellWidth = response.cell_width || 0;
        this.montageCellHeight = response.cell_height || 0;
        this.montageZoomLevel = 1.0;
        this.showingMontage = true;

        // Store in cache
        this.montageCache.set(cacheKey, {
          preview: this.montagePreview,
          imageCount: this.montageImageCount,
          gridRows: this.montageGridRows,
          gridCols: this.montageGridCols,
          cellWidth: this.montageCellWidth,
          cellHeight: this.montageCellHeight,
        });

        console.log('Montage displayed successfully', {
          imageCount: this.montageImageCount,
          rows: this.montageGridRows,
          cols: this.montageGridCols,
          cellWidth: this.montageCellWidth,
          cellHeight: this.montageCellHeight
        });
        this.cdr.markForCheck();
      } else {
        console.error('No montage_base64 in response:', response);
      }
    } catch (error) {
      console.error('Failed to generate montage:', error);
      this.showingMontage = false;
    } finally {
      this.generatingMontage = false;
      this.cdr.markForCheck();
    }
  }

  closeMontage(): void {
    this.showingMontage = false;
    // Keep montagePreview in memory for cache - don't null it
    this.montageZoomLevel = 1.0;
  }

  onMontageImageLoaded(): void {
    console.log('Montage image loaded');
    // Initialize the zoom transform after the image is loaded
    setTimeout(() => {
      this.applyMontageTransform();
    }, 0);
  }

  onMontageImageClick(event: MouseEvent): void {
    if (!this.montagePreview || this.montageGridRows === 0 || this.montageGridCols === 0) {
      return;
    }

    const montageImg = (event.target as HTMLImageElement);
    const rect = montageImg.getBoundingClientRect();
    
    // Click position in browser viewport
    const clickX = event.clientX - rect.left;
    const clickY = event.clientY - rect.top;
    
    // Scale to original image coordinates
    const scaleX = montageImg.naturalWidth / rect.width;
    const scaleY = montageImg.naturalHeight / rect.height;
    const imgX = clickX * scaleX;
    const imgY = clickY * scaleY;

    // Calculate grid position with 2px padding/spacing
    const padding = 2;
    const localX = imgX - padding;
    const localY = imgY - padding;

    if (localX < 0 || localY < 0) {
      return;
    }

    // Each cell: width + 2px spacing
    const cellStride = this.montageCellWidth + 2;
    const rowStride = this.montageCellHeight + 30 + 2; // 30 = label height

    const col = Math.floor(localX / cellStride);
    const row = Math.floor(localY / rowStride);

    if (col >= this.montageGridCols || row >= this.montageGridRows) {
      return;
    }

    const imageIndex = row * this.montageGridCols + col;
    if (imageIndex < this.montageImageCount) {
      this.pipelineState.setPreviewImageIndex(imageIndex);
      this.closeMontage();
      this.cdr.markForCheck();
    }
  }

  // === Unified tool event handlers ===

  onToolClick(event: MouseEvent): void {
    this.showRulerContextMenu = false;
    if (this.rulerActive) {
      this.handleRulerClick(event);
    } else if (this.scaleActive) {
      this.handleScaleClick(event);
    } else if (this.pixelActive) {
      this.handlePixelClick(event);
    }
  }

  onToolMouseMove(event: MouseEvent): void {
    if (this.rulerActive) {
      this.handleRulerMouseMove(event);
    } else if (this.scaleActive) {
      this.handleScaleMouseMove(event);
    } else if (this.pixelActive) {
      this.handlePixelMouseMove(event);
    }
  }

  onToolMouseLeave(): void {
    if (this.rulerActive && this.rulerDrawingStart) {
      this.rulerDrawingStart = null;
      this.rulerDrawingCurrent = null;
      this.rulerDrawingDistance = 0;
    }
    if (this.scaleActive && this.scaleStart && !this.scaleEnd) {
      this.scaleStart = null;
      this.scaleCurrentPos = null;
    }
    if (this.pixelActive && !this.pixelFrozenPos) {
      this.pixelCurrentPos = null;
    }
  }

  onToolMouseDown(event: MouseEvent): void {
    if (this.rulerActive || this.scaleActive) {
      event.preventDefault();
      event.stopPropagation();
    }
  }

  // === Ruler click/move handlers ===

  private handleRulerClick(event: MouseEvent): void {
    event.preventDefault();
    event.stopPropagation();

    const pos = this.toolSvgCoords(event);
    if (!pos) return;

    if (!this.rulerDrawingStart) {
      if (this.rulerLines.length >= this.RULER_MAX_LINES) return;
      this.rulerDrawingStart = pos;
      this.rulerDrawingCurrent = pos;
      this.rulerDrawingDistance = 0;
    } else {
      const dist = this.calcDistance(this.rulerDrawingStart, pos);
      this.rulerLines = [...this.rulerLines, { start: this.rulerDrawingStart, end: pos, distance: dist }];
      this.rulerDrawingStart = null;
      this.rulerDrawingCurrent = null;
      this.rulerDrawingDistance = 0;
    }
  }

  private handleRulerMouseMove(event: MouseEvent): void {
    if (!this.rulerDrawingStart) return;

    const pos = this.toolSvgCoords(event);
    if (!pos) return;

    this.rulerDrawingCurrent = pos;
    this.rulerDrawingDistance = this.calcDistance(this.rulerDrawingStart, pos);
  }

  // === Ruler line context menu ===

  onRulerLineContextMenu(event: MouseEvent, index: number): void {
    event.preventDefault();
    event.stopPropagation();
    this.rulerContextLineIndex = index;
    const container = this.imageRoiContainer?.nativeElement;
    if (container) {
      const containerRect = container.getBoundingClientRect();
      this.rulerContextMenuScreenX = event.clientX - containerRect.left;
      this.rulerContextMenuScreenY = event.clientY - containerRect.top;
    }
    this.showRulerContextMenu = true;
  }

  deleteRulerLineFromContext(): void {
    this.showRulerContextMenu = false;
    if (this.rulerContextLineIndex >= 0 && this.rulerContextLineIndex < this.rulerLines.length) {
      this.rulerLines = this.rulerLines.filter((_, i) => i !== this.rulerContextLineIndex);
    }
    this.rulerContextLineIndex = -1;
  }

  // === Scale click/move handlers ===

  private handleScaleClick(event: MouseEvent): void {
    event.preventDefault();
    event.stopPropagation();

    const pos = this.toolSvgCoords(event);
    if (!pos) return;

    if (this.scaleEnd) {
      // Restart scale line
      this.scaleStart = pos;
      this.scaleEnd = null;
      this.scaleCurrentPos = pos;
      this.scaleLinePx = 0;
    } else if (!this.scaleStart) {
      this.scaleStart = pos;
      this.scaleCurrentPos = pos;
    } else {
      this.scaleEnd = pos;
      this.scaleCurrentPos = null;
      this.scaleLinePx = this.calcDistance(this.scaleStart, this.scaleEnd);
      this.computeScaleBar();
    }
  }

  private handleScaleMouseMove(event: MouseEvent): void {
    if (!this.scaleStart || this.scaleEnd) return;

    const pos = this.toolSvgCoords(event);
    if (!pos) return;

    this.scaleCurrentPos = pos;
  }

  // === Save image with annotations ===

  async saveAnnotatedImage(): Promise<void> {
    if (this.showGraphViewer && this.graphCanvasRef?.nativeElement) {
      await this.saveCanvasAsPng(this.graphCanvasRef.nativeElement, 'pipeline_preview_graph.png');
      return;
    }
    if (this.showingMontage && this.montagePreview) {
      await this.saveImageSrcAsPng(this.montagePreview, 'pipeline_preview_montage.png');
      return;
    }
    if (this.referenceCropStripActive && this.referenceCropImages.length > 0) {
      const canvas = await this.renderReferenceSequencePreview();
      if (canvas) await this.saveCanvasAsPng(canvas, 'pipeline_preview_reference_sequence.png');
      return;
    }

    if (!this.imageSrc) return;

    const img = this.previewImg?.nativeElement;
    if (!img) return;

    const w = img.naturalWidth;
    const h = img.naturalHeight;
    const canvas = document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.drawImage(img, 0, 0);

    const s = this.rulerScale;

    // Draw completed ruler lines
    for (const line of this.rulerLines) {
      ctx.beginPath();
      ctx.moveTo(line.start.x, line.start.y);
      ctx.lineTo(line.end.x, line.end.y);
      ctx.strokeStyle = '#1a5fb4';
      ctx.lineWidth = 2 * s;
      ctx.stroke();

      for (const pt of [line.start, line.end]) {
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, 6 * s, 0, Math.PI * 2);
        ctx.fillStyle = '#1a5fb4';
        ctx.fill();
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 1.5 * s;
        ctx.stroke();
      }

      const mx = (line.start.x + line.end.x) / 2 + 12 * s;
      const my = (line.start.y + line.end.y) / 2 - 10 * s;
      const text = line.distance.toFixed(1) + ' px';
      ctx.font = `${13 * s}px monospace`;
      ctx.lineJoin = 'round';
      ctx.strokeStyle = '#000';
      ctx.lineWidth = 3 * s;
      ctx.strokeText(text, mx, my);
      ctx.fillStyle = '#fff';
      ctx.fillText(text, mx, my);
    }

    // Draw scale line (only if not fully calibrated)
    if (this.scaleStart && this.scaleEnd) {
      ctx.beginPath();
      ctx.moveTo(this.scaleStart.x, this.scaleStart.y);
      ctx.lineTo(this.scaleEnd.x, this.scaleEnd.y);
      ctx.strokeStyle = '#e67e22';
      ctx.lineWidth = 2 * s;
      ctx.stroke();

      for (const pt of [this.scaleStart, this.scaleEnd]) {
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, 6 * s, 0, Math.PI * 2);
        ctx.fillStyle = '#e67e22';
        ctx.fill();
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 1.5 * s;
        ctx.stroke();
      }
    }

    // Draw scale bar
    if (this.scaleBarOverlayActive && this.scaleBarOverlay) {
      this.drawScaleBarOverlayToCanvas(ctx, this.scaleBarOverlay, s);
    } else if (this.showScaleBar && this.pxPerMm > 0 && this.scaleBarPx > 0) {
      const barX = w - this.scaleBarPx - 20 * s;
      const barEndX = w - 20 * s;
      const barY = h - 25 * s;

      // Background
      ctx.fillStyle = 'rgba(0,0,0,0.55)';
      ctx.fillRect(barX - 15 * s, h - 58 * s, this.scaleBarPx + 30 * s, 44 * s);

      // Bar line
      ctx.beginPath();
      ctx.moveTo(barX, barY);
      ctx.lineTo(barEndX, barY);
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 3 * s;
      ctx.stroke();

      // End caps
      for (const capX of [barX, barEndX]) {
        ctx.beginPath();
        ctx.moveTo(capX, barY - 9 * s);
        ctx.lineTo(capX, barY + 9 * s);
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 2 * s;
        ctx.stroke();
      }

      // Label
      const label = this.formatScaleBarLabel();
      ctx.font = `${13 * s}px sans-serif`;
      ctx.fillStyle = '#fff';
      ctx.textAlign = 'center';
      ctx.fillText(label, barX + this.scaleBarPx / 2, h - 42 * s);
      ctx.textAlign = 'start';
    }

    await this.saveCanvasAsPng(canvas, 'pipeline_preview.png');
  }

  private async saveImageSrcAsPng(src: string, filename: string): Promise<void> {
    const img = await this.loadPreviewImage(src);
    const canvas = document.createElement('canvas');
    canvas.width = img.naturalWidth || img.width;
    canvas.height = img.naturalHeight || img.height;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.drawImage(img, 0, 0);
    await this.saveCanvasAsPng(canvas, filename);
  }

  private async renderReferenceSequencePreview(): Promise<HTMLCanvasElement | null> {
    const imgs = await Promise.all(this.referenceCropImages.map((src) => this.loadPreviewImage(src)));
    if (!imgs.length) return null;

    const tileW = 180;
    const tileH = 170;
    const gap = 12;
    const pad = 18;
    const histW = this.referenceSequenceComponents.length ? 320 : 0;
    const cols = Math.min(4, Math.max(1, imgs.length));
    const rows = Math.ceil(imgs.length / cols);
    const gridW = cols * tileW + (cols - 1) * gap;
    const gridH = rows * tileH + (rows - 1) * gap;
    const canvas = document.createElement('canvas');
    canvas.width = pad * 2 + gridW + (histW ? gap + histW : 0);
    canvas.height = pad * 2 + Math.max(gridH, 240);
    const ctx = canvas.getContext('2d');
    if (!ctx) return null;

    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.font = '13px sans-serif';
    ctx.textBaseline = 'middle';

    imgs.forEach((img, index) => {
      const col = index % cols;
      const row = Math.floor(index / cols);
      const x = pad + col * (tileW + gap);
      const y = pad + row * (tileH + gap);
      this.drawReferenceTile(ctx, img, x, y, tileW, tileH, this.getReferenceCropLabel(index), this.getReferenceCropScoreLabel(index));
    });

    if (histW) {
      this.drawReferenceHistogram(ctx, pad + gridW + gap, pad, histW, Math.max(gridH, 240) - pad);
    }
    return canvas;
  }

  private drawReferenceTile(ctx: CanvasRenderingContext2D, img: HTMLImageElement, x: number, y: number, w: number, h: number, label: string, score: string): void {
    const footerH = score ? 24 : 0;
    ctx.fillStyle = '#111827';
    ctx.fillRect(x, y, w, h);
    const maxImgH = h - footerH;
    const scale = Math.min(w / img.width, maxImgH / img.height);
    const iw = img.width * scale;
    const ih = img.height * scale;
    ctx.drawImage(img, x + (w - iw) / 2, y + (maxImgH - ih) / 2, iw, ih);
    ctx.fillStyle = 'rgba(15,23,42,0.82)';
    ctx.fillRect(x + 6, y + 6, Math.min(w - 12, Math.max(34, label.length * 7 + 14)), 22);
    ctx.fillStyle = '#f8fafc';
    ctx.fillText(label, x + 13, y + 17);
    if (score) {
      ctx.fillStyle = 'rgba(2,6,23,0.9)';
      ctx.fillRect(x, y + h - footerH, w, footerH);
      ctx.fillStyle = '#f8fafc';
      ctx.textAlign = 'center';
      ctx.fillText(score, x + w / 2, y + h - footerH / 2);
      ctx.textAlign = 'start';
    }
  }

  private drawReferenceHistogram(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number): void {
    ctx.fillStyle = '#111827';
    ctx.fillRect(x, y, w, h);
    let yy = y + 18;
    for (const component of this.referenceSequenceComponents) {
      ctx.fillStyle = '#f8fafc';
      ctx.font = '700 13px sans-serif';
      ctx.fillText(component, x + 12, yy);
      ctx.font = '11px sans-serif';
      yy += 26;
      for (let index = 0; index < this.getReferenceComponentScores(component).length; index++) {
        if (yy > y + h - 10) return;
        const score = this.getReferenceComponentScores(component)[index];
        const barX = x + 62;
        const barW = w - 150;
        ctx.fillStyle = '#d1d5db';
        ctx.fillText(this.getReferenceCropLabel(index), x + 12, yy);
        ctx.fillStyle = 'rgba(255,255,255,0.12)';
        ctx.fillRect(barX, yy - 5, barW, 9);
        ctx.fillStyle = this.getReferenceSequenceColor(component);
        ctx.fillRect(barX, yy - 5, barW * this.getReferenceComponentBarWidth(component, score) / 100, 9);
        ctx.fillStyle = '#f8fafc';
        ctx.textAlign = 'right';
        ctx.fillText(`${this.getReferenceComponentScoreLabel(component, index)} ${this.getReferenceComponentDiffLabel(component, index)}`, x + w - 12, yy);
        ctx.textAlign = 'start';
        yy += 22;
      }
      yy += 10;
    }
  }

  private loadPreviewImage(src: string): Promise<HTMLImageElement> {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = reject;
      img.src = src;
    });
  }

  private async saveCanvasAsPng(canvas: HTMLCanvasElement, filename: string): Promise<void> {
    const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, 'image/png'));
    if (!blob) return;

    if ('showSaveFilePicker' in window) {
      try {
        const handle = await (window as any).showSaveFilePicker({
          suggestedName: filename,
          types: [{
            description: 'PNG Image',
            accept: { 'image/png': ['.png'] },
          }],
        });
        const writable = await handle.createWritable();
        await writable.write(blob);
        await writable.close();
        return;
      } catch (e: any) {
        if (e?.name === 'AbortError') return;
      }
    }

    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(a.href);
  }

  // === Shared utility ===

  private toolSvgCoords(event: MouseEvent): { x: number; y: number } | null {
    const svg = (event.currentTarget ?? event.target) as SVGSVGElement;
    if (!svg) return null;
    const rect = svg.getBoundingClientRect();
    const sx = this.rulerImgW / rect.width;
    const sy = this.rulerImgH / rect.height;
    return {
      x: Math.max(0, Math.min(this.rulerImgW, (event.clientX - rect.left) * sx)),
      y: Math.max(0, Math.min(this.rulerImgH, (event.clientY - rect.top) * sy)),
    };
  }

  private calcDistance(
    a: { x: number; y: number },
    b: { x: number; y: number }
  ): number {
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    return Math.sqrt(dx * dx + dy * dy);
  }

  // === Pixel measurement tool ===

  togglePixelTool(): void {
    this.pixelActive = !this.pixelActive;
    if (this.pixelActive) {
      this.deactivateMeasurementTools('pixel');
    }
  }

  private deactivateMeasurementTools(activeTool: 'ruler' | 'scale' | 'pixel' | null = null): void {
    this.rulerActive = activeTool === 'ruler';
    this.scaleActive = activeTool === 'scale';
    this.pixelActive = activeTool === 'pixel';

    this.rulerDrawingStart = null;
    this.rulerDrawingCurrent = null;
    this.rulerDrawingDistance = 0;
    this.scaleStart = null;
    this.scaleEnd = null;
    this.scaleCurrentPos = null;
    this.scaleLinePx = 0;
    this.pixelCurrentPos = null;
    this.pixelFrozenPos = null;
    this.pixelCanvasCache = null;
    this.pixelImageDataCache = null;
  }

  private handlePixelClick(event: MouseEvent): void {
    event.preventDefault();
    event.stopPropagation();

    const pos = this.toolSvgCoords(event);
    if (!pos) return;

    if (this.pixelFrozenPos) {
      this.pixelFrozenPos = null;
    } else {
      this.pixelFrozenPos = pos;
      this.updatePixelGrid(pos);
    }
  }

  private handlePixelMouseMove(event: MouseEvent): void {
    if (this.pixelFrozenPos) return;

    const pos = this.toolSvgCoords(event);
    if (!pos) return;

    this.pixelCurrentPos = pos;
    this.updatePixelGrid(pos);
  }

  private updatePixelGrid(centerPos: {x: number; y: number}): void {
    const img = this.previewImg?.nativeElement;
    if (!img || !img.complete) return;

    const x = Math.round(centerPos.x);
    const y = Math.round(centerPos.y);

    const imageData = this.getImageData(img);
    if (!imageData) return;

    const width = imageData.width;
    const height = imageData.height;
    const data = imageData.data;
    const channels = data.length / (width * height);

    this.pixelGridValues = [];
    this.pixelGridColors = [];

    for (let row = -1; row <= 1; row++) {
      for (let col = -1; col <= 1; col++) {
        const px = x + col;
        const py = y + row;

        if (px < 0 || px >= width || py < 0 || py >= height) {
          this.pixelGridValues.push('---');
          this.pixelGridColors.push('#333');
          continue;
        }

        const idx = (py * width + px) * channels;
        const values: number[] = [];
        for (let c = 0; c < Math.min(channels, 4); c++) {
          values.push(data[idx + c] || 0);
        }

          const displayValue = this.formatPixelValue(values, channels);
        const hexColor = this.rgbToHex(values[0] || 0, values[1] || 0, values[2] || 0);

        this.pixelGridValues.push(displayValue);
        this.pixelGridColors.push(hexColor);
      }
    }

    this.detectColorSpace();
    this.cdr.markForCheck();
  }

  private getImageData(img: HTMLImageElement): ImageData | null {
    try {
      if (!img.complete || !img.naturalWidth || !img.naturalHeight) {
        return null;
      }

      if (this.pixelImageDataCache) {
        return this.pixelImageDataCache;
      }

      if (!this.pixelCanvasCache) {
        this.pixelCanvasCache = document.createElement('canvas');
      }

      const canvas = this.pixelCanvasCache;
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;

      const ctx = canvas.getContext('2d', { willReadFrequently: true });
      if (!ctx) return null;

      ctx.drawImage(img, 0, 0);
      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      this.pixelImageDataCache = imageData;
      return imageData;
    } catch (e) {
      console.error('Failed to get image data:', e);
      return null;
    }
  }

  private formatPixelValue(values: number[], channels: number): string {
    if (this.pixelOutputType === 'GRAYSCALE' || this.pixelOutputType === 'MASK') {
      return values[0].toString();
    }

    if (channels >= 3) {
      return `${values[0]},${values[1]},${values[2]}`;
    }

    return values.join(',');
  }

  private detectColorSpace(): void {
    if (this.pixelOutputType === 'GRAYSCALE') {
      this.pixelColorSpace = 'Grayscale';
      return;
    }

    if (this.pixelOutputType === 'MASK') {
      this.pixelColorSpace = 'Mask';
      return;
    }

    this.pixelColorSpace = 'RGB';
  }

  private rgbToHex(r: number, g: number, b: number): string {
    return '#' + [r, g, b].map(x => {
      const hex = x.toString(16);
      return hex.length === 1 ? '0' + hex : hex;
    }).join('');
  }

  isColorBright(hexColor: string): boolean {
    const r = parseInt(hexColor.substring(1, 3), 16);
    const g = parseInt(hexColor.substring(3, 5), 16);
    const b = parseInt(hexColor.substring(5, 7), 16);
    const brightness = (r * 299 + g * 587 + b * 114) / 1000;
    return brightness > 128;
  }

  getPixelDisplayLines(val: string): string[] {
    if (!val || val === '---') {
      return ['-'];
    }

    return val.split(',').map((part) => part.trim());
  }

  getPixelTextStartY(centerY: number, lineCount: number): number {
    return centerY - ((lineCount - 1) * this.pixelGridLineHeight) / 2 + this.pixelGridFontSize * 0.35;
  }

  getPixelDisplayValue(val: string): string {
    if (val === '---') return '~';
    const parts = val.split(',');
    return parts.map(p => {
      const n = parseInt(p, 10);
      if (isNaN(n)) return '~';
      return Math.round(n / 51).toString();
    }).join(',');
  }

  copyPixelValues(): void {
    if (this.pixelGridValues.length === 0) return;
    const values = this.pixelGridValues.join(' | ');
    navigator.clipboard.writeText(values).catch(() => undefined);
  }
}
