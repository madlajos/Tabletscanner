import { Component, OnInit, OnDestroy, ViewChild, ElementRef, AfterViewInit, ChangeDetectorRef } from '@angular/core';
import { CommonModule, DecimalPipe } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Subscription, combineLatest } from 'rxjs';
import { DataType } from '../../models/pipeline.models';
import { PipelineStateService } from '../../services/pipeline-state.service';
import { RecipeService } from '../../services/recipe.service';

@Component({
  selector: 'app-pipeline-preview',
  standalone: true,
  imports: [CommonModule, FormsModule, DecimalPipe],
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
                    (click)="toggleScale()" title="Sk\u00e1la eszk\u00f6z">
              \uD83D\uDCD0
            </button>
            <button class="tool-btn icon-btn icon-tool-btn" (click)="saveAnnotatedImage()" title="K\u00e9p ment\u00e9se"
                    [disabled]="!imageSrc">
              \uD83D\uDCBE
            </button>
            <button class="tool-btn icon-tool-btn" [class.active]="pixelActive"
                    (click)="togglePixelTool()" title="Pixel m\u00e9r\u00e9s">
              \uD83D\uDD0D
            </button>
          </div>
          @if (pixelActive) {
            <span class="pixel-color-space">{{ pixelColorSpace }}</span>
            <div class="pixel-values-display">
              @for (val of pixelGridValues; track $index) {
                <span class="pixel-value-item">{{ val }}</span>
              }
            </div>
            <button class="tool-btn icon-btn" (click)="copyPixelValues()"
                    title="Értékek másolása" [disabled]="!pixelCurrentPos && !pixelFrozenPos">
              \u2398
            </button>
          }
          @if (rulerActive) {
            @for (i of rulerSlots; track i) {
              <input type="text" class="ruler-measurement-box" readonly
                     [value]="getRulerMeasurement(i)"
                     [class.used]="i < rulerLines.length || (i === rulerLines.length && rulerDrawingStart)">
            }
            <button class="tool-btn icon-btn" (click)="copyRulerMeasurements()"
                    title="M\u00e9r\u00e9sek m\u00e1sol\u00e1sa" [disabled]="rulerLines.length === 0">
              \u2398
            </button>
            <button class="tool-btn icon-btn" (click)="clearAllRulerLines()"
                    title="Vonalak t\u00f6rl\u00e9se" [disabled]="rulerLines.length === 0">
              \u2715
            </button>
          }
          @if (scaleActive) {
            @if (pxPerMm > 0) {
              <span class="scale-ratio-display">{{ pxPerMm | number:'1.2-2' }} px/mm</span>
            }
            <span class="scale-label">Val\u00f3s t\u00e1vols\u00e1g:</span>
            <input type="number" class="scale-mm-input" [(ngModel)]="scaleMm"
                   placeholder="0" min="0" step="0.1"
                   (ngModelChange)="onScaleMmChange()">
            <span class="scale-unit">mm</span>
            <label class="scale-checkbox-label">
              <input type="checkbox" [(ngModel)]="showScaleBar"
                     (ngModelChange)="onShowScaleBarChange()">
              <span>Sk\u00e1la mutat\u00e1sa</span>
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
                   [style.pointer-events]="(rulerActive || scaleActive || pixelActive) ? 'auto' : 'none'"
                   [style.cursor]="(rulerActive || scaleActive || pixelActive) ? 'crosshair' : 'default'"
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
                @if (scaleStart && !(scaleEnd && scaleMm > 0)) {
                  <circle [attr.cx]="scaleStart.x" [attr.cy]="scaleStart.y" [attr.r]="6 * rulerScale"
                          fill="#e67e22" stroke="#fff" [attr.stroke-width]="1.5 * rulerScale"/>
                  @if (scaleEnd) {
                    <line [attr.x1]="scaleStart.x" [attr.y1]="scaleStart.y"
                          [attr.x2]="scaleEnd.x" [attr.y2]="scaleEnd.y"
                          stroke="#e67e22" [attr.stroke-width]="2 * rulerScale"/>
                    <circle [attr.cx]="scaleEnd.x" [attr.cy]="scaleEnd.y" [attr.r]="6 * rulerScale"
                            fill="#e67e22" stroke="#fff" [attr.stroke-width]="1.5 * rulerScale"/>
                  } @else if (scaleCurrentPos) {
                    <line [attr.x1]="scaleStart.x" [attr.y1]="scaleStart.y"
                          [attr.x2]="scaleCurrentPos.x" [attr.y2]="scaleCurrentPos.y"
                          stroke="#e67e22" [attr.stroke-width]="2 * rulerScale"
                          [attr.stroke-dasharray]="(8 * rulerScale) + ' ' + (5 * rulerScale)"/>
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
                @if (showScaleBar && pxPerMm > 0 && scaleBarPx > 0) {
                  <rect [attr.x]="rulerImgW - scaleBarPx - 35 * rulerScale"
                        [attr.y]="rulerImgH - 58 * rulerScale"
                        [attr.width]="scaleBarPx + 30 * rulerScale"
                        [attr.height]="44 * rulerScale"
                        fill="rgba(0,0,0,0.55)" [attr.rx]="4 * rulerScale"/>
                  <line [attr.x1]="rulerImgW - scaleBarPx - 20 * rulerScale"
                        [attr.y1]="rulerImgH - 25 * rulerScale"
                        [attr.x2]="rulerImgW - 20 * rulerScale"
                        [attr.y2]="rulerImgH - 25 * rulerScale"
                        stroke="#fff" [attr.stroke-width]="3 * rulerScale"/>
                  <line [attr.x1]="rulerImgW - scaleBarPx - 20 * rulerScale"
                        [attr.y1]="rulerImgH - 34 * rulerScale"
                        [attr.x2]="rulerImgW - scaleBarPx - 20 * rulerScale"
                        [attr.y2]="rulerImgH - 16 * rulerScale"
                        stroke="#fff" [attr.stroke-width]="2 * rulerScale"/>
                  <line [attr.x1]="rulerImgW - 20 * rulerScale"
                        [attr.y1]="rulerImgH - 34 * rulerScale"
                        [attr.x2]="rulerImgW - 20 * rulerScale"
                        [attr.y2]="rulerImgH - 16 * rulerScale"
                        stroke="#fff" [attr.stroke-width]="2 * rulerScale"/>
                  <text [attr.x]="rulerImgW - scaleBarPx / 2 - 20 * rulerScale"
                        [attr.y]="rulerImgH - 42 * rulerScale"
                        fill="#fff" [attr.font-size]="13 * rulerScale" font-family="sans-serif"
                        text-anchor="middle">
                    {{ scaleBarMm }} mm
                  </text>
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
  `],
})
export class PipelinePreviewComponent implements OnInit, OnDestroy {
  @ViewChild('previewContainer') previewContainer!: ElementRef<HTMLDivElement>;
  @ViewChild('scrollArea') scrollArea!: ElementRef<HTMLDivElement>;
  @ViewChild('previewImg') previewImg!: ElementRef<HTMLImageElement>;
  @ViewChild('graphCanvas') graphCanvasRef!: ElementRef<HTMLCanvasElement>;
  @ViewChild('imageRoiContainer') imageRoiContainer!: ElementRef<HTMLDivElement>;

  imageSrc: string | null = null;
  isGrayscale = false;
  loading = false;
  imageCount = 0;
  currentIndex = 0;

  // Zoom/pan for gallery image
  zoomLevel = 1.0;
  private baseFitScale = 1;
  private isDragging = false;
  private dragStart = { x: 0, y: 0, scrollLeft: 0, scrollTop: 0 };

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
  showScaleBar = false;
  scaleBarPx = 0;
  scaleBarMm = 0;

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

  readonly Math = Math;

  get pxPerMm(): number {
    if (this.scaleLinePx > 0 && this.scaleMm > 0) {
      return this.scaleLinePx / this.scaleMm;
    }
    return 0;
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
           (this.scaleStart !== null && !(this.scaleEnd && this.scaleMm > 0)) ||
           (this.showScaleBar && this.pxPerMm > 0) ||
           (this.pixelActive && (this.pixelCurrentPos !== null || this.pixelFrozenPos !== null));
  }

  private subs: Subscription[] = [];

  constructor(
    private pipelineState: PipelineStateService,
    private cdr: ChangeDetectorRef,
    private recipeService: RecipeService,
  ) {}

  ngOnInit(): void {
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
      this.pipelineState.previewImageIsGrayscale$.subscribe((isGray) => {
        this.isGrayscale = isGray;
      }),
      this.pipelineState.previewLoading$.subscribe((l) => {
        if (this.particleClickPending) return;
        this.loading = l;
      }),
      this.pipelineState.imageCount$.subscribe((c) => (this.imageCount = c)),
      this.pipelineState.previewImageIndex$.subscribe((i) => (this.currentIndex = i)),
      combineLatest([
        this.pipelineState.sideOutputs$,
        this.pipelineState.selectedStepIndex$,
        this.pipelineState.pipeline$,
        this.pipelineState.previewImageIndex$
      ]).subscribe(([so, stepIdx, pipeline, imgIdx]) => {
        this.imageNames = so?.['loaded_paths'] ?? [];
        this.selectedStepIndex = stepIdx;
        this.currentPipeline = pipeline;
        
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
          if (Array.isArray(circles) && Array.isArray(circles[imgIdx])) {
            this.circlesForOverlay = circles[imgIdx];
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
    
    // Determine if we're in montage view or regular image view
    if (this.showingMontage && this.montagePreview) {
      // Zoom montage
      this.montageZoomLevel = Math.max(1.0, Math.min(5.0, this.montageZoomLevel * factor));
      this.applyMontageTransform();
    } else {
      // Zoom regular image
      this.zoomLevel = Math.max(1.0, Math.min(5.0, this.zoomLevel * factor));
      this.applyImageTransform();
    }
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

  // --- ROI interaction ---

  onImageLoad(): void {
    const img = this.previewImg?.nativeElement;
    if (img) {
      this.roiImgW = img.naturalWidth;
      this.roiImgH = img.naturalHeight;
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

  // === Ruler tool methods (multi-ruler, up to 5 lines) ===

  toggleRuler(): void {
    this.rulerActive = !this.rulerActive;
    if (this.rulerActive) {
      this.scaleActive = false;
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
    this.scaleActive = !this.scaleActive;
    if (this.scaleActive) {
      this.rulerActive = false;
      this.rulerDrawingStart = null;
      this.rulerDrawingCurrent = null;
      this.rulerDrawingDistance = 0;
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

  onShowScaleBarChange(): void {
    this.computeScaleBar();
  }

  private computeScaleBar(): void {
    const pm = this.pxPerMm;
    if (pm <= 0) {
      this.scaleBarPx = 0;
      this.scaleBarMm = 0;
      return;
    }
    const targetPx = this.rulerImgW / 5;
    const targetMm = targetPx / pm;
    const niceValues = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000];
    let best = niceValues[0];
    let bestDiff = Math.abs(targetMm - best);
    for (const v of niceValues) {
      const diff = Math.abs(targetMm - v);
      if (diff < bestDiff) {
        bestDiff = diff;
        best = v;
      }
    }
    this.scaleBarMm = best;
    this.scaleBarPx = best * pm;
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
        this.montageZoomLevel = 1.0;  // Reset zoom when displaying new montage
        this.showingMontage = true;
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
    this.montagePreview = null;
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
    if (this.scaleStart && this.scaleEnd && !(this.scaleMm > 0)) {
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
    if (this.showScaleBar && this.pxPerMm > 0 && this.scaleBarPx > 0) {
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
      const label = `${this.scaleBarMm} mm`;
      ctx.font = `${13 * s}px sans-serif`;
      ctx.fillStyle = '#fff';
      ctx.textAlign = 'center';
      ctx.fillText(label, barX + this.scaleBarPx / 2, h - 42 * s);
      ctx.textAlign = 'start';
    }

    // Save via file picker or download fallback
    const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, 'image/png'));
    if (!blob) return;

    if ('showSaveFilePicker' in window) {
      try {
        const handle = await (window as any).showSaveFilePicker({
          suggestedName: 'annotated_image.png',
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
    a.download = 'annotated_image.png';
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
      this.rulerActive = false;
      this.scaleActive = false;
      this.rulerDrawingStart = null;
      this.scaleStart = null;
      this.pixelCurrentPos = null;
      this.pixelFrozenPos = null;
      this.pixelCanvasCache = null;
      this.pixelImageDataCache = null;
    }
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
