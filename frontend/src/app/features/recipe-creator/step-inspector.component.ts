import {
  Component,
  OnInit,
  OnDestroy,
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { MatIconModule } from '@angular/material/icon';
import { HttpClient } from '@angular/common/http';
import { Subscription, combineLatest } from 'rxjs';
import { PipelineStateService } from '../../services/pipeline-state.service';
import { RecipeService, CalibrationRecord } from '../../services/recipe.service';
import {
  StepDefinition,
  StepInstance,
  StepError,
  ParamSchema,
} from '../../models/pipeline.models';
import { HistogramChartComponent } from './histogram-chart.component';
import { ScatterChartComponent, CurveFitData } from './scatter-chart.component';
import { PCAChartComponent, PCAData } from './pca-chart.component';

interface NodeHelpParameter {
  name: string;
  description: string;
}

interface NodeHelpPlacement {
  before?: string;
  after?: string;
  restrictions?: string[];
}

interface NodeHelpContent {
  purpose: string;
  usage: string;
  inputs: string;
  parameters?: NodeHelpParameter[];
  outputs: string;
  placement?: NodeHelpPlacement;
}

@Component({
  selector: 'app-step-inspector',
  standalone: true,
  imports: [CommonModule, FormsModule, MatIconModule, HistogramChartComponent, ScatterChartComponent, PCAChartComponent],
  template: `
    <div class="inspector-wrapper">
      @if (!definition) {
        <div class="no-selection">Válasszon egy lépést a szerkesztéshez</div>
      } @else {
        <div class="step-header">
          <mat-icon class="step-icon">{{ definition.icon }}</mat-icon>
          <button
            type="button"
            class="step-name-btn"
            [class.has-help]="hasNodeHelp()"
            [class.expanded]="nodeHelpExpanded"
            [disabled]="!hasNodeHelp()"
            (click)="toggleNodeHelp()"
            [title]="hasNodeHelp() ? 'Részletes leírás megnyitása' : 'Ehhez a lépéshez még nincs részletes leírás'"
          >
            <span class="step-name">{{ definition.name }}</span>
            @if (hasNodeHelp()) {
              <mat-icon class="step-help-chevron">chevron_right</mat-icon>
            }
          </button>
        </div>

        @if (nodeHelpExpanded && getNodeHelpContent(); as nodeHelp) {
          <div class="node-help-card" role="note" aria-label="Lépés részletes leírása">
            <div class="node-help-section">
              <div class="node-help-title">Cél</div>
              <p class="node-help-text">{{ nodeHelp.purpose }}</p>
            </div>
            <div class="node-help-section">
              <div class="node-help-title">Használat</div>
              <p class="node-help-text">{{ nodeHelp.usage }}</p>
            </div>
            <div class="node-help-section">
              <div class="node-help-title">Bemenet</div>
              <p class="node-help-text">{{ nodeHelp.inputs }}</p>
            </div>
            @if (nodeHelp.parameters && nodeHelp.parameters.length > 0) {
              <div class="node-help-section">
                <div class="node-help-title">Paraméterek</div>
                @for (param of nodeHelp.parameters; track param.name) {
                  <div class="node-param-block">
                    <span class="node-param-name">{{ param.name }}</span>
                    <span class="node-param-desc">{{ param.description }}</span>
                  </div>
                }
              </div>
            }
            <div class="node-help-section">
              <div class="node-help-title">Kimenet</div>
              <p class="node-help-text">{{ nodeHelp.outputs }}</p>
            </div>
            @if (nodeHelp.placement) {
              <div class="node-help-section">
                <div class="node-help-title">Elhelyezési szabályok</div>
                @if (nodeHelp.placement.before) {
                  <div class="node-rule-row">
                    <span class="node-rule-label">Előtte:</span>
                    <span class="node-rule-text">{{ nodeHelp.placement.before }}</span>
                  </div>
                }
                @if (nodeHelp.placement.after) {
                  <div class="node-rule-row">
                    <span class="node-rule-label">Utána:</span>
                    <span class="node-rule-text">{{ nodeHelp.placement.after }}</span>
                  </div>
                }
                @if (nodeHelp.placement.restrictions && nodeHelp.placement.restrictions.length > 0) {
                  <div class="node-rule-list">
                    @for (rule of nodeHelp.placement.restrictions; track rule) {
                      <div class="node-rule-item">• {{ rule }}</div>
                    }
                  </div>
                }
              </div>
            }
          </div>
        }

        @if (stepErrors.length > 0) {
          <div class="error-list">
            @for (err of stepErrors; track err.error_code) {
              <div class="error-item">⚠ {{ err.message }}</div>
            }
          </div>
        }

        <fieldset class="params-section" [disabled]="isPreviewMode" [class.preview-locked]="isPreviewMode">
          @if (isRoiStep()) {
            <div class="roi-shape-selector">
              <button class="roi-shape-btn" [class.active]="getParamValue('roi_type') === 'rect'" (click)="onParamChange('roi_type', 'rect')" title="Téglalap">
                <svg viewBox="0 0 24 24" width="20" height="20">
                  <rect x="3" y="5" width="18" height="14" fill="none" stroke="currentColor" stroke-width="1.5" stroke-dasharray="3 2" rx="1"/>
                </svg>
              </button>
              <button class="roi-shape-btn" [class.active]="getParamValue('roi_type') === 'ellipse'" (click)="onParamChange('roi_type', 'ellipse')" title="Ellipszis">
                <svg viewBox="0 0 24 24" width="20" height="20">
                  <ellipse cx="12" cy="12" rx="10" ry="7" fill="none" stroke="currentColor" stroke-width="1.5" stroke-dasharray="3 2"/>
                </svg>
              </button>
              <button class="roi-shape-btn" [class.active]="getParamValue('roi_type') === 'polygon'" (click)="onParamChange('roi_type', 'polygon')" title="Sokszög">
                <svg viewBox="0 0 24 24" width="20" height="20">
                  <polygon points="12,2 22,8 19,20 5,20 2,8" fill="none" stroke="currentColor" stroke-width="1.5" stroke-dasharray="3 2"/>
                </svg>
              </button>
            </div>
            <div class="param-row roi-crop-row">
              <label class="param-label" [attr.for]="'param-output_mode'">Kivágás</label>
              <div class="param-control">
                <label class="toggle-wrap">
                  <input
                    type="checkbox"
                    id="param-output_mode"
                    [checked]="getParamValue('output_mode') === 'crop'"
                    (change)="onParamChange('output_mode', $any($event.target).checked ? 'crop' : 'mask')"
                  />
                  <span class="toggle-label">{{ getParamValue('output_mode') === 'crop' ? 'Be' : 'Ki' }}</span>
                </label>
              </div>
            </div>
            <div class="param-row roi-crop-row">
              <label class="param-label" [attr.for]="'param-shape_only'">Csak körvonal</label>
              <div class="param-control">
                <label class="toggle-wrap">
                  <input
                    type="checkbox"
                    id="param-shape_only"
                    [checked]="!!getParamValue('shape_only')"
                    (change)="onParamChange('shape_only', $any($event.target).checked)"
                  />
                  <span class="toggle-label">{{ getParamValue('shape_only') ? 'Be' : 'Ki' }}</span>
                </label>
              </div>
            </div>
            @if (getParamValue('shape_only')) {
              <div class="param-row roi-crop-row">
                <label class="param-label" [attr.for]="'param-shape_outline_color'">Körvonal színe</label>
                <div class="param-control">
                  <select
                    id="param-shape_outline_color"
                    [ngModel]="getParamValue('shape_outline_color')"
                    (ngModelChange)="onParamChange('shape_outline_color', $event)"
                  >
                    <option value="fekete">Fekete</option>
                    <option value="fehér">Fehér</option>
                  </select>
                </div>
              </div>
              <div class="param-row roi-crop-row">
                <label class="param-label" [attr.for]="'param-shape_outline_thickness'">Körvonal vastagsága</label>
                <div class="param-control">
                  <input
                    type="number"
                    id="param-shape_outline_thickness"
                    [min]="1"
                    [max]="100"
                    [step]="1"
                    [ngModel]="getParamValue('shape_outline_thickness')"
                    (ngModelChange)="onNumericTextChange(getParamByName('shape_outline_thickness')!, $event)"
                  />
                </div>
              </div>
            }
            @if (isRoiEmpty()) {
              <div class="roi-empty-warning">⚠ Nincs kijelölt ROI terület</div>
            }
          }
          @if (isReferenceCropStep()) {
            <div class="reference-crop-actions">
              <button
                type="button"
                class="reference-crop-toggle"
                [class.active]="!!getParamValue('show_references')"
                (click)="toggleReferenceCropView()"
              >
                {{ getParamValue('show_references') ? 'Teljes kep mutatasa' : 'Kivagott referenciak mutatasa' }}
              </button>
              <div class="reference-crop-count">
                {{ getReferenceCropCountLabel() }}
              </div>
            </div>
            @if (getReferenceCropRows().length > 0) {
              <div class="reference-crop-list">
                @for (crop of getReferenceCropRows(); track crop.key) {
                  <div class="reference-crop-row">
                    <span class="reference-crop-index">{{ crop.index + 1 }}</span>
                    <input
                      type="text"
                      class="reference-crop-name"
                      [ngModel]="crop.name"
                      (ngModelChange)="onReferenceCropNameChange(crop, $event)"
                      [placeholder]="'Referencia ' + (crop.index + 1)"
                    />
                    <button
                      type="button"
                      class="reference-crop-delete"
                      (click)="removeReferenceCrop(crop)"
                      [attr.aria-label]="'Referencia ' + (crop.index + 1) + ' torlese'"
                      title="Referencia torlese"
                    >
                      &times;
                    </button>
                  </div>
                }
              </div>
            }
          }
          @if (step?.step_def_id === 'cluster_reference_map') {
            <div class="cluster-components">
              <div class="section-label">Elfogadott térképek</div>
              @for (component of getAcceptedClusterMaps(); track $index) {
                <div class="cluster-component-card">
                  @if (getAcceptedClusterMapImage($index); as mapImage) {
                    <img class="cluster-component-preview" [src]="mapImage" [alt]="component.name" />
                  }
                  <input
                    type="text"
                    [ngModel]="component.name"
                    (ngModelChange)="updateAcceptedClusterMap($index, 'name', $event)"
                  />
                  <div class="cluster-component-summary">
                    Labelek: {{ component.selected_labels }} · referencia: {{ component.reference_label }}
                  </div>
                  <label>
                    Szorzó
                    <input
                      type="number"
                      min="0"
                      max="1"
                      step="0.05"
                      [ngModel]="component.map_multiplier"
                      (ngModelChange)="updateAcceptedClusterMap($index, 'map_multiplier', +$event)"
                    />
                  </label>
                  <button type="button" (click)="removeAcceptedClusterMap($index)">Eltávolítás</button>
                </div>
              } @empty {
                <div class="dev-empty">Még nincs elfogadott térkép.</div>
              }
              @if (getParamValue('remainder_as_last')) {
                <div class="cluster-component-card remainder">
                  @if (getAcceptedClusterMapImage(getAcceptedClusterMaps().length); as remainderImage) {
                    <img class="cluster-component-preview" [src]="remainderImage" alt="Maradék térkép" />
                  }
                  <input
                    type="text"
                    [ngModel]="getParamValue('remainder_name')"
                    (ngModelChange)="onParamChange('remainder_name', $event)"
                    placeholder="Maradék"
                  />
                  <div class="cluster-component-summary">
                    100% − az összes eltárolt térkép
                  </div>
                  <label>
                    Megjelenítési szorzó
                    <input
                      type="number"
                      min="0"
                      max="1"
                      step="0.05"
                      [ngModel]="getParamValue('remainder_display_multiplier')"
                      (ngModelChange)="onParamChange('remainder_display_multiplier', +$event)"
                    />
                  </label>
                  <label class="toggle-wrap">
                    <input
                      type="checkbox"
                      [ngModel]="getParamValue('remainder_invert')"
                      (ngModelChange)="onParamChange('remainder_invert', $event)"
                    />
                    <span>Színskála megfordítása</span>
                  </label>
                  <button type="button" (click)="removeClusterMapRemainder()">Maradék törlése</button>
                </div>
              } @else {
                <button
                  type="button"
                  class="cluster-remainder-button"
                  [disabled]="getAcceptedClusterMaps().length < 2"
                  (click)="calculateClusterMapRemainder()"
                >
                  Maradék kiszámítása
                </button>
                @if (getAcceptedClusterMaps().length < 2) {
                  <div class="cluster-component-summary">
                    Előbb tárolj el legalább két térképet a „Kész” gombbal.
                  </div>
                }
              }
            </div>
          }
          @for (param of getVisibleParams(); track param.name) {
                @if (param.name !== 'file_order' && param.name !== 'group_colors' && param.name !== 'output_mode' && param.name !== 'shape_only' && param.name !== 'shape_outline_color' && param.name !== 'shape_outline_thickness' && !shouldHideParam(param)) {
            <div class="param-row">
              <label class="param-label" [attr.for]="'param-' + param.name">
                {{ getDisplayParamLabel(param) }}
              </label>

              @switch (param.type) {
                @case ('int') {
                  @if (step?.step_def_id === 'color_thresh' && isColorThreshMaxParam(param.name); as isMaxParam) {
                    <!-- Range slider for color_thresh channels -->
                    @let minParamName = param.name.slice(0, -3) + 'min';
                    @let minValue = getParamValue(minParamName);
                    @let maxValue = getParamValue(param.name);
                    @let minMax = getSliderMinMax(param);
                    <div class="param-control range-slider-control">
                      <div class="range-slider-container">
                        <input
                          type="range"
                          class="range-slider-min"
                          [min]="minMax.min"
                          [max]="minMax.max"
                          [step]="param.step ?? 1"
                          [value]="minValue ?? minMax.min"
                          (input)="onRangeMinChange(minParamName, $event)"
                        />
                        <input
                          type="range"
                          class="range-slider-max"
                          [min]="minMax.min"
                          [max]="minMax.max"
                          [step]="param.step ?? 1"
                          [value]="maxValue ?? minMax.max"
                          (input)="onRangeMaxChange(param.name, $event)"
                        />
                      </div>
                      <div class="range-slider-values">
                        <input
                          type="number"
                          class="range-number-input"
                          [min]="minMax.min"
                          [max]="minMax.max"
                          [value]="minValue ?? minMax.min"
                          (ngModelChange)="onParamChange(minParamName, +$event)"
                        />
                        <span class="range-separator">–</span>
                        <input
                          type="number"
                          class="range-number-input"
                          [min]="minMax.min"
                          [max]="minMax.max"
                          [value]="maxValue ?? minMax.max"
                          (ngModelChange)="onParamChange(param.name, +$event)"
                        />
                      </div>
                    </div>

                    @if (getColorThreshHistogramForParam(param.name); as hist) {
                      <div class="color-thresh-histogram">
                        <app-histogram-chart
                          [data]="hist.values"
                          [rangeMin]="hist.rangeMin"
                          [rangeMax]="hist.rangeMax"
                          [label]="hist.channel + ' - Kép ' + (previewImageIndex + 1)"
                          [markerLines]="getColorThreshMarkerLines(hist.channel)"
                        />
                      </div>
                    }
                  } @else {
                    <!-- Standard int slider -->
                    <div class="param-control slider-control">
                      <input
                        type="range"
                        [id]="'param-' + param.name"
                        [min]="getSliderMin(param)"
                        [max]="getSliderMax(param)"
                        [step]="param.odd_only ? 2 : (param.step ?? 1)"
                        [ngModel]="getParamValue(param.name)"
                        (ngModelChange)="onParamChange(param.name, $event)"
                      />
                      <input
                        type="number"
                        class="slider-number"
                        [min]="getSliderMin(param)"
                        [max]="getSliderMax(param)"
                        [ngModel]="getParamValue(param.name)"
                        (ngModelChange)="onNumericTextChange(param, $event)"
                      />
                      @if (isFitCurveValidationRatioParam(param.name)) {
                        <span class="inline-unit">%</span>
                      }
                    </div>
                  }
                }
                @case ('float') {
                  <div class="param-control slider-control">
                    <input
                      type="range"
                      [id]="'param-' + param.name"
                      [min]="getSliderMin(param)"
                      [max]="getSliderMax(param)"
                      [step]="param.step ?? 0.01"
                      [ngModel]="getParamValue(param.name)"
                      (ngModelChange)="onParamChange(param.name, +$event)"
                    />
                    <input
                      type="number"
                      class="slider-number"
                      [min]="getSliderMin(param)"
                      [max]="getSliderMax(param)"
                      [step]="param.step ?? 0.01"
                      [ngModel]="getParamValue(param.name)"
                      (ngModelChange)="onNumericTextChange(param, $event)"
                    />
                  </div>
                }
                @case ('bool') {
                  @if (isFitCurveDataMergeParam(param.name)) {
                    <div class="aggregation-radio-grid">
                      <label class="merge-mode-option">
                        <input type="radio" name="fit-merge-mode" [checked]="getFitCurveMergeMode() === 'none'" (change)="setFitCurveMergeMode('none')" />
                        <span>Nincs</span>
                      </label>
                      <label class="merge-mode-option">
                        <input type="radio" name="fit-merge-mode" [checked]="getFitCurveMergeMode() === 'tablet'" (change)="setFitCurveMergeMode('tablet')" />
                        <span>Tablettánként</span>
                      </label>
                      <label class="merge-mode-option with-inline-select">
                        <input type="radio" name="fit-merge-mode" [checked]="getFitCurveMergeMode() === 'level'" (change)="setFitCurveMergeMode('level')" />
                        <span>Szintenként</span>
                        @if (getFitCurveMergeMode() === 'level') {
                          <select
                            id="param-agg_method"
                            [ngModel]="getParamValue('agg_method')"
                            (ngModelChange)="onParamChange('agg_method', $event)"
                          >
                            @for (opt of getFitCurveAggMethodOptions(); track opt) {
                              <option [value]="opt">{{ getFitCurveAggMethodDisplayLabel(opt) }}</option>
                            }
                          </select>
                        }
                      </label>
                    </div>
                  } @else {
                    <div class="param-control">
                      <label class="toggle-wrap">
                        <input
                          type="checkbox"
                          [id]="'param-' + param.name"
                          [ngModel]="getParamValue(param.name)"
                          (ngModelChange)="onParamChange(param.name, $event)"
                          [disabled]="isBoolParamDisabled(param.name)"
                        />
                        <span class="toggle-label">{{ getParamValue(param.name) ? 'Be' : 'Ki' }}</span>
                      </label>
                    </div>
                  }
                }
                @case ('enum') {
                  <div class="param-control">
                    <select
                      [id]="'param-' + param.name"
                      [ngModel]="getParamValue(param.name)"
                      (ngModelChange)="onParamChange(param.name, $event)"
                    >
                      @for (opt of getFilteredOptions(param); track opt) {
                        <option [value]="opt">{{ getOptionDisplayLabel(param, opt) }}</option>
                      }
                    </select>
                    @if (step?.step_def_id === 'apply_threshold' && param.name === 'mode' && getThresholdInputHistogram()) {
                      <div class="threshold-histogram-wrap">
                        <app-histogram-chart
                          [data]="getThresholdInputHistogram()!"
                          [rangeMin]="0"
                          [rangeMax]="256"
                          [label]="''"
                          [markerLines]="getThresholdMarkerLines()"
                        />
                      </div>
                    }
                  </div>
                }
                @case ('string') {
                  @if (step?.step_def_id === 'cluster_reference_map' && param.name === 'selected_labels') {
                    <div class="param-control label-picker">
                      @for (label of getClusterMapLabelOptions(); track label) {
                        <button type="button" class="label-chip" [class.active]="isClusterMapLabelSelected(label)"
                          [attr.aria-pressed]="isClusterMapLabelSelected(label)"
                          (click)="toggleClusterMapLabel(label)">
                          <span class="label-chip-swatch" [style.background]="getClusterLabelColor(label)">
                            {{ isClusterMapLabelSelected(label) ? '✓' : '' }}
                          </span>
                          <span>Label {{ label }}</span>
                        </button>
                      }
                    </div>
                  } @else if (step?.step_def_id === 'cluster_reference_map' && param.name === 'reference_label') {
                    <div class="param-control">
                      <select
                        [id]="'param-' + param.name"
                        [ngModel]="getParamValue(param.name)"
                        (ngModelChange)="onParamChange(param.name, $event)"
                      >
                        @for (label of getSelectedClusterMapLabelOptions(); track label) {
                          <option [value]="label">Label {{ label }}</option>
                        }
                      </select>
                    </div>
                  } @else if (isFitCurveYAxisParam(param.name)) {
                    <div class="param-control">
                      <select
                        [id]="'param-' + param.name"
                        [ngModel]="getParamValue(param.name)"
                        (ngModelChange)="onParamChange(param.name, $event)"
                      >
                        @for (opt of getFitCurveYOptions(); track opt) {
                          <option [value]="opt">{{ getYKeyLabel(opt) }}</option>
                        }
                      </select>
                    </div>
                  } @else if (isPredictYAxisParam(param.name)) {
                    <div class="param-control">
                      <select
                        [id]="'param-' + param.name"
                        [ngModel]="getParamValue(param.name)"
                        (ngModelChange)="onParamChange(param.name, $event)"
                      >
                        @for (opt of getPredictYOptions(); track opt) {
                          <option [value]="opt">{{ getYKeyLabel(opt) }}</option>
                        }
                      </select>
                    </div>
                  } @else if (isKmeansReferenceSourceParam(param.name)) {
                    <div class="param-control">
                      <select
                        [id]="'param-' + param.name"
                        [ngModel]="getParamValue(param.name) || 'auto'"
                        (ngModelChange)="onParamChange(param.name, $event)"
                      >
                        @for (opt of getKmeansReferenceSourceOptions(); track opt.value) {
                          <option [value]="opt.value">{{ opt.label }}</option>
                        }
                      </select>
                    </div>
                  } @else if (isPredictEquationParam(param.name)) {
                    <div class="param-control file-path-control">
                      <input
                        type="text"
                        [id]="'param-' + param.name"
                        [ngModel]="getParamValue(param.name)"
                        (change)="onParamChange(param.name, $any($event.target).value)"
                        placeholder="Pl. y = 1.2x + 3.4"
                      />
                      <button class="browse-btn" (click)="openCalibrationBrowser()" title="Kalibráció kiválasztása">
                        <mat-icon>manage_search</mat-icon>
                      </button>
                    </div>
                  } @else {
                    <div class="param-control" [class.file-path-control]="isReferenceValuesParam(param.name)">
                      <input
                        type="text"
                        [id]="'param-' + param.name"
                        [ngModel]="getParamValue(param.name)"
                        (change)="onParamChange(param.name, $any($event.target).value)"
                      />
                      @if (isReferenceValuesParam(param.name)) {
                        <button class="browse-btn" (click)="importReferenceValuesFromFile()" title="CSV/TXT import"><mat-icon>upload_file</mat-icon></button>
                      }
                    </div>
                  }
                }
                @case ('file_path') {
                  <div class="param-control file-path-control">
                    <input
                      type="text"
                      [id]="'param-' + param.name"
                      [ngModel]="getParamValue(param.name)"
                      (change)="onParamChange(param.name, $any($event.target).value)"
                      placeholder="Fájl elérési útja..."
                    />
                    <button class="browse-btn" (click)="browseFile(param.name)" title="Fájl tallózása"><mat-icon>image</mat-icon></button>
                    <button class="browse-btn" (click)="browseFolder(param.name)" title="Mappa tallózása"><mat-icon>folder_open</mat-icon></button>
                  </div>
                }
              }
            </div>
            }
          }

          @if (isSaveImagesStep()) {
            <div class="save-images-section">
              <div class="param-row">
                <label class="param-label" for="save-output-folder">Kimeneti mappa</label>
                <div class="param-control file-path-control">
                  <input
                    type="text"
                    id="save-output-folder"
                    [ngModel]="getParamValue('output_folder')"
                    (ngModelChange)="onParamChange('output_folder', $event)"
                    placeholder="Mappa elérési útja..."
                  />
                  <button class="browse-btn" (click)="browseFolder('output_folder')" title="Mappa tallózása"><mat-icon>folder_open</mat-icon></button>
                </div>
              </div>

              <div class="param-row">
                <label class="param-label" for="save-name-prefix">Név előtag</label>
                <div class="param-control">
                  <input
                    type="text"
                    id="save-name-prefix"
                    [ngModel]="getParamValue('name_prefix')"
                    (ngModelChange)="onParamChange('name_prefix', $event)"
                    placeholder="Pl. feldolgozott_"
                  />
                </div>
              </div>

              <div class="param-row">
                <label class="param-label" for="save-name-suffix">Név utótag</label>
                <div class="param-control">
                  <input
                    type="text"
                    id="save-name-suffix"
                    [ngModel]="getParamValue('name_suffix')"
                    (ngModelChange)="onParamChange('name_suffix', $event)"
                    placeholder="Pl. _szerkesztett"
                  />
                </div>
              </div>

              <div class="user-result-item">
                <span class="user-result-label">Névminta előnézet:</span>
                <span class="user-result-value">{{ getSaveNamePreview() }}</span>
              </div>
            </div>
          }

          @if (isSaveArrayStep()) {
            <div class="save-images-section">
              <div class="param-row">
                <label class="param-label" for="save-array-folder">Mentési hely</label>
                <div class="param-control file-path-control">
                  <input
                    type="text"
                    id="save-array-folder"
                    [ngModel]="getParamValue('output_folder')"
                    (ngModelChange)="onParamChange('output_folder', $event)"
                    placeholder="Mappa elérési útja..."
                  />
                  <button class="browse-btn" (click)="browseFolder('output_folder')" title="Mappa tallózása"><mat-icon>folder_open</mat-icon></button>
                </div>
              </div>

              <div class="param-row">
                <label class="param-label" for="save-array-filename">Fájlnév</label>
                <div class="param-control">
                  <input
                    type="text"
                    id="save-array-filename"
                    [ngModel]="getParamValue('filename')"
                    (ngModelChange)="onParamChange('filename', $event)"
                    placeholder="adattomb.csv"
                  />
                </div>
              </div>
            </div>
          }

          @if (step?.step_def_id === 'detect_particles') {
            <div class="section-label" style="margin-top: 6px;">Szemcsék szűrése</div>

            <div class="filter-subgroup">
              <div class="filter-subgroup-head">
                <span class="filter-subgroup-title">Terület alapján</span>
                <label class="toggle-wrap">
                  <input
                    type="checkbox"
                    id="param-filter_by_area"
                    [ngModel]="getParamValue('filter_by_area')"
                    (ngModelChange)="onParamChange('filter_by_area', $event)"
                  />
                  <span class="toggle-label">{{ getParamValue('filter_by_area') ? 'Be' : 'Ki' }}</span>
                </label>
              </div>
              @if (getParamValue('filter_by_area')) {
                @if (getParamByName('filter_min_area'); as minAreaParam) {
                  <div class="sub-param-row">
                    <label class="param-label" for="param-filter_min_area">Min.</label>
                    <div class="param-control slider-control">
                      <input
                        type="range"
                        id="param-filter_min_area"
                        [min]="getSliderMin(minAreaParam)"
                        [max]="getSliderMax(minAreaParam)"
                        [step]="minAreaParam.odd_only ? 2 : (minAreaParam.step ?? 1)"
                        [ngModel]="getParamValue('filter_min_area')"
                        (ngModelChange)="onParamChange('filter_min_area', $event)"
                      />
                      <input
                        type="number"
                        class="slider-number"
                        [ngModel]="getParamValue('filter_min_area')"
                        (ngModelChange)="onNumericTextChange(minAreaParam, $event)"
                      />
                    </div>
                  </div>
                }
                @if (getParamByName('filter_max_area'); as maxAreaParam) {
                  <div class="sub-param-row">
                    <label class="param-label" for="param-filter_max_area">Max.</label>
                    <div class="param-control slider-control">
                      <input
                        type="range"
                        id="param-filter_max_area"
                        [min]="getSliderMin(maxAreaParam)"
                        [max]="getSliderMax(maxAreaParam)"
                        [step]="maxAreaParam.odd_only ? 2 : (maxAreaParam.step ?? 1)"
                        [ngModel]="getParamValue('filter_max_area')"
                        (ngModelChange)="onParamChange('filter_max_area', $event)"
                      />
                      <input
                        type="number"
                        class="slider-number"
                        [ngModel]="getParamValue('filter_max_area')"
                        (ngModelChange)="onNumericTextChange(maxAreaParam, $event)"
                      />
                    </div>
                  </div>
                }
              }
            </div>

            <div class="filter-subgroup">
              <div class="filter-subgroup-head">
                <span class="filter-subgroup-title">Kerekdedség alapján</span>
                <label class="toggle-wrap">
                  <input
                    type="checkbox"
                    id="param-filter_by_circularity"
                    [ngModel]="getParamValue('filter_by_circularity')"
                    (ngModelChange)="onParamChange('filter_by_circularity', $event)"
                  />
                  <span class="toggle-label">{{ getParamValue('filter_by_circularity') ? 'Be' : 'Ki' }}</span>
                </label>
              </div>
              @if (getParamValue('filter_by_circularity')) {
                @if (getParamByName('filter_min_circularity'); as minCircParam) {
                  <div class="sub-param-row">
                    <label class="param-label" for="param-filter_min_circularity">Min.</label>
                    <div class="param-control slider-control">
                      <input
                        type="range"
                        id="param-filter_min_circularity"
                        [min]="getSliderMin(minCircParam)"
                        [max]="getSliderMax(minCircParam)"
                        [step]="minCircParam.step ?? 0.01"
                        [ngModel]="getParamValue('filter_min_circularity')"
                        (ngModelChange)="onParamChange('filter_min_circularity', +$event)"
                      />
                      <input
                        type="number"
                        class="slider-number"
                        [step]="minCircParam.step ?? 0.01"
                        [ngModel]="getParamValue('filter_min_circularity')"
                        (ngModelChange)="onNumericTextChange(minCircParam, $event)"
                      />
                    </div>
                  </div>
                }
                @if (getParamByName('filter_max_circularity'); as maxCircParam) {
                  <div class="sub-param-row">
                    <label class="param-label" for="param-filter_max_circularity">Max.</label>
                    <div class="param-control slider-control">
                      <input
                        type="range"
                        id="param-filter_max_circularity"
                        [min]="getSliderMin(maxCircParam)"
                        [max]="getSliderMax(maxCircParam)"
                        [step]="maxCircParam.step ?? 0.01"
                        [ngModel]="getParamValue('filter_max_circularity')"
                        (ngModelChange)="onParamChange('filter_max_circularity', +$event)"
                      />
                      <input
                        type="number"
                        class="slider-number"
                        [step]="maxCircParam.step ?? 0.01"
                        [ngModel]="getParamValue('filter_max_circularity')"
                        (ngModelChange)="onNumericTextChange(maxCircParam, $event)"
                      />
                    </div>
                  </div>
                }
              }
            </div>

            <div class="filter-subgroup">
              <div class="filter-subgroup-head">
                <span class="filter-subgroup-title">Konvexitás alapján</span>
                <label class="toggle-wrap">
                  <input
                    type="checkbox"
                    id="param-filter_by_convexity"
                    [ngModel]="getParamValue('filter_by_convexity')"
                    (ngModelChange)="onParamChange('filter_by_convexity', $event)"
                  />
                  <span class="toggle-label">{{ getParamValue('filter_by_convexity') ? 'Be' : 'Ki' }}</span>
                </label>
              </div>
              @if (getParamValue('filter_by_convexity')) {
                <div class="convexity-checkboxes">
                  <label class="toggle-wrap">
                    <input type="checkbox" [ngModel]="getParamValue('filter_convex')" (ngModelChange)="onParamChange('filter_convex', $event)" />
                    <span class="toggle-label">Konvex</span>
                  </label>
                  <label class="toggle-wrap">
                    <input type="checkbox" [ngModel]="getParamValue('filter_concave')" (ngModelChange)="onParamChange('filter_concave', $event)" />
                    <span class="toggle-label">Konkáv</span>
                  </label>
                </div>
              }
            </div>
          }

          @if (isReferenceStep()) {
            <div class="generator-group">
              <div class="generator-group-title">Értékek generálása</div>
              <div class="generator-fields">
                <div class="gen-field">
                  <label class="gen-label">Szintek száma</label>
                  <input type="number" class="gen-input" [ngModel]="getParamValue('num_levels')" (ngModelChange)="onNumericTextChange(getParamByName('num_levels')!, $event)" min="1" />
                </div>
                <div class="gen-field">
                  <label class="gen-label">Kezdőérték</label>
                  <input type="number" class="gen-input" [ngModel]="getParamValue('start')" (ngModelChange)="onNumericTextChange(getParamByName('start')!, $event)" [step]="0.1" />
                </div>
                <div class="gen-field">
                  <label class="gen-label">Lépésköz</label>
                  <input type="number" class="gen-input" [ngModel]="getParamValue('step_val')" (ngModelChange)="onNumericTextChange(getParamByName('step_val')!, $event)" [step]="0.1" />
                </div>
              </div>
              <button class="run-fit-btn" (click)="generateReferenceValues()">
                Értékek generálása
              </button>
            </div>

            @if (referenceGroups.length > 0) {
              <div class="group-colors-section">
                <div class="omitted-title">Csoport színek</div>
                @for (grp of referenceGroups; track grp.key) {
                  <div class="group-color-row">
                    <span class="group-color-label">{{ grp.label }}</span>
                    <input
                      type="color"
                      class="group-color-picker"
                      [ngModel]="grp.color"
                      (ngModelChange)="onReferenceGroupColorChange(grp.key, $event)"
                    />
                  </div>
                }
              </div>
            }
          }
        </fieldset>

        @if (isLoadImageStep()) {
          <div class="image-manager-section">
            <button class="image-manager-btn" [disabled]="isPreviewMode" (click)="showImageManager = true">
              <mat-icon>photo_library</mat-icon> Képek kezelése
            </button>
          </div>
        }

        @if (showImageManager) {
          <div class="img-manager-overlay" (click)="showImageManager = false">
            <div class="img-manager-dialog" (click)="$event.stopPropagation()">
              <div class="img-manager-header">
                <span class="img-manager-title">Képek sorrendje</span>
                <button class="img-manager-close" (click)="showImageManager = false"><mat-icon>close</mat-icon></button>
              </div>
              @if (loadedImageNames.length === 0) {
                <p class="img-manager-empty">Nincs betöltött kép. Először válasszon forrás útvonalat.</p>
              } @else {
                <div class="img-manager-list">
                  @for (name of loadedImageNames; track $index) {
                    <div
                      class="img-manager-item"
                      [class.selected]="selectedImageIdx === $index"
                      (click)="selectedImageIdx = $index"
                    >
                      <span class="img-idx">{{ $index + 1 }}.</span>
                      <span class="img-name">{{ name }}</span>
                    </div>
                  }
                </div>
                <div class="img-manager-actions">
                  <button (click)="moveImage('top')" [disabled]="selectedImageIdx <= 0" title="Legelejére"><mat-icon>vertical_align_top</mat-icon></button>
                  <button (click)="moveImage('up')" [disabled]="selectedImageIdx <= 0" title="Fel"><mat-icon>arrow_upward</mat-icon></button>
                  <button (click)="moveImage('down')" [disabled]="selectedImageIdx >= loadedImageNames.length - 1" title="Le"><mat-icon>arrow_downward</mat-icon></button>
                  <button (click)="moveImage('bottom')" [disabled]="selectedImageIdx >= loadedImageNames.length - 1" title="Legvégére"><mat-icon>vertical_align_bottom</mat-icon></button>
                </div>
                <div class="img-manager-footer">
                  <button class="img-manager-apply" (click)="applyImageOrder()">Alkalmaz</button>
                </div>
              }
            </div>
          </div>
        }

        <!-- User-friendly results -->
        @if (hasUserFriendlyResults()) {
          <div class="user-results-section">
            <div class="section-label">Eredmények</div>

              @if (step?.step_def_id === 'load_image') {
                <div class="user-result-item">
                  <span class="user-result-label">Betöltött képek száma:</span>
                  <span class="user-result-value">{{ getLoadedImageCount() }}</span>
                </div>
              }

              @if (step?.step_def_id === 'calculate_histograms' && getHistogramData()) {
                <app-histogram-chart
                  [data]="getHistogramData()!"
                  [rangeMin]="getParamValue('range_min') ?? 0"
                  [rangeMax]="getParamValue('range_max') ?? 256"
                  [label]="'Kép ' + (previewImageIndex + 1) + ' hisztogramja'"
                />
              }

              @if (step?.step_def_id === 'histogram_equalization') {
                @if (getHisteqInputData()) {
                  <app-histogram-chart
                    [data]="getHisteqInputData()!"
                    [rangeMin]="0"
                    [rangeMax]="256"
                    [label]="'Bemenet – Kép ' + (previewImageIndex + 1)"
                  />
                }
                @if (getHisteqOutputData()) {
                  <app-histogram-chart
                    [data]="getHisteqOutputData()!"
                    [rangeMin]="0"
                    [rangeMax]="256"
                    [label]="'Kimenet – Kép ' + (previewImageIndex + 1)"
                  />
                }
              }

              @if (step?.step_def_id === 'calculate_intensity_stats' && getIntensityStatsEntries().length > 0) {
                <div class="stats-grid">
                  @for (entry of getIntensityStatsEntries(); track entry.key) {
                    <div class="stat-item">
                      <span class="stat-label">{{ entry.label }}</span>
                      <span class="stat-value">{{ entry.value }}</span>
                    </div>
                  }
                </div>
              }

              @if (step?.step_def_id === 'kmeans_cluster') {
                @if (getKmeansReferenceInfo(); as refInfo) {
                  <div class="user-result-item">
                    <span class="user-result-label">Referencia cropok a pipeline-ban:</span>
                    <span class="user-result-value">{{ refInfo.reference_crops_available }}</span>
                  </div>
                  <div class="user-result-item">
                    <span class="user-result-label">Referencia cropok hasznalata:</span>
                    <span class="user-result-value">{{ refInfo.uses_reference_crops ? 'Igen' : 'Nem' }}</span>
                  </div>
                  <div class="user-result-item">
                    <span class="user-result-label">Forras:</span>
                    <span class="user-result-value">{{ refInfo.reference_source_label || (refInfo.reference_sequence_used ? 'Reference sequence' : (refInfo.reference_crops_available > 0 ? 'Reference crop' : '-')) }}</span>
                  </div>
                  <div class="user-result-item">
                    <span class="user-result-label">Tenyleges klaszterszam:</span>
                    <span class="user-result-value">{{ refInfo.effective_k }}</span>
                  </div>
                  @if (getKmeansReferenceSequenceEntries().length > 0) {
                    <div class="stats-grid">
                      @for (entry of getKmeansReferenceSequenceEntries(); track entry.key) {
                        <div class="stat-item">
                          <span class="stat-label">{{ entry.label }}</span>
                          <span class="stat-value">{{ entry.value }}</span>
                        </div>
                      }
                    </div>
                  }
                }
                @if (getKmeansClusterEntries().length > 0) {
                  <div class="stats-grid">
                    @for (entry of getKmeansClusterEntries(); track entry.key) {
                      <div class="stat-item">
                        <span class="stat-label">{{ entry.label }}</span>
                        <span class="stat-value">{{ entry.value }}</span>
                      </div>
                    }
                  </div>
                }
              }

              @if (step?.step_def_id === 'fit_curve') {
                <button class="run-fit-btn" (click)="runCurveFit()" [disabled]="previewLoading || isPreviewMode">
                  {{ previewLoading ? '⏳ Futtatás...' : '▶ Görbe illesztés futtatása' }}
                </button>
                @if (getLatestCurveFit()) {
                  <button class="run-fit-btn" (click)="openSaveCalibrationDialog()">
                    Kalibrációs görbe mentése
                  </button>
                  <div class="chart-with-maximize">
                    <app-scatter-chart
                      [data]="getLatestCurveFit()!"
                      [label]="'Görbe illesztés'"
                      [omittedIndices]="getOmittedForCurrentChart()"
                    />
                    <button class="maximize-btn" (click)="maximizeChart(getLatestCurveFit()!)" title="Nagyítás">
                      <mat-icon>open_in_full</mat-icon>
                    </button>
                  </div>
                }
                @if (omittedEntries.length > 0) {
                  <div class="omitted-section">
                    <div class="omitted-header">
                      <div class="omitted-title">Kihagyott adatpontok ({{ omittedEntries.length }})</div>
                      <button class="omitted-restore-btn" (click)="restoreAllOmittedPoints()">Visszaállítás</button>
                    </div>
                    @for (entry of omittedEntries; track entry.index) {
                      <div class="omitted-item">
                        <span class="omitted-idx">#{{ entry.index + 1 }}</span>
                        <span class="omitted-name">{{ entry.name }}</span>
                      </div>
                    }
                  </div>
                }
              }

              @if (step?.step_def_id === 'histogram_pca') {
                <button class="run-fit-btn" (click)="runPCA()" [disabled]="previewLoading || isPreviewMode">
                  {{ previewLoading ? '⏳ PCA futtatás...' : '▶ PCA futtatása' }}
                </button>
                <div class="chart-with-maximize">
                  @if (getPCAData()) {
                    <app-pca-chart 
                      [data]="getPCAData()!" 
                      (componentChanged)="onPCAComponentChanged($event)"
                    />
                    <button class="maximize-btn" (click)="maximizeChart(getPCAData()!)" title="Nagyítás">
                      <mat-icon>open_in_full</mat-icon>
                    </button>
                  } @else {
                    <div class="no-data-message">PCA adatok nincsenek rendelkezésre. Futtasd a PCA-t!</div>
                  }
                </div>
              }

              @if (step?.step_def_id === 'save_images') {
                <button class="run-fit-btn" (click)="savePipelineImages()" [disabled]="previewLoading || saveImagesInProgress || isPreviewMode">
                  {{ saveImagesInProgress ? '⏳ Mentés...' : '💾 Képek mentése' }}
                </button>
                @if (saveImagesResultText) {
                  <div class="user-result-item">
                    <span class="user-result-label">Mentés eredménye:</span>
                    <span class="user-result-value">{{ saveImagesResultText }}</span>
                  </div>
                }
              }

              @if (step?.step_def_id === 'save_array') {
                <button class="run-fit-btn" (click)="savePipelineArray()" [disabled]="previewLoading || saveArrayInProgress || isPreviewMode">
                  {{ saveArrayInProgress ? '⏳ Mentés...' : '💾 Adattömb mentése' }}
                </button>
                @if (saveArrayResultText) {
                  <div class="user-result-item">
                    <span class="user-result-label">Mentés eredménye:</span>
                    <span class="user-result-value">{{ saveArrayResultText }}</span>
                  </div>
                }
                @if (getArraySavePreview(); as preview) {
                  @if (preview.source_key) {
                    <div class="user-result-item">
                      <span class="user-result-label">Forrás:</span>
                      <span class="user-result-value">{{ preview.source_key }}</span>
                    </div>
                  }
                  <div class="array-preview-wrap">
                    <div class="array-preview-scroll">
                      <table class="array-preview-table">
                        <thead>
                          <tr>
                            @for (h of preview.headers; track $index) {
                              <th>{{ h }}</th>
                            }
                          </tr>
                        </thead>
                        <tbody>
                          @for (row of preview.rows; track $index) {
                            <tr>
                              @for (cell of row; track $index) {
                                <td>{{ cell }}</td>
                              }
                            </tr>
                          }
                        </tbody>
                      </table>
                    </div>
                    <div class="array-preview-meta">
                      Előnézet: {{ preview.rows?.length || 0 }} sor × {{ preview.headers?.length || 0 }} oszlop
                      @if (preview.total_rows && preview.total_cols) {
                        <span>(teljes: {{ preview.total_rows }} × {{ preview.total_cols }})</span>
                      }
                    </div>
                  </div>
                }
              }

              @if (step?.step_def_id === 'predict_node' && getPredictions()?.length) {
                @for (pred of getPredictions(); track $index) {
                  @if (pred) {
                    <div class="user-result-item">
                      <span class="user-result-label">Kép {{ $index + 1 }}:</span>
                      <span class="user-result-value">{{ pred.predicted_x != null ? pred.predicted_x.toFixed(4) : '-' }}</span>
                    </div>
                  }
                }
              }
          </div>
        }

        <!-- Developer results (collapsible) -->
        <div class="side-outputs-section">
          <div class="dev-results-header" (click)="devResultsExpanded = !devResultsExpanded">
            <mat-icon class="expand-icon" [class.expanded]="devResultsExpanded">chevron_right</mat-icon>
            <span class="section-label">Fejlesztői eredmények</span>
            <button class="copy-all-btn" (click)="copyAllResults($event)" title="Összes másolása">
              <mat-icon>content_copy</mat-icon>
            </button>
          </div>
          @if (devResultsExpanded) {
            <div class="dev-results-body">
              @if (!hasSideOutputs()) {
                <div class="dev-empty">Nincsenek még fejlesztői eredmények.</div>
              } @else {
                @for (key of sideOutputKeys(); track key) {
                  <div class="side-output-item">
                    <span class="side-key">{{ key }}:</span>
                    <span class="side-value" [title]="formatSideOutput(sideOutputs[key])">{{ formatSideOutput(sideOutputs[key]) }}</span>
                    <button class="copy-row-btn" (click)="copyResult(key)" title="Másolás">
                      <mat-icon>content_copy</mat-icon>
                    </button>
                  </div>
                }
              }
            </div>
          }
        </div>

        @if (copyNotification) {
          <div class="copy-toast">{{ copyNotification }}</div>
        }

        @if (showSaveCalibrationDialog) {
          <div class="img-manager-overlay" (click)="closeSaveCalibrationDialog()">
            <div class="img-manager-dialog calibration-dialog" (click)="$event.stopPropagation()">
              <div class="img-manager-header">
                <span class="img-manager-title">Kalibrációs görbe mentése</span>
                <button class="img-manager-close" (click)="closeSaveCalibrationDialog()"><mat-icon>close</mat-icon></button>
              </div>
              <div class="param-row">
                <label class="param-label">Egyenlet</label>
                <div class="calibration-eq-box">{{ pendingCalibrationEquation }}</div>
              </div>
              <div class="param-row">
                <label class="param-label">Y paraméter</label>
                <div class="calibration-eq-box">{{ getYKeyLabel(pendingCalibrationYKey) }}</div>
              </div>
              <div class="param-row">
                <label class="param-label" for="calibration-name">Név</label>
                <input id="calibration-name" class="gen-input" type="text" [(ngModel)]="pendingCalibrationName" />
              </div>
              <div class="param-row">
                <label class="param-label" for="calibration-comment">Megjegyzés</label>
                <textarea id="calibration-comment" class="calibration-comment" [(ngModel)]="pendingCalibrationComment"></textarea>
              </div>
              <div class="img-manager-footer">
                <button class="img-manager-apply" (click)="saveCurrentCalibration()" [disabled]="savingCalibration">
                  {{ savingCalibration ? 'Mentés...' : 'Mentés' }}
                </button>
              </div>
            </div>
          </div>
        }

        @if (showCalibrationBrowser) {
          <div class="img-manager-overlay" (click)="closeCalibrationBrowser()">
            <div class="img-manager-dialog" (click)="$event.stopPropagation()">
              <div class="img-manager-header">
                <span class="img-manager-title">Kalibrációk</span>
                <button class="img-manager-close" (click)="closeCalibrationBrowser()"><mat-icon>close</mat-icon></button>
              </div>
              @if (calibrationRecords.length === 0) {
                <p class="img-manager-empty">Nincs mentett kalibráció.</p>
              } @else {
                <div class="img-manager-list">
                  @for (cal of calibrationRecords; track cal.id) {
                    <div class="img-manager-item" [class.selected]="selectedCalibrationId === cal.id" (click)="selectedCalibrationId = cal.id">
                      <div class="cal-list-main">
                        <div class="cal-list-name">{{ cal.name }}</div>
                        <div class="cal-list-eq">{{ cal.equation }}</div>
                        @if (cal.y_key) {
                          <div class="cal-list-comment">Y: {{ getYKeyLabel(cal.y_key) }}</div>
                        }
                        @if (cal.comment) {
                          <div class="cal-list-comment">{{ cal.comment }}</div>
                        }
                      </div>
                    </div>
                  }
                </div>
                <div class="img-manager-footer">
                  <button class="img-manager-apply" (click)="applySelectedCalibration()" [disabled]="!selectedCalibrationId">Kiválaszt</button>
                </div>
              }
            </div>
          </div>
        }
      }
    </div>
  `,
  styles: [`
    :host {
      display: block;
      height: 100%;
      overflow-y: auto;
      scrollbar-width: thin;
      scrollbar-color: #444 #1a1a1a;
    }
    :host::-webkit-scrollbar {
      width: 10px;
      height: 10px;
    }
    :host::-webkit-scrollbar-track {
      background: #1a1a1a;
      border-radius: 8px;
    }
    :host::-webkit-scrollbar-thumb {
      background: #444;
      border-radius: 8px;
      border: 2px solid #1a1a1a;
    }
    :host::-webkit-scrollbar-thumb:hover {
      background: #5a5a5a;
    }

    .inspector-wrapper {
      padding: 12px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .no-selection {
      color: #7a7a7a;
      font-size: 12px;
      text-align: center;
      padding: 40px 16px;
    }

    .step-header {
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 10px 12px;
      background: linear-gradient(180deg, #2d3136 0%, #26292d 100%);
      border: 1px solid #3d434c;
      border-radius: 10px;
    }
    .step-icon {
      font-size: 18px;
      width: 18px;
      height: 18px;
      color: #b7c9dc;
    }
    .step-name {
      font-size: 14px;
      font-weight: 600;
      color: #eef2f7;
      line-height: 1.2;
    }

    .step-name-btn {
      display: inline-flex;
      align-items: center;
      gap: 4px;
      margin: 0;
      padding: 0;
      border: 0;
      background: transparent;
      color: inherit;
      cursor: default;
      font: inherit;
    }

    .step-name-btn.has-help {
      cursor: pointer;
    }

    .step-name-btn.has-help:hover .step-name {
      color: #cfe4fb;
    }

    .step-name-btn:disabled {
      opacity: 1;
    }

    .step-help-chevron {
      font-size: 18px;
      width: 18px;
      height: 18px;
      color: #9fb3c7;
      transition: transform 0.15s ease;
    }

    .step-name-btn.expanded .step-help-chevron {
      transform: rotate(90deg);
    }

    .node-help-card {
      padding: 10px;
      border: 1px solid #3a4b60;
      border-radius: 10px;
      background: linear-gradient(180deg, #1f262e 0%, #1b2229 100%);
      display: flex;
      flex-direction: column;
      gap: 10px;
    }

    .node-help-section {
      display: flex;
      flex-direction: column;
      gap: 6px;
    }

    .node-help-title {
      font-size: 11px;
      font-weight: 700;
      color: #bcd0e5;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }

    .node-help-text {
      margin: 0;
      color: #d5dee8;
      font-size: 12px;
      line-height: 1.45;
    }

    .node-param-block {
      display: flex;
      flex-direction: column;
      gap: 3px;
      padding: 6px 0;
    }

    .node-param-name {
      color: #e9edf2;
      font-weight: 600;
      font-size: 12px;
    }

    .node-param-desc {
      color: #c9d1db;
      font-size: 12px;
      line-height: 1.4;
    }

    .node-rule-row {
      display: flex;
      gap: 6px;
      align-items: flex-start;
      font-size: 12px;
      line-height: 1.4;
    }

    .node-rule-label {
      color: #e9edf2;
      font-weight: 600;
      min-width: 46px;
      flex-shrink: 0;
    }

    .node-rule-text {
      color: #c9d1db;
    }

    .node-rule-list {
      display: flex;
      flex-direction: column;
      gap: 4px;
      margin-top: 2px;
    }

    .node-rule-item {
      color: #d6dee9;
      font-size: 12px;
      line-height: 1.35;
    }

    .error-list { display: flex; flex-direction: column; gap: 6px; }
    .error-item {
      font-size: 11px;
      color: #f87171;
      padding: 8px 10px;
      background: rgba(127, 29, 29, 0.28);
      border: 1px solid rgba(248, 113, 113, 0.22);
      border-radius: 8px;
    }

    .params-section {
      display: flex;
      flex-direction: column;
      gap: 10px;
      margin: 0;
      padding: 0;
      border: 0;
      min-inline-size: 0;
    }

    .params-section.preview-locked {
      opacity: 0.75;
    }

    .params-section.preview-locked .param-row,
    .params-section.preview-locked .generator-group,
    .params-section.preview-locked .group-colors-section {
      filter: saturate(0.55);
    }

    .roi-shape-selector {
      display: flex;
      gap: 6px;
    }
    .roi-shape-btn {
      flex: 1;
      display: flex;
      align-items: center;
      justify-content: center;
      min-height: 36px;
      padding: 8px;
      background: #23262a;
      border: 1px solid #42474f;
      border-radius: 8px;
      color: #9099a5;
      cursor: pointer;
      transition: all 0.15s;
    }
    .roi-shape-btn:hover { background: #333; color: #ccc; border-color: #555; }
    .roi-shape-btn.active { background: #224477; border-color: #3b82f6; color: #fff; }
    .roi-empty-warning {
      font-size: 12px; color: #ef4444; padding: 8px 10px;
      background: rgba(239,68,68,0.1); border: 1px solid rgba(239,68,68,0.3);
      border-radius: 8px; text-align: center;
    }
    .reference-crop-actions {
      display: flex;
      align-items: center;
      gap: 10px;
      margin: 8px 0 12px;
    }
    .reference-crop-toggle {
      border: 1px solid #3b82f6;
      background: rgba(59, 130, 246, 0.12);
      color: #dbeafe;
      border-radius: 6px;
      padding: 7px 10px;
      font-size: 12px;
      cursor: pointer;
    }
    .reference-crop-toggle.active {
      background: #2563eb;
      color: #fff;
    }
    .reference-crop-count {
      font-size: 12px;
      color: #9ca3af;
    }
    .reference-crop-list {
      display: flex;
      flex-direction: column;
      gap: 8px;
      margin: -4px 0 12px;
    }
    .reference-crop-row {
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .reference-crop-index {
      width: 24px;
      height: 24px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      flex: 0 0 auto;
      color: #e5e7eb;
      font-size: 12px;
      font-weight: 700;
      background: #374151;
      border-radius: 999px;
    }
    .reference-crop-name {
      flex: 1;
      min-width: 0;
      height: 30px;
      padding: 0 9px;
      background: #1f2227;
      border: 1px solid #3a3f46;
      border-radius: 6px;
      color: #e5e7eb;
      font-size: 12px;
    }
    .reference-crop-name:focus {
      outline: none;
      border-color: #3b82f6;
    }
    .reference-crop-delete {
      width: 26px;
      height: 26px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      flex: 0 0 auto;
      padding: 0;
      border: 1px solid transparent;
      border-radius: 6px;
      background: transparent;
      color: #9ca3af;
      font-size: 20px;
      line-height: 1;
      cursor: pointer;
    }
    .reference-crop-delete:hover {
      border-color: rgba(239, 68, 68, 0.45);
      background: rgba(239, 68, 68, 0.12);
      color: #f87171;
    }
    .reference-crop-delete:focus-visible {
      outline: 2px solid #ef4444;
      outline-offset: 1px;
    }
    .param-row {
      display: flex;
      flex-direction: column;
      gap: 8px;
      padding: 10px;
      background: linear-gradient(180deg, #26282c 0%, #232427 100%);
      border: 1px solid #373c42;
      border-radius: 10px;
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.02);
    }
    .param-label {
      font-size: 11px;
      font-weight: 600;
      color: #c2c9d3;
      letter-spacing: 0.03em;
      line-height: 1.25;
    }
    .param-control { width: 100%; }

    .label-picker {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
    }

    .label-chip {
      min-width: 0;
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 8px 10px;
      color: #cbd5e1;
      background: #202733;
      border: 1px solid #3a4658;
      border-radius: 7px;
      font: inherit;
      cursor: pointer;
      transition: border-color 120ms ease, background 120ms ease, color 120ms ease;
    }

    .label-chip:hover {
      border-color: #718096;
      background: #293241;
    }

    .label-chip.active {
      color: #f8fbff;
      background: #174b73;
      border-color: #3b9bdd;
    }

    .label-chip-swatch {
      width: 18px;
      height: 18px;
      display: inline-grid;
      place-items: center;
      flex: 0 0 18px;
      color: #fff;
      border: 1px solid rgba(255, 255, 255, 0.75);
      border-radius: 4px;
      font-size: 11px;
      line-height: 1;
      text-shadow: 0 1px 2px #000;
    }

    .threshold-histogram-wrap {
      margin-top: 8px;
    }

    .color-thresh-histogram {
      margin-top: 8px;
      padding: 8px 0;
      border-top: 1px solid #373c42;
    }

    .slider-control { display: flex; align-items: center; gap: 10px; }
    .slider-control input[type="range"] { flex: 1; accent-color: #224477; }

    .range-slider-control {
      width: 100%;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }

    .range-slider-container {
      position: relative;
      display: flex;
      align-items: center;
      height: 24px;
    }

    .range-slider-min,
    .range-slider-max {
      position: absolute;
      width: 100%;
      height: 6px;
      border-radius: 3px;
      background: none;
      pointer-events: none;
      -webkit-appearance: none;
      appearance: none;
      outline: none;
    }

    .range-slider-min {
      z-index: 4;
    }

    .range-slider-max {
      z-index: 5;
    }

    .range-slider-min::-webkit-slider-thumb,
    .range-slider-max::-webkit-slider-thumb {
      -webkit-appearance: none;
      appearance: none;
      pointer-events: auto;
      width: 16px;
      height: 16px;
      border-radius: 50%;
      background: #3b82f6;
      border: 2px solid #1e40af;
      cursor: pointer;
      box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
    }

    .range-slider-min::-moz-range-thumb,
    .range-slider-max::-moz-range-thumb {
      pointer-events: auto;
      width: 16px;
      height: 16px;
      border-radius: 50%;
      background: #3b82f6;
      border: 2px solid #1e40af;
      cursor: pointer;
      box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
    }

    .range-slider-min::-webkit-slider-thumb:hover,
    .range-slider-max::-webkit-slider-thumb:hover {
      background: #60a5fa;
    }

    .range-slider-values {
      display: flex;
      align-items: center;
      gap: 8px;
      justify-content: center;
    }

    .range-number-input {
      width: 70px;
      padding: 6px;
      background: #1e2023;
      border: 1px solid #474b52;
      border-radius: 6px;
      color: #d4d9e3;
      font-size: 12px;
      text-align: center;
    }

    .range-separator {
      color: #888;
      font-weight: 600;
    }

    .slider-number {
      width: 86px;
      min-height: 36px;
      padding: 7px 9px;
      background: #1e2023;
      border: 1px solid #474b52;
      border-radius: 8px;
      color: #e6e6e6;
      font-size: 13px;
      box-sizing: border-box;
    }

    .inline-unit {
      color: #aeb6c2;
      font-size: 12px;
      min-width: 14px;
      text-align: left;
    }

    .slider-number,
    .gen-input[type="number"],
    .param-control input[type="number"] {
      appearance: textfield;
      -moz-appearance: textfield;
    }

    .slider-number::-webkit-outer-spin-button,
    .slider-number::-webkit-inner-spin-button,
    .gen-input[type="number"]::-webkit-outer-spin-button,
    .gen-input[type="number"]::-webkit-inner-spin-button,
    .param-control input[type="number"]::-webkit-outer-spin-button,
    .param-control input[type="number"]::-webkit-inner-spin-button {
      -webkit-appearance: none;
      margin: 0;
    }

    .slider-number:hover,
    .slider-number:focus,
    .gen-input[type="number"]:hover,
    .gen-input[type="number"]:focus,
    .param-control input[type="number"]:hover,
    .param-control input[type="number"]:focus {
      border-color: #5d7694;
      box-shadow: 0 0 0 1px rgba(59, 130, 246, 0.18);
    }

    .param-control input[type="text"], .param-control select {
      width: 100%;
      min-height: 36px;
      padding: 7px 10px;
      background: #1e2023;
      border: 1px solid #474b52;
      border-radius: 8px;
      color: #e0e0e0;
      font-size: 13px;
      box-sizing: border-box;
    }

    .param-control input[type="number"],
    .gen-input[type="number"] {
      width: 100%;
      min-height: 36px;
      padding: 7px 10px;
      background:
        linear-gradient(180deg, rgba(255, 255, 255, 0.02) 0%, rgba(255, 255, 255, 0) 100%),
        #1e2023;
      border: 1px solid #474b52;
      border-radius: 8px;
      color: #e0e0e0;
      font-size: 13px;
      box-sizing: border-box;
    }
    .param-control select { cursor: pointer; }
    .toggle-wrap {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      cursor: pointer;
      min-height: 36px;
      padding: 0 2px;
    }
    .toggle-label { font-size: 13px; color: #d3d8de; font-weight: 600; }

    .filter-subgroup {
      display: flex;
      flex-direction: column;
      gap: 8px;
      margin-bottom: 8px;
    }

    .filter-subgroup-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }

    .filter-subgroup-title {
      font-size: 12px;
      font-weight: 700;
      color: #d0d8e2;
      letter-spacing: 0.02em;
      line-height: 1.2;
    }

    .sub-param-row {
      display: grid;
      grid-template-columns: 40px minmax(165px, 1fr);
      align-items: center;
      gap: 8px;
      padding-left: 10px;
    }

    .sub-param-row .slider-control {
      gap: 8px;
    }

    .sub-param-row .slider-control input[type="range"] {
      max-width: 130px;
      min-width: 90px;
    }

    .sub-param-row .slider-number {
      width: 64px;
      min-height: 32px;
      padding: 5px 7px;
      font-size: 12px;
    }

    .convexity-checkboxes {
      display: flex;
      gap: 16px;
      padding: 0 2px 0 12px;
    }

    .aggregation-block {
      display: flex;
      flex-direction: column;
      gap: 8px;
      padding: 8px;
      background: #1f2226;
      border: 1px solid #353a40;
      border-radius: 8px;
    }

    .aggregation-radio-grid {
      display: flex;
      flex-direction: column;
      gap: 6px;
    }

    .merge-mode-option {
      display: flex;
      align-items: center;
      gap: 8px;
      color: #d3d8de;
      font-size: 12px;
      cursor: pointer;
      padding: 4px 6px;
      border-radius: 6px;
      transition: background 0.12s;
    }
    .merge-mode-option:hover { background: rgba(59, 130, 246, 0.08); }

    .merge-mode-option input[type="radio"] {
      appearance: none;
      -webkit-appearance: none;
      width: 16px;
      height: 16px;
      border: 2px solid #474b52;
      border-radius: 50%;
      background: #1e2023;
      cursor: pointer;
      position: relative;
      flex-shrink: 0;
      transition: border-color 0.15s, background 0.15s;
    }
    .merge-mode-option input[type="radio"]:checked {
      border-color: #3b82f6;
      background: #1e2023;
    }
    .merge-mode-option input[type="radio"]:checked::after {
      content: '';
      position: absolute;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: #3b82f6;
    }
    .merge-mode-option input[type="radio"]:hover {
      border-color: #6ba1f7;
    }

    .merge-mode-option.with-inline-select select {
      margin-left: auto;
      max-width: 130px;
      min-height: 28px;
      background: #1e2023;
      border: 1px solid #474b52;
      border-radius: 6px;
      color: #e0e0e0;
      padding: 4px 6px;
    }

    .side-outputs-section,
    .user-results-section {
      padding: 12px;
      background: linear-gradient(180deg, #222427 0%, #1f2023 100%);
      border: 1px solid #32363b;
      border-radius: 10px;
    }

    .section-label {
      font-size: 11px; font-weight: 600; color: #9ba6b2;
      text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 8px;
    }

    .side-output-item { display: flex; align-items: flex-start; gap: 6px; font-size: 12px; margin-bottom: 4px; min-width: 0; }
    .side-key { color: #888; flex-shrink: 0; }
    .side-value { color: #e0e0e0; font-variant-numeric: tabular-nums; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; min-width: 0; flex: 1; }

    .dev-results-header {
      display: flex; align-items: center; gap: 4px; cursor: pointer; user-select: none;
    }
    .dev-results-header .section-label { margin-bottom: 0; flex: 1; }
    .expand-icon { font-size: 18px; width: 18px; height: 18px; color: #888; transition: transform 0.15s ease; }
    .expand-icon.expanded { transform: rotate(90deg); }
    .dev-results-body { margin-top: 8px; }
    .dev-empty {
      color: #8a8f97;
      font-size: 12px;
      padding: 6px 2px;
    }

    /* Copy buttons */
    .copy-all-btn, .copy-row-btn {
      background: none;
      border: 1px solid transparent;
      border-radius: 3px;
      color: #666;
      cursor: pointer;
      padding: 2px;
      display: flex;
      align-items: center;
      flex-shrink: 0;
    }
    .copy-all-btn mat-icon, .copy-row-btn mat-icon { font-size: 14px; width: 14px; height: 14px; }
    .copy-all-btn:hover, .copy-row-btn:hover { color: #ccc; border-color: #555; }

    .copy-toast {
      position: fixed; bottom: 24px; left: 50%; transform: translateX(-50%);
      background: #333; color: #e0e0e0; padding: 6px 16px; border-radius: 6px;
      font-size: 12px; z-index: 9999; pointer-events: none;
    }

    .user-result-item { display: flex; gap: 6px; font-size: 12px; margin-bottom: 4px; }
    .user-result-label { color: #aaa; }
    .user-result-value { color: #e0e0e0; font-weight: 600; font-variant-numeric: tabular-nums; }

    .stats-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 4px 12px; }
    .stat-item { display: flex; justify-content: space-between; font-size: 11px; padding: 2px 0; }
    .stat-label { color: #888; }
    .stat-value { color: #e0e0e0; font-variant-numeric: tabular-nums; }

    .gray-map-panel {
      display: flex;
      flex-direction: column;
      gap: 8px;
    }

    .gray-map-toolbar {
      display: flex;
      flex-direction: column;
      gap: 6px;
    }

    .gray-map-toolbar-row {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 8px;
    }

    .gray-map-toolbar-label {
      font-size: 11px;
      color: #9ba6b2;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }

    .gray-map-select {
      min-width: 220px;
      padding: 7px 10px;
      border-radius: 6px;
      border: 1px solid #3a414b;
      background: #14161a;
      color: #e0e0e0;
      font-size: 12px;
    }

    .gray-map-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 10px;
    }

    .gray-map-card {
      display: flex;
      flex-direction: column;
      gap: 6px;
      padding: 8px;
      background: #1d1f22;
      border: 1px solid #333842;
      border-radius: 8px;
      cursor: pointer;
      transition: border-color 120ms ease, transform 120ms ease, box-shadow 120ms ease;
    }

    .gray-map-card:hover {
      border-color: #556170;
      transform: translateY(-1px);
    }

    .gray-map-card.selected {
      border-color: #3b82f6;
      box-shadow: 0 0 0 1px rgba(59, 130, 246, 0.25) inset;
    }

    .gray-map-title {
      font-size: 11px;
      color: #9ba6b2;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }

    .gray-map-count {
      margin-left: 6px;
      color: #6f8aa6;
      font-weight: 500;
      letter-spacing: 0;
      text-transform: none;
    }

    .gray-map-image {
      width: 100%;
      height: auto;
      display: block;
      border-radius: 4px;
      background: #0f1114;
      image-rendering: pixelated;
    }

    .gray-map-summary {
      margin-top: 8px;
      font-size: 11px;
      color: #aab3be;
      font-variant-numeric: tabular-nums;
    }

    .jet-colorbar-wrap {
      display: flex;
      flex-direction: column;
      gap: 3px;
      margin-top: 4px;
    }

    .jet-colorbar-bar {
      width: 100%;
      height: 13px;
      border-radius: 4px;
      border: 1px solid #2a2e36;
      background: linear-gradient(
        to right,
        #000080 0%,
        #0000ff 12%,
        #00ffff 37%,
        #00ff00 50%,
        #ffff00 62%,
        #ff8000 75%,
        #ff0000 88%,
        #800000 100%
      );
    }

    .jet-colorbar-ticks {
      display: flex;
      justify-content: space-between;
      font-size: 9px;
      color: #7a8590;
      font-variant-numeric: tabular-nums;
      padding: 0 1px;
    }

    .dual-map-panel {
      display: flex;
      flex-direction: column;
      gap: 16px;
    }

    .dual-map-section {
      border: 1px solid #2c3440;
      border-radius: 8px;
      padding: 10px 10px 8px 10px;
      background: #15181c;
    }

    .dual-map-section--sub {
      border-color: #3a4a2a;
      background: #141a12;
    }

    .dual-map-section-title {
      font-size: 11px;
      font-weight: 700;
      letter-spacing: .06em;
      text-transform: uppercase;
      color: #8fa8c8;
      margin-bottom: 8px;
    }

    .file-path-control { display: flex; gap: 4px; align-items: center; }
    .chart-with-maximize { position: relative; }

    .run-fit-btn {
      width: 100%; padding: 8px 12px; margin-bottom: 10px;
      background: #224477; border: 1px solid #336699; border-radius: 6px;
      color: #e0e0e0; cursor: pointer; font-size: 12px; font-weight: 600; text-align: center;
    }
    .run-fit-btn:hover:not(:disabled) { background: #1f5ba8; border-color: #3b82f6; }
    .run-fit-btn:disabled { opacity: 0.5; cursor: default; }

    .array-preview-wrap {
      margin-top: 8px;
      border: 1px solid #3a3f46;
      border-radius: 6px;
      background: #1d1f22;
      overflow: hidden;
    }

    .array-preview-scroll {
      max-width: 100%;
      max-height: 220px;
      overflow: auto;
    }

    .array-preview-table {
      border-collapse: collapse;
      width: max-content;
      min-width: 100%;
      font-size: 11px;
      color: #dde2ea;
    }

    .array-preview-table th,
    .array-preview-table td {
      border: 1px solid #363b42;
      padding: 4px 6px;
      white-space: nowrap;
      font-variant-numeric: tabular-nums;
    }

    .array-preview-table th {
      position: sticky;
      top: 0;
      background: #2a2d33;
      z-index: 1;
      font-weight: 600;
    }

    .array-preview-meta {
      font-size: 11px;
      color: #aab3bf;
      padding: 6px 8px;
      border-top: 1px solid #363b42;
    }

    .maximize-btn {
      position: absolute; top: 2px; right: 2px;
      background: rgba(40,40,40,0.8); border: 1px solid #555; border-radius: 4px;
      color: #999; cursor: pointer; padding: 2px;
      display: flex; opacity: 0; transition: opacity 0.15s;
    }
    .chart-with-maximize:hover .maximize-btn { opacity: 1; }
    .maximize-btn:hover { color: #fff; background: #3b82f6; border-color: #3b82f6; }
    .maximize-btn mat-icon { font-size: 16px; width: 16px; height: 16px; }

    /* Omitted datapoints */
    .omitted-section {
      margin-top: 8px;
      padding: 8px;
      background: #1e1e1e;
      border-radius: 4px;
      border: 1px solid #333;
    }

    .omitted-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      margin-bottom: 4px;
    }

    .omitted-title {
      font-size: 10px;
      font-weight: 600;
      color: #ef4444;
      text-transform: uppercase;
      min-width: 0;
    }

    .omitted-restore-btn {
      background: #2a2a2a;
      border: 1px solid #555;
      border-radius: 4px;
      color: #d1d5db;
      cursor: pointer;
      font-size: 10px;
      font-weight: 600;
      line-height: 1;
      padding: 4px 8px;
      text-transform: uppercase;
      white-space: nowrap;
    }

    .omitted-restore-btn:hover {
      background: #374151;
      border-color: #9ca3af;
      color: #fff;
    }

    .omitted-item { display: flex; gap: 6px; font-size: 11px; padding: 1px 0; }
    .omitted-idx { color: #ef4444; font-weight: 600; min-width: 24px; }
    .omitted-name { color: #999; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

    .file-path-control input[type="text"] { flex: 1; }

    .browse-btn {
      background: #333; border: 1px solid #555; border-radius: 4px;
      color: #e0e0e0; cursor: pointer; padding: 3px 6px;
      line-height: 1; flex-shrink: 0; display: flex;
    }
    .browse-btn mat-icon { font-size: 16px; width: 16px; height: 16px; }
    .browse-btn:hover { background: #444; border-color: #3b82f6; }

    .group-colors-section {
      margin-top: 2px;
      padding: 10px;
      border: 1px solid #363b42;
      border-radius: 10px;
      background: #202226;
    }
    .group-color-row { display: flex; align-items: center; justify-content: space-between; gap: 8px; margin-bottom: 6px; }
    .group-color-label { color: #bbb; font-size: 11px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

    .group-color-picker { width: 28px; height: 22px; border: 1px solid #555; border-radius: 4px; padding: 0; background: transparent; cursor: pointer; }

    .generator-group {
      margin-top: 2px;
      padding: 10px;
      border: 1px solid #46505d;
      border-radius: 10px;
      background: linear-gradient(180deg, #2b3138 0%, #262b32 100%);
    }

    .generator-group-title {
      font-size: 11px;
      font-weight: 600;
      color: #d0d8e2;
      text-transform: uppercase;
      letter-spacing: 0.03em;
      margin-bottom: 6px;
    }

    .generator-fields {
      display: flex;
      flex-direction: column;
      gap: 6px;
      margin-bottom: 8px;
    }

    .gen-field {
      display: grid;
      grid-template-columns: minmax(88px, auto) minmax(0, 1fr);
      align-items: center;
      gap: 8px;
    }

    .gen-label {
      font-size: 12px;
      color: #c7cfda;
      min-width: 0;
    }

    .gen-input {
      width: 100%;
      min-width: 0;
      min-height: 36px;
      padding: 7px 9px;
      box-sizing: border-box;
      background: #1f2328;
      border: 1px solid #596474;
      border-radius: 8px;
      color: #e0e0e0;
      font-size: 13px;
    }

    /* Image manager button */
    .image-manager-section {
      margin-top: 12px;
    }

    .image-manager-btn {
      width: 100%; padding: 8px 12px;
      background: #333; border: 1px solid #555; border-radius: 6px;
      color: #e0e0e0; cursor: pointer; font-size: 12px; font-weight: 600;
      display: flex; align-items: center; gap: 6px; justify-content: center;
    }
    .image-manager-btn:hover { background: #3b82f6; border-color: #3b82f6; }
    .image-manager-btn:disabled {
      opacity: 0.5;
      cursor: default;
    }

    .img-manager-overlay {
      position: fixed; inset: 0; background: rgba(0,0,0,0.6);
      display: flex; align-items: center; justify-content: center; z-index: 1000;
    }

    .img-manager-dialog { background: #2a2a2a; border: 1px solid #555; border-radius: 8px; padding: 16px; min-width: 340px; max-width: 440px; max-height: 70vh; display: flex; flex-direction: column; }

    .img-manager-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
    .img-manager-title { font-size: 15px; font-weight: 600; color: #e0e0e0; }
    .img-manager-close { background: none; border: none; color: #888; cursor: pointer; padding: 2px 6px; display: flex; }
    .img-manager-close:hover { color: #fff; }
    .img-manager-empty { color: #888; font-size: 12px; text-align: center; padding: 20px; }

    .img-manager-list { flex: 1; overflow-y: auto; max-height: 40vh; border: 1px solid #444; border-radius: 4px; background: #1e1e1e; }

    .img-manager-item {
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 6px 10px;
      cursor: pointer;
      font-size: 12px;
      color: #ccc;
      border-bottom: 1px solid #333;
    }

    .img-manager-item:hover { background: #333; }
    .img-manager-item.selected { background: #224477; color: #fff; }

    .img-idx {
      color: #888;
      min-width: 24px;
      font-variant-numeric: tabular-nums;
    }
    .img-manager-item.selected .img-idx { color: #bfdbfe; }

    .img-name {
      flex: 1;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .img-manager-actions {
      display: flex;
      justify-content: center;
      gap: 6px;
      padding: 10px 0;
    }
    .img-manager-actions button {
      background: #333; border: 1px solid #555; border-radius: 4px;
      color: #e0e0e0; cursor: pointer; padding: 6px 12px;
      display: flex; align-items: center;
    }
    .img-manager-actions button mat-icon { font-size: 20px; width: 20px; height: 20px; }
    .img-manager-actions button:hover:not(:disabled) { background: #444; border-color: #3b82f6; }
    .img-manager-actions button:disabled { opacity: 0.3; cursor: default; }

    .img-manager-footer {
      display: flex; justify-content: flex-end; padding-top: 8px; border-top: 1px solid #444;
    }
    .img-manager-apply {
      padding: 6px 16px; border: 1px solid #224477; border-radius: 4px;
      cursor: pointer; font-size: 12px; font-weight: 600; background: #224477; color: #fff;
    }
    .img-manager-apply:hover { background: #1f4b8f; }

    .calibration-comment {
      width: 100%;
      min-height: 72px;
      padding: 7px 10px;
      background: #1e2023;
      border: 1px solid #474b52;
      border-radius: 8px;
      color: #e0e0e0;
      font-size: 13px;
      font-family: inherit;
      box-sizing: border-box;
      resize: vertical;
    }
    .calibration-comment:focus {
      outline: none;
      border-color: #3b82f6;
    }

    .no-data-message {
      padding: 24px;
      text-align: center;
      color: #999;
      font-size: 14px;
      background: #f5f5f5;
      border-radius: 8px;
    }

    .cluster-components {
      display: flex;
      flex-direction: column;
      gap: 8px;
      margin: 10px 0 14px;
      padding: 10px;
      border: 1px solid #474b52;
      border-radius: 8px;
    }
    .cluster-component-card {
      display: flex;
      flex-direction: column;
      gap: 6px;
      padding: 8px;
      border-left: 4px solid #2e7d32;
      background: #24282a;
      border-radius: 5px;
    }
    .cluster-component-card input[type="text"] { font-weight: 600; }
    .cluster-component-summary { color: #aab3bf; font-size: 12px; }
    .cluster-component-preview {
      display: block;
      width: 100%;
      max-height: 180px;
      object-fit: contain;
      background: #111;
      border-radius: 5px;
    }
    .cluster-component-card.remainder { border-left-color: #f59e0b; }
    .cluster-remainder-button {
      padding: 9px 12px;
      border: 0;
      border-radius: 6px;
      color: #fff;
      background: #b46a08;
      font-weight: 600;
      cursor: pointer;
    }
    .cluster-remainder-button:disabled { opacity: .45; cursor: default; }
  `],
})
export class StepInspectorComponent implements OnInit, OnDestroy {
  nodeHelpExpanded = false;
  isPreviewMode = false;
  private lastSelectedStepInstanceId = '';

  definition: StepDefinition | undefined;
  step: StepInstance | undefined;
  stepErrors: StepError[] = [];
  sideOutputs: Record<string, any> = {};

  // Image manager state
  showImageManager = false;
  loadedImageNames: string[] = [];
  /** Tracks the original index for each position so reordering can be applied. */
  imageOrderIndices: number[] = [];
  selectedImageIdx = 0;

  // Developer results toggle (collapsed by default)
  devResultsExpanded = false;

  // Copy toast
  copyNotification = '';
  private copyTimeout: any;

  // Calibration dialogs
  showSaveCalibrationDialog = false;
  savingCalibration = false;
  pendingCalibrationEquation = '';
  pendingCalibrationName = '';
  pendingCalibrationComment = '';
  pendingCalibrationYKey = '';

  showCalibrationBrowser = false;
  calibrationRecords: CalibrationRecord[] = [];
  selectedCalibrationId = '';

  // Current preview image index (for histogram display)
  previewImageIndex = 0;

  // Omitted datapoints from graph viewer
  omittedEntries: { index: number; name: string }[] = [];
  currentOmittedIndices: Set<number> = new Set();

  // Preview loading state (to disable run button)
  previewLoading = false;
  saveImagesInProgress = false;
  saveImagesResultText = '';
  saveArrayInProgress = false;
  saveArrayResultText = '';

  // Image dimensions for ROI slider limits
  private imgDimsW = 0;
  private imgDimsH = 0;

  // Reference groups and colors (add_sequence_values)
  referenceGroups: Array<{ key: string; label: string; color: string }> = [];

  /** Steps that only output images and have no user-friendly results */
  private readonly IMAGE_ONLY_STEPS = new Set([
    'select_channel', 'apply_threshold', 'apply_blur', 'apply_clahe',
    'normalize_images', 'brightness_contrast', 'gamma_correction',
    'flat_field_correction', 'robust_stretch_gamma', 'advanced_illumin_corr',
    'mask_rect_roi', 'apply_range_mask', 'add_sequence_values',
  ]);

  /** User-friendly labels for intensity stat Y-axis keys */
  private readonly Y_KEY_LABELS: Record<string, string> = {
    mean: 'Átlag (mean)',
    median: 'Medián (median)',
    std: 'Szórás (std)',
    min: 'Minimum (min)',
    max: 'Maximum (max)',
    pixel_count: 'Pixelszám',
    dynamic_range: 'Dinamikus tart. (P95–P5)',
  };

  private selectedIndex = -1;
  private subs: Subscription[] = [];
  private nodeDescriptions: Record<string, NodeHelpContent> = {};
  selectedGrayMapKey = 'soft_membership_jet';
  selectedGrayMapComponentIndex = 0;
  private grayMapSelectionInitialized = false;

  // dual_map state
  selectedDualMapGrayKey = 'soft_membership_jet';
  selectedDualMapGrayCompIdx = 0;
  selectedDualMapRgbKey = 'rgb_soft_membership_jet';
  selectedDualMapRgbCompIdx = 0;
  selectedDualMapSubKey = 'sub_soft_membership_jet';
  selectedDualMapSubCompIdx = 0;
  private dualMapInitialized = false;

  constructor(
    private pipelineState: PipelineStateService,
    private recipeService: RecipeService,
    private http: HttpClient,
  ) {}

  private loadNodeDescriptions(): void {
    this.http.get<Record<string, NodeHelpContent>>('assets/node-descriptions.json').subscribe({
      next: (data) => {
        this.nodeDescriptions = data;
      },
      error: (err) => {
        console.warn('Failed to load node descriptions:', err);
        this.nodeDescriptions = {};
      },
    });
  }

  ngOnInit(): void {
    this.loadNodeDescriptions();
    this.subs.push(
      combineLatest([
        this.pipelineState.pipeline$,
        this.pipelineState.selectedStepIndex$,
        this.pipelineState.toolboxPreviewStepId$,
        this.pipelineState.validationErrors$,
        this.pipelineState.sideOutputs$,
        this.pipelineState.previewImageIndex$,
      ]).subscribe(([pipeline, idx, previewStepId, errors, sideOutputs, imgIdx]) => {
        this.selectedIndex = idx;
        this.previewImageIndex = imgIdx;
        if (previewStepId) {
          this.isPreviewMode = true;
          this.definition = this.pipelineState.getStepDefinition(previewStepId);
          if (this.definition) {
            const defaults: Record<string, any> = {};
            for (const p of this.definition.params) {
              defaults[p.name] = p.default;
            }
            this.step = {
              instance_id: `preview-${this.definition.id}`,
              step_def_id: this.definition.id,
              param_values: defaults,
              order: -1,
            };
          } else {
            this.step = undefined;
          }
          this.lastSelectedStepInstanceId = this.step?.instance_id ?? '';
          this.stepErrors = [];
          this.sideOutputs = {};
          this.loadedImageNames = [];
          this.imageOrderIndices = [];
          this.referenceGroups = [];
          this.saveImagesResultText = '';
          this.saveArrayResultText = '';
          return;
        }

        this.isPreviewMode = false;
        if (idx >= 0 && idx < pipeline.steps.length) {
          this.step = pipeline.steps[idx];
          this.definition = this.pipelineState.getStepDefinition(this.step.step_def_id);
          const currentStepId = this.step.instance_id ?? '';
          if (currentStepId !== this.lastSelectedStepInstanceId) {
            this.nodeHelpExpanded = false;
            this.lastSelectedStepInstanceId = currentStepId;
          }
          this.stepErrors = errors.filter((e) => e.step_index === idx);
          // Clear results if this step has validation errors
          this.sideOutputs = this.stepErrors.length > 0 ? {} : (sideOutputs ?? {});
          if (this.step.step_def_id !== 'save_images') {
            this.saveImagesResultText = '';
          }
          if (this.step.step_def_id !== 'save_array') {
            this.saveArrayResultText = '';
          }
          this.syncFitCurveDefaultsFromContext();
          this.syncPredictNodeDefaultsFromContext();
          // Populate loaded image names from side outputs
          const paths: string[] = Array.isArray(sideOutputs?.['loaded_paths']) ? sideOutputs['loaded_paths'] : [];
          const pathsChanged =
            paths.length !== this.loadedImageNames.length ||
            paths.some((path, index) => path !== this.loadedImageNames[index]);
          if (pathsChanged) {
            this.loadedImageNames = [...paths];
            this.imageOrderIndices = paths.map((_, i) => i);
            this.selectedImageIdx = 0;
          }
        } else {
          this.step = undefined;
          this.definition = undefined;
          this.lastSelectedStepInstanceId = '';
          this.stepErrors = [];
          this.sideOutputs = {};
          this.loadedImageNames = [];
          this.imageOrderIndices = [];
          this.saveImagesResultText = '';
          this.saveArrayResultText = '';
        }

        if (this.step?.step_def_id === 'add_sequence_values') {
          this.refreshReferenceGroupsFromParams(this.step.param_values);
        } else {
          this.referenceGroups = [];
        }

      }),
      this.pipelineState.omittedPoints$.subscribe(({ indices, imageNames }) => {
        this.currentOmittedIndices = new Set(indices);
        this.omittedEntries = Array.from(indices)
          .sort((a, b) => a - b)
          .map(i => ({ index: i, name: imageNames[i] ?? `Kép ${i + 1}` }));
      }),
      this.pipelineState.previewLoading$.subscribe((l) => {
        this.previewLoading = l;
      }),
      this.pipelineState.imageDims$.subscribe((dims) => {
        this.imgDimsW = dims.w;
        this.imgDimsH = dims.h;
      }),
    );
  }

  ngOnDestroy(): void {
    this.subs.forEach((s) => s.unsubscribe());
  }

  getParamValue(paramName: string): any {
    if (!this.step) return undefined;
    const explicit = this.step.param_values?.[paramName];

    if (this.step.step_def_id === 'fit_curve' && (paramName === 'validation_ratio' || paramName === 'split_method')) {
      if (explicit === undefined || explicit === null || explicit === '') {
        return this.getParamDefaultValue(paramName);
      }
    }

    if (explicit !== undefined) return explicit;
    return this.getParamDefaultValue(paramName);
  }

  private getParamDefaultValue(paramName: string): any {
    const param = this.definition?.params.find(p => p.name === paramName);
    return param?.default;
  }

  getSliderMin(param: ParamSchema): number {
    if (this.step?.step_def_id === 'resize_images' && param.name === 'scale') {
      return 0;
    }
    return Number(param.min ?? 0);
  }

  getSliderMax(param: ParamSchema): number {
    if (this.step?.step_def_id === 'resize_images' && param.name === 'scale') {
      return 1;
    }
    if (this.step?.step_def_id === 'detect_particles' && param.name === 'filter_min_area') {
      return 1000;
    }
    if (this.step?.step_def_id === 'detect_particles' && param.name === 'filter_max_area') {
      return 10000;
    }
    // ROI sliders: max based on actual image dimensions
    if (this.step?.step_def_id === 'mask_rect_roi' && this.imgDimsW > 0 && this.imgDimsH > 0) {
      if (param.name === 'roi_x' || param.name === 'roi_width' || param.name === 'roi_cx' || param.name === 'roi_rx') {
        return this.imgDimsW;
      }
      if (param.name === 'roi_y' || param.name === 'roi_height' || param.name === 'roi_cy' || param.name === 'roi_ry') {
        return this.imgDimsH;
      }
    }
    return Number(param.max ?? (param.type === 'float' ? 1 : 100));
  }

  onNumericTextChange(param: ParamSchema, value: any): void {
    if (this.isPreviewMode) return;
    if (value === null || value === undefined || value === '') return;
    const normalized = String(value).trim().replace(',', '.');
    const num = param.type === 'int' ? parseInt(normalized, 10) : parseFloat(normalized);
    if (Number.isNaN(num)) return;
    this.onParamChange(param.name, num);
  }

  onParamChange(paramName: string, value: any): void {
    if (this.isPreviewMode) return;
    if (!this.step) return;
    const updated = { ...this.step.param_values, [paramName]: value };

    // When color space changes, auto-select first valid channel
    if (this.step.step_def_id === 'select_channel' && paramName === 'space') {
      const validChannels = this.CHANNEL_MAP[value] ?? ['GRAY'];
      if (!validChannels.includes(updated['channel'])) {
        updated['channel'] = validChannels[0];
      }
    }

    // Mutual exclusivity: aggregate vs merge_ab_pairs
    if (this.step.step_def_id === 'fit_curve') {
      if (paramName === 'split_enabled' && value) {
        if (updated['validation_ratio'] === undefined || updated['validation_ratio'] === null || updated['validation_ratio'] === '') {
          updated['validation_ratio'] = this.getParamDefaultValue('validation_ratio') ?? 20;
        }
        if (updated['split_method'] === undefined || updated['split_method'] === null || updated['split_method'] === '') {
          updated['split_method'] = this.getParamDefaultValue('split_method') ?? 'random';
        }
      }
      if (paramName === 'aggregate' && value) {
        updated['merge_ab_pairs'] = false;
      } else if (paramName === 'merge_ab_pairs' && value) {
        updated['aggregate'] = false;
      }
    }

    this.pipelineState.updateParams(this.selectedIndex, updated);

    if (this.step.step_def_id === 'add_sequence_values') {
      this.refreshReferenceGroupsFromParams(updated);
    }
  }

  formatFloat(val: any): string {
    if (val == null) return '-';
    return Number(val).toFixed(2);
  }

  hasSideOutputs(): boolean {
    return Object.keys(this.sideOutputs).length > 0;
  }

  sideOutputKeys(): string[] {
    return Object.keys(this.sideOutputs);
  }

  formatSideOutput(value: any): string {
    if (value == null) return '-';
    if (typeof value === 'number') return Number(value).toFixed(4);
    if (typeof value === 'object') return JSON.stringify(value);
    return String(value);
  }

  hasNodeHelp(): boolean {
    return !!this.getNodeHelpContent();
  }

  toggleNodeHelp(): void {
    if (!this.hasNodeHelp()) return;
    this.nodeHelpExpanded = !this.nodeHelpExpanded;
  }

  getNodeHelpContent(): NodeHelpContent | null {
    const stepDefId = this.step?.step_def_id;
    if (!stepDefId || !this.nodeDescriptions[stepDefId]) {
      return null;
    }
    return this.nodeDescriptions[stepDefId];
  }

  // --- File/folder browsing ---

  browseFile(paramName: string): void {
    if (this.isPreviewMode) return;
    this.recipeService.browseFile().subscribe({
      next: (res) => {
        if (res.path) {
          this.onParamChange(paramName, res.path);
        }
      },
    });
  }

  browseFolder(paramName: string): void {
    if (this.isPreviewMode) return;
    this.recipeService.browseFolder().subscribe({
      next: (res) => {
        if (res.path) {
          this.onParamChange(paramName, res.path);
        }
      },
    });
  }

  isReferenceStep(): boolean {
    return this.step?.step_def_id === 'add_sequence_values';
  }

  isRoiStep(): boolean {
    return this.step?.step_def_id === 'mask_rect_roi';
  }

  getClusterMapLabelOptions(): number[] {
    const previousK = this.getPreviousKmeansK();
    return Array.from({ length: Math.max(2, previousK) }, (_, i) => i + 1);
  }

  private getPreviousKmeansK(): number {
    const pipeline = this.pipelineState.getPipeline();
    for (let i = this.selectedIndex - 1; i >= 0; i--) {
      const candidate = pipeline.steps[i];
      if (candidate.step_def_id === 'kmeans_cluster') return Number(candidate.param_values?.['k'] ?? 3);
    }
    return 3;
  }

  isClusterMapLabelSelected(label: number): boolean {
    return String(this.getParamValue('selected_labels') ?? '').split(',').map(v => Number(v.trim())).includes(label);
  }

  getSelectedClusterMapLabelOptions(): number[] {
    const selected = this.getClusterMapLabelOptions().filter(
      label => this.isClusterMapLabelSelected(label),
    );
    return selected.length ? selected : [1];
  }

  getClusterLabelColor(label: number): string {
    const colors = [
      '#ff0000', '#00ff00', '#0000ff', '#ffff00',
      '#ff00ff', '#00ffff', '#ff0080', '#0080ff',
      '#ff8000', '#80ff00', '#8000ff', '#00ff80',
    ];
    return colors[(Math.max(1, label) - 1) % colors.length];
  }

  toggleClusterMapLabel(label: number): void {
    const selected = new Set(String(this.getParamValue('selected_labels') ?? '').split(',').map(v => Number(v.trim())).filter(v => v > 0));
    if (selected.has(label)) {
      if (selected.size === 1) return;
      selected.delete(label);
    } else {
      selected.add(label);
    }
    const selectedLabels = Array.from(selected).sort((a, b) => a - b);
    this.onParamChange('selected_labels', selectedLabels.join(','));
    const referenceLabel = Number(this.getParamValue('reference_label') ?? 1);
    if (!selected.has(referenceLabel)) {
      this.onParamChange('reference_label', String(selectedLabels[0]));
    }
  }

  getAcceptedClusterMaps(): any[] {
    try {
      const parsed = JSON.parse(String(this.getParamValue('accepted_components') ?? '[]'));
      return Array.isArray(parsed) ? parsed : [];
    } catch {
      return [];
    }
  }

  getAcceptedClusterMapImage(componentIndex: number): string | null {
    const rows = this.sideOutputs?.['cluster_map_component_images_base64'];
    if (!Array.isArray(rows) || rows.length === 0) return null;
    const imageIndex = Math.min(Math.max(this.previewImageIndex, 0), rows.length - 1);
    const row = rows[imageIndex];
    const encoded = Array.isArray(row) ? row[componentIndex] : null;
    return typeof encoded === 'string' && encoded
      ? `data:image/png;base64,${encoded}`
      : null;
  }

  updateAcceptedClusterMap(index: number, key: string, value: any): void {
    const components = this.getAcceptedClusterMaps().map((component) => ({ ...component }));
    if (!components[index]) return;
    components[index][key] = key === 'map_multiplier'
      ? Math.max(0, Math.min(1, Number(value) || 0))
      : value;
    this.onParamChange('accepted_components', JSON.stringify(components));
  }

  removeAcceptedClusterMap(index: number): void {
    const components = this.getAcceptedClusterMaps();
    components.splice(index, 1);
    this.onParamChange('accepted_components', JSON.stringify(components));
  }

  calculateClusterMapRemainder(): void {
    if (this.getAcceptedClusterMaps().length < 2) return;
    this.onParamChange('remainder_as_last', true);
  }

  removeClusterMapRemainder(): void {
    this.onParamChange('remainder_as_last', false);
  }

  isReferenceCropStep(): boolean {
    return this.step?.step_def_id === 'reference_crop';
  }

  toggleReferenceCropView(): void {
    this.onParamChange('show_references', !this.getParamValue('show_references'));
  }

  getReferenceCropCount(): number {
    return this.getReferenceCropSquares().length;
  }

  getReferenceCropCountLabel(): string {
    const currentCount = this.getReferenceCropCount();
    const totalCount = this.getReferenceCropTotalCount();
    if (totalCount > currentCount) {
      return `${currentCount} ezen a kepen / ${totalCount} osszesen`;
    }
    return `${currentCount} kijeloles`;
  }

  getReferenceCropRows(): Array<{ key: string; index: number; localIndex: number; imageKey: string; name: string }> {
    const overrides = this.getReferenceCropOverrides();
    const keys = Object.keys(overrides).sort((a, b) => Number(a) - Number(b));
    if (keys.length > 0) {
      const rows: Array<{ key: string; index: number; localIndex: number; imageKey: string; name: string }> = [];
      for (const imageKey of keys) {
        const squares = Array.isArray(overrides[imageKey]) ? overrides[imageKey] : [];
        squares.forEach((sq, localIndex) => {
          const index = rows.length;
          rows.push({
            key: `${imageKey}:${localIndex}`,
            index,
            localIndex,
            imageKey,
            name: String(sq?.name ?? ''),
          });
        });
      }
      return rows;
    }
    return this.getReferenceCropSquares().map((sq, index) => ({
      key: `base:${index}`,
      index,
      localIndex: index,
      imageKey: '',
      name: String(sq?.name ?? ''),
    }));
  }

  onReferenceCropNameChange(
    crop: { index: number; localIndex: number; imageKey: string },
    name: string
  ): void {
    if (!this.step || this.selectedIndex < 0) return;
    const overrides = this.getReferenceCropOverrides();
    let squares: any[];
    if (crop.imageKey) {
      squares = Array.isArray(overrides[crop.imageKey]) ? [...overrides[crop.imageKey]] : [];
      if (crop.localIndex < 0 || crop.localIndex >= squares.length) return;
      squares[crop.localIndex] = { ...squares[crop.localIndex], name };
      overrides[crop.imageKey] = squares;
    } else {
      squares = this.getReferenceCropSquares();
      if (crop.index < 0 || crop.index >= squares.length) return;
      squares[crop.index] = { ...squares[crop.index], name };
    }
    this.pipelineState.updateParams(this.selectedIndex, {
      ...this.step.param_values,
      reference_squares: JSON.stringify(this.getAllReferenceCropSquares(overrides, squares)),
      reference_square_overrides: JSON.stringify(overrides),
    });
  }

  removeReferenceCrop(crop: { index: number; localIndex: number; imageKey: string }): void {
    if (!this.step || this.selectedIndex < 0) return;
    const overrides = this.getReferenceCropOverrides();
    let squares: any[];
    if (crop.imageKey) {
      const imageSquares = Array.isArray(overrides[crop.imageKey]) ? overrides[crop.imageKey] : [];
      if (crop.localIndex < 0 || crop.localIndex >= imageSquares.length) return;
      overrides[crop.imageKey] = imageSquares.filter((_, index) => index !== crop.localIndex);
      squares = this.getReferenceCropSquares();
    } else {
      squares = this.getReferenceCropSquares();
      if (crop.index < 0 || crop.index >= squares.length) return;
      squares = squares.filter((_, index) => index !== crop.index);
    }
    this.pipelineState.updateParams(this.selectedIndex, {
      ...this.step.param_values,
      reference_squares: JSON.stringify(this.getAllReferenceCropSquares(overrides, squares)),
      reference_square_overrides: JSON.stringify(overrides),
    });
  }

  private getReferenceCropSquares(): any[] {
    const overrides = this.getReferenceCropOverrides();
    const current = overrides[String(this.previewImageIndex)];
    if (Array.isArray(current)) return current;
    if (Object.keys(overrides).length > 0) return [];
    const raw = this.getParamValue('reference_squares') ?? '[]';
    try {
      const squares = typeof raw === 'string' ? JSON.parse(raw) : raw;
      return Array.isArray(squares) ? squares : [];
    } catch {
      return [];
    }
  }

  private getReferenceCropTotalCount(): number {
    const overrides = this.getReferenceCropOverrides();
    const keys = Object.keys(overrides);
    if (keys.length === 0) return this.getReferenceCropSquares().length;
    return keys.reduce((sum, key) => {
      const row = overrides[key];
      return sum + (Array.isArray(row) ? row.length : 0);
    }, 0);
  }

  private getReferenceCropOverrides(): Record<string, any[]> {
    const raw = this.getParamValue('reference_square_overrides') ?? '{}';
    try {
      const overrides = typeof raw === 'string' ? JSON.parse(raw) : raw;
      return overrides && typeof overrides === 'object' && !Array.isArray(overrides) ? overrides : {};
    } catch {
      return {};
    }
  }

  private getAllReferenceCropSquares(overrides: Record<string, any[]>, fallbackSquares: any[]): any[] {
    const keys = Object.keys(overrides).sort((a, b) => Number(a) - Number(b));
    if (keys.length === 0) return fallbackSquares;
    return keys.flatMap((key) => Array.isArray(overrides[key]) ? overrides[key] : []);
  }

  isRoiEmpty(): boolean {
    if (!this.isRoiStep()) return false;
    const t = this.normalizeRoiType(this.getParamValue('roi_type'));
    if (t === 'rect') {
      return !(this.getParamValue('roi_width') > 0 && this.getParamValue('roi_height') > 0);
    }
    if (t === 'ellipse') {
      return !(this.getParamValue('roi_rx') > 0 && this.getParamValue('roi_ry') > 0);
    }
    if (t === 'polygon') {
      const raw = this.getParamValue('roi_points') ?? '[]';
      try {
        const pts = typeof raw === 'string' ? JSON.parse(raw) : raw;
        return !Array.isArray(pts) || pts.length < 3;
      } catch {
        return true;
      }
    }
    return true;
  }

  isReferenceValuesParam(paramName: string): boolean {
    return this.isReferenceStep() && paramName === 'values';
  }

  /** Params hidden from the generic loop because they are shown in the generator group */
  private readonly REFERENCE_GENERATOR_PARAMS = new Set(['num_levels', 'start', 'step_val']);

  private readonly ROI_RECT_PARAMS = new Set(['roi_x', 'roi_y', 'roi_width', 'roi_height']);
  private readonly ROI_ELLIPSE_PARAMS = new Set(['roi_cx', 'roi_cy', 'roi_rx', 'roi_ry']);
  private readonly ROI_POLYGON_PARAMS = new Set(['roi_points']);
  private readonly DETECT_FILTER_PARAMS = new Set([
    'filter_by_area',
    'filter_min_area',
    'filter_max_area',
    'filter_by_circularity',
    'filter_min_circularity',
    'filter_max_circularity',
    'filter_by_convexity',
    'filter_convex',
    'filter_concave',
  ]);

  private normalizeRoiType(value: any): 'rect' | 'ellipse' | 'polygon' {
    if (value === 'circle') return 'ellipse';
    if (value === 'ellipse' || value === 'polygon' || value === 'rect') return value;
    return 'rect';
  }

  private isReferenceSliderParam(paramName: string): boolean {
    return this.REFERENCE_GENERATOR_PARAMS.has(paramName);
  }

  /** Returns params in display order, filtering out hidden ones */
  getVisibleParams(): ParamSchema[] {
    if (!this.definition) return [];
    const params = this.definition.params;
    if (this.step?.step_def_id === 'add_sequence_values') {
      return params.filter(p => !this.REFERENCE_GENERATOR_PARAMS.has(p.name));
    }
    if (this.step?.step_def_id === 'mask_rect_roi') {
      const roiType = this.normalizeRoiType(this.getParamValue('roi_type'));
      const applyMask = this.getParamValue('apply_mask') ?? true;
      const shapeOnly = this.getParamValue('shape_only') ?? false;
      return params.filter(p => {
        if (p.name === 'roi_type') return false; // shown as shape buttons
        if (p.name === 'roi_overrides') return false; // managed automatically
        if (p.name === 'shape_outline_color' || p.name === 'shape_outline_thickness') return false;
        if (shapeOnly && (p.name === 'apply_mask' || p.name === 'background_color' || p.name === 'invert_mask')) return false;
        if (!applyMask && (p.name === 'background_color' || p.name === 'invert_mask')) return false;
        // roi_angle is now supported for all ROI types including polygon
        if (roiType === 'rect') return !this.ROI_ELLIPSE_PARAMS.has(p.name) && !this.ROI_POLYGON_PARAMS.has(p.name);
        if (roiType === 'ellipse') return !this.ROI_RECT_PARAMS.has(p.name) && !this.ROI_POLYGON_PARAMS.has(p.name);
        if (roiType === 'polygon') return !this.ROI_RECT_PARAMS.has(p.name) && !this.ROI_ELLIPSE_PARAMS.has(p.name);
        return true;
      });
    }
    if (this.step?.step_def_id === 'reference_crop') {
      return params.filter(p => p.name !== 'reference_squares' && p.name !== 'reference_square_overrides' && p.name !== 'show_references');
    }
    return params;
  }

  getParamByName(name: string): ParamSchema | undefined {
    return this.definition?.params.find(p => p.name === name);
  }

  shouldHideParam(param: ParamSchema): boolean {
    if (this.step?.step_def_id === 'save_images') {
      return param.name === 'output_folder' || param.name === 'name_prefix' || param.name === 'name_suffix';
    }
    if (this.step?.step_def_id === 'save_array') {
      return param.name === 'output_folder' || param.name === 'filename';
    }

    if (this.step?.step_def_id === 'predict_node' && param.name === 'fit_index') return true;

    if (this.step?.step_def_id === 'reference_crop') {
      return param.name === 'reference_squares' || param.name === 'reference_square_overrides' || param.name === 'show_references';
    }

    if (this.step?.step_def_id === 'reference_color_align') {
      const usesBranchReference = this.getParamValue('reference_branch') !== 'auto';
      if (param.name === 'output_dark' || param.name === 'output_light') return true;
      if (!usesBranchReference && param.name === 'mode') return true;
    }

    if (this.step?.step_def_id === 'cluster_reference_map') {
      return param.name === 'accepted_components'
        || param.name === 'remainder_as_last'
        || param.name === 'remainder_name'
        || param.name === 'remainder_display_multiplier'
        || param.name === 'remainder_invert';
    }

    if (this.step?.step_def_id === 'color_thresh') {
      if (param.name === 'space') return true;
      // Hide all _min parameters (they're combined in range slider at _max)
      if (param.name.endsWith('_min')) return true;
      const channelParams = this.getColorThreshVisibleParams();
      return !channelParams.has(param.name);
    }

    if (this.step?.step_def_id === 'detect_particles') {
      if (this.DETECT_FILTER_PARAMS.has(param.name)) return true;
      return false;
    }

    if (this.step?.step_def_id !== 'fit_curve') return false;
    if (param.name === 'x_name') return true;
    if (param.name === 'agg_method') return true;
    if (param.name === 'merge_ab_pairs') return true;
    if (param.name === 'degree' && this.getParamValue('model') !== 'poly') return true;
    if (!this.getParamValue('split_enabled')) {
      if (param.name === 'validation_ratio') return true;
      if (param.name === 'split_method') return true;
    }
    return false;
  }

  getDisplayParamLabel(param: ParamSchema): string {
    if (this.step?.step_def_id === 'predict_node' && param.name === 'y_name') return 'Bemeneti Y mező';
    if (this.step?.step_def_id !== 'fit_curve') return param.label;
    if (param.name === 'y_name') return 'Y tengely értékei';
    if (param.name === 'model') return 'Illesztett görbe';
    if (param.name === 'aggregate') return 'Adatösszevonás';
    return param.label;
  }

  isFitCurveDataMergeParam(paramName: string): boolean {
    return this.step?.step_def_id === 'fit_curve' && paramName === 'aggregate';
  }

  isFitCurveYAxisParam(paramName: string): boolean {
    return this.step?.step_def_id === 'fit_curve' && paramName === 'y_name';
  }

  isFitCurveValidationRatioParam(paramName: string): boolean {
    return this.step?.step_def_id === 'fit_curve' && paramName === 'validation_ratio';
  }

  isPredictYAxisParam(paramName: string): boolean {
    return this.step?.step_def_id === 'predict_node' && paramName === 'y_name';
  }

  isKmeansReferenceSourceParam(paramName: string): boolean {
    return this.step?.step_def_id === 'kmeans_cluster' && paramName === 'reference_source';
  }

  isPredictEquationParam(paramName: string): boolean {
    return this.step?.step_def_id === 'predict_node' && paramName === 'equation';
  }

  getKmeansReferenceSourceOptions(): Array<{ value: string; label: string }> {
    const options: Array<{ value: string; label: string }> = [
      { value: 'auto', label: 'Auto - aktualis referencia a pipeline-bol' },
    ];
    const pipeline = this.pipelineState.getPipeline();
    const stop = this.selectedIndex >= 0 ? this.selectedIndex : pipeline.steps.length;
    for (let i = 0; i < stop; i++) {
      const step = pipeline.steps[i];
      if (step.step_def_id !== 'reference_crop' && step.step_def_id !== 'reference_sequence') continue;
      const label = step.step_def_id === 'reference_sequence'
        ? `Reference sequence #${i + 1}`
        : `Reference crop #${i + 1}`;
      options.push({ value: step.instance_id, label });
    }
    const current = String(this.step?.param_values?.['reference_source'] ?? '').trim();
    if (current && current !== 'auto' && !options.some((opt) => opt.value === current)) {
      options.push({ value: current, label: `Ismeretlen forras (${current.slice(0, 8)})` });
    }
    return options;
  }

  getFitCurveAggMethodOptions(): string[] {
    const param = this.getParamByName('agg_method');
    return param?.options ?? ['mean', 'median'];
  }

  getFitCurveAggMethodDisplayLabel(option: string): string {
    const map: Record<string, string> = {
      mean: 'Átlag',
      median: 'Medián',
    };
    return map[option] ?? option;
  }

  getFitCurveMergeMode(): 'none' | 'tablet' | 'level' {
    if (this.getParamValue('aggregate')) return 'level';
    if (this.getParamValue('merge_ab_pairs')) return 'tablet';
    return 'none';
  }

  setFitCurveMergeMode(mode: 'none' | 'tablet' | 'level'): void {
    if (this.isPreviewMode) return;
    if (!this.step || this.step.step_def_id !== 'fit_curve') return;
    const updated = { ...this.step.param_values };
    if (mode === 'level') {
      updated['aggregate'] = true;
      updated['merge_ab_pairs'] = false;
    } else if (mode === 'tablet') {
      updated['aggregate'] = false;
      updated['merge_ab_pairs'] = true;
    } else {
      updated['aggregate'] = false;
      updated['merge_ab_pairs'] = false;
    }
    this.pipelineState.updateParams(this.selectedIndex, updated);
  }

  getPredictYOptions(): string[] {
    const defaults = this.getParamByName('y_name')?.options ?? ['mean', 'median', 'std', 'min', 'max'];
    const options = this.getFitCurveYOptions(defaults);
    const current = String(this.getParamValue('y_name') ?? '').trim();
    if (current && !options.includes(current)) {
      options.push(current);
    }
    return options;
  }

  getYKeyLabel(key: string): string {
    if (this.Y_KEY_LABELS[key]) return this.Y_KEY_LABELS[key];
    // Percentile keys like p5, p25, p50, p75, p95
    const pMatch = key.match(/^p(\d+(?:\.\d+)?)$/i);
    if (pMatch) return `P${pMatch[1]} percentilis`;
    return key;
  }

  isBoolParamDisabled(paramName: string): boolean {
    return false;
  }

  // --- Image manager ---

  isLoadImageStep(): boolean {
    return this.step?.step_def_id === 'load_image';
  }

  isSaveImagesStep(): boolean {
    return this.step?.step_def_id === 'save_images';
  }

  isSaveArrayStep(): boolean {
    return this.step?.step_def_id === 'save_array';
  }

  getSaveNamePreview(): string {
    const original = this.getCurrentLoadedImageName();
    const extIdx = original.lastIndexOf('.');
    const stem = extIdx > 0 ? original.slice(0, extIdx) : original;
    const ext = extIdx > 0 ? original.slice(extIdx) : '.png';
    const prefix = String(this.getParamValue('name_prefix') ?? '');
    const suffix = String(this.getParamValue('name_suffix') ?? '');
    return `${prefix}${stem}${suffix}${ext}`;
  }

  private getCurrentLoadedImageName(): string {
    if (this.loadedImageNames.length > 0) {
      const idx = Math.max(0, Math.min(this.previewImageIndex, this.loadedImageNames.length - 1));
      return this.loadedImageNames[idx];
    }
    return 'image_001.png';
  }

  getArraySavePreview(): any | null {
    const preview = this.sideOutputs['array_save_preview'];
    if (!preview || typeof preview !== 'object') return null;
    return preview;
  }

  moveImage(direction: 'up' | 'down' | 'top' | 'bottom'): void {
    if (this.isPreviewMode) return;
    const idx = this.selectedImageIdx;
    const arr = this.loadedImageNames;
    const orderArr = this.imageOrderIndices;
    if (arr.length < 2) return;

    let newIdx = idx;
    switch (direction) {
      case 'top':
        newIdx = 0;
        break;
      case 'up':
        newIdx = idx - 1;
        break;
      case 'down':
        newIdx = idx + 1;
        break;
      case 'bottom':
        newIdx = arr.length - 1;
        break;
    }
    if (newIdx < 0 || newIdx >= arr.length || newIdx === idx) return;

    // Swap in names
    const tmpName = arr[idx];
    arr[idx] = arr[newIdx];
    arr[newIdx] = tmpName;

    // Swap in order indices
    const tmpOrder = orderArr[idx];
    orderArr[idx] = orderArr[newIdx];
    orderArr[newIdx] = tmpOrder;

    this.selectedImageIdx = newIdx;
  }

  applyImageOrder(): void {
    if (this.isPreviewMode) return;
    this.pipelineState.reorderLoadedImages(this.imageOrderIndices);
    this.showImageManager = false;
  }

  // --- User-friendly results ---

  hasUserFriendlyResults(): boolean {
    if (this.isPreviewMode) return false;
    if (!this.step) return false;
    if (this.IMAGE_ONLY_STEPS.has(this.step.step_def_id)) return false;
    const id = this.step.step_def_id;
    if (id === 'load_image') return this.getLoadedImageCount() !== '-';
    if (id === 'calculate_histograms') return !!this.getHistogramData();
    if (id === 'histogram_equalization') return !!this.getHisteqInputData() || !!this.getHisteqOutputData();
    if (id === 'calculate_intensity_stats') return this.getIntensityStatsEntries().length > 0;
    if (id === 'fit_curve') return true;
    if (id === 'histogram_pca') return true;
    if (id === 'save_images') return true;
    if (id === 'save_array') return true;
    if (id === 'predict_node') return (this.getPredictions()?.length ?? 0) > 0;
    if (id === 'kmeans_cluster') return !!this.getKmeansReferenceInfo() || this.getKmeansClusterEntries().length > 0;
    return false;
  }

  getLoadedImageCount(): number | string {
    if (this.loadedImageNames.length > 0) {
      return this.loadedImageNames.length;
    }

    const imageCount = this.pipelineState.getImageCount();
    if (imageCount > 0) {
      return imageCount;
    }

    const sideOutputCount = Number(this.sideOutputs['image_count']);
    return Number.isFinite(sideOutputCount) && sideOutputCount > 0 ? sideOutputCount : '-';
  }

  getHistogramData(): number[] | null {
    const histograms = this.sideOutputs['histograms'];
    if (!Array.isArray(histograms) || histograms.length === 0) return null;
    const idx = Math.min(this.previewImageIndex, histograms.length - 1);
    const h = histograms[idx];
    return Array.isArray(h) ? h : null;
  }

  getHisteqInputData(): number[] | null {
    const histograms = this.sideOutputs['histeq_input_histograms'];
    if (!Array.isArray(histograms) || histograms.length === 0) return null;
    const idx = Math.min(this.previewImageIndex, histograms.length - 1);
    return Array.isArray(histograms[idx]) ? histograms[idx] : null;
  }

  getHisteqOutputData(): number[] | null {
    const histograms = this.sideOutputs['histeq_output_histograms'];
    if (!Array.isArray(histograms) || histograms.length === 0) return null;
    const idx = Math.min(this.previewImageIndex, histograms.length - 1);
    return Array.isArray(histograms[idx]) ? histograms[idx] : null;
  }

  getIntensityStatsEntries(): { key: string; label: string; value: string }[] {
    const stats = this.sideOutputs['intensity_stats'];
    if (!Array.isArray(stats) || stats.length === 0) return [];
    const idx = Math.min(this.previewImageIndex, stats.length - 1);
    const s = stats[idx];
    if (!s || typeof s !== 'object') return [];
    const values = s as Record<string, any>;

    const baseOrder = ['min', 'max', 'mean', 'median', 'std', 'pixel_count'];
    const baseLabels: Record<string, string> = {
      min: 'Min',
      max: 'Max',
      mean: 'Átlag',
      median: 'Medián',
      std: 'Szórás',
      pixel_count: 'Pixelszám',
      dynamic_range: 'Dinamikus tart.',
    };

    const buildEntriesFromStat = (
      stat: Record<string, any>,
      prefix: string,
      labelPrefix: string,
    ): { key: string; label: string; value: string }[] => {
      const result: { key: string; label: string; value: string }[] = [];
      for (const key of baseOrder) {
        const v = stat[key];
        if (v == null) continue;
        result.push({
          key: prefix + key,
          label: labelPrefix + (baseLabels[key] ?? key),
          value: key === 'pixel_count' ? String(v) : Number(v).toFixed(2),
        });
      }
      const percentileKeys = Object.keys(stat)
        .filter((k) => /^p\d+(?:\.\d+)?$/i.test(k) && stat[k] != null)
        .sort((a, b) => Number(a.slice(1)) - Number(b.slice(1)));
      for (const key of percentileKeys) {
        result.push({ key: prefix + key, label: labelPrefix + key.toUpperCase(), value: Number(stat[key]).toFixed(2) });
      }
      if (stat['dynamic_range'] != null) {
        result.push({ key: prefix + 'dynamic_range', label: labelPrefix + baseLabels['dynamic_range'], value: Number(stat['dynamic_range']).toFixed(2) });
      }
      const known = new Set([...baseOrder, ...percentileKeys, 'dynamic_range']);
      for (const key of Object.keys(stat).filter((k) => !known.has(k) && typeof stat[k] === 'number').sort()) {
        result.push({ key: prefix + key, label: labelPrefix + key.replace(/_/g, ' '), value: Number(stat[key]).toFixed(2) });
      }
      return result;
    };

    // RGB: stat has a "channels" array
    if (Array.isArray(values['channels'])) {
      const channelCount: number = values['channel_count'] ?? values['channels'].length;
      const channelNames: string[] =
        channelCount === 3 ? ['B', 'G', 'R'] :
        channelCount === 4 ? ['B', 'G', 'R', 'A'] :
        (values['channels'] as any[]).map((_, i) => `Ch${i}`);
      const entries: { key: string; label: string; value: string }[] = [];
      (values['channels'] as any[]).forEach((chStat, i) => {
        if (!chStat || typeof chStat !== 'object') return;
        const name = channelNames[i] ?? `Ch${i}`;
        entries.push(...buildEntriesFromStat(chStat as Record<string, any>, `ch${i}_`, `${name} – `));
      });
      return entries;
    }

    // Grayscale: flat stat object
    return buildEntriesFromStat(values, '', '');
  }

  getKmeansReferenceInfo(): {
    mode?: string;
    reference_crops_available: number;
    uses_reference_crops: boolean;
    effective_k: number;
    reference_sequence_used?: boolean;
    reference_source_label?: string;
    reference_source_type?: string;
    reference_sequence?: any;
  } | null {
    const rows = this.sideOutputs['kmeans_reference_info'];
    if (!Array.isArray(rows) || rows.length === 0) return null;
    const idx = Math.min(this.previewImageIndex, rows.length - 1);
    const row = rows[idx];
    if (!row || typeof row !== 'object') return null;
    const info = row as Record<string, any>;
    return {
      mode: String(info['mode'] ?? ''),
      reference_crops_available: Number(info['reference_crops_available'] ?? 0),
      uses_reference_crops: Boolean(info['uses_reference_crops']),
      effective_k: Number(info['effective_k'] ?? 0),
      reference_sequence_used: Boolean(info['reference_sequence_used']),
      reference_source_label: String(info['reference_source_label'] ?? ''),
      reference_source_type: String(info['reference_source_type'] ?? ''),
      reference_sequence: info['reference_sequence'] ?? null,
    };
  }

  getKmeansReferenceSequenceEntries(): { key: string; label: string; value: string }[] {
    const info = this.getKmeansReferenceInfo();
    const sequenceRows = info?.reference_sequence;
    if (!Array.isArray(sequenceRows) || sequenceRows.length === 0) return [];
    const idx = Math.min(this.previewImageIndex, sequenceRows.length - 1);
    const row = sequenceRows[idx];
    const items = row?.items;
    if (!Array.isArray(items)) return [];
    return items.map((item: any, i: number) => {
      const scores = item?.scores && typeof item.scores === 'object' ? item.scores : {};
      const scoreParts = Object.keys(scores)
        .sort()
        .map((key) => `${key}: ${Number(scores[key]).toFixed(1)}`);
      const label = String(item?.label ?? i + 1);
      return {
        key: `ref_seq_${i}`,
        label: `Ref ${i + 1} (${label})`,
        value: scoreParts.length ? scoreParts.join(', ') : Number(item?.score ?? 0).toFixed(1),
      };
    });
  }

  getKmeansClusterEntries(): { key: string; label: string; value: string }[] {
    const countsRows = this.sideOutputs['kmeans_counts'];
    const percentageRows = this.sideOutputs['kmeans_percentages'];
    if (!Array.isArray(countsRows) || countsRows.length === 0) return [];
    const idx = Math.min(this.previewImageIndex, countsRows.length - 1);
    const counts = countsRows[idx];
    const percentages = Array.isArray(percentageRows)
      ? percentageRows[Math.min(this.previewImageIndex, percentageRows.length - 1)]
      : [];
    if (!Array.isArray(counts)) return [];
    return counts.map((count, i) => {
      const pct = Array.isArray(percentages) && percentages[i] != null ? Number(percentages[i]) : null;
      return {
        key: `cluster_${i + 1}`,
        label: `Klaszter ${i + 1}`,
        value: pct == null || Number.isNaN(pct) ? `${count} px` : `${count} px (${pct.toFixed(1)}%)`,
      };
    });
  }

  getLatestCurveFit(): CurveFitData | null {
    const fits = this.sideOutputs['curve_fits'];
    if (!Array.isArray(fits) || fits.length === 0) return null;
    return fits[fits.length - 1] as CurveFitData;
  }

  getPCAData(): PCAData | null {
    const scores = this.sideOutputs['histogram_pca_scores'];
    const explained_ratio = this.sideOutputs['histogram_pca_explained_ratio'];
    const cumulative_ratio = this.sideOutputs['histogram_pca_cumulative_ratio'];

    console.log('[DEBUG PCA]', {
      scores: scores ? `Array(${Array.isArray(scores) ? scores.length : '?'})` : 'undefined',
      explained_ratio: explained_ratio ? `Array(${Array.isArray(explained_ratio) ? explained_ratio.length : '?'})` : 'undefined',
      cumulative_ratio: cumulative_ratio ? `Array(${Array.isArray(cumulative_ratio) ? cumulative_ratio.length : '?'})` : 'undefined',
      allSideOutputKeys: Object.keys(this.sideOutputs).filter(k => k.includes('pca') || k.includes('histogram'))
    });

    if (!Array.isArray(scores) || scores.length === 0) return null;
    if (!Array.isArray(explained_ratio) || explained_ratio.length < 2) return null;
    if (!Array.isArray(cumulative_ratio) || cumulative_ratio.length < 2) return null;

    return {
      scores,
      explained_ratio,
      cumulative_ratio,
    };
  }

  getOmittedForCurrentChart(): Set<number> {
    const fit: any = this.getLatestCurveFit();
    if (fit?.aggregation?.enabled) {
      return new Set<number>();
    }
    return this.currentOmittedIndices;
  }

  restoreAllOmittedPoints(): void {
    if (this.isPreviewMode || this.omittedEntries.length === 0) return;
    this.pipelineState.notifyOmittedPoints(new Set<number>(), []);
  }

  getPredictions(): any[] | null {
    const preds = this.sideOutputs['predictions'];
    if (!Array.isArray(preds)) return null;
    return preds;
  }

  getGrayMapSummary(): string {
    if (this.step?.step_def_id !== 'gray_map') return '';

    const circleInfo = this.sideOutputs['circle_info'];
    const previewIndex = Math.max(0, this.previewImageIndex || 0);
    if (!Array.isArray(circleInfo) || !circleInfo[previewIndex]) return '';

    const info = circleInfo[previewIndex];
    const cx = Number(info?.cx_px);
    const cy = Number(info?.cy_px);
    const radius = Number(info?.r_px);
    if (!Number.isFinite(cx) || !Number.isFinite(cy) || !Number.isFinite(radius)) return '';

    return `Körközép: (${cx.toFixed(1)}, ${cy.toFixed(1)}), sugár: ${radius.toFixed(1)} px`;
  }

  // ── dual_map helpers ──────────────────────────────────────────────────────

  getDualMapGrayOptions(): Array<{ key: string; label: string }> {
    const so = this.sideOutputs;
    const imgIdx = Math.max(0, this.previewImageIndex || 0);
    const opts: Array<{ key: string; label: string }> = [];
    if (this._dualMapKeyHasData(so, 'soft_membership_jet_base64', imgIdx))  opts.push({ key: 'soft_membership_jet',  label: 'Soft membership JET' });
    if (this._dualMapKeyHasData(so, 'component_map_jet_base64', imgIdx))    opts.push({ key: 'component_map_jet',    label: 'Komponens térkép JET' });
    if (this._dualMapKeyHasData(so, 'hard_jet_base64', imgIdx))             opts.push({ key: 'hard_jet',             label: 'Hard térkép JET' });
    if (this._dualMapKeyHasData(so, 'hard_composite_rgb_base64', imgIdx))   opts.push({ key: 'hard_composite_rgb',   label: 'Hard kompozit' });
    return opts;
  }

  getDualMapRgbOptions(): Array<{ key: string; label: string }> {
    const so = this.sideOutputs;
    const imgIdx = Math.max(0, this.previewImageIndex || 0);
    const opts: Array<{ key: string; label: string }> = [];
    if (this._dualMapKeyHasData(so, 'rgb_soft_membership_jet_base64', imgIdx))  opts.push({ key: 'rgb_soft_membership_jet',  label: 'RGB Soft membership JET' });
    if (this._dualMapKeyHasData(so, 'rgb_component_map_jet_base64', imgIdx))    opts.push({ key: 'rgb_component_map_jet',    label: 'RGB Komponens térkép JET' });
    if (this._dualMapKeyHasData(so, 'rgb_hard_jet_base64', imgIdx))             opts.push({ key: 'rgb_hard_jet',             label: 'RGB Hard térkép JET' });
    if (this._dualMapKeyHasData(so, 'rgb_hard_composite_rgb_base64', imgIdx))   opts.push({ key: 'rgb_hard_composite_rgb',   label: 'RGB Hard kompozit' });
    return opts;
  }

  private _dualMapKeyHasData(so: Record<string, any>, b64Key: string, imgIdx: number): boolean {
    const source = so[b64Key];
    if (!Array.isArray(source) || source.length === 0) return false;
    // Clamp: dual_map produces 1 entry per pair, not 1 per loaded image
    const idx = Math.max(0, Math.min(imgIdx, source.length - 1));
    const item = source[idx];
    if (item === null || item === undefined) return false;
    if (Array.isArray(item)) return item.some((x: unknown) => typeof x === 'string' && x.length > 0);
    return typeof item === 'string' && item.length > 0;
  }

  hasDualMapData(): boolean {
    // Check history first (most reliable indicator that dual_map actually ran)
    const history: string[] = this.sideOutputs['history'] ?? [];
    if (history.includes('dual_map_node')) return true;
    // Fallback: check for any output key from either side
    return !!(
      this.sideOutputs['soft_membership_jet_base64'] ||
      this.sideOutputs['rgb_soft_membership_jet_base64'] ||
      this.sideOutputs['hard_composite_rgb_base64'] ||
      this.sideOutputs['gray_source_images_base64']
    );
  }

  getDualMapGrayItems(): Array<{ key: string; familyKey: string; componentIndex?: number; componentCount?: number; label: string; src: string }> {
    if (this.step?.step_def_id !== 'dual_map') return [];
    const imgIdx = Math.max(0, this.previewImageIndex || 0);
    return this._getDualMapItems(this.selectedDualMapGrayKey, imgIdx, 'gray');
  }

  getDualMapRgbItems(): Array<{ key: string; familyKey: string; componentIndex?: number; componentCount?: number; label: string; src: string }> {
    if (this.step?.step_def_id !== 'dual_map') return [];
    const imgIdx = Math.max(0, this.previewImageIndex || 0);
    return this._getDualMapItems(this.selectedDualMapRgbKey, imgIdx, 'rgb');
  }

  private _getDualMapItems(
    familyKey: string,
    imgIdx: number,
    side: 'gray' | 'rgb',
  ): Array<{ key: string; familyKey: string; componentIndex?: number; componentCount?: number; label: string; src: string }> {
    // Map familyKey → base64 output key
    const b64Key = `${familyKey}_base64`;
    const source = this.sideOutputs[b64Key];
    if (!Array.isArray(source) || source.length === 0) return [];

    const idx = Math.max(0, Math.min(imgIdx, source.length - 1));
    const item = source[idx];

    // Per-component array (jet maps)
    if (Array.isArray(item)) {
      return item
        .map((comp: unknown, ci: number) => {
          if (typeof comp !== 'string' || !comp.trim()) return null;
          return {
            key: `${familyKey}:${idx}:${ci}`,
            familyKey,
            componentIndex: ci,
            componentCount: item.length,
            label: `${side === 'gray' ? 'Komponens' : 'RGB Komponens'} ${ci + 1}`,
            src: `data:image/jpeg;base64,${comp}`,
          };
        })
        .filter((x): x is NonNullable<typeof x> => x !== null);
    }

    // Single image
    if (typeof item === 'string' && item.trim()) {
      return [{ key: `${familyKey}:${idx}`, familyKey, label: familyKey, src: `data:image/jpeg;base64,${item}` }];
    }
    return [];
  }

  onDualMapGrayKeyChange(key: string): void {
    this.selectedDualMapGrayKey = key;
    this.selectedDualMapGrayCompIdx = 0;
    this._pushDualMapPreview();
  }

  onDualMapRgbKeyChange(key: string): void {
    this.selectedDualMapRgbKey = key;
    this.selectedDualMapRgbCompIdx = 0;
    this._pushDualMapPreview();
  }

  onDualMapSubKeyChange(key: string): void {
    this.selectedDualMapSubKey = key;
    this.selectedDualMapSubCompIdx = 0;
    this._pushDualMapPreview();
  }

  showDualMapInPreview(side: 'gray' | 'rgb' | 'sub', familyKey: string, compIdx: number): void {
    if (this.isPreviewMode) return;
    if (side === 'gray') {
      this.selectedDualMapGrayKey = familyKey;
      this.selectedDualMapGrayCompIdx = compIdx;
    } else if (side === 'rgb') {
      this.selectedDualMapRgbKey = familyKey;
      this.selectedDualMapRgbCompIdx = compIdx;
    } else {
      this.selectedDualMapSubKey = familyKey;
      this.selectedDualMapSubCompIdx = compIdx;
    }
    this._pushDualMapPreview();
  }

  getDualMapSubOptions(): Array<{ key: string; label: string }> {
    const so = this.sideOutputs;
    const imgIdx = Math.max(0, this.previewImageIndex || 0);
    const opts: Array<{ key: string; label: string }> = [];
    if (this._dualMapKeyHasData(so, 'sub_soft_membership_jet_base64', imgIdx))
      opts.push({ key: 'sub_soft_membership_jet',  label: 'Sub Soft membership JET' });
    if (this._dualMapKeyHasData(so, 'sub_component_map_jet_base64', imgIdx))
      opts.push({ key: 'sub_component_map_jet',    label: 'Sub Komponens térkép JET' });
    if (this._dualMapKeyHasData(so, 'sub_hard_jet_base64', imgIdx))
      opts.push({ key: 'sub_hard_jet',             label: 'Sub Hard térkép JET' });
    if (this._dualMapKeyHasData(so, 'sub_hard_composite_rgb_base64', imgIdx))
      opts.push({ key: 'sub_hard_composite_rgb',   label: 'Sub Hard kompozit' });
    if (this._dualMapKeyHasData(so, 'sub_rgb_soft_membership_jet_base64', imgIdx))
      opts.push({ key: 'sub_rgb_soft_membership_jet',  label: 'Sub RGB Soft membership JET' });
    if (this._dualMapKeyHasData(so, 'sub_rgb_component_map_jet_base64', imgIdx))
      opts.push({ key: 'sub_rgb_component_map_jet',    label: 'Sub RGB Komponens térkép JET' });
    if (this._dualMapKeyHasData(so, 'sub_rgb_hard_jet_base64', imgIdx))
      opts.push({ key: 'sub_rgb_hard_jet',             label: 'Sub RGB Hard térkép JET' });
    if (this._dualMapKeyHasData(so, 'sub_rgb_hard_composite_rgb_base64', imgIdx))
      opts.push({ key: 'sub_rgb_hard_composite_rgb',   label: 'Sub RGB Hard kompozit' });
    return opts;
  }

  hasDualMapSubData(): boolean {
    return this.getDualMapSubOptions().length > 0;
  }

  getDualMapSubItems(): Array<{ key: string; familyKey: string; componentIndex?: number; componentCount?: number; label: string; src: string }> {
    if (this.step?.step_def_id !== 'dual_map') return [];
    const imgIdx = Math.max(0, this.previewImageIndex || 0);
    const side = this.selectedDualMapSubKey.startsWith('sub_rgb_') ? 'rgb' : 'gray';
    return this._getDualMapItems(this.selectedDualMapSubKey, imgIdx, side as 'gray' | 'rgb');
  }

  private _getDualMapSourceImage(sourceKey: string, imgIdx: number): string | null {
    const source = this.sideOutputs[sourceKey];
    if (!Array.isArray(source) || source.length === 0) return null;
    const idx = Math.max(0, Math.min(imgIdx, source.length - 1));
    const item = source[idx];
    if (typeof item === 'string' && item.trim()) return `data:image/jpeg;base64,${item}`;
    return null;
  }

  private _getDualMapOverlay(familyKey: string, compIdx: number, imgIdx: number): string | null {
    const items = this._getDualMapItems(familyKey, imgIdx, familyKey.startsWith('rgb_') ? 'rgb' : 'gray');
    if (!items.length) return null;
    const ci = Math.max(0, Math.min(compIdx, items.length - 1));
    return items[ci]?.src ?? null;
  }

  private _pushDualMapPreview(): void {
    if (this.isPreviewMode) return;
    const imgIdx = Math.max(0, this.previewImageIndex || 0);

    const grayBase = this._getDualMapSourceImage('gray_source_images_base64', imgIdx);
    const rgbBase  = this._getDualMapSourceImage('rgb_source_images_base64', imgIdx);

    const grayItems = this._getDualMapItems(this.selectedDualMapGrayKey, imgIdx, 'gray');
    const rgbItems  = this._getDualMapItems(this.selectedDualMapRgbKey,  imgIdx, 'rgb');
    const grayOverlays = grayItems.map(it => it.src);
    const rgbOverlays  = rgbItems.map(it => it.src);

    // Sub overlays
    let subBase: string | null = null;
    let subOverlays: string[] = [];
    let subLabel = 'Aláosztályozás';
    const subOpts = this.getDualMapSubOptions();
    if (subOpts.length > 0) {
      const isSub = subOpts.some(o => o.key === this.selectedDualMapSubKey);
      const subKey = isSub ? this.selectedDualMapSubKey : subOpts[0].key;
      if (!isSub) this.selectedDualMapSubKey = subKey;
      const subSide = subKey.startsWith('sub_rgb_') ? 'rgb' : 'gray';
      subBase = subSide === 'rgb'
        ? this._getDualMapSourceImage('rgb_source_images_base64', imgIdx)
        : this._getDualMapSourceImage('gray_source_images_base64', imgIdx);
      const subItems = this._getDualMapItems(subKey, imgIdx, subSide as 'gray' | 'rgb');
      subOverlays = subItems.map(it => it.src);
      subLabel = subSide === 'rgb' ? 'Sub RGB' : 'Sub szürke';
    }

    this.pipelineState.setDualMapPreview({ grayBase, grayOverlays, rgbBase, rgbOverlays, subBase, subOverlays, subLabel });
  }

  private syncDualMapPreview(): void {
    if (this.isPreviewMode) return;
    if (this.step?.step_def_id !== 'dual_map') return;
    if (!this.hasDualMapData()) return;

    if (!this.dualMapInitialized) {
      const grayOpts = this.getDualMapGrayOptions();
      const rgbOpts  = this.getDualMapRgbOptions();
      const subOpts  = this.getDualMapSubOptions();
      this.selectedDualMapGrayKey = grayOpts.length ? grayOpts[0].key : 'soft_membership_jet';
      this.selectedDualMapRgbKey  = rgbOpts.length  ? rgbOpts[0].key  : 'rgb_soft_membership_jet';
      this.selectedDualMapSubKey  = subOpts.length  ? subOpts[0].key  : 'sub_soft_membership_jet';
      this.selectedDualMapGrayCompIdx = 0;
      this.selectedDualMapRgbCompIdx  = 0;
      this.selectedDualMapSubCompIdx  = 0;
      this.dualMapInitialized = true;
    }

    this._pushDualMapPreview();
  }

  // ── end dual_map helpers ───────────────────────────────────────────────────

  savePipelineImages(): void {
    if (this.isPreviewMode) return;
    if (!this.step || this.step.step_def_id !== 'save_images') return;

    const outputFolder = String(this.getParamValue('output_folder') ?? '').trim();
    if (!outputFolder) {
      alert('A kimeneti mappa kötelező.');
      return;
    }

    const pipeline = this.pipelineState.getPipeline();
    const scaleBarOverlay = this.pipelineState.getScaleBarExportParams();
    this.saveImagesInProgress = true;

    this.recipeService.savePipelineImages(pipeline, this.selectedIndex, scaleBarOverlay).subscribe({
      next: (res) => {
        this.saveImagesInProgress = false;
        this.saveImagesResultText = `${res.saved_count} kép mentve`;
        this.showCopyToast(`Képek mentve (${res.saved_count})`);
      },
      error: (err) => {
        this.saveImagesInProgress = false;
        const msg = err?.error?.error ?? 'Képek mentése sikertelen.';
        alert(msg);
      },
    });
  }

  savePipelineArray(): void {
    if (this.isPreviewMode) return;
    if (!this.step || this.step.step_def_id !== 'save_array') return;

    const outputFolder = String(this.getParamValue('output_folder') ?? '').trim();
    if (!outputFolder) {
      alert('A mentési hely kötelező.');
      return;
    }

    const pipeline = this.pipelineState.getPipeline();
    this.saveArrayInProgress = true;

    this.recipeService.savePipelineArray(pipeline, this.selectedIndex).subscribe({
      next: (res) => {
        this.saveArrayInProgress = false;
        this.saveArrayResultText = `${res.row_count} sor mentve (${res.col_count} oszlop)`;
        this.showCopyToast('Adattömb CSV mentve');
      },
      error: (err) => {
        this.saveArrayInProgress = false;
        const msg = err?.error?.error ?? 'Adattömb mentése sikertelen.';
        alert(msg);
      },
    });
  }

  formatFloat4(val: any): string {
    if (val == null) return '-';
    return Number(val).toFixed(4);
  }

  // --- Copy functionality ---

  copyAllResults(event: Event): void {
    if (this.isPreviewMode) return;
    event.stopPropagation();
    const text = JSON.stringify(this.sideOutputs, null, 2);
    navigator.clipboard.writeText(text).then(() => this.showCopyToast('Eredmények másolva'));
  }

  copyResult(key: string): void {
    if (this.isPreviewMode) return;
    const val = this.sideOutputs[key];
    const text = typeof val === 'object' ? JSON.stringify(val, null, 2) : String(val);
    navigator.clipboard.writeText(text).then(() => this.showCopyToast(`„${key}" másolva`));
  }

  private showCopyToast(message: string): void {
    this.copyNotification = message;
    clearTimeout(this.copyTimeout);
    this.copyTimeout = setTimeout(() => {
      this.copyNotification = '';
    }, 1500);
  }

  // --- Dynamic enum filtering ---

  private readonly CHANNEL_MAP: Record<string, string[]> = {
    BGR: ['B', 'G', 'R', 'ALL'],
    HSV: ['H', 'S', 'V', 'ALL'],
    LAB: ['L', 'A', 'B', 'ALL'],
    GRAY: ['GRAY'],
  };

  getFilteredOptions(param: ParamSchema): string[] {
    if (this.step?.step_def_id === 'pseudo_image' &&
        ['blue_source', 'green_source', 'red_source'].includes(param.name)) {
      const imageCount = Math.max(1, this.loadedImageNames.length);
      return Array.from({ length: imageCount }, (_, index) =>
        ['B', 'G', 'R', 'GRAY'].map(channel => `${index + 1}-${channel}`)
      ).flat();
    }
    if (this.step?.step_def_id === 'reference_color_align' && param.name === 'reference_branch') {
      const pipeline = this.pipelineState.getPipeline();
      const available: string[] = [];
      for (let i = 0; i < pipeline.steps.length; i++) {
        if (pipeline.steps[i].step_def_id !== 'load_image' || pipeline.steps[i].enabled === false) continue;
        const end = pipeline.steps.findIndex((candidate, index) => index > i && candidate.step_def_id === 'load_image');
        const branchEnd = end < 0 ? pipeline.steps.length : end;
        if (pipeline.steps.slice(i + 1, branchEnd).some(candidate => candidate.step_def_id === 'reference_crop' && candidate.enabled !== false)) {
          available.push(pipeline.steps[i].instance_id);
        }
      }
      return ['auto', ...available];
    }
    if (this.step?.step_def_id === 'select_channel' && param.name === 'channel') {
      const space = this.getParamValue('space') ?? 'GRAY';
      return this.CHANNEL_MAP[space] ?? param.options ?? [];
    }
    if (this.step?.step_def_id === 'fit_curve' && param.name === 'y_name') {
      return this.getFitCurveYOptions(param.options ?? []);
    }
    return param.options ?? [];
  }

  getOptionDisplayLabel(param: ParamSchema, option: string): string {
    if (this.step?.step_def_id === 'pseudo_image' &&
        ['blue_source', 'green_source', 'red_source'].includes(param.name)) {
      const [imageNumber, channel] = option.split('-', 2);
      const imageName = this.loadedImageNames[Number(imageNumber) - 1];
      return imageName ? `${imageNumber}. kép (${imageName}) – ${channel}` : `${imageNumber}. kép – ${channel}`;
    }
    if (this.step?.step_def_id === 'reference_color_align' && param.name === 'reference_branch') {
      if (option === 'auto') return 'Regi referencia cropok';
      const pipeline = this.pipelineState.getPipeline();
      const branchStarts = pipeline.steps.filter(candidate => candidate.step_def_id === 'load_image');
      const index = branchStarts.findIndex(candidate => candidate.instance_id === option);
      return pipeline.branch_names?.[option] ?? `Ag ${index + 1}`;
    }
    if (this.step?.step_def_id === 'reference_color_align' && param.name === 'mode') {
      const map: Record<string, string> = {
        location: 'Csak tonusillesztes',
        location_scale: 'Tonus- es szinillesztes',
      };
      return map[option] ?? option;
    }

    if (this.step?.step_def_id === 'cluster_reference_map' && param.name === 'center_mode') {
      const map: Record<string, string> = {
        min_max_midpoint: 'Minimum és maximum közepe',
        cluster_median: 'Klaszter mediánja',
        reference_mean: 'Referencia átlaga',
        reference_mean_half: 'Referenciaátlag fele',
      };
      return map[option] ?? option;
    }

    if (this.step?.step_def_id === 'apply_blur' && param.name === 'method') {
      const map: Record<string, string> = {
        gaussian: 'Gauss elmosás',
        median: 'Medián szűrés',
        bilateral: 'Bilaterális szűrés',
        average: 'Átlagoló szűrés',
      };
      return map[option] ?? option;
    }

    if (this.step?.step_def_id === 'flat_field_correction' && param.name === 'method') {
      const map: Record<string, string> = {
        gaussian: 'Gauss háttérbecslés',
        downsampled: 'Lekicsinyített háttérbecslés',
      };
      return map[option] ?? option;
    }

    if (this.step?.step_def_id === 'advanced_illumin_corr' && param.name === 'bg_method') {
      const map: Record<string, string> = {
        gaussian: 'Gauss háttérbecslés',
        downsampled: 'Lekicsinyített háttérbecslés',
      };
      return map[option] ?? option;
    }

    if (this.step?.step_def_id === 'apply_range_mask' && param.name === 'keep_mode') {
      const map: Record<string, string> = {
        inside: 'Tartományon belüli értékek megtartása',
        outside: 'Tartományon kívüli értékek megtartása',
      };
      return map[option] ?? option;
    }

    if (this.step?.step_def_id === 'apply_threshold' && param.name === 'mode') {
      const map: Record<string, string> = {
        binary: 'Bináris (küszöb felett fehér)',
        binary_inv: 'Fordított bináris (küszöb alatt fehér)',
        trunc: 'Levágás (küszöb felett korlátoz)',
        tozero: 'Nullázás (küszöb alatt 0)',
        tozero_inv: 'Fordított nullázás (küszöb felett 0)',
      };
      return map[option] ?? option;
    }

    if (this.step?.step_def_id === 'histogram_equalization' && param.name === 'output_mode') {
      const map: Record<string, string> = {
        image: 'Korrigált kép',
        histogram: 'Korrigált hisztogram',
      };
      return map[option] ?? option;
    }

    if (this.step?.step_def_id === 'kmeans_cluster' && param.name === 'init_mode') {
      const map: Record<string, string> = {
        auto: 'Automatikus',
        reference_fixed: 'Fix referencia cropok',
        reference_seeded: 'Referenciaval inditott k-means',
      };
      return map[option] ?? option;
    }

    if (this.step?.step_def_id === 'kmeans_cluster' && param.name === 'output_mode') {
      const map: Record<string, string> = {
        palette: 'Kontrasztos paletta',
        centroid: 'Klaszterkozep szine',
      };
      return map[option] ?? option;
    }

    if (this.step?.step_def_id === 'kmeans_cluster' && param.name === 'background') {
      const map: Record<string, string> = {
        black: 'Fekete',
        white: 'Feher',
        original: 'Eredeti kep',
      };
      return map[option] ?? option;
    }

    if (this.step?.step_def_id === 'normalize_images' && param.name === 'norm_type') {
      const map: Record<string, string> = {
        minmax: 'Min-Max',
        l1: 'L1 norma',
        l2: 'L2 norma',
      };
      return map[option] ?? option;
    }

    if (this.step?.step_def_id === 'detect_particles' && param.name === 'draw_label_key') {
      const map: Record<string, string> = {
        label: 'Azonosító',
        area_px: 'Terület (px)',
        perimeter_px: 'Kerület (px)',
        equivalent_diameter_px: 'Ekvivalens átmérő (px)',
        circularity: 'Körösség',
        intensity_mean: 'Átlagos intenzitás',
      };
      return map[option] ?? option;
    }

    if (this.step?.step_def_id === 'fit_curve' && param.name === 'model') {
      const map: Record<string, string> = {
        linear: 'Lineáris',
        poly: 'Polinom',
        log: 'Logaritmikus',
        exp: 'Exponenciális',
      };
      return map[option] ?? option;
    }

    if (this.step?.step_def_id === 'fit_curve' && param.name === 'split_method') {
      const map: Record<string, string> = {
        random: 'Véletlenszerű',
        ordered: 'Sorrend szerinti',
      };
      return map[option] ?? option;
    }

    if (this.step?.step_def_id === 'fit_curve' && param.name === 'agg_method') {
      const map: Record<string, string> = {
        mean: 'Átlag',
        median: 'Medián',
      };
      return map[option] ?? option;
    }

    if (this.step?.step_def_id !== 'select_channel') {
      return option;
    }

    if (param.name === 'space') {
      const map: Record<string, string> = {
        BGR: 'BGR',
        HSV: 'HSV',
        LAB: 'Lab',
        GRAY: 'Szürkeárnyalatos',
      };
      return map[option] ?? option;
    }

    if (param.name === 'channel') {
      const space = this.getParamValue('space') ?? 'BGR';
      const map: Record<string, string> = {
        B: space === 'LAB' ? 'b' : 'Blue',
        G: 'Green',
        R: 'Red',
        H: 'Hue',
        S: 'Saturation',
        V: 'Value',
        L: 'L',
        A: 'a',
        GRAY: 'szürkeárnyalat',
        ALL: 'Összes csatorna',
      };
      return map[option] ?? option;
    }

    return option;
  }

  getFitCurveYOptions(defaultOptions: string[] = []): string[] {
    const options: string[] = [];
    const seen = new Set<string>();
    const add = (key: string) => {
      const k = String(key ?? '').trim();
      if (!k || seen.has(k)) return;
      seen.add(k);
      options.push(k);
    };

    for (const opt of defaultOptions) add(opt);

    const addNumericKeysFromSeries = (series: any) => {
      if (!Array.isArray(series) || series.length === 0) return;
      const sample = series.find((v) => v && typeof v === 'object' && !Array.isArray(v));
      if (!sample || typeof sample !== 'object') return;
      for (const [k, v] of Object.entries(sample)) {
        if (typeof v === 'number') add(k);
      }
    };

    addNumericKeysFromSeries(this.sideOutputs['intensity_stats']);
    for (const [, value] of Object.entries(this.sideOutputs ?? {})) {
      addNumericKeysFromSeries(value);
    }

    const current = String(this.getParamValue('y_name') ?? '').trim();
    if (current) add(current);

    return options;
  }

  getThresholdInputHistogram(): number[] | null {
    if (this.step?.step_def_id !== 'apply_threshold') return null;
    const histograms = this.sideOutputs['threshold_input_histograms'];
    if (!Array.isArray(histograms) || histograms.length === 0) return null;
    const idx = Math.min(this.previewImageIndex, histograms.length - 1);
    const h = histograms[idx];
    return Array.isArray(h) ? h : null;
  }

  getThresholdMarkerLines(): Array<{ value: number; label?: string; color?: string }> {
    if (this.step?.step_def_id !== 'apply_threshold') return [];
    const thresh = Number(this.getParamValue('thresh'));
    const maxval = Number(this.getParamValue('maxval'));
    const lines: Array<{ value: number; label?: string; color?: string }> = [];

    if (Number.isFinite(thresh)) {
      lines.push({ value: thresh, label: 'Küszöb', color: '#f59e0b' });
    }
    if (Number.isFinite(maxval)) {
      lines.push({ value: maxval, label: 'Max', color: '#34d399' });
    }
    return lines;
  }

  private makeAxisLabel(key: string): string {
    const trimmed = String(key ?? '').trim();
    if (!trimmed) return '';
    return trimmed
      .replace(/_/g, ' ')
      .replace(/\s+/g, ' ')
      .trim();
  }

  private getCurveFitEquation(fit: CurveFitData | null): string {
    if (!fit || !Array.isArray(fit.coefficients) || fit.coefficients.length === 0) return '';
    const c = fit.coefficients;
    if (fit.model === 'linear' || (fit.model === 'poly' && fit.degree === 1)) {
      return `y = ${c[0].toFixed(6)}x + ${c[1].toFixed(6)}`;
    }
    if (fit.model === 'log') {
      return `y = ${c[0].toFixed(6)}\u00B7ln(x) + ${c[1].toFixed(6)}`;
    }
    if (fit.model === 'exp') {
      return `y = ${c[0].toFixed(6)}\u00B7e^(${c[1].toFixed(6)}x)`;
    }
    const expr = c
      .map((coeff, i) => {
        const power = c.length - 1 - i;
        const val = coeff.toFixed(6);
        if (power === 0) return val;
        if (power === 1) return `${val}x`;
        return `${val}x^${power}`;
      })
      .join(' + ');
    return `y = ${expr}`;
  }

  private syncFitCurveDefaultsFromContext(): void {
    if (this.isPreviewMode) return;
    if (!this.step || this.step.step_def_id !== 'fit_curve') return;

    const currentY = String(this.step.param_values?.['y_name'] ?? '').trim();
    const yOptions = this.getFitCurveYOptions(this.getParamByName('y_name')?.options ?? []);
    const nextY = yOptions.includes(currentY) ? currentY : (yOptions[0] ?? currentY ?? '');

    if (nextY && nextY !== currentY) {
      const updated = {
        ...this.step.param_values,
        y_name: nextY || currentY || 'mean',
      };
      this.pipelineState.updateParams(this.selectedIndex, updated);
    }
  }

  private syncPredictNodeDefaultsFromContext(): void {
    if (this.isPreviewMode) return;
    if (!this.step || this.step.step_def_id !== 'predict_node') return;

    const currentY = String(this.step.param_values?.['y_name'] ?? '').trim();
    const yOptions = this.getPredictYOptions();
    const nextY = yOptions.includes(currentY) ? currentY : (yOptions[0] ?? currentY ?? '');

    if (nextY && nextY !== currentY) {
      const updated = {
        ...this.step.param_values,
        y_name: nextY || currentY || 'mean',
      };
      this.pipelineState.updateParams(this.selectedIndex, updated);
    }
  }

  private normalizeColorMap(map: Record<string, string>): Record<string, string> {
    const out: Record<string, string> = {};
    for (const [k, v] of Object.entries(map)) {
      if (typeof v === 'string' && /^#[0-9a-fA-F]{6}$/.test(v.trim())) {
        out[String(k)] = v.trim();
      }
    }
    return out;
  }

  private parseColorMap(raw: any): Record<string, string> {
    if (!raw || typeof raw !== 'string') return {};
    try {
      const parsed = JSON.parse(raw);
      if (!parsed || typeof parsed !== 'object') return {};
      return this.normalizeColorMap(parsed as Record<string, string>);
    } catch {
      return {};
    }
  }

  private defaultColor(index: number): string {
    const palette = ['#60a5fa', '#34d399', '#f59e0b', '#f87171', '#a78bfa', '#22d3ee', '#f472b6', '#84cc16'];
    return palette[index % palette.length];
  }

  private toNum(v: any, fallback: number): number {
    const n = Number(v);
    return Number.isFinite(n) ? n : fallback;
  }

  private formatNum(v: number): string {
    if (Number.isInteger(v)) return String(v);
    return Number(v.toFixed(6)).toString();
  }

  private buildReferenceGroups(values: number[]): Array<{ key: string; label: string }> {
    if (!values.length) return [];
    const imageCount = Math.max(1, this.pipelineState.getImageCount());
    const nLevels = values.length;
    const perLevel = Math.floor(imageCount / nLevels) || 1;
    return values.map((v, idx) => {
      const key = this.formatNum(v);
      const startImg = idx * perLevel + 1;
      const endImg = idx === nLevels - 1 ? imageCount : (idx + 1) * perLevel;
      return { key, label: `${key} (${startImg}-${endImg})` };
    });
  }

  private refreshReferenceGroupsFromParams(params?: Record<string, any>): void {
    const p = params ?? this.step?.param_values ?? {};
    const csv = String(p['values'] ?? '').trim();
    const vals = csv
      .split(',')
      .map((s) => s.trim())
      .filter((s) => s.length > 0)
      .map((s) => Number(s))
      .filter((n) => Number.isFinite(n));
    const colorMap = this.parseColorMap(p['group_colors']);
    this.referenceGroups = this.buildReferenceGroups(vals).map((g, idx) => ({
      key: g.key,
      label: g.label,
      color: colorMap[g.key] ?? this.defaultColor(idx),
    }));
    this.persistReferenceGroupColors();
  }

  private persistReferenceGroupColors(): void {
    if (!this.step || this.step.step_def_id !== 'add_sequence_values') return;
    const colorMap: Record<string, string> = {};
    for (const g of this.referenceGroups) {
      colorMap[g.key] = g.color;
    }
    const nextGroupColors = JSON.stringify(colorMap);
    if ((this.step.param_values?.['group_colors'] ?? '') === nextGroupColors) {
      return;
    }
    const next = { ...this.step.param_values, group_colors: nextGroupColors };
    this.pipelineState.updateParams(this.selectedIndex, next);
  }

  onReferenceGroupColorChange(groupKey: string, color: string): void {
    if (this.isPreviewMode) return;
    this.referenceGroups = this.referenceGroups.map((g) =>
      g.key === groupKey ? { ...g, color } : g
    );
    this.persistReferenceGroupColors();
  }

  generateReferenceValues(): void {
    if (this.isPreviewMode) return;
    if (!this.step || this.step.step_def_id !== 'add_sequence_values') return;
    const p = this.step.param_values ?? {};
    const numLevels = Math.max(1, Math.floor(this.toNum(p['num_levels'], 5)));
    const start = this.toNum(p['start'], 1);
    const stepVal = this.toNum(p['step_val'], 1);

    // Generate unique level values: start, start+step, start+2*step, ...
    const values: number[] = [];
    for (let i = 0; i < numLevels; i++) {
      values.push(start + i * stepVal);
    }

    if (!values.length) return;

    // Store short-form (unique levels only) - backend expands
    const csv = values.map((v) => this.formatNum(v)).join(', ');
    const updated = { ...this.step.param_values, values: csv, num_levels: numLevels };
    this.pipelineState.updateParams(this.selectedIndex, updated);
    this.refreshReferenceGroupsFromParams(updated);
  }

  importReferenceValuesFromFile(): void {
    if (this.isPreviewMode) return;
    this.recipeService.browseValuesFile().subscribe({
      next: (res) => {
        const path = res.path ?? '';
        if (!path) return;
        this.recipeService.importExplicitValues(path).subscribe({
          next: (out) => {
            const updated = { ...(this.step?.param_values ?? {}), values: out.values_csv };
            this.pipelineState.updateParams(this.selectedIndex, updated);
            this.refreshReferenceGroupsFromParams(updated);
          },
          error: (err) => {
            const msg = err?.error?.error ?? 'Érvénytelen értékfájl.';
            alert(msg);
          },
        });
      },
    });
  }

  openSaveCalibrationDialog(): void {
    if (this.isPreviewMode) return;
    const fit = this.getLatestCurveFit();
    if (!fit) return;
    this.pendingCalibrationEquation = this.getCurveFitEquation(fit);
    this.pendingCalibrationName = `Kalibráció ${new Date().toLocaleString('hu-HU')}`;
    this.pendingCalibrationComment = '';
    this.pendingCalibrationYKey = (fit as any).y_key ?? String(this.step?.param_values?.['y_name'] ?? fit.y_name ?? 'mean');
    this.showSaveCalibrationDialog = true;
  }

  closeSaveCalibrationDialog(): void {
    this.showSaveCalibrationDialog = false;
  }

  saveCurrentCalibration(): void {
    if (this.isPreviewMode) return;
    const fit = this.getLatestCurveFit();
    if (!fit) return;
    const name = this.pendingCalibrationName.trim();
    if (!name) {
      alert('A kalibráció neve kötelező.');
      return;
    }

    const equation = this.pendingCalibrationEquation.trim() || this.getCurveFitEquation(fit);
    if (!equation) {
      alert('Az egyenlet nem lehet üres.');
      return;
    }

    const yKey = (fit as any).y_key ?? String(this.step?.param_values?.['y_name'] ?? fit.y_name ?? 'mean');

    this.savingCalibration = true;
    this.recipeService.saveCalibration({
      name,
      equation,
      comment: this.pendingCalibrationComment.trim(),
      x_name: fit.x_name,
      y_name: fit.y_name,
      y_key: yKey,
      model: fit.model,
      degree: fit.degree ?? undefined,
      coefficients: fit.coefficients,
      x_min: (fit as any).x_min,
      x_max: (fit as any).x_max,
    }).subscribe({
      next: () => {
        this.savingCalibration = false;
        this.showSaveCalibrationDialog = false;
        this.showCopyToast('Kalibráció mentve');
      },
      error: (err) => {
        this.savingCalibration = false;
        const msg = err?.error?.error ?? 'Kalibráció mentése sikertelen.';
        alert(msg);
      },
    });
  }

  openCalibrationBrowser(): void {
    if (this.isPreviewMode) return;
    this.recipeService.listCalibrations().subscribe({
      next: (records) => {
        this.calibrationRecords = records;
        this.selectedCalibrationId = records[0]?.id ?? '';
        this.showCalibrationBrowser = true;
      },
      error: (err) => {
        const msg = err?.error?.error ?? 'Kalibrációk lekérése sikertelen.';
        alert(msg);
      },
    });
  }

  closeCalibrationBrowser(): void {
    this.showCalibrationBrowser = false;
  }

  applySelectedCalibration(): void {
    if (this.isPreviewMode) return;
    if (!this.step || this.step.step_def_id !== 'predict_node') return;
    if (!this.selectedCalibrationId) return;
    const selected = this.calibrationRecords.find(c => c.id === this.selectedCalibrationId);
    if (!selected) return;

    const updated = {
      ...this.step.param_values,
      equation: selected.equation ?? '',
      y_name: selected.y_key || selected.y_name || this.step.param_values['y_name'] || 'mean',
    };
    this.pipelineState.updateParams(this.selectedIndex, updated);
    this.showCalibrationBrowser = false;
    this.showCopyToast('Kalibráció kiválasztva');
  }

  // --- Maximize chart ---

  maximizeChart(data: any): void {
    if (this.isPreviewMode) return;
    // Determine chart type and send to pipeline preview
    if (data.x_values && data.y_values) {
      // Scatter chart (curve fit data)
      this.pipelineState.showExpandedChart(data, 'scatter', 'Görbe illesztés');
    } else if (data.scores) {
      // PCA chart
      this.pipelineState.showExpandedChart(data, 'pca', 'PCA Vizualizáció');
    }
  }

  closeExpandedChart(): void {
    // Expanded chart state is managed by pipeline-preview via PipelineStateService
    // Nothing to do here
  }

  // --- Run curve fit manually ---

  runCurveFit(): void {
    if (this.isPreviewMode) return;
    const previousFits = this.sideOutputs['curve_fits'];
    const previousFitCount = Array.isArray(previousFits) ? previousFits.length : 0;

    const sub = this.pipelineState.sideOutputs$.subscribe((so) => {
      const fits = so?.['curve_fits'];
      if (!Array.isArray(fits) || fits.length === 0) return;
      const latest = fits[fits.length - 1] as CurveFitData;
      if (!latest) return;
      if (fits.length <= previousFitCount) return;
      this.maximizeChart(latest);
      sub.unsubscribe();
    });

    this.pipelineState.requestPreview(true);

    setTimeout(() => {
      sub.unsubscribe();
    }, 12000);
  }

  // --- Run PCA manually ---

  runPCA(): void {
    if (this.isPreviewMode) return;
    const previousScores = this.sideOutputs['histogram_pca_scores'];
    const previousScoresCount = Array.isArray(previousScores) ? previousScores.length : 0;

    const sub = this.pipelineState.sideOutputs$.subscribe((so) => {
      const scores = so?.['histogram_pca_scores'];
      if (!Array.isArray(scores) || scores.length === 0) return;
      if (scores.length <= previousScoresCount) return;
      sub.unsubscribe();
    });

    this.pipelineState.requestPreview(true);

    setTimeout(() => {
      sub.unsubscribe();
    }, 12000);
  }

  onPCAComponentChanged(event: { pcX: number; pcY: number }): void {
    // This could be used to store the user's component selection preference
    // For now, it's just a passthrough from the PCA chart component
  }

  // --- Color Threshold methods ---

  private COLOR_SPACE_CHANNELS: Record<string, { channels: string[]; ranges: Record<string, [number, number]> }> = {
    BGR: {
      channels: ['B', 'G', 'R'],
      ranges: { B: [0, 255], G: [0, 255], R: [0, 255] }
    },
    HSV: {
      channels: ['H', 'S', 'V'],
      ranges: { H: [0, 179], S: [0, 255], V: [0, 255] }
    },
    LAB: {
      channels: ['L', 'A', 'B'],
      ranges: { L: [0, 255], A: [0, 255], B: [0, 255] }
    },
    GRAY: {
      channels: ['GRAY'],
      ranges: { GRAY: [0, 255] }
    }
  };

  getColorThreshSpace(): string {
    if (this.step?.step_def_id !== 'color_thresh') return 'HSV';
    
    // Get the pipeline to find the previous select_channel step
    const pipeline = this.pipelineState.getPipeline();
    const currentIdx = pipeline.steps.findIndex(s => s.instance_id === this.step?.instance_id);
    
    if (currentIdx <= 0) return 'HSV';
    
    // Look backwards for select_channel step
    for (let i = currentIdx - 1; i >= 0; i--) {
      const step = pipeline.steps[i];
      if (step.step_def_id === 'select_channel') {
        const space = step.param_values?.['space'] as string;
        return space || 'HSV';
      }
    }
    
    return 'HSV';
  }

  isColorThreshMaxParam(paramName: string): boolean {
    if (this.step?.step_def_id !== 'color_thresh') return false;
    return paramName.endsWith('_max');
  }

  getColorThreshHistogramForParam(paramName: string): { channel: string; values: number[]; rangeMin: number; rangeMax: number } | null {
    if (!paramName.endsWith('_max')) return null;
    
    const histograms = this.getColorThreshHistograms();
    if (!histograms) return null;
    
    // Extract channel from param name (e.g., 'H_max' -> 'H', 'Lab_B_max' -> 'B')
    let channel = '';
    if (paramName === 'Lab_B_max') {
      channel = 'B';
    } else if (paramName === 'GRAY_max') {
      channel = 'GRAY';
    } else {
      channel = paramName.slice(0, -4);
    }
    
    return histograms.find(h => h.channel === channel) || null;
  }

  getColorThreshVisibleParams(space?: string): Set<string> {
    const detectedSpace = space || this.getColorThreshSpace();
    const paramSet = new Set<string>();
    const channels = this.COLOR_SPACE_CHANNELS[detectedSpace]?.channels ?? [];
    
    for (const ch of channels) {
      if (detectedSpace === 'LAB' && ch === 'B') {
        paramSet.add('Lab_B_min');
        paramSet.add('Lab_B_max');
      } else if (detectedSpace === 'GRAY') {
        paramSet.add('GRAY_min');
        paramSet.add('GRAY_max');
      } else {
        paramSet.add(`${ch}_min`);
        paramSet.add(`${ch}_max`);
      }
    }
    paramSet.add('invert');
    return paramSet;
  }

  getColorThreshHistograms(): Array<{ channel: string; values: number[]; rangeMin: number; rangeMax: number }> | null {
    if (this.step?.step_def_id !== 'color_thresh') return null;
    const space = this.getColorThreshSpace();
    const histograms = this.sideOutputs['color_thresh_channel_histograms'];
    
    if (!Array.isArray(histograms) || histograms.length === 0) return null;
    
    const idx = Math.min(this.previewImageIndex, histograms.length - 1);
    const histData = histograms[idx];
    if (!histData || typeof histData !== 'object') return null;

    const channels = this.COLOR_SPACE_CHANNELS[space]?.channels ?? [];
    const config = this.COLOR_SPACE_CHANNELS[space];
    
    const result: Array<{ channel: string; values: number[]; rangeMin: number; rangeMax: number }> = [];
    
    for (const ch of channels) {
      if (histData[ch]) {
        const [rangeMin, rangeMax] = config?.ranges[ch] ?? [0, 255];
        result.push({
          channel: ch,
          values: Array.isArray(histData[ch]) ? histData[ch] : [],
          rangeMin,
          rangeMax
        });
      }
    }
    
    return result.length > 0 ? result : null;
  }

  getColorThreshInputImage(): string | null {
    if (this.step?.step_def_id !== 'color_thresh') return null;
    const inputImages = this.sideOutputs['color_thresh_input_images'];
    if (!Array.isArray(inputImages) || inputImages.length === 0) return null;
    const idx = Math.min(this.previewImageIndex, inputImages.length - 1);
    return inputImages[idx] || null;
  }

  getColorThreshMarkerLines(forChannel?: string): Array<{ value: number; label?: string; color?: string }> {
    if (this.step?.step_def_id !== 'color_thresh') return [];
    
    const space = this.getColorThreshSpace();
    const lines: Array<{ value: number; label?: string; color?: string }> = [];
    
    // If called with specific channel, only show that channel's lines
    if (forChannel) {
      let minKey = '', maxKey = '';
      
      if (space === 'LAB' && forChannel === 'B') {
        minKey = 'Lab_B_min';
        maxKey = 'Lab_B_max';
      } else if (space === 'GRAY') {
        minKey = 'GRAY_min';
        maxKey = 'GRAY_max';
      } else {
        minKey = `${forChannel}_min`;
        maxKey = `${forChannel}_max`;
      }
      
      const minVal = Number(this.getParamValue(minKey));
      const maxVal = Number(this.getParamValue(maxKey));
      
      if (Number.isFinite(minVal)) {
        lines.push({ value: minVal, label: `${forChannel} min`, color: '#60a5fa' });
      }
      if (Number.isFinite(maxVal)) {
        lines.push({ value: maxVal, label: `${forChannel} max`, color: '#34d399' });
      }
      
      return lines;
    }
    
    // Show all channels (fallback, shouldn't be used)
    const channels = this.COLOR_SPACE_CHANNELS[space]?.channels ?? [];
    for (const ch of channels) {
      let minKey = '', maxKey = '';
      
      if (space === 'LAB' && ch === 'B') {
        minKey = 'Lab_B_min';
        maxKey = 'Lab_B_max';
      } else if (space === 'GRAY') {
        minKey = 'GRAY_min';
        maxKey = 'GRAY_max';
      } else {
        minKey = `${ch}_min`;
        maxKey = `${ch}_max`;
      }
      
      const minVal = Number(this.getParamValue(minKey));
      const maxVal = Number(this.getParamValue(maxKey));
      
      if (Number.isFinite(minVal)) {
        lines.push({ value: minVal, label: `${ch} min`, color: '#60a5fa' });
      }
      if (Number.isFinite(maxVal)) {
        lines.push({ value: maxVal, label: `${ch} max`, color: '#34d399' });
      }
    }
    
    return lines;
  }

  getSliderMinMax(param: any): { min: number; max: number } {
    return {
      min: this.getSliderMin(param),
      max: this.getSliderMax(param)
    };
  }

  onRangeMinChange(paramName: string, event: Event): void {
    const value = +(event.target as HTMLInputElement).value;
    this.onParamChange(paramName, value);
  }

  onRangeMaxChange(paramName: string, event: Event): void {
    const value = +(event.target as HTMLInputElement).value;
    this.onParamChange(paramName, value);
  }
}
