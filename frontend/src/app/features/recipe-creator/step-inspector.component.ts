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
  imports: [CommonModule, FormsModule, MatIconModule, HistogramChartComponent, ScatterChartComponent],
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
            @if (isRoiEmpty()) {
              <div class="roi-empty-warning">⚠ Nincs kijelölt ROI terület</div>
            }
          }
          @for (param of getVisibleParams(); track param.name) {
            @if (param.name !== 'file_order' && param.name !== 'group_colors' && !shouldHideParam(param)) {
            <div class="param-row">
              <label class="param-label" [attr.for]="'param-' + param.name">
                {{ getDisplayParamLabel(param) }}
              </label>

              @switch (param.type) {
                @case ('int') {
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
                      [ngModel]="getParamValue(param.name)"
                      (ngModelChange)="onNumericTextChange(param, $event)"
                    />
                    @if (isFitCurveValidationRatioParam(param.name)) {
                      <span class="inline-unit">%</span>
                    }
                  </div>
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
                  @if (isFitCurveYAxisParam(param.name)) {
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
                  } @else if (isPredictEquationParam(param.name)) {
                    <div class="param-control file-path-control">
                      <input
                        type="text"
                        [id]="'param-' + param.name"
                        [ngModel]="getParamValue(param.name)"
                        (ngModelChange)="onParamChange(param.name, $event)"
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
                        (ngModelChange)="onParamChange(param.name, $event)"
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
                      (ngModelChange)="onParamChange(param.name, $event)"
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

    .threshold-histogram-wrap {
      margin-top: 8px;
    }

    .slider-control { display: flex; align-items: center; gap: 10px; }
    .slider-control input[type="range"] { flex: 1; accent-color: #224477; }

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
    const num = param.type === 'int' ? parseInt(String(value), 10) : parseFloat(String(value));
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

  isRoiEmpty(): boolean {
    if (!this.isRoiStep()) return false;
    const t = this.getParamValue('roi_type') ?? 'rect';
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
      const roiType = this.getParamValue('roi_type') ?? 'rect';
      return params.filter(p => {
        if (p.name === 'roi_type') return false; // shown as shape buttons
        if (roiType === 'rect') return !this.ROI_ELLIPSE_PARAMS.has(p.name) && !this.ROI_POLYGON_PARAMS.has(p.name);
        if (roiType === 'ellipse') return !this.ROI_RECT_PARAMS.has(p.name) && !this.ROI_POLYGON_PARAMS.has(p.name);
        if (roiType === 'polygon') return !this.ROI_RECT_PARAMS.has(p.name) && !this.ROI_ELLIPSE_PARAMS.has(p.name);
        return true;
      });
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

  isPredictEquationParam(paramName: string): boolean {
    return this.step?.step_def_id === 'predict_node' && paramName === 'equation';
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
    if (id === 'save_images') return true;
    if (id === 'save_array') return true;
    if (id === 'predict_node') return (this.getPredictions()?.length ?? 0) > 0;
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
    const entries: { key: string; label: string; value: string }[] = [];
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

    for (const key of baseOrder) {
      const v = values[key];
      if (v == null) continue;
      entries.push({
        key,
        label: baseLabels[key] ?? key,
        value: key === 'pixel_count' ? String(v) : Number(v).toFixed(2),
      });
    }

    const percentileKeys = Object.keys(values)
      .filter((k) => /^p\d+(?:\.\d+)?$/i.test(k) && values[k] != null)
      .sort((a, b) => Number(a.slice(1)) - Number(b.slice(1)));

    for (const key of percentileKeys) {
      entries.push({
        key,
        label: key.toUpperCase(),
        value: Number(values[key]).toFixed(2),
      });
    }

    if (values['dynamic_range'] != null) {
      entries.push({
        key: 'dynamic_range',
        label: baseLabels['dynamic_range'],
        value: Number(values['dynamic_range']).toFixed(2),
      });
    }

    const known = new Set([...baseOrder, ...percentileKeys, 'dynamic_range']);
    const otherNumeric = Object.keys(values)
      .filter((k) => !known.has(k) && typeof values[k] === 'number')
      .sort();

    for (const key of otherNumeric) {
      entries.push({
        key,
        label: key.replace(/_/g, ' '),
        value: Number(values[key]).toFixed(2),
      });
    }

    return entries;
  }

  getLatestCurveFit(): CurveFitData | null {
    const fits = this.sideOutputs['curve_fits'];
    if (!Array.isArray(fits) || fits.length === 0) return null;
    return fits[fits.length - 1] as CurveFitData;
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

  savePipelineImages(): void {
    if (this.isPreviewMode) return;
    if (!this.step || this.step.step_def_id !== 'save_images') return;

    const outputFolder = String(this.getParamValue('output_folder') ?? '').trim();
    if (!outputFolder) {
      alert('A kimeneti mappa kötelező.');
      return;
    }

    const pipeline = this.pipelineState.getPipeline();
    this.saveImagesInProgress = true;

    this.recipeService.savePipelineImages(pipeline, this.selectedIndex).subscribe({
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
    BGR: ['B', 'G', 'R'],
    HSV: ['H', 'S', 'V'],
    LAB: ['L', 'A', 'B'],
    GRAY: ['GRAY'],
  };

  getFilteredOptions(param: ParamSchema): string[] {
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
      const map: Record<string, string> = {
        B: 'Blue',
        G: 'Green',
        R: 'Red',
        H: 'Hue',
        S: 'Saturation',
        V: 'Value',
        L: 'L',
        A: 'a',
        GRAY: 'szürkeárnyalat',
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
    const omitted = this.pipelineState.getOmittedPoints();
    this.pipelineState.requestMaximizeGraph(data, omitted.indices, this.selectedIndex);
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
}
