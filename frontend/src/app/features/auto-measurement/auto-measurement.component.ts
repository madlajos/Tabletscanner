import { Component, signal, OnInit, OnDestroy, AfterViewInit, ElementRef, ViewChild } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Subscription, from, timer, EMPTY } from 'rxjs';
import { catchError, concatMap, last, finalize, switchMap } from 'rxjs/operators';
import { HttpClient } from '@angular/common/http';
import {
  AutoMeasurementService,
  CameraParameterRange,
  TabletStepRequest
} from '../../services/auto-measurement.service';
import { FilterSettingsService } from '../../services/filter-settings.service';
import { AutofocusSettingsService } from '../../services/autofocus-settings.service';
import { MotionSettingsService } from '../../services/motion-settings.service';
import { FilterSettings } from '../../models/filter-settings.models';
import { AutofocusSettings } from '../../models/autofocus-settings.models';
import { SharedService } from '../../shared.service';
import { ErrorNotificationService } from '../../services/error-notification.service';
import { BASE_URL } from '../../api-config';
import { CapturePlanRow, CaptureRequestRow, LightChannel, LIGHT_CHANNEL_LABELS, UvBrightnessMode } from '../../models/light.models';
import { AdvancedMotionSettings } from '../../models/motion.models';
import { ToolheadPositionComponent } from './toolhead-position.component';
import { CameraCombinationSettingsService } from '../../services/camera-combination-settings.service';
import { CameraCombinationSettings } from '../../models/camera-combination-settings.models';
import { ErrorPopupListComponent } from '../../components/error-popup-list/error-popup-list.component';
import { AnalysisRecipeSelectorComponent } from '../../components/analysis-recipe-selector/analysis-recipe-selector.component';
import { PipelineDocument } from '../../models/pipeline.models';

// Type declaration for Electron API (exposed via preload.js)
declare global {
  interface Window {
    electronAPI?: {
      selectFolder: () => Promise<string>;
      selectFile: () => Promise<string>;
      saveTssFile: (suggestedName: string, jsonContent: string) => Promise<{ canceled: boolean; fileName?: string }>;
    };
  }
}

/** Normalize a filesystem path to use forward slashes for JSON portability. */
function normalizePath(p: string): string {
  if (!p) return p;
  return p.replace(/\\/g, '/');
}

// Interface for tablet position calculation
interface TabletPosition {
  index: number;
  x: number;
  y: number;
}

@Component({
  selector: 'app-auto-measurement',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    ToolheadPositionComponent,
    ErrorPopupListComponent,
    AnalysisRecipeSelectorComponent,
  ],
  templateUrl: './auto-measurement.component.html',
  styleUrls: ['./auto-measurement.component.css']
})
export class AutoMeasurementComponent implements OnInit, AfterViewInit, OnDestroy {

  @ViewChild('capturePlanScroll') private capturePlanScroll?: ElementRef<HTMLElement>;

  readonly gridSize = 10; // change to 9 later if needed

  // Connection status (from shared service)
  cameraConnected = false;
  motionConnected = false;
  private cameraStatusInitialized = false;
  private motionStatusInitialized = false;
  private cameraSub?: Subscription;
  private motionSub?: Subscription;
  private autofocusErrorSub?: Subscription;
  private autofocusActiveSub?: Subscription;
  private scannerOperationSub?: Subscription;
  isAutofocusing = false;
  scannerOperationActive = false;

  // Tablet IDs, bottom-left = 1, top-right = gridSize^2
  readonly tablets = Array.from(
    { length: this.gridSize * this.gridSize },
    (_, i) => {
      const rowFromTop = Math.floor(i / this.gridSize); // 0..gridSize-1
      const col = i % this.gridSize;                    // 0..gridSize-1
      const rowFromBottom = this.gridSize - 1 - rowFromTop;
      return rowFromBottom * this.gridSize + col + 1;
    }
  );

  // Measurement settings
  autofocus = false;
  backgroundSubtraction = false;
  selectedAnalysisRecipe: PipelineDocument | null = null;
  readonly wavelengthOptions: readonly LightChannel[] = ['uv255', 'uv310', 'uv365', 'vis'];
  readonly lightOptions: ReadonlyArray<{ value: string; label: string }> = [
    ...(['uv255', 'uv310', 'uv365'] as const).flatMap(wavelength => [
      { value: `${wavelength}:dimmed`, label: `${LIGHT_CHANNEL_LABELS[wavelength]} – Tompított` },
      { value: `${wavelength}:full`, label: `${LIGHT_CHANNEL_LABELS[wavelength]} – Teljes` }
    ]),
    { value: 'vis:full', label: 'VIS – Teljes' }
  ];
  readonly filterOptions = [1, 2, 3, 4, 5, 6] as const;
  readonly lightLabels = LIGHT_CHANNEL_LABELS;
  private defaultExposureTime = 100000;
  private defaultGain = 0;
  capturePlan: CapturePlanRow[] = [this.createCapturePlanRow('vis', 'full', 1)];
  private autofocusSettings: AutofocusSettings = {
    channel: 'vis',
    brightness: 'full',
    filter_position: 1
  };
  private autofocusSettingsLoaded = false;
  private capturePlanLoaded = false;
  private cameraCombinationSettings: CameraCombinationSettings | null = null;
  filterSettings: FilterSettings = {
    filters: [],
    slots: [null, null, null, null, null, null],
    height_offsets_mm: { empty: { uv255: 0, uv310: 0, uv365: 0, vis: 0 } }
  };
  exposureRange?: CameraParameterRange;
  gainRange?: CameraParameterRange;

  // Save location and measurement name
  saveLocation = '';
  measurementName = '';

  // First tablet position and spacing (from settings)
  firstTabletX = 2.9;
  firstTabletY = 0;
  firstTabletZ = 20.0;
  tabletSpacing = 18.3;

  // Set of selected tablet IDs (supports multiple ranges + gaps)
  private selectedSignal = signal<Set<number>>(new Set<number>());

  // Range anchor for adding ranges (first click)
  private rangeAnchorSignal = signal<number | null>(null);

  // Hover index for range preview
  private hoverIndexSignal = signal<number | null>(null);

  // Is measurement currently running (toggled "on")
  measurementActive = false;

  // Flag to signal stop request
  private stopRequested = false;

  // Progress tracking
  currentTabletIndex = 0;  // 1-based index of current tablet being processed
  private completedTablets = new Set<number>();  // Set of completed tablet IDs
  private failedTablets = new Set<number>();      // Set of tablets with AF errors (E2000/E2002/E2003)
  private tabletErrors = new Map<number, string>(); // Tablet ID -> error message
  private tabletImages = new Map<number, string[]>(); // Tablet ID -> saved image paths (non-masked)
  private currentTabletId: number | null = null;  // ID of tablet currently being processed

  // Measurement folder path (created at start)
  private measurementFolder = '';

  // Subscription to current tablet measurement (for immediate cancellation on stop)
  private currentTabletSubscription: Subscription | null = null;
  private currentProgressSubscription: Subscription | null = null;
  private emittedProgressPaths = new Set<string>();
  activePlanRowIndex: number | null = null;
  private planRowScrollTimer: ReturnType<typeof setTimeout> | null = null;

  // Subscription to homing operation (for immediate cancellation on stop)
  private homingSubscription: Subscription | null = null;

  errorMessage: string | null = null;
  successMessage: string | null = null;
  validationMessage: string | null = null;

  // Reconnection state during auto-measurement
  reconnecting = false;
  reconnectMessage: string | null = null;
  private reconnectTimer: ReturnType<typeof setInterval> | null = null;
  private reconnectAttemptCount = 0;
  private static readonly MAX_RECONNECT_CYCLES = 3;

  // Context menu state for tablet grid
  tabletContextMenuVisible = false;
  tabletContextMenuX = 0;
  tabletContextMenuY = 0;
  tabletContextMenuId: number | null = null;
  tabletHomed = false;
  private homedSub?: Subscription;
  private traySettingsSub?: Subscription;
  private cameraCombinationSettingsSub?: Subscription;
  private filterSettingsSub?: Subscription;

  constructor(
    private autoService: AutoMeasurementService,
    private autofocusSettingsService: AutofocusSettingsService,
    private filterSettingsService: FilterSettingsService,
    private cameraCombinationSettingsService: CameraCombinationSettingsService,
    private motionSettingsService: MotionSettingsService,
    private sharedService: SharedService,
    private errorNotificationService: ErrorNotificationService,
    private http: HttpClient
  ) {}

  onAnalysisRecipeSelected(recipe: PipelineDocument): void {
    this.selectedAnalysisRecipe = recipe;
  }

  ngOnInit(): void {
    // Subscribe to connection status from shared service
    this.cameraSub = this.sharedService.cameraConnectionStatus$.subscribe(status => {
      const disconnectedNow = this.cameraStatusInitialized && this.cameraConnected && !status;
      this.cameraConnected = status;
      this.cameraStatusInitialized = true;
      if (disconnectedNow) {
        this.errorNotificationService.addError({
          code: 'E1111', message: this.errorNotificationService.getMessage('E1111')
        });
      } else if (status) {
        this.errorNotificationService.removeError('E1111');
      }
    });
    this.motionSub = this.sharedService.motionPlatformConnectionStatus$.subscribe(status => {
      const disconnectedNow = this.motionStatusInitialized && this.motionConnected && !status;
      this.motionConnected = status;
      this.motionStatusInitialized = true;
      if (disconnectedNow) {
        this.errorNotificationService.addError({
          code: 'E1201', message: this.errorNotificationService.getMessage('E1201')
        });
      } else if (status) {
        this.errorNotificationService.removeError('E1201');
      }
    });
    this.scannerOperationSub = this.sharedService.measurementActive$.subscribe(active => {
      this.scannerOperationActive = active;
    });

    // Subscribe to homed state from SharedService (published by motion-control component)
    this.homedSub = this.sharedService.motionHomed$.subscribe(homed => {
      this.tabletHomed = homed;
      console.log('Auto-measurement received homed state:', homed);
    });

    // Subscribe to manual autofocus active state
    this.autofocusActiveSub = this.sharedService.autofocusActive$.subscribe(active => {
      this.isAutofocusing = active;
    });

    // Subscribe to manual autofocus errors from motion-control
    this.autofocusErrorSub = this.sharedService.autofocusError$.subscribe(msg => {
      if (!this.measurementActive) {
        this.errorMessage = msg;
        this.publishToolbarNotice('error', msg);
      }
    });

    this.autoService.getCameraConfig().subscribe({
      next: config => {
        this.defaultExposureTime = config.values.ExposureTime;
        this.defaultGain = config.values.Gain;
        this.exposureRange = config.ranges.ExposureTime;
        this.gainRange = config.ranges.Gain;
      },
      error: err => console.warn('Failed to load camera parameter ranges:', err)
    });

    this.filterSettingsSub = this.filterSettingsService.settings$.subscribe(settings => {
      if (!settings) return;
      this.filterSettings = settings;
      this.refreshPlanCameraCombinations();
    });
    this.filterSettingsService.get().subscribe({
      error: err => console.warn('Failed to load filter settings:', err)
    });

    this.cameraCombinationSettingsSub = this.cameraCombinationSettingsService.settings$.subscribe(settings => {
      this.cameraCombinationSettings = settings;
      this.refreshPlanCameraCombinations();
    });
    this.cameraCombinationSettingsService.get().subscribe({
      error: err => console.warn('Failed to load camera combination settings:', err)
    });

    this.autofocusSettingsService.get().subscribe({
      next: response => {
        this.autofocusSettings = response.autofocus_settings;
        this.autofocusSettingsLoaded = true;
        this.syncAutofocusReferenceRow();
      },
      error: err => console.warn('Failed to load autofocus settings:', err)
    });

    this.traySettingsSub = this.motionSettingsService.advanced$.subscribe(settings => {
      if (settings) this.applyTrayGeometry(settings);
    });
    this.motionSettingsService.getAdvanced().subscribe({
      error: err => console.warn('Failed to load tray geometry settings:', err)
    });

    // Load saved settings from backend
    this.autoService.getSettings().subscribe({
      next: (res) => {
        if (res.auto_measurement_settings) {
          const settings = res.auto_measurement_settings;
          this.saveLocation = normalizePath(settings.save_location || '');
          if (Array.isArray(settings.capture_plan) && settings.capture_plan.length > 0) {
            this.capturePlan = settings.capture_plan
              .filter(row => this.isValidCaptureRequestRow(row))
              .map(row => this.createCapturePlanRow(
                row.wavelength,
                row.brightness,
                row.filter_position,
                row.exposure_time,
                row.gain
              ));
          }
          this.capturePlanLoaded = true;
          this.syncAutofocusReferenceRow();
          this.refreshPlanCameraCombinations();
        }
      },
      error: (err) => {
        console.warn('Failed to load auto measurement settings:', err);
      }
    });
  }

  ngAfterViewInit(): void {
    // Hide context menu when clicking outside
    document.addEventListener('click', () => this.hideTabletContextMenu());
  }

  ngOnDestroy(): void {
    this.stopProgressPolling();
    if (this.planRowScrollTimer) clearTimeout(this.planRowScrollTimer);
    this.cameraSub?.unsubscribe();
    this.motionSub?.unsubscribe();
    this.homedSub?.unsubscribe();
    this.autofocusErrorSub?.unsubscribe();
    this.autofocusActiveSub?.unsubscribe();
    this.scannerOperationSub?.unsubscribe();
    this.traySettingsSub?.unsubscribe();
    this.cameraCombinationSettingsSub?.unsubscribe();
    this.filterSettingsSub?.unsubscribe();
    
    // Ensure measurement is marked inactive on destroy
    if (this.measurementActive) {
      this.sharedService.setMeasurementActive(false);
    }
  }

  // ===== Derived properties =====

  get selectedCount(): number {
    return this.selectedSignal().size;
  }

  hasSelection(): boolean {
    return this.selectedSignal().size > 0;
  }

  get measurementButtonPrimary(): string {
    return this.measurementActive ? 'Mérés leállítása' : 'Mérés indítása';
  }

  get measurementButtonDetail(): string | null {
    if (this.measurementActive) {
      if (this.reconnecting && this.reconnectMessage) return this.reconnectMessage;
      if (this.validationMessage) return this.validationMessage;
      return `Mérés folyamatban… (${this.currentTabletIndex}/${this.selectedCount})`;
    }
    if (this.isAutofocusing) return 'Autofókusz folyamatban…';
    if (this.scannerOperationActive) return 'Másik művelet folyamatban…';
    return this.getValidationMessage();
  }

  private publishToolbarNotice(
    severity: 'info' | 'success' | 'error', message: string | null
  ): void {
    if (message) this.sharedService.setToolbarNotice({ severity, message });
  }

  // Get validation message based on current state
  getValidationMessage(): string | null {
    if (!this.cameraConnected && !this.motionConnected) {
      return 'Kamera és mozgásplatform nincs csatlakoztatva.';
    }
    if (!this.cameraConnected) {
      return 'Kamera nincs csatlakoztatva.';
    }
    if (!this.motionConnected) {
      return 'Mozgásplatform nincs csatlakoztatva.';
    }
    if (!this.saveLocation || this.saveLocation.trim() === '') {
      return 'Válasszon mentési helyet.';
    }
    if (!this.measurementName || this.measurementName.trim() === '') {
      return 'Adja meg a mérés nevét.';
    }
    if (this.capturePlan.length === 0) {
      return 'Adjon hozzá legalább egy mérési sort.';
    }
    if (this.capturePlan.some(row => !this.isCapturePlanRowValid(row))) {
      return 'Adjon meg érvényes záridő- és erősítésértéket minden mérési sorban.';
    }
    if (this.selectedSignal().size === 0) {
      return 'Válasszon legalább egy tablettát.';
    }
    return null;
  }

  // Start/Stop button enabled state
  canStart(): boolean {
    if (this.measurementActive) {
      return true; // allow stopping
    }

    if (this.isAutofocusing || this.scannerOperationActive) {
      return false;
    }

    // Update validation message
    this.validationMessage = this.getValidationMessage();

    // Check all conditions
    const hasSelected = this.selectedSignal().size > 0;
    const connected = this.cameraConnected && this.motionConnected;
    const hasSaveLocation = !!(this.saveLocation && this.saveLocation.trim() !== '');
    const hasMeasurementName = !!(this.measurementName && this.measurementName.trim() !== '');
    const hasCapturePlan = this.capturePlan.length > 0
      && this.capturePlan.every(row => this.isCapturePlanRowValid(row));

    return hasSelected && connected && hasSaveLocation && hasMeasurementName && hasCapturePlan;
  }

  addCapturePlanRow(): void {
    const previous = this.capturePlan[this.capturePlan.length - 1];
    const row = this.createCapturePlanRow(
      previous?.wavelength ?? 'vis',
      previous?.brightness ?? 'full',
      previous?.filter_position ?? 1
    );
    this.capturePlan.push(row);
    this.applyCameraCombination(row);
    this.persistCapturePlan();
  }

  removeCapturePlanRow(index: number): void {
    if (index === 0 || this.measurementActive) {
      return;
    }
    this.capturePlan.splice(index, 1);
    this.persistCapturePlan();
  }

  clearCapturePlan(): void {
    if (this.measurementActive || this.capturePlan.length <= 1) {
      return;
    }
    this.capturePlan.splice(1);
    this.persistCapturePlan();
  }

  onCapturePlanChanged(): void {
    if (!this.measurementActive) {
      this.enforceAutofocusReferenceRow();
      this.persistCapturePlan();
    }
  }

  trackCapturePlanRow(_: number, row: CapturePlanRow): string {
    return row.id;
  }

  private createCapturePlanRow(
    wavelength: LightChannel,
    brightness: UvBrightnessMode,
    filterPosition: number,
    exposureTime = this.defaultExposureTime,
    gain = this.defaultGain
  ): CapturePlanRow {
    return {
      id: crypto.randomUUID?.() ?? `${Date.now()}-${Math.random()}`,
      wavelength,
      brightness: wavelength === 'vis' ? 'full' : brightness,
      filter_position: this.filterOptions.includes(filterPosition as 1 | 2 | 3 | 4 | 5 | 6)
        ? filterPosition as 1 | 2 | 3 | 4 | 5 | 6
        : 1,
      exposure_time: exposureTime,
      gain,
      exposure_time_text: this.formatExposureText(String(exposureTime)),
      gain_text: gain.toString()
    };
  }

  private isValidCaptureRequestRow(row: unknown): row is CaptureRequestRow {
    if (!row || typeof row !== 'object') return false;
    const candidate = row as CaptureRequestRow;
    return this.wavelengthOptions.includes(candidate.wavelength)
      && (candidate.brightness === 'dimmed' || candidate.brightness === 'full')
      && (candidate.wavelength !== 'vis' || candidate.brightness === 'full')
      && this.filterOptions.includes(candidate.filter_position as 1 | 2 | 3 | 4 | 5 | 6)
      && this.isFinitePositive(candidate.exposure_time)
      && this.isFiniteNonNegative(candidate.gain);
  }

  private persistCapturePlan(): void {
    this.enforceAutofocusReferenceRow();
    if (!this.capturePlan.every(row => this.isCapturePlanRowValid(row))) return;
    const capturePlan = this.capturePlan.map(
      ({ wavelength, brightness, filter_position, exposure_time, gain }) =>
        ({ wavelength, brightness, filter_position, exposure_time, gain })
    );
    this.autoService.updateSettings('capture_plan', capturePlan).subscribe({
      error: err => console.warn('Failed to save capture plan:', err)
    });
  }

  onCaptureOpticsChanged(row: CapturePlanRow): void {
    this.applyCameraCombination(row);
    this.onCapturePlanChanged();
  }

  onFilterSelectionChanged(row: CapturePlanRow, filterPosition: number): void {
    if (!this.filterOptions.includes(filterPosition as 1 | 2 | 3 | 4 | 5 | 6)) return;
    row.filter_position = filterPosition as 1 | 2 | 3 | 4 | 5 | 6;
    this.onCaptureOpticsChanged(row);
  }

  getLightSelectionValue(row: CapturePlanRow): string {
    return `${row.wavelength}:${row.brightness}`;
  }

  getLightSelectionLabel(row: CapturePlanRow): string {
    const mode = row.brightness === 'dimmed' ? 'Tompított' : 'Teljes';
    return `${this.lightLabels[row.wavelength]} – ${mode}`;
  }

  onLightSelectionChanged(row: CapturePlanRow, selection: string): void {
    const [wavelength, brightness] = selection.split(':') as [LightChannel, UvBrightnessMode];
    if (!this.wavelengthOptions.includes(wavelength)) return;
    if (brightness !== 'dimmed' && brightness !== 'full') return;
    row.wavelength = wavelength;
    row.brightness = wavelength === 'vis' ? 'full' : brightness;
    this.onCaptureOpticsChanged(row);
  }

  private refreshPlanCameraCombinations(): void {
    if (!this.capturePlanLoaded || !this.cameraCombinationSettings) return;
    for (const row of this.capturePlan) this.applyCameraCombination(row);
    // Do not persist until the fixed autofocus row has also loaded; otherwise
    // asynchronous startup responses could briefly save the fallback optics.
    if (this.autofocusSettingsLoaded) this.persistCapturePlan();
  }

  private applyCameraCombination(row: CapturePlanRow): void {
    const group = this.cameraFilterGroup(row.filter_position);
    const cell = group ? this.cameraCombinationSettings?.[group]?.[row.wavelength] : null;
    if (!cell) return;
    row.exposure_time = cell.exposure_time;
    row.gain = cell.gain;
    row.exposure_time_text = this.formatExposureText(String(cell.exposure_time));
    row.gain_text = String(cell.gain);
  }

  private cameraFilterGroup(position: number): string | null {
    const filterId = this.filterSettings.slots[position - 1];
    if (!filterId) return 'empty';
    const definition = this.filterSettings.filters.find(item => item.id === filterId);
    if (!definition) return null;
    const name = definition.name.normalize('NFD').replace(/[\u0300-\u036f]/g, '')
      .toLocaleLowerCase('hu').replace(/[\s-]/g, '');
    if (['kek', 'zold', 'piros', 'blue', 'green', 'red'].includes(name)) return 'rgb';
    if (name === '255nm' || name === '265nm') return 'filter_255nm';
    if (name === '365nm') return 'filter_365nm';
    return null;
  }

  private enforceAutofocusReferenceRow(): void {
    const firstRow = this.capturePlan[0];
    if (!firstRow) return;
    firstRow.wavelength = this.autofocusSettings.channel;
    firstRow.brightness = this.autofocusSettings.channel === 'vis'
      ? 'full'
      : this.autofocusSettings.brightness;
    firstRow.filter_position = this.autofocusSettings.filter_position;
  }

  private syncAutofocusReferenceRow(): void {
    if (!this.autofocusSettingsLoaded || !this.capturePlanLoaded) return;
    this.enforceAutofocusReferenceRow();
    if (this.capturePlan[0]) this.applyCameraCombination(this.capturePlan[0]);
    this.persistCapturePlan();
  }

  getFilterSlotLabel(position: number): string {
    const filterId = this.filterSettings.slots[position - 1];
    if (!filterId) return position === 1 ? 'Üres' : '—';
    const filter = this.filterSettings.filters.find(item => item.id === filterId);
    return filter?.name || '—';
  }

  onCaptureNumberBlur(row: CapturePlanRow): void {
    if (this.isExposureValid(row)) {
      row.exposure_time = this.parseDecimal(row.exposure_time_text)!;
      row.exposure_time_text = this.formatExposureText(String(row.exposure_time));
    }
    if (this.isCapturePlanRowValid(row)) {
      this.persistCapturePlan();
    }
  }

  onGainBlur(row: CapturePlanRow): void {
    if (this.isGainValid(row)) {
      row.gain = this.parseDecimal(row.gain_text)!;
      row.gain_text = String(row.gain);
    }
    if (this.isCapturePlanRowValid(row)) {
      this.persistCapturePlan();
    }
  }

  onCaptureNumberInput(
    row: CapturePlanRow,
    field: 'exposure_time' | 'gain',
    value: string
  ): void {
    if (field === 'exposure_time') {
      const ungroupedValue = value.replace(/\s/g, '');
      if (!/^\d*(?:[.,]\d*)?$/.test(ungroupedValue)) return;
      row.exposure_time_text = this.formatExposureText(ungroupedValue);
      const parsed = this.parseDecimal(ungroupedValue);
      if (parsed !== null) row.exposure_time = parsed;
    } else {
      if (!/^\d*(?:[.,]\d*)?$/.test(value)) return;
      row.gain_text = value;
      const parsed = this.parseDecimal(value);
      if (parsed !== null) row.gain = parsed;
    }
  }

  blockInvalidNumberKey(event: KeyboardEvent): void {
    if (event.ctrlKey || event.metaKey || event.altKey) return;
    const allowedControlKeys = [
      'Backspace', 'Delete', 'Tab', 'ArrowLeft', 'ArrowRight', 'Home', 'End'
    ];
    if (allowedControlKeys.includes(event.key) || /^\d$/.test(event.key)) return;
    if (event.key === '.' || event.key === ',') {
      const input = event.target as HTMLInputElement;
      if (!input.value.includes('.') && !input.value.includes(',')) return;
    }
    event.preventDefault();
  }

  blockInvalidNumberPaste(event: ClipboardEvent): void {
    const pastedText = event.clipboardData?.getData('text') ?? '';
    const input = event.target as HTMLInputElement;
    const selectionStart = input.selectionStart ?? input.value.length;
    const selectionEnd = input.selectionEnd ?? selectionStart;
    const prospectiveValue =
      input.value.slice(0, selectionStart) + pastedText + input.value.slice(selectionEnd);
    const valueToValidate = input.closest('.exposure-column')
      ? prospectiveValue.replace(/\s/g, '')
      : prospectiveValue;
    if (!/^\d*(?:[.,]\d*)?$/.test(valueToValidate)) {
      event.preventDefault();
    }
  }

  isCapturePlanRowValid(row: CapturePlanRow): boolean {
    return this.isExposureValid(row) && this.isGainValid(row);
  }

  isExposureValid(row: CapturePlanRow): boolean {
    return this.isValueInRange(this.parseDecimal(row.exposure_time_text), this.exposureRange);
  }

  isGainValid(row: CapturePlanRow): boolean {
    return this.isValueInRange(this.parseDecimal(row.gain_text), this.gainRange, true);
  }

  private parseDecimal(value: string): number | null {
    const ungroupedValue = value.replace(/\s/g, '');
    if (!/^\d+(?:[.,]\d+)?$/.test(ungroupedValue)) return null;
    const parsed = Number(ungroupedValue.replace(',', '.'));
    return Number.isFinite(parsed) ? parsed : null;
  }

  private formatExposureText(value: string): string {
    const separatorIndex = value.search(/[.,]/);
    const integerPart = separatorIndex >= 0 ? value.slice(0, separatorIndex) : value;
    const decimalPart = separatorIndex >= 0 ? value.slice(separatorIndex) : '';
    return integerPart.replace(/\B(?=(\d{3})+(?!\d))/g, ' ') + decimalPart;
  }

  private isValueInRange(
    value: unknown,
    range?: CameraParameterRange,
    allowZero = false
  ): boolean {
    if (!(allowZero ? this.isFiniteNonNegative(value) : this.isFinitePositive(value))) return false;
    const numericValue = Number(value);
    if (!range) return true;
    if (numericValue < range.min || numericValue > range.max) return false;
    const increment = range.inc || 0;
    if (increment <= 0) return true;
    const steps = Math.round((numericValue - range.min) / increment);
    const accepted = range.min + steps * increment;
    return Math.abs(accepted - numericValue) <= 1e-6;
  }

  private isFinitePositive(value: unknown): boolean {
    return typeof value === 'number' && Number.isFinite(value) && value > 0;
  }

  private isFiniteNonNegative(value: unknown): boolean {
    return typeof value === 'number' && Number.isFinite(value) && value >= 0;
  }

  // ===== Tablet context menu =====

  onTabletContextMenu(event: MouseEvent, id: number): void {
    // Prevent context menu if measurement is active
    if (this.measurementActive) {
      event.preventDefault();
      return;
    }

    event.preventDefault();

    // Estimated menu dimensions
    const menuWidth = 180;
    const menuHeight = 80;

    // Get viewport dimensions
    const viewportWidth = window.innerWidth;
    const viewportHeight = window.innerHeight;

    // Calculate position, adjusting if menu would go off-screen
    let menuX = event.clientX;
    let menuY = event.clientY;

    // Adjust horizontal position if menu would overflow right edge
    if (menuX + menuWidth > viewportWidth) {
      menuX = viewportWidth - menuWidth - 5;
    }

    // Adjust vertical position if menu would overflow bottom edge
    if (menuY + menuHeight > viewportHeight) {
      menuY = viewportHeight - menuHeight - 5;
    }

    // Ensure menu doesn't go off left/top edges
    menuX = Math.max(5, menuX);
    menuY = Math.max(5, menuY);

    this.tabletContextMenuVisible = true;
    this.tabletContextMenuX = menuX;
    this.tabletContextMenuY = menuY;
    this.tabletContextMenuId = id;
  }

  hideTabletContextMenu(): void {
    this.tabletContextMenuVisible = false;
    this.tabletContextMenuId = null;
  }

  selectTablet(): void {
    if (this.tabletContextMenuId !== null) {
      this.onDotClick(this.tabletContextMenuId);
    }
    this.hideTabletContextMenu();
  }

  moveToTablet(): void {
    if (this.tabletContextMenuId === null || !this.tabletHomed) {
      console.warn('moveToTablet blocked: tabletHomed =', this.tabletHomed);
      return;
    }

    const pos = this.getTabletPosition(this.tabletContextMenuId);
    if (pos) {
      this.sharedService.invalidateAutofocus();
      this.http.post(`${BASE_URL}/move_toolhead_absolute`, {
        x: pos.x,
        y: pos.y
      }).subscribe({
        next: () => {
          console.log(`Moved to tablet ${this.tabletContextMenuId}`);
          this.hideTabletContextMenu();
        },
        error: (err) => {
          console.error('Failed to move toolhead:', err);
        }
      });
    }
  }

  private getTabletPosition(tabletId: number): { x: number; y: number } | null {
    // Tablet IDs: 1-10 = bottom row, 11-20 = next row up, etc.
    // For gridSize=10: tablet 1 is at (row=0, col=0), tablet 2 at (row=0, col=1), etc.
    
    const tabletIndex = tabletId - 1; // Convert 1-based to 0-based
    if (tabletIndex < 0 || tabletIndex >= this.gridSize * this.gridSize) {
      return null;
    }

    // Calculate row from bottom and column from left
    const rowFromBottom = Math.floor(tabletIndex / this.gridSize);
    const col = tabletIndex % this.gridSize;

    // Calculate position
    const x = this.firstTabletX + col * this.tabletSpacing;
    const y = this.firstTabletY + rowFromBottom * this.tabletSpacing;

    return { x, y };
  }

  /**
   * Convert tablet ID to label format (A1, B1, C1, etc.)
   * Columns are letters (A, B, C...), rows are numbers (1, 2, 3...)
   * ID 1 = A1, ID 2 = B1, ID 10 = J1, ID 11 = A2, etc. (for 10x10 grid)
   */
  getTabletLabel(id: number): string {
    const index = id - 1; // Convert to 0-based index
    const col = index % this.gridSize; // Column determines letter
    const row = Math.floor(index / this.gridSize) + 1; // Row determines number (1-based)
    const letter = String.fromCharCode(65 + col); // 65 = 'A'
    return letter + row;
  }

  /**
   * Returns the tooltip text for a tablet based on its measurement state.
   */
  getTabletTooltip(id: number): string {
    const label = this.getTabletLabel(id);
    if (this.completedTablets.has(id)) {
      return `${label} - Sikeres mérés`;
    }
    if (this.currentTabletId === id) {
      return `${label} - Mérés folyamatban`;
    }
    if (this.failedTablets.has(id)) {
      const error = this.tabletErrors.get(id);
      return error ? `${label} - ${error}` : `${label} - Hiba`;
    }
    return label;
  }

  // ===== Folder selection =====

  async selectSaveFolder(): Promise<void> {
    if (this.measurementActive) return;
    
    // Electron environment: use native dialog via preload API
    if (window.electronAPI?.selectFolder) {
      try {
        const folder = await window.electronAPI.selectFolder();
        if (folder) {
          const normalized = normalizePath(folder);
          this.saveLocation = normalized;
          this.autoService.updateSettings('save_location', normalized).subscribe({
            error: (err) => console.warn('Failed to save location setting:', err)
          });
        }
      } catch (e) {
        console.error('Folder selection error (Electron):', e);
      }
      return;
    }

    // Fallback for dev/browser: call backend to open a Tkinter dialog
    this.autoService.selectFolder().subscribe({
      next: (res) => {
        if (res.folder) {
          const normalized = normalizePath(res.folder);
          this.saveLocation = normalized;
          // Persist to settings
          this.autoService.updateSettings('save_location', normalized).subscribe({
            error: (err) => console.warn('Failed to save location setting:', err)
          });
        }
      },
      error: (err) => {
        console.error('Failed to select folder:', err);
      }
    });
  }

  // ===== DOT STATE METHODS =====

  isDotSelected(id: number): boolean {
    const selected = this.selectedSignal();
    if (selected.has(id)) {
      return true;
    }

    const anchor = this.rangeAnchorSignal();
    const hover = this.hoverIndexSignal();
    if (anchor != null && hover != null) {
      const start = Math.min(anchor, hover);
      const end = Math.max(anchor, hover);
      return id >= start && id <= end;
    }

    return false;
  }

  isDotCompleted(id: number): boolean {
    return this.completedTablets.has(id);
  }

  isDotFailed(id: number): boolean {
    return this.failedTablets.has(id);
  }

  isDotInProgress(id: number): boolean {
    return this.currentTabletId === id;
  }

  // Determine dot state for CSS class binding
  getDotState(id: number): 'completed' | 'failed' | 'in-progress' | 'pending' | 'none' {
    if (this.isDotCompleted(id)) {
      return 'completed';
    }
    if (this.isDotFailed(id)) {
      return 'failed';
    }
    if (this.isDotInProgress(id)) {
      return 'in-progress';
    }
    if (this.isDotSelected(id)) {
      return 'pending';
    }
    return 'none';
  }

  // ===== CLICK SELECTION LOGIC =====

  onDotClick(id: number): void {
    // If the tablet was successfully measured, open its saved images (works during and after measurement)
    if (this.completedTablets.has(id)) {
      this.openTabletImages(id);
      return;
    }

    if (this.measurementActive) {
      // Selection is locked while measurement is running
      return;
    }

    const selected = new Set(this.selectedSignal());
    const anchor = this.rangeAnchorSignal();

    if (anchor === null) {
      if (selected.has(id)) {
        selected.delete(id);
        this.selectedSignal.set(selected);
        this.hoverIndexSignal.set(null);
      } else {
        selected.add(id);
        this.selectedSignal.set(selected);
        this.rangeAnchorSignal.set(id);
        this.hoverIndexSignal.set(null);
      }
      return;
    }

    if (id === anchor) {
      this.rangeAnchorSignal.set(null);
      this.hoverIndexSignal.set(null);
      return;
    }

    const start = Math.min(anchor, id);
    const end = Math.max(anchor, id);
    for (let v = start; v <= end; v++) {
      selected.add(v);
    }

    this.selectedSignal.set(selected);
    this.rangeAnchorSignal.set(null);
    this.hoverIndexSignal.set(null);
  }

  onDotMouseEnter(id: number): void {
    if (this.measurementActive) {
      this.hoverIndexSignal.set(null);
      return;
    }
    if (this.rangeAnchorSignal() !== null) {
      this.hoverIndexSignal.set(id);
    } else {
      this.hoverIndexSignal.set(null);
    }
  }

  onGridMouseLeave(): void {
    this.hoverIndexSignal.set(null);
  }

  clearSelection(): void {
    if (this.measurementActive) {
      return;
    }
    this.selectedSignal.set(new Set<number>());
    this.rangeAnchorSignal.set(null);
    this.hoverIndexSignal.set(null);
    this.completedTablets.clear();
    this.failedTablets.clear();
    this.tabletErrors.clear();
    this.currentTabletId = null;
    this.currentTabletIndex = 0;
    this.errorMessage = null;
    this.successMessage = null;
    this.validationMessage = null;
    this.sharedService.clearToolbarNotice();
  }

  // ===== Position calculation =====

  private calculateTabletPosition(tabletIndex: number): TabletPosition {
    const zeroBasedIndex = tabletIndex - 1;
    const rowFromBottom = Math.floor(zeroBasedIndex / this.gridSize);
    const col = zeroBasedIndex % this.gridSize;
    
    return {
      index: tabletIndex,
      x: this.firstTabletX + col * this.tabletSpacing,
      y: this.firstTabletY + rowFromBottom * this.tabletSpacing
    };
  }

  // ===== Start / Stop measurement =====

  startMeasurement(): void {
    // Toggle behavior - if running, stop
    if (this.measurementActive) {
      this.stopMeasurement();
      return;
    }
    if (this.scannerOperationActive) {
      return;
    }

    const indices = Array.from(this.selectedSignal()).sort((a, b) => a - b);
    if (indices.length === 0) {
      return;
    }

    // Clear messages and reset state
    this.errorMessage = null;
    this.successMessage = null;
    this.validationMessage = null;
    this.sharedService.clearToolbarNotice();
    this.stopRequested = false;
    this.reconnectAttemptCount = 0;
    this.completedTablets.clear();
    this.failedTablets.clear();
    this.tabletErrors.clear();
    this.currentTabletId = null;
    this.currentTabletIndex = 0;

    // Create measurement folder path (normalized for cross-platform consistency)
    const safeLocation = normalizePath(this.saveLocation);
    const safeName = this.measurementName.trim().replace(/[<>:"|?*\\]/g, '_');
    this.measurementFolder = `${safeLocation}/${safeName}`;

    // Set measurement active (locks UI)
    this.measurementActive = true;
    this.sharedService.setMeasurementActive(true);

    // Check if already homed, then home if needed or proceed directly
    this.checkHomedThenProceed(indices);
  }

  private checkHomedThenProceed(indices: number[]): void {
    // Always home at the start of auto measurement to ensure a known reference.
    this.homeMotionPlatformThenProceed(indices);
  }

  private homeMotionPlatformThenProceed(indices: number[]): void {
    this.validationMessage = 'Mozgásplatform pozicionálása...';

    // Signal motion-control component that homing is in progress
    this.sharedService.setMotionHoming(true);

    const axesOrder: Array<'z' | 'y' | 'x' | 'a'> = ['z', 'y', 'x', 'a'];

    const homing$ = from(axesOrder).pipe(
      concatMap((axis) => this.http.post(`${BASE_URL}/home_toolhead`, {
        axes: [axis],
        select_autofocus_filter: axis === 'a'
      })),
      last(),
      switchMap(() => this.http.get<{ x?: number | null; y?: number | null; z?: number | null }>(`${BASE_URL}/get_motion_platform_position`)),
      finalize(() => {
        this.sharedService.setMotionHoming(false);
        this.homingSubscription = null;
      })
    );

    this.homingSubscription = homing$.subscribe({
      next: (position) => {
        console.log('Motion platform and filter revolver homed successfully (Z→Y→X→A). Position:', position);
        this.validationMessage = null;
        
        // Update motion position and homed state via SharedService
        this.sharedService.setMotionPosition({
          x: position?.x ?? null,
          y: position?.y ?? null,
          z: position?.z ?? null
        });
        
        // Explicitly set homed state to true after successful homing
        this.sharedService.setMotionHomed(true);
        
        this.processTabletQueue(indices, 0);
      },
      error: (err) => {
        console.error('Failed to home motion platform:', err);

        if (this.stopRequested) {
          this.finishMeasurement(false);
          return;
        }

        // Detect USB disconnect during homing
        if (this.isDeviceDisconnectError(err)) {
          const device = this.detectDisconnectedDevice(err);
          this.attemptDeviceReconnect(
            device,
            'nullázás',
            () => this.homeMotionPlatformThenProceed(indices)
          );
          return;
        }

        this.errorMessage = 'Hiba: Nem sikerült pozicionálni a mozgásplatformot.';
        this.publishToolbarNotice('error', this.errorMessage);
        this.finishMeasurement(false);
      }
    });
  }

  stopMeasurement(): void {
    this.stopRequested = true;

    // Signal backend to abort autofocus immediately
    this.http.post(`${BASE_URL}/abort-autofocus`, {}).subscribe({
      next: () => console.log('Autofocus abort signal sent to backend'),
      error: (err) => console.warn('Could not send abort signal:', err)
    });

    // Note: Lights will be turned off in finishMeasurement()
    // No need to duplicate the call here

    // Cancel any active operations immediately
    this.cancelActiveOperations();

    // Clear homing flag so UI unlocks promptly
    this.sharedService.setMotionHoming(false);

    // Finish measurement immediately so UI resets
    this.finishMeasurement(false);
  }

  private cancelActiveOperations(): void {
    // Cancel current tablet measurement
    if (this.currentTabletSubscription) {
      this.currentTabletSubscription.unsubscribe();
      this.currentTabletSubscription = null;
    }
    this.stopProgressPolling();

    // Cancel homing operation
    if (this.homingSubscription) {
      this.homingSubscription.unsubscribe();
      this.homingSubscription = null;
    }

    // Cancel reconnection timer
    if (this.reconnectTimer) {
      clearInterval(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    this.reconnecting = false;
    this.reconnectMessage = null;
  }

  private async processTabletQueue(indices: number[], queueIndex: number): Promise<void> {
    // Check if stopped or finished
    if (this.stopRequested || queueIndex >= indices.length) {
      this.finishMeasurement(queueIndex >= indices.length);
      return;
    }

    const tabletId = indices[queueIndex];
    const position = this.calculateTabletPosition(tabletId);
    
    // Update progress
    this.currentTabletIndex = queueIndex + 1;
    this.currentTabletId = tabletId;
    
    // Determine if this is the first tablet (for coarse vs fine autofocus)
    const isFirstTablet = queueIndex === 0;

    const requestId = crypto.randomUUID?.() ?? `${Date.now()}-${Math.random()}`;
    const req: TabletStepRequest = {
      request_id: requestId,
      tablet_index: tabletId,
      x: position.x,
      y: position.y,
      z: this.firstTabletZ,
      measurement_folder: this.measurementFolder,
      measurement_name: this.measurementName.trim(),
      autofocus: this.autofocus,
      capture_plan: this.capturePlan.map(
        ({ wavelength, brightness, filter_position, exposure_time, gain }) =>
          ({ wavelength, brightness, filter_position, exposure_time, gain })
      ),
      is_first_tablet: isFirstTablet,
      background_subtraction: this.backgroundSubtraction
    };

    // If stop was requested before starting this tablet, exit early
    if (this.stopRequested) {
      this.finishMeasurement(false);
      return;
    }

    this.startProgressPolling(requestId, tabletId);
    this.currentTabletSubscription = this.autoService.measureSingleTablet(req).subscribe({
      next: (resp) => {
        if (this.stopRequested) {
          this.currentTabletSubscription?.unsubscribe();
          this.currentTabletSubscription = null;
          this.finishMeasurement(false);
          return;
        }

        if (resp.status === 'success') {
          this.stopProgressPolling();
          // Check if any E2xxx error was flagged (tablet missing, exposure, position, quality, etc.)
          if (resp.af_error_code && resp.af_error_code.startsWith('E2')) {
            // Mark tablet as failed (red) and store the error message
            this.failedTablets.add(tabletId);
            this.tabletErrors.set(tabletId, resp.af_error_message ?? resp.af_error_code);
            this.errorMessage = `${this.getTabletLabel(tabletId)} tabletta: ${resp.af_error_message ?? resp.af_error_code}`;
            this.errorNotificationService.addError({
              code: `AUTO_TABLET_${tabletId}_${resp.af_error_code}`,
              message: this.errorMessage,
              severity: 'warning',
            });
          } else {
            // Mark tablet as completed (green)
            this.completedTablets.add(tabletId);
          }
          
          // Emit saved images to gallery and store paths (exclude background-subtracted _masked images)
          if (resp.saved_images && resp.saved_images.length > 0) {
            const nonMaskedPaths: string[] = [];
            for (const imagePath of resp.saved_images) {
              if (imagePath.includes('_masked')) {
                continue;
              }
              nonMaskedPaths.push(imagePath);
              this.emitGalleryImage(imagePath, tabletId);
            }
            if (nonMaskedPaths.length > 0) {
              this.tabletImages.set(tabletId, nonMaskedPaths);
            }
          }

          // Process next tablet
          this.currentTabletSubscription?.unsubscribe();
          this.currentTabletSubscription = null;
          this.processTabletQueue(indices, queueIndex + 1);
        } else {
          this.stopProgressPolling();
          // Error during measurement
          this.errorMessage = `A(z) ${this.getTabletLabel(tabletId)} tabletta mérése sikertelen.`;
          this.publishToolbarNotice('error', this.errorMessage);
          this.currentTabletSubscription?.unsubscribe();
          this.currentTabletSubscription = null;
          this.finishMeasurement(false);
        }
      },
      error: (err) => {
        this.stopProgressPolling();
        this.currentTabletSubscription?.unsubscribe();
        this.currentTabletSubscription = null;

        if (this.stopRequested) {
          this.finishMeasurement(false);
          return;
        }

        // Check if this is a device disconnect
        if (this.isDeviceDisconnectError(err)) {
          const device = this.detectDisconnectedDevice(err);
          this.attemptDeviceReconnect(
            device,
            `tabletta ${tabletId}`,
            () => this.processTabletQueue(indices, queueIndex)
          );
          return;
        }

        const errorCode = err?.error?.code;
        this.errorMessage = errorCode
          ? this.errorNotificationService.getMessage(errorCode)
          : `Szerverhiba a(z) ${this.getTabletLabel(tabletId)} tabletta mérésekor.`;
        // Error responses are already presented by the HTTP interceptor. Do
        // not add a second notice containing low-level backend exception text.
        this.finishMeasurement(false);
      }
    });
  }

  private startProgressPolling(requestId: string, tabletId: number): void {
    this.stopProgressPolling();
    this.emittedProgressPaths.clear();
    this.currentProgressSubscription = timer(100, 250).pipe(
      switchMap(() => this.autoService.getProgress(requestId).pipe(
        catchError(err => {
          if (err?.status !== 404) console.warn('Failed to poll measurement image progress:', err);
          return EMPTY;
        })
      ))
    ).subscribe({
      next: progress => {
        this.setActivePlanRow(progress.active_plan_row_index);
        this.errorNotificationService.addWarnings(progress.warnings);
        for (const image of progress.images) {
          if (!image.masked) this.emitGalleryImage(image.path, tabletId);
        }
      }
    });
  }

  private stopProgressPolling(): void {
    this.currentProgressSubscription?.unsubscribe();
    this.currentProgressSubscription = null;
    this.setActivePlanRow(null);
  }

  private setActivePlanRow(rowIndex: number | null): void {
    if (this.activePlanRowIndex === rowIndex) return;
    this.activePlanRowIndex = rowIndex;
    if (this.planRowScrollTimer) clearTimeout(this.planRowScrollTimer);
    this.planRowScrollTimer = null;
    if (rowIndex === null) return;

    this.planRowScrollTimer = setTimeout(() => {
      this.planRowScrollTimer = null;
      const container = this.capturePlanScroll?.nativeElement;
      const row = container?.querySelector<HTMLElement>(`[data-plan-row-index="${rowIndex}"]`);
      row?.scrollIntoView({ block: 'nearest', inline: 'nearest', behavior: 'smooth' });
    });
  }

  private emitGalleryImage(path: string, tabletId: number): void {
    const key = path.replace(/\\/g, '/').toLowerCase();
    if (this.emittedProgressPaths.has(key)) return;
    this.emittedProgressPaths.add(key);
    this.sharedService.emitSavedImage({ path, tabletIndex: tabletId });
    const current = this.tabletImages.get(tabletId) ?? [];
    if (!current.some(existing => existing.replace(/\\/g, '/').toLowerCase() === key)) {
      this.tabletImages.set(tabletId, [...current, path]);
    }
  }

  private applyTrayGeometry(settings: AdvancedMotionSettings): void {
    this.firstTabletX = settings.first_tablet_x_mm;
    this.firstTabletY = settings.first_tablet_y_mm;
    this.firstTabletZ = settings.first_tablet_z_mm;
    this.tabletSpacing = settings.tablet_spacing_mm;
  }

  // ===== Device disconnect detection helpers =====

  /**
   * Check if an HTTP error indicates a device (motion platform or camera) disconnect.
   */
  private isDeviceDisconnectError(err: any): boolean {
    const code = err?.error?.code;
    const details = err?.error?.details || err?.error?.message || err?.error?.error || '';
    const status = err?.status;

    // Motion platform: E1201 with 503
    if (code === 'E1201' && status === 503) return true;

    // Camera: E1111 with 503
    if (code === 'E1111' && status === 503) return true;

    // Serial exception wrapped in 500
    if (status === 500 && (
      details.includes('SerialException') ||
      details.includes('PermissionError') ||
      details.includes('WriteFile failed') ||
      details.includes('ClearCommError')
    )) return true;

    // Camera disconnect in 500
    if (status === 500 && (
      details.includes('Camera not ready') ||
      details.includes('Camera disconnected') ||
      details.includes('Grab failed') ||
      details.includes('Failed to grab') ||
      details.includes('physically removed') ||
      details.includes('not open')
    )) return true;

    return false;
  }

  /**
   * Determine which device is disconnected based on the error.
   * Returns 'motion' or 'camera'.
   */
  private detectDisconnectedDevice(err: any): 'motion' | 'camera' {
    const code = err?.error?.code;
    if (code === 'E1111') return 'camera';

    const details = err?.error?.details || err?.error?.message || err?.error?.error || '';
    if (
      details.includes('Camera not ready') ||
      details.includes('Camera disconnected') ||
      details.includes('Grab failed') ||
      details.includes('Failed to grab') ||
      details.includes('physically removed') ||
      details.includes('not open')
    ) return 'camera';

    return 'motion';
  }

  /**
   * Attempt to reconnect to the specified device for up to 30 seconds.
   * If reconnection succeeds, call resumeCallback to continue the measurement.
   * If it fails after 30s, show a toolbar error and stop.
   *
   * @param device 'motion' or 'camera'
   * @param context Hungarian operator-facing context (e.g. 'nullázás' or 'tabletta 3')
   * @param resumeCallback Function to call after successful reconnection
   */
  private attemptDeviceReconnect(
    device: 'motion' | 'camera',
    context: string,
    resumeCallback: () => void
  ): void {
    const RECONNECT_TIMEOUT_S = 30;
    const RECONNECT_INTERVAL_MS = 3000;
    const startTime = Date.now();

    const deviceName = device === 'motion' ? 'Mozgásplatform' : 'Kamera';
    const errorCode = device === 'motion' ? 'E1201' : 'E1111';

    // Track how many full reconnect cycles have been attempted.
    // If we keep cycling (connect returns 200 but device is still dead),
    // stop after MAX_RECONNECT_CYCLES to prevent an infinite loop.
    this.reconnectAttemptCount++;
    if (this.reconnectAttemptCount > AutoMeasurementComponent.MAX_RECONNECT_CYCLES) {
      console.error(`${deviceName} reconnect cycle limit reached (${AutoMeasurementComponent.MAX_RECONNECT_CYCLES}). Stopping measurement.`);
      this.errorNotificationService.addError({
        code: errorCode,
        message: `${deviceName} kapcsolat megszakadt (${context}). Többszöri újracsatlakozás sikertelen.`
      });
      this.errorMessage = `${deviceName} újracsatlakozás többszöri sikertelen próbálkozás után. Mérés megszakítva.`;
      this.reconnectAttemptCount = 0;
      this.finishMeasurement(false);
      return;
    }

    this.reconnecting = true;
    this.reconnectMessage = `${deviceName} kapcsolat megszakadt (${context}). Újracsatlakozás... (0/${RECONNECT_TIMEOUT_S}s) [${this.reconnectAttemptCount}/${AutoMeasurementComponent.MAX_RECONNECT_CYCLES}]`;

    this.reconnectTimer = setInterval(() => {
      const elapsedMs = Date.now() - startTime;
      const elapsedS = Math.round(elapsedMs / 1000);

      // If stop was requested, cancel reconnection
      if (this.stopRequested) {
        this.clearReconnectState();
        this.finishMeasurement(false);
        return;
      }

      // Timeout reached — give up
      if (elapsedMs >= RECONNECT_TIMEOUT_S * 1000) {
        this.clearReconnectState();

        // Show the same compact toolbar error used by other scanner failures.
        this.errorNotificationService.addError({
          code: errorCode,
          message: `${deviceName} kapcsolat megszakadt (${context}). Újracsatlakozás sikertelen (${RECONNECT_TIMEOUT_S}s).`
        });

        this.errorMessage = `${deviceName} újracsatlakozás sikertelen (${RECONNECT_TIMEOUT_S}s). Mérés megszakítva.`;
        this.finishMeasurement(false);
        return;
      }

      // Update message with countdown
      this.reconnectMessage = `${deviceName} kapcsolat megszakadt (${context}). Újracsatlakozás... (${elapsedS}/${RECONNECT_TIMEOUT_S}s) [${this.reconnectAttemptCount}/${AutoMeasurementComponent.MAX_RECONNECT_CYCLES}]`;

      // Try to reconnect to the appropriate device
      const reconnect$ = device === 'motion'
        ? this.autoService.reconnectMotionPlatform()
        : this.autoService.reconnectCamera();

      reconnect$.subscribe({
        next: (resp: any) => {
          const msg = resp?.message || '';
          if (msg.includes('failed')) return; // not actually connected

          console.info(`${deviceName} reconnected during auto-measurement:`, msg);
          this.clearReconnectState();
          // Reset cycle counter on successful reconnection
          this.reconnectAttemptCount = 0;

          // Remove the disconnect error popup immediately so it doesn't linger.
          // Without this, the motion-control component's reconnection polling
          // must independently discover the reconnection to clear the error,
          // which is a race condition that sometimes fails.
          this.errorNotificationService.removeError(errorCode);

          // Update shared service so UI reflects the reconnected state
          if (device === 'motion') {
            this.sharedService.setMotionPlatformConnectionStatus(true);
          } else {
            this.sharedService.setCameraConnectionStatus(true);
          }

          // Resume the operation
          this.reconnectMessage = `Újracsatlakozás sikeres. Mérés folytatása...`;
          setTimeout(() => {
            this.reconnectMessage = null;
            resumeCallback();
          }, 500);
        },
        error: () => {
          // Reconnection attempt failed — timer will try again
          console.warn(`${deviceName} reconnect attempt failed (${elapsedS}s elapsed)`);
        }
      });
    }, RECONNECT_INTERVAL_MS);
  }

  private clearReconnectState(): void {
    if (this.reconnectTimer) {
      clearInterval(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    this.reconnecting = false;
  }

  private finishMeasurement(success: boolean): void {
    // Clean up reconnection state
    this.clearReconnectState();
    this.reconnectMessage = null;

    // Turn off all lights whenever measurement finishes (success or stopped)
    this.http.post(`${BASE_URL}/turn-off-all-lights`, {}).subscribe({
      next: () => {
        console.log('All lights turned off after measurement completion');
        // Emit event to notify other UI components that lights are off
        this.sharedService.emitLightsOff();
      },
      error: (err) => console.warn('Could not turn off lights after measurement:', err)
    });

    this.measurementActive = false;
    this.sharedService.setMeasurementActive(false);
    this.currentTabletId = null;
    
    if (this.stopRequested) {
      const failedNote = this.failedTablets.size > 0 ? ` ${this.failedTablets.size} hibás.` : '';
      this.successMessage = `Mérés leállítva. ${this.completedTablets.size} tabletta mérése kész.${failedNote}`;
      this.publishToolbarNotice('info', this.successMessage);
    } else if (success) {
      const failedNote = this.failedTablets.size > 0 ? ` ${this.failedTablets.size} hibás.` : '';
      this.successMessage = `Mérés sikeresen befejezve. ${this.completedTablets.size} tabletta mérése kész.${failedNote}`;
      this.publishToolbarNotice('success', this.successMessage);
      this.scheduleBedMoveToZero();
    }
    // Error message is set in processTabletQueue if there was an error
    
    this.stopRequested = false;
  }

  private openTabletImages(id: number): void {
    const paths = this.tabletImages.get(id);
    if (!paths || paths.length === 0) {
      return;
    }
    for (const path of paths) {
      this.http.post(`${BASE_URL}/open_image`, { path }).subscribe({
        error: (err) => console.error('Failed to open image:', err)
      });
    }
  }

  private scheduleBedMoveToZero(): void {
    if (!this.motionConnected) {
      return;
    }

    setTimeout(() => {
      if (!this.motionConnected) {
        return;
      }

      this.http.post(`${BASE_URL}/move_toolhead_absolute`, { z: 0 }).subscribe({
        next: () => console.log('Auto-measurement complete: moved bed to Z=0.'),
        error: (err) => console.warn('Failed to move bed to Z=0 after auto-measurement:', err)
      });
    }, 2000);
  }
}
