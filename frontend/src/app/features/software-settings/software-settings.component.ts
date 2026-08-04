import { CommonModule } from '@angular/common';
import { Component, EventEmitter, HostListener, OnDestroy, OnInit, Output } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { MatIconModule } from '@angular/material/icon';
import {
  Observable,
  Subject,
  Subscription,
  catchError,
  concatMap,
  debounceTime,
  forkJoin,
  map,
  of,
  switchMap,
  takeUntil,
  tap
} from 'rxjs';
import { AdvancedLampSettings, LampSettings, UvLampSettings } from '../../models/light.models';
import { LampSettingsService } from '../../services/lamp-settings.service';
import { FilterSettingsService } from '../../services/filter-settings.service';
import {
  FilterDefinition,
  FilterSettings,
  HeightOffsetChannel,
  HeightOffsetRow
} from '../../models/filter-settings.models';
import { MotionSettingsService } from '../../services/motion-settings.service';
import { SharedService } from '../../shared.service';
import { FilterRevolverComponent } from '../../components/filter-revolver/filter-revolver.component';
import { CameraImageSettingsService } from '../../services/camera-image-settings.service';
import { CameraImageSettings, CameraIntegerLimit } from '../../models/camera-image-settings.models';

type SettingsType = 'filter' | 'lamp' | 'camera' | 'tray' | 'advanced';
type UvChannel = 'uv255' | 'uv310' | 'uv365';

interface LampRow {
  channel: UvChannel | 'vis';
  label: string;
  settings: Partial<UvLampSettings>;
}

@Component({
  selector: 'app-software-settings',
  standalone: true,
  imports: [CommonModule, FormsModule, MatIconModule, FilterRevolverComponent],
  templateUrl: './software-settings.component.html',
  styleUrls: ['./software-settings.component.css']
})
export class SoftwareSettingsComponent implements OnInit, OnDestroy {
  @Output() close = new EventEmitter<void>();

  selectedType: SettingsType = 'filter';
  measurementActive = false;
  loadingLampSettings = false;
  loadingFilterSettings = false;
  savingFilterSettings = false;
  filterError = '';
  filterSaved = false;
  savingLampSettings = false;
  lampError = '';
  lampSaved = false;
  loadingCameraSettings = false;
  savingCameraSettings = false;
  centeringCameraAxis: 'x' | 'y' | null = null;
  cameraError = '';
  cameraSaved = false;
  cameraConnected = false;
  cameraImageSettings: CameraImageSettings = {
    override_enabled: false, width: 4000, height: 4000, offset_x: 0, offset_y: 0
  };
  cameraLimits: Partial<Record<'width' | 'height' | 'offset_x' | 'offset_y', CameraIntegerLimit>> = {};
  loadingAdvancedSettings = false;
  savingAdvancedSettings = false;
  advancedError = '';
  advancedSaved = false;
  useVirtualComPort = false;
  maxHeightOffsetUpMm = 5;
  maxHeightOffsetDownMm = -5;
  firstTabletXMm = 2.9;
  firstTabletYMm = 10.6;
  firstTabletZMm = 20;
  tabletSpacingMm = 18.3;
  virtualConnectionLabel = '';
  private advancedSettingsLoaded = false;
  private filterSettingsLoaded = false;
  private lampSettingsLoaded = false;
  private cameraSettingsLoaded = false;
  private nextFilterId = 1;
  private readonly destroy$ = new Subject<void>();
  private readonly filterSaveRequests$ = new Subject<void>();
  private readonly lampSaveRequests$ = new Subject<void>();
  private readonly advancedSaveRequests$ = new Subject<void>();
  private readonly cameraSaveRequests$ = new Subject<void>();
  private filterDirty = false;
  private lampDirty = false;
  private advancedDirty = false;
  private measurementSubscription?: Subscription;

  readonly settingsTypes: { id: SettingsType; label: string; icon: string }[] = [
    { id: 'filter', label: 'Szűrőváltó', icon: 'tune' },
    { id: 'lamp', label: 'Lámpa', icon: 'lightbulb' },
    { id: 'camera', label: 'Kamera', icon: 'photo_camera' },
    { id: 'tray', label: 'Tálca', icon: 'table_restaurant' },
    { id: 'advanced', label: 'Haladó', icon: 'build' }
  ];

  lampRows: LampRow[] = [
    { channel: 'uv255', label: '255 nm', settings: {} },
    { channel: 'uv310', label: '310 nm', settings: {} },
    { channel: 'uv365', label: '365 nm', settings: {} },
    { channel: 'vis', label: 'VIS', settings: {} }
  ];
  readonly fixedAdvancedSelectors: Record<UvChannel | 'vis', string> = {
    uv255: 'P2', uv310: 'P3', uv365: 'P1', vis: 'P0'
  };
  readonly heightOffsetChannels: ReadonlyArray<{ id: HeightOffsetChannel; label: string }> = [
    { id: 'uv255', label: '255 nm' },
    { id: 'uv310', label: '310 nm' },
    { id: 'uv365', label: '365 nm' },
    { id: 'vis', label: 'VIS' }
  ];
  advancedSelectors: Record<UvChannel | 'vis', string> = { ...this.fixedAdvancedSelectors };
  filterSettings: FilterSettings = {
    filters: [],
    slots: [null, null, null, null, null, null],
    height_offsets_mm: { empty: this.createEmptyHeightOffsetRow() }
  };

  constructor(
    private readonly lampSettingsService: LampSettingsService,
    private readonly filterSettingsService: FilterSettingsService,
    private readonly motionSettingsService: MotionSettingsService,
    private readonly cameraImageSettingsService: CameraImageSettingsService,
    private readonly sharedService: SharedService
  ) {}

  ngOnInit(): void {
    this.filterSaveRequests$.pipe(
      debounceTime(500),
      concatMap(() => this.persistFilterSettings()),
      takeUntil(this.destroy$)
    ).subscribe();
    this.lampSaveRequests$.pipe(
      debounceTime(500),
      concatMap(() => this.persistLampSettings()),
      takeUntil(this.destroy$)
    ).subscribe();
    this.advancedSaveRequests$.pipe(
      debounceTime(500),
      concatMap(() => this.persistAdvancedSettings()),
      takeUntil(this.destroy$)
    ).subscribe();
    this.cameraSaveRequests$.pipe(
      debounceTime(500),
      concatMap(() => this.persistCameraSettings()),
      takeUntil(this.destroy$)
    ).subscribe();
    this.measurementSubscription = this.sharedService.measurementActive$.subscribe(active => {
      this.measurementActive = active;
    });
    this.loadFilterSettings();
    this.loadAdvancedSettings();
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
    this.measurementSubscription?.unsubscribe();
  }

  selectType(type: SettingsType): void {
    this.selectedType = type;
    if (type === 'lamp' && !this.loadingLampSettings && !this.hasLoadedLampSettings()) {
      this.loadLampSettings();
    }
    if (type === 'filter' && !this.loadingFilterSettings && !this.hasLoadedFilterSettings()) {
      this.loadFilterSettings();
    }
    if (type === 'camera' && !this.loadingCameraSettings && !this.cameraSettingsLoaded) {
      this.loadCameraSettings();
    }
    if (type === 'advanced' && !this.loadingAdvancedSettings && !this.hasLoadedAdvancedSettings()) {
      this.loadAdvancedSettings();
    }
  }

  dismiss(): void {
    // Closing a settings dialog must never be blocked by a delayed or failed
    // automatic save. Pending debounce timers are cancelled in ngOnDestroy.
    this.close.emit();
  }

  @HostListener('document:keydown.escape')
  onEscape(): void {
    this.dismiss();
  }

  onBackdropClick(event: MouseEvent): void {
    if (event.target === event.currentTarget) {
      this.dismiss();
    }
  }

  onLampChanged(): void {
    this.lampError = '';
    this.lampSaved = false;
    this.lampDirty = true;
    this.lampSaveRequests$.next();
  }

  private persistLampSettings(): Observable<boolean> {
    const channels: Partial<Record<UvChannel, UvLampSettings>> = {};

    for (const row of this.lampRows) {
      if (row.channel === 'vis') {
        continue;
      }
      const settings = row.settings;
      if (!this.isValidPercentage(settings.dim_percent) || !this.isValidPercentage(settings.full_percent)
        || !this.isValidTimeout(settings.dim_timeout_seconds) || !this.isValidTimeout(settings.full_timeout_seconds)) {
        this.lampError = 'A fényerőnek 10 és 100% között, a lekapcsolási időnek 0 másodpercnél nagyobbnak kell lennie.';
        return of(false);
      }
      channels[row.channel] = {
        dim_percent: Number(settings.dim_percent),
        full_percent: Number(settings.full_percent),
        dim_timeout_seconds: Number(settings.dim_timeout_seconds),
        full_timeout_seconds: Number(settings.full_timeout_seconds)
      };
    }

    this.savingLampSettings = true;
    return this.lampSettingsService.update({ channels }).pipe(
      tap(() => {
        this.lampSaved = true;
        this.lampDirty = false;
        this.savingLampSettings = false;
      }),
      map(() => true),
      catchError(error => {
        this.lampError = error?.error?.error || 'A lámpabeállítások mentése sikertelen.';
        this.savingLampSettings = false;
        return of(false);
      })
    );
  }

  onCameraSettingsChanged(): void {
    this.cameraError = '';
    this.cameraSaved = false;
    if (!this.hasNumericCameraGeometry()) {
      this.cameraError = 'A képgeometria mezőiben érvényes számokat adjon meg.';
      return;
    }
    this.cameraSaveRequests$.next();
  }

  centerCamera(axis: 'x' | 'y'): void {
    if (!this.cameraImageSettings.override_enabled || !this.cameraConnected || this.centeringCameraAxis) return;
    this.cameraError = '';
    this.cameraSaved = false;
    this.centeringCameraAxis = axis;
    this.cameraImageSettingsService.center(axis).subscribe({
      next: response => {
        this.applyCameraSettingsResponse(response);
        this.cameraSaved = true;
        this.centeringCameraAxis = null;
      },
      error: error => {
        this.cameraError = error?.error?.error || 'A kamerakép középre igazítása sikertelen.';
        this.centeringCameraAxis = null;
      }
    });
  }

  cameraLimit(name: 'width' | 'height' | 'offset_x' | 'offset_y'): CameraIntegerLimit | undefined {
    return this.cameraLimits[name];
  }

  private persistCameraSettings(): Observable<boolean> {
    this.savingCameraSettings = true;
    const payload: CameraImageSettings = {
      override_enabled: this.cameraImageSettings.override_enabled,
      width: Number(this.cameraImageSettings.width),
      height: Number(this.cameraImageSettings.height),
      offset_x: Number(this.cameraImageSettings.offset_x),
      offset_y: Number(this.cameraImageSettings.offset_y)
    };
    return this.cameraImageSettingsService.update(payload).pipe(
      tap(response => {
        this.applyCameraSettingsResponse(response);
        this.cameraSaved = true;
        this.savingCameraSettings = false;
      }),
      map(() => true),
      catchError(error => {
        this.cameraError = error?.error?.error || 'A kamerakép-beállítások mentése sikertelen.';
        this.savingCameraSettings = false;
        return of(false);
      })
    );
  }

  private loadCameraSettings(): void {
    this.loadingCameraSettings = true;
    this.cameraError = '';
    this.cameraImageSettingsService.get().subscribe({
      next: response => {
        this.applyCameraSettingsResponse(response);
        this.cameraSettingsLoaded = true;
        this.loadingCameraSettings = false;
      },
      error: error => {
        this.cameraError = error?.error?.error || 'A kamerabeállítások betöltése sikertelen.';
        this.loadingCameraSettings = false;
      }
    });
  }

  private applyCameraSettingsResponse(response: {
    camera_image_settings: CameraImageSettings;
    limits: Partial<Record<'width' | 'height' | 'offset_x' | 'offset_y', CameraIntegerLimit>>;
    connected: boolean;
  }): void {
    this.cameraImageSettings = { ...response.camera_image_settings };
    this.cameraLimits = { ...response.limits };
    this.cameraConnected = response.connected;
  }

  private hasNumericCameraGeometry(): boolean {
    if (!this.cameraImageSettings.override_enabled) return true;
    if (!this.cameraConnected) return false;
    return (['width', 'height', 'offset_x', 'offset_y'] as const)
      .every(name => Number.isFinite(Number(this.cameraImageSettings[name])));
  }

  addFilter(): void {
    const id = this.createFilterId();
    this.filterSettings.filters.push({
      id,
      name: '',
      wavelength_range: '',
      color: '#ffffff'
    });
    this.filterSettings.height_offsets_mm[id] = this.createEmptyHeightOffsetRow();
    this.filterSaved = false;
  }

  removeFilter(index: number): void {
    const removedId = this.filterSettings.filters[index]?.id;
    this.filterSettings.filters.splice(index, 1);
    if (removedId) {
      this.filterSettings.slots = this.filterSettings.slots.map(slot => slot === removedId ? null : slot);
      delete this.filterSettings.height_offsets_mm[removedId];
    }
    this.onFilterChanged();
  }

  onFilterChanged(): void {
    this.filterSaved = false;
    this.filterError = '';
    this.filterDirty = true;
    this.filterSaveRequests$.next();
  }

  assignFilterToSlot(index: number, filterId: string | null): void {
    if (index === 0) return;
    const slots = [...this.filterSettings.slots];
    slots[index] = filterId;
    this.filterSettings = { ...this.filterSettings, slots };
    this.onFilterChanged();
  }

  assignFilterToSlotFromEvent(index: number, event: Event): void {
    const value = (event.target as HTMLSelectElement).value;
    this.assignFilterToSlot(index, value || null);
  }

  trackSlotIndex(index: number): number {
    return index;
  }

  get selectableFilters(): FilterDefinition[] {
    return this.filterSettings.filters.filter(filter => filter.name.trim().length > 0);
  }

  filterForSlot(filterId: string | null): FilterDefinition | undefined {
    return this.filterSettings.filters.find(filter => filter.id === filterId);
  }

  isAutofocusZeroCell(filterKey: string, channel: HeightOffsetChannel): boolean {
    return filterKey === 'empty' && channel === 'vis';
  }

  private persistFilterSettings(): Observable<boolean> {
    this.filterError = '';
    this.filterSaved = false;
    const normalizedFilters = this.filterSettings.filters.map(filter => ({
      ...filter,
      name: filter.name.trim(),
      wavelength_range: filter.wavelength_range.trim()
    }));
    const names = new Set<string>();
    for (const filter of normalizedFilters) {
      const normalizedName = filter.name.toLocaleLowerCase('hu-HU');
      if (!filter.name || !filter.wavelength_range) {
        return of(true);
      }
      if (names.has(normalizedName)) {
        this.filterError = 'A szűrők neve legyen egyedi.';
        return of(false);
      }
      names.add(normalizedName);
    }

    const normalizedHeightOffsets: Record<string, HeightOffsetRow> = {};
    for (const key of ['empty', ...normalizedFilters.map(filter => filter.id)]) {
      const sourceRow = this.filterSettings.height_offsets_mm[key];
      if (!sourceRow) {
        this.filterError = 'Az egyik szűrő magasság-eltolás sora hiányzik.';
        return of(false);
      }
      const normalizedRow = {} as HeightOffsetRow;
      for (const channel of this.heightOffsetChannels) {
        const value = key === 'empty' && channel.id === 'vis'
          ? 0
          : Number(sourceRow[channel.id]);
        if (
          !Number.isFinite(value)
          || value < this.maxHeightOffsetDownMm
          || value > this.maxHeightOffsetUpMm
        ) {
          this.filterError = `A magasság-eltolás ${this.maxHeightOffsetDownMm} és ${this.maxHeightOffsetUpMm} mm közötti szám lehet.`;
          return of(false);
        }
        normalizedRow[channel.id] = value;
      }
      normalizedHeightOffsets[key] = normalizedRow;
    }

    const validIds = new Set(normalizedFilters.map(filter => filter.id));
    if (this.filterSettings.slots.some(slot => slot !== null && !validIds.has(slot))) {
      this.filterError = 'Az egyik pozíció nem létező szűrőre hivatkozik.';
      return of(false);
    }

    const payload: FilterSettings = {
      filters: normalizedFilters,
      slots: [null, ...this.filterSettings.slots.slice(1)],
      height_offsets_mm: normalizedHeightOffsets
    };
    this.savingFilterSettings = true;
    return this.filterSettingsService.update(payload).pipe(
      tap(() => {
        this.filterSaved = true;
        this.filterDirty = false;
        this.savingFilterSettings = false;
      }),
      map(() => true),
      catchError(error => {
        this.filterError = error?.error?.error || 'A szűrőbeállítások mentése sikertelen.';
        this.savingFilterSettings = false;
        return of(false);
      })
    );
  }

  onAdvancedChanged(): void {
    this.advancedSaved = false;
    this.advancedError = this.getAdvancedValidationError() || '';

    // Emit for invalid changes too: this resets any pending debounce started
    // by the preceding valid keystroke. persistAdvancedSettings validates
    // again and will not send an invalid payload.
    this.advancedSaveRequests$.next();
    if (this.advancedError) {
      return;
    }
    this.advancedDirty = true;
  }

  private persistAdvancedSettings(): Observable<boolean> {
    const validationError = this.getAdvancedValidationError();
    if (validationError) {
      this.advancedError = validationError;
      return of(false);
    }

    const output_selectors = Object.fromEntries(
      Object.entries(this.advancedSelectors).map(([channel, value]) => [channel, value.trim().toUpperCase()])
    ) as AdvancedLampSettings['output_selectors'];
    if (Object.entries(this.fixedAdvancedSelectors).some(
      ([channel, selector]) => output_selectors[channel as UvChannel | 'vis'] !== selector
    )) {
      this.advancedError = 'A lámpakimeneteket a jóváhagyott firmware rögzített kiosztása határozza meg.';
      return of(false);
    }

    this.savingAdvancedSettings = true;
    return this.motionSettingsService.updateAdvanced({
      use_virtual_com_port: this.useVirtualComPort,
      max_height_offset_up_mm: Number(this.maxHeightOffsetUpMm),
      max_height_offset_down_mm: Number(this.maxHeightOffsetDownMm),
      first_tablet_x_mm: Number(this.firstTabletXMm),
      first_tablet_y_mm: Number(this.firstTabletYMm),
      first_tablet_z_mm: Number(this.firstTabletZMm),
      tablet_spacing_mm: Number(this.tabletSpacingMm)
    }).pipe(
      // Change adapters before applying new output selectors so a real board
      // can still receive its all-off commands through the previous mapping.
      switchMap(motion => this.lampSettingsService.updateAdvanced({ output_selectors }).pipe(
        map(lamp => ({ lamp, motion }))
      ))
    ).pipe(
      tap(response => {
        this.virtualConnectionLabel = response.motion.connection.connected
          ? `Csatlakoztatva: ${response.motion.connection.port}`
          : 'Nincs csatlakoztatott mozgásvezérlő.';
        this.sharedService.setMotionPlatformConnectionStatus(response.motion.connection.connected);
        this.advancedSaved = true;
        this.advancedDirty = false;
        this.savingAdvancedSettings = false;
      }),
      map(() => true),
      catchError(error => {
        this.advancedError = error?.error?.code === 'E1204'
          ? 'A 10×10-es tálca koordinátáinak az X/Y mozgástartományon belül kell maradniuk.'
          : 'A haladó beállítások mentése sikertelen.';
        this.savingAdvancedSettings = false;
        return of(false);
      })
    );
  }

  private loadLampSettings(): void {
    this.loadingLampSettings = true;
    this.lampError = '';
    this.lampSettingsService.get().subscribe({
      next: response => {
        this.applyLampSettings(response.lamp_settings);
        this.lampSettingsLoaded = true;
        this.loadingLampSettings = false;
      },
      error: error => {
        this.lampError = error?.error?.error || 'A lámpabeállítások betöltése sikertelen.';
        this.loadingLampSettings = false;
      }
    });
  }

  private loadFilterSettings(): void {
    this.loadingFilterSettings = true;
    this.filterError = '';
    this.filterSettingsService.get().subscribe({
      next: response => {
        this.filterSettings = {
          ...response.filter_settings,
          slots: [null, ...response.filter_settings.slots.slice(1)]
        };
        this.filterSettingsLoaded = true;
        this.loadingFilterSettings = false;
      },
      error: error => {
        this.filterError = error?.error?.error || 'A szűrőbeállítások betöltése sikertelen.';
        this.loadingFilterSettings = false;
      }
    });
  }

  private loadAdvancedSettings(): void {
    this.loadingAdvancedSettings = true;
    this.advancedError = '';
    forkJoin({
      lamp: this.lampSettingsService.getAdvanced(),
      motion: this.motionSettingsService.getAdvanced()
    }).subscribe({
      next: response => {
        this.advancedSelectors = {
          ...this.advancedSelectors,
          ...response.lamp.advanced_lamp_settings.output_selectors
        };
        this.useVirtualComPort = response.motion.advanced_motion_settings.use_virtual_com_port;
        this.maxHeightOffsetUpMm = response.motion.advanced_motion_settings.max_height_offset_up_mm;
        this.maxHeightOffsetDownMm = response.motion.advanced_motion_settings.max_height_offset_down_mm;
        this.firstTabletXMm = response.motion.advanced_motion_settings.first_tablet_x_mm;
        this.firstTabletYMm = response.motion.advanced_motion_settings.first_tablet_y_mm;
        this.firstTabletZMm = response.motion.advanced_motion_settings.first_tablet_z_mm;
        this.tabletSpacingMm = response.motion.advanced_motion_settings.tablet_spacing_mm;
        this.advancedSettingsLoaded = true;
        this.loadingAdvancedSettings = false;
      },
      error: error => {
        this.advancedError = error?.error?.error || 'A haladó lámpabeállítások betöltése sikertelen.';
        this.loadingAdvancedSettings = false;
      }
    });
  }

  private applyLampSettings(lampSettings: LampSettings): void {
    for (const row of this.lampRows) {
      if (row.channel !== 'vis') {
        row.settings = { ...(lampSettings.channels[row.channel] || {}) };
      }
    }
  }

  private hasLoadedLampSettings(): boolean {
    return this.lampSettingsLoaded;
  }

  private hasLoadedAdvancedSettings(): boolean {
    return this.advancedSettingsLoaded;
  }

  private hasLoadedFilterSettings(): boolean {
    return this.filterSettingsLoaded;
  }

  private createFilterId(): string {
    if (typeof globalThis.crypto?.randomUUID === 'function') {
      return globalThis.crypto.randomUUID();
    }
    return `filter-${Date.now()}-${this.nextFilterId++}`;
  }

  private createEmptyHeightOffsetRow(): HeightOffsetRow {
    return { uv255: 0, uv310: 0, uv365: 0, vis: 0 };
  }

  private isValidPercentage(value: unknown): boolean {
    const number = Number(value);
    return Number.isFinite(number) && number >= 10 && number <= 100;
  }

  private isValidTimeout(value: unknown): boolean {
    const number = Number(value);
    return Number.isFinite(number) && number > 0;
  }

  private hasValidHeightOffsetLimits(): boolean {
    const maxUp = Number(this.maxHeightOffsetUpMm);
    const maxDown = Number(this.maxHeightOffsetDownMm);
    return Number.isFinite(maxUp) && maxUp > 0 && Number.isFinite(maxDown) && maxDown < 0;
  }

  private hasValidTrayGeometry(): boolean {
    const x = Number(this.firstTabletXMm);
    const y = Number(this.firstTabletYMm);
    const z = Number(this.firstTabletZMm);
    const spacing = Number(this.tabletSpacingMm);
    return [x, y, z, spacing].every(Number.isFinite)
      && x >= 0 && y >= 0 && z >= 0 && z <= 30 && spacing > 0
      && x + 9 * spacing <= 175.5
      && y + 9 * spacing <= 175.5;
  }

  private getAdvancedValidationError(): string | null {
    if (!this.hasValidHeightOffsetLimits()) {
      return 'A felfelé határ pozitív, a lefelé határ negatív szám legyen.';
    }
    if (!this.hasValidTrayGeometry()) {
      return 'Az első tabletta és a 10×10-es kiosztás minden koordinátája maradjon a gép mozgástartományában.';
    }
    const offsetsOutsideLimits = Object.values(this.filterSettings.height_offsets_mm)
      .some(row => Object.values(row).some(value => {
        const offset = Number(value);
        return !Number.isFinite(offset)
          || offset < Number(this.maxHeightOffsetDownMm)
          || offset > Number(this.maxHeightOffsetUpMm);
      }));
    if (offsetsOutsideLimits) {
      return 'A megadott határokon kívül eső magasság-eltolásokat előbb módosítsa.';
    }
    return null;
  }
}
