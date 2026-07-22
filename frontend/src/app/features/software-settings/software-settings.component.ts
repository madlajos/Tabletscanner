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
import { FilterDefinition, FilterSettings } from '../../models/filter-settings.models';
import { MotionSettingsService } from '../../services/motion-settings.service';
import { SharedService } from '../../shared.service';
import { FilterRevolverComponent } from '../../components/filter-revolver/filter-revolver.component';

type SettingsType = 'filter' | 'lamp' | 'tray' | 'advanced';
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
  loadingAdvancedSettings = false;
  savingAdvancedSettings = false;
  advancedError = '';
  advancedSaved = false;
  useVirtualComPort = false;
  maxHeightOffsetUpMm = 5;
  maxHeightOffsetDownMm = -5;
  virtualConnectionLabel = '';
  private advancedSettingsLoaded = false;
  private filterSettingsLoaded = false;
  private lampSettingsLoaded = false;
  private nextFilterId = 1;
  private readonly destroy$ = new Subject<void>();
  private readonly filterSaveRequests$ = new Subject<void>();
  private readonly lampSaveRequests$ = new Subject<void>();
  private readonly advancedSaveRequests$ = new Subject<void>();
  private filterDirty = false;
  private lampDirty = false;
  private advancedDirty = false;
  private measurementSubscription?: Subscription;

  readonly settingsTypes: { id: SettingsType; label: string; icon: string }[] = [
    { id: 'filter', label: 'Szűrőváltó', icon: 'tune' },
    { id: 'lamp', label: 'Lámpa', icon: 'lightbulb' },
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
  advancedSelectors: Record<UvChannel | 'vis', string> = { ...this.fixedAdvancedSelectors };
  filterSettings: FilterSettings = { filters: [], slots: [null, null, null, null, null, null] };

  constructor(
    private readonly lampSettingsService: LampSettingsService,
    private readonly filterSettingsService: FilterSettingsService,
    private readonly motionSettingsService: MotionSettingsService,
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

  addFilter(): void {
    this.filterSettings.filters.push({
      id: this.createFilterId(),
      name: '',
      wavelength_range: '',
      height_offset_mm: 0,
      color: '#ffffff'
    });
    this.filterSaved = false;
  }

  removeFilter(index: number): void {
    const removedId = this.filterSettings.filters[index]?.id;
    this.filterSettings.filters.splice(index, 1);
    if (removedId) {
      this.filterSettings.slots = this.filterSettings.slots.map(slot => slot === removedId ? null : slot);
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

  private persistFilterSettings(): Observable<boolean> {
    this.filterError = '';
    this.filterSaved = false;
    const normalizedFilters = this.filterSettings.filters.map(filter => ({
      ...filter,
      name: filter.name.trim(),
      wavelength_range: filter.wavelength_range.trim(),
      height_offset_mm: Number(filter.height_offset_mm)
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
      if (
        !Number.isFinite(filter.height_offset_mm)
        || filter.height_offset_mm < this.maxHeightOffsetDownMm
        || filter.height_offset_mm > this.maxHeightOffsetUpMm
      ) {
        this.filterError = `A magasság-eltolás ${this.maxHeightOffsetDownMm} és ${this.maxHeightOffsetUpMm} mm közötti szám lehet.`;
        return of(false);
      }
      names.add(normalizedName);
    }

    const validIds = new Set(normalizedFilters.map(filter => filter.id));
    if (this.filterSettings.slots.some(slot => slot !== null && !validIds.has(slot))) {
      this.filterError = 'Az egyik pozíció nem létező szűrőre hivatkozik.';
      return of(false);
    }

    const payload: FilterSettings = {
      filters: normalizedFilters,
      slots: [...this.filterSettings.slots]
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
    this.advancedError = '';
    this.advancedSaved = false;
    if (!this.hasValidHeightOffsetLimits()) {
      this.advancedError = 'A felfelé határ pozitív, a lefelé határ negatív szám legyen.';
      return;
    }
    const filtersOutsideLimits = this.filterSettings.filters.some(filter => {
      const offset = Number(filter.height_offset_mm);
      return offset < Number(this.maxHeightOffsetDownMm) || offset > Number(this.maxHeightOffsetUpMm);
    });
    if (filtersOutsideLimits) {
      this.advancedError = 'A megadott határokon kívül eső szűrő-magasságokat előbb módosítsa.';
      return;
    }
    this.advancedDirty = true;
    this.advancedSaveRequests$.next();
  }

  private persistAdvancedSettings(): Observable<boolean> {
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
      max_height_offset_down_mm: Number(this.maxHeightOffsetDownMm)
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
        this.advancedError = error?.error?.error || 'A haladó lámpabeállítások mentése sikertelen.';
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
        this.filterSettings = response.filter_settings;
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
}
