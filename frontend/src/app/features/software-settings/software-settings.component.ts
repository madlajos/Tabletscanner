import { CommonModule } from '@angular/common';
import { Component, EventEmitter, OnDestroy, OnInit, Output } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { MatIconModule } from '@angular/material/icon';
import { Subscription } from 'rxjs';
import { AdvancedLampSettings, LampSettings, UvLampSettings } from '../../models/light.models';
import { LampSettingsService } from '../../services/lamp-settings.service';
import { SharedService } from '../../shared.service';

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
  imports: [CommonModule, FormsModule, MatIconModule],
  templateUrl: './software-settings.component.html',
  styleUrls: ['./software-settings.component.css']
})
export class SoftwareSettingsComponent implements OnInit, OnDestroy {
  @Output() close = new EventEmitter<void>();

  selectedType: SettingsType = 'filter';
  measurementActive = false;
  loadingLampSettings = false;
  savingLampSettings = false;
  lampError = '';
  lampSaved = false;
  loadingAdvancedSettings = false;
  savingAdvancedSettings = false;
  advancedError = '';
  advancedSaved = false;
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
  advancedSelectors: Record<UvChannel | 'vis', string> = {
    uv255: 'P1', uv310: 'P2', uv365: 'P3', vis: 'P0'
  };

  constructor(
    private readonly lampSettingsService: LampSettingsService,
    private readonly sharedService: SharedService
  ) {}

  ngOnInit(): void {
    this.measurementSubscription = this.sharedService.measurementActive$.subscribe(active => {
      this.measurementActive = active;
    });
  }

  ngOnDestroy(): void {
    this.measurementSubscription?.unsubscribe();
  }

  selectType(type: SettingsType): void {
    this.selectedType = type;
    if (type === 'lamp' && !this.loadingLampSettings && !this.hasLoadedLampSettings()) {
      this.loadLampSettings();
    }
    if (type === 'advanced' && !this.loadingAdvancedSettings && !this.hasLoadedAdvancedSettings()) {
      this.loadAdvancedSettings();
    }
  }

  dismiss(): void {
    this.close.emit();
  }

  onBackdropClick(event: MouseEvent): void {
    if (event.target === event.currentTarget) {
      this.dismiss();
    }
  }

  saveLampSettings(): void {
    this.lampError = '';
    this.lampSaved = false;
    const channels: Partial<Record<UvChannel, UvLampSettings>> = {};

    for (const row of this.lampRows) {
      if (row.channel === 'vis') {
        continue;
      }
      const settings = row.settings;
      if (!this.isValidPercentage(settings.dim_percent) || !this.isValidPercentage(settings.full_percent)
        || !this.isValidTimeout(settings.dim_timeout_seconds) || !this.isValidTimeout(settings.full_timeout_seconds)) {
        this.lampError = 'A fényerőnek 10 és 100% között, a lekapcsolási időnek 0 másodpercnél nagyobbnak kell lennie.';
        return;
      }
      channels[row.channel] = {
        dim_percent: Number(settings.dim_percent),
        full_percent: Number(settings.full_percent),
        dim_timeout_seconds: Number(settings.dim_timeout_seconds),
        full_timeout_seconds: Number(settings.full_timeout_seconds)
      };
    }

    this.savingLampSettings = true;
    this.lampSettingsService.update({ channels }).subscribe({
      next: response => {
        this.applyLampSettings(response.lamp_settings);
        this.lampSaved = true;
        this.savingLampSettings = false;
      },
      error: error => {
        this.lampError = error?.error?.error || 'A lámpabeállítások mentése sikertelen.';
        this.savingLampSettings = false;
      }
    });
  }

  saveAdvancedSettings(): void {
    this.advancedError = '';
    this.advancedSaved = false;
    const output_selectors = { ...this.advancedSelectors };
    if (Object.values(output_selectors).some(value => !/^P\d+$/i.test(value.trim()))) {
      this.advancedError = 'Minden kimenethez P után egy nem negatív számot adjon meg, például P0.';
      return;
    }

    this.savingAdvancedSettings = true;
    this.lampSettingsService.updateAdvanced({ output_selectors }).subscribe({
      next: response => {
        this.advancedSelectors = { ...response.advanced_lamp_settings.output_selectors };
        this.advancedSaved = true;
        this.savingAdvancedSettings = false;
      },
      error: error => {
        this.advancedError = error?.error?.error || 'A haladó lámpabeállítások mentése sikertelen.';
        this.savingAdvancedSettings = false;
      }
    });
  }

  private loadLampSettings(): void {
    this.loadingLampSettings = true;
    this.lampError = '';
    this.lampSettingsService.get().subscribe({
      next: response => {
        this.applyLampSettings(response.lamp_settings);
        this.loadingLampSettings = false;
      },
      error: error => {
        this.lampError = error?.error?.error || 'A lámpabeállítások betöltése sikertelen.';
        this.loadingLampSettings = false;
      }
    });
  }

  private loadAdvancedSettings(): void {
    this.loadingAdvancedSettings = true;
    this.advancedError = '';
    this.lampSettingsService.getAdvanced().subscribe({
      next: response => {
        this.advancedSelectors = { ...this.advancedSelectors, ...response.advanced_lamp_settings.output_selectors };
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
    return this.lampRows.some(row => row.channel !== 'vis' && Object.keys(row.settings).length > 0);
  }

  private hasLoadedAdvancedSettings(): boolean {
    return Object.values(this.advancedSelectors).some(value => value.length > 0);
  }

  private isValidPercentage(value: unknown): boolean {
    const number = Number(value);
    return Number.isFinite(number) && number >= 10 && number <= 100;
  }

  private isValidTimeout(value: unknown): boolean {
    const number = Number(value);
    return Number.isFinite(number) && number > 0;
  }
}
