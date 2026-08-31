import { Component, HostListener, OnInit, OnDestroy } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { interval, Subscription, of } from 'rxjs';
import { switchMap, catchError, timeout, finalize } from 'rxjs/operators';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ErrorNotificationService } from '../../services/error-notification.service';
import { SharedService } from '../../shared.service';
import { firstValueFrom } from 'rxjs';
import { BASE_URL } from '../../api-config';
import { LightChannel, LightStatus } from '../../models/light.models';
import {
  FilterDefinition,
  HeightOffsetApplication,
  FilterRevolverDirection,
  FilterRevolverStatus,
  FilterSettings
} from '../../models/filter-settings.models';
import { FilterSettingsService } from '../../services/filter-settings.service';
import { MotionSettingsService } from '../../services/motion-settings.service';
import { FilterRevolverService } from '../../services/filter-revolver.service';
import { FilterRevolverComponent } from '../../components/filter-revolver/filter-revolver.component';


@Component({
  selector: 'app-motion-control',
  // Important for Angular 15+ when using `imports` here:
  standalone: true,
  imports: [CommonModule, FormsModule, FilterRevolverComponent],
  templateUrl: './motion-control.html',
  styleUrls: ['./motion-control.scss'], // fixed key (plural)
})
export class MotionControl implements OnInit, OnDestroy {
  movementAmount: number = 1;
  firstTabletX = 2.9;
  firstTabletY = 0;
  firstTabletZ = 20;
  tabletSpacing = 18.3;

  motorOffState: boolean = false;

  xPosition: number | string = '?';
  yPosition: number | string = '?';
  zPosition: number | string = '?';

  xMin: number = 0;
  xMax: number = 175;
  yMin: number = 0;
  yMax: number = 165;
  zMin: number = 0;
  zMax: number = 40;

  private originalOnFocus = { x: undefined as any, y: undefined as any, z: undefined as any };
  private skipNextBlurRevert = false;
  isEditing = { x: false, y: false, z: false };


  connectionPolling: Subscription | undefined;
  positionPolling: Subscription | undefined;
  reconnectionPolling: Subscription | undefined;
  private measurementActiveSub?: Subscription;
  private motionPositionSub?: Subscription;
  private lightsOffSub?: Subscription;
  private externalConnectionSub?: Subscription;
  private fourChannelLightPolling?: Subscription;
  private filterStatusPolling?: Subscription;
  private filterSettingsSub?: Subscription;
  private traySettingsSub?: Subscription;
  private autofocusInvalidateSub?: Subscription;
  private filterStatusGeneration = 0;
  isConnected: boolean = false;

  // Flag to lock controls during auto-measurement
  measurementActive: boolean = false;

  isEditingX: boolean = false;
  isEditingY: boolean = false;
  isEditingZ: boolean = false;

  lightBusy = false;
  lightStatus: LightStatus = {
    active_channel: null,
    active_mode: null,
    channels: { uv255: false, uv310: false, uv365: false, vis: false },
    auto_turned_off: []
  };
  private lampClickTimers: Partial<Record<LightChannel, ReturnType<typeof setTimeout>>> = {};
  readonly lampButtons: ReadonlyArray<{ channel: LightChannel; label: string; onAsset: string; offAsset: string }> = [
    { channel: 'uv255', label: '255 nm', onAsset: '255nm_on.svg', offAsset: '255nm_off.svg' },
    { channel: 'uv310', label: '310 nm', onAsset: '310nm_on.svg', offAsset: '310nm_off.svg' },
    { channel: 'uv365', label: '365 nm', onAsset: '365nm_on.svg', offAsset: '365nm_off.svg' },
    { channel: 'vis', label: 'VIS', onAsset: 'vis_on.svg', offAsset: 'vis_off.svg' }
  ];


  isHoming = false;
  // The A axis may travel almost a full revolution before finding its Hall sensor.
  // Keep the client timeout above the backend's 60-second A-axis limit.
  private readonly HOMING_TIMEOUT_MS = 70000;
  homeContextMenuVisible = false;
  homeContextMenuX = 0;
  homeContextMenuY = 0;
  xHomed: boolean = false;
  yHomed: boolean = false;
  zHomed: boolean = false;
  filterRevolverBusy = false;
  filterRevolverRotationDegrees = 0;
  private filterRevolverRotationInitialized = false;
  filterSettings: FilterSettings = {
    filters: [],
    slots: [null, null, null, null, null, null],
    height_offsets_mm: {
      empty: { uv255: 0, uv310: 0, uv365: 0, vis: 0 }
    }
  };
  filterRevolverStatus: FilterRevolverStatus = {
    position: null,
    homed: false,
    motion_platform_homed: false,
    busy: false
  };

  isAutofocusing = false;
  autofocusDone = false;
  private autofocusAbortedByUser = false;

  constructor(
    private http: HttpClient,
    private errorNotificationService: ErrorNotificationService,
    private sharedService: SharedService,
    private filterSettingsService: FilterSettingsService,
    private motionSettingsService: MotionSettingsService,
    private filterRevolverService: FilterRevolverService
  ) { }

  ngOnInit(): void {
    this.startConnectionPolling();
    
    // Subscribe to measurement active state for UI lockdown
    this.measurementActiveSub = this.sharedService.measurementActive$.subscribe(
      (active) => {
        this.measurementActive = active;
      }
    );

    // Subscribe to immediate motion position updates (e.g., after homing)
    this.motionPositionSub = this.sharedService.motionPosition$.subscribe((pos) => {
      if (!pos) return;
      const round3 = (v: number | null) => (typeof v === 'number' ? Math.round(v * 1000) / 1000 : '?');
      if (pos.x !== null) { this.xPosition = round3(pos.x) as number | string; this.xHomed = true; }
      if (pos.y !== null) { this.yPosition = round3(pos.y) as number | string; this.yHomed = true; }
      if (pos.z !== null) { this.zPosition = round3(pos.z) as number | string; this.zHomed = true; }
      // Publish homed state to SharedService
      this.updateSharedHomedState();
    });

    // Subscribe to motion homing status (from auto-measurement or manual home)
    this.sharedService.motionHomingStatus$.subscribe(
      (isHoming) => {
        this.isHoming = isHoming;
      }
    );

    // Subscribe to lights-off event (when auto-measurement is stopped)
    this.lightsOffSub = this.sharedService.lightsOff$.subscribe(() => {
      console.log('Lights off event received; updating UI state');
      this.lightStatus = {
        ...this.lightStatus,
        active_channel: null,
        active_mode: null,
        channels: { uv255: false, uv310: false, uv365: false, vis: false }
      };
    });

    // Subscribe to autofocus invalidation (e.g., when auto-measurement moves the platform)
    this.autofocusInvalidateSub = this.sharedService.autofocusInvalidate$.subscribe(() => {
      this.autofocusDone = false;
    });

    this.startFourChannelLightPolling();
    this.filterSettingsSub = this.filterSettingsService.settings$.subscribe(settings => {
      if (settings) this.filterSettings = settings;
    });
    this.filterSettingsService.get().subscribe({
      error: error => console.error('Failed to load filter settings:', error)
    });
    this.traySettingsSub = this.motionSettingsService.advanced$.subscribe(settings => {
      if (!settings) return;
      this.firstTabletX = settings.first_tablet_x_mm;
      this.firstTabletY = settings.first_tablet_y_mm;
      this.firstTabletZ = settings.first_tablet_z_mm;
      this.tabletSpacing = settings.tablet_spacing_mm;
    });
    this.motionSettingsService.getAdvanced().subscribe({
      error: error => console.error('Failed to load tray geometry settings:', error)
    });
    this.startFilterRevolverStatusPolling();

    // Listen for external reconnections (e.g., auto-measurement reconnects the platform).
    // Without this, the error popup can stay visible because only motion-control's
    // own reconnection polling would clear it, which is a race condition.
    this.externalConnectionSub = this.sharedService.motionPlatformConnectionStatus$.subscribe(
      (connected) => {
        if (connected && !this.isConnected) {
          console.info('Motion platform connection restored externally – syncing state.');
          this.isConnected = true;
          this.errorNotificationService.removeError('E1201');
          this.stopReconnectionPolling();
          if (!this.connectionPolling) {
            this.startConnectionPolling();
          }
          this.startPollingPosition();
        }
      }
    );
  }

  ngOnDestroy(): void {
    this.stopConnectionPolling();
    this.stopReconnectionPolling();
    this.stopPositionPolling();
    this.fourChannelLightPolling?.unsubscribe();
    this.filterStatusPolling?.unsubscribe();
    this.filterSettingsSub?.unsubscribe();
    this.traySettingsSub?.unsubscribe();
    this.autofocusInvalidateSub?.unsubscribe();
    Object.values(this.lampClickTimers).forEach(timer => timer && clearTimeout(timer));
    this.measurementActiveSub?.unsubscribe();
    this.motionPositionSub?.unsubscribe();
    this.lightsOffSub?.unsubscribe();
    this.externalConnectionSub?.unsubscribe();
  }

  // Check if controls should be disabled
  get controlsDisabled(): boolean {
    return !this.isConnected || this.measurementActive || this.isHoming || this.isAutofocusing;
  }

  // ---------- Polling ----------

  startPollingPosition(): void {
    if (this.positionPolling && !this.positionPolling.closed) {
      return;
    }
    this.positionPolling = interval(3000).subscribe(() => {
      this.updateMotionPlatformPosition();
    });
  }


  stopPositionPolling(): void {
    if (this.positionPolling && !this.positionPolling.closed) {
      this.positionPolling.unsubscribe();
    }
    this.positionPolling = undefined;
  }

  updateMotionPlatformPosition(): void {
    if (!(this.isConnected && !this.motorOffState)) return;

    this.http
      .get<{ x?: number | null; y?: number | null; z?: number | null }>(`${BASE_URL}/get_motion_platform_position`)
      .subscribe({
        next: (position) => {
          const hasNum = (v: any) => typeof v === 'number' && Number.isFinite(v);
          const round3 = (v: number) => Math.round(v * 1000) / 1000;

          // DOM-focused fallback (prevents overwrite if focus flags ever desync)
          const activeId = (document.activeElement && (document.activeElement as HTMLElement).id) || '';

          // X
          if (!this.isEditingX && activeId !== 'x-position') {
            if (!this.xHomed) {
              this.xPosition = '?';
            } else if (hasNum(position?.x)) {
              this.xPosition = round3(position!.x as number);
            }
          }

          // Y
          if (!this.isEditingY && activeId !== 'y-position') {
            if (!this.yHomed) {
              this.yPosition = '?';
            } else if (hasNum(position?.y)) {
              this.yPosition = round3(position!.y as number);
            }
          }

          // Z
          if (!this.isEditingZ && activeId !== 'z-position') {
            if (!this.zHomed) {
              this.zPosition = '?';
            } else if (hasNum(position?.z)) {
              this.zPosition = round3(position!.z as number);
            }
          }
        },
        error: (error) => {
          console.error('Failed to get Motion platform position!', error);
        },
      });
  }



  startConnectionPolling(): void {
    if (this.connectionPolling) return;

    this.connectionPolling = interval(5000)
      .pipe(
        switchMap(() =>
          this.http
            .get<{ connected: boolean }>(`${BASE_URL}/status/serial/motionplatform`)
            .pipe(
              timeout(5000),
              catchError((err) => {
                console.warn('Motion platform connection polling timed out or failed.', err);
                return of({ connected: false });
              })
            )
        )
      )
      .subscribe({
        next: (response) => {
          const wasConnected = this.isConnected;
          this.isConnected = response.connected;
          this.sharedService.setMotionPlatformConnectionStatus(response.connected);

          if (!this.isConnected && !this.reconnectionPolling) {
            console.warn('Motion platform disconnected – starting reconnection polling.');
            // Reset homed state on disconnect
            this.xHomed = this.yHomed = this.zHomed = false;
            this.filterRevolverStatus = {
              position: null,
              homed: false,
              motion_platform_homed: false,
              busy: false
            };
            this.updateSharedHomedState();
            this.errorNotificationService.addError({
              code: 'E1201',
              message: this.errorNotificationService.getMessage('E1201'),
            });
            this.stopPositionPolling();
            this.stopConnectionPolling();
            this.startReconnectionPolling();
          } else if (this.isConnected && !wasConnected) {
            console.info('Motion platform reconnected.');
            this.errorNotificationService.removeError('E1201');
            this.stopReconnectionPolling();
            this.startConnectionPolling();
            this.startPollingPosition(); // begin position updates when connected
          }
        },
        error: (error) => {
          console.error('Unexpected polling error!', error);
          this.isConnected = false;
          this.filterRevolverStatus = {
            position: null,
            homed: false,
            motion_platform_homed: false,
            busy: false
          };
          this.sharedService.setMotionPlatformConnectionStatus(false);
          this.stopPositionPolling();
        },
      });
  }

  stopConnectionPolling(): void {
    if (this.connectionPolling) {
      this.connectionPolling.unsubscribe();
      this.connectionPolling = undefined;
    }
  }

  startReconnectionPolling(): void {
    if (this.reconnectionPolling) return;
    this.reconnectionPolling = interval(3000).subscribe(() => {
      this.tryReconnectMotionPlatform();
    });
  }

  stopReconnectionPolling(): void {
    if (this.reconnectionPolling) {
      this.reconnectionPolling.unsubscribe();
      this.reconnectionPolling = undefined;
    }
  }

  startFourChannelLightPolling(): void {
    this.syncFourChannelLightStatus();
    this.fourChannelLightPolling = interval(1000).subscribe(() => this.syncFourChannelLightStatus());
  }

  private syncFourChannelLightStatus(): void {
    if (!this.isConnected || this.lightBusy) return;
    this.http.get<LightStatus>(`${BASE_URL}/lights/status`).subscribe({
      next: status => {
        this.lightStatus = status;
        if (!this.isAutofocusing && status.height_offset_reference) {
          this.autofocusDone = status.height_offset_reference.available;
        }
        this.applyHeightOffsetPosition(status.height_offset);
      },
      error: () => undefined
    });
  }

  private startFilterRevolverStatusPolling(): void {
    this.syncFilterRevolverStatus();
    this.filterStatusPolling = interval(1000).subscribe(() => this.syncFilterRevolverStatus());
  }

  private syncFilterRevolverStatus(): void {
    if (!this.isConnected || this.filterRevolverBusy) return;
    const requestGeneration = this.filterStatusGeneration;
    this.filterRevolverService.getStatus().subscribe({
      next: status => {
        // Ignore a poll that started before a rotate/home command. Otherwise
        // its older position can overwrite the acknowledged command response.
        if (requestGeneration === this.filterStatusGeneration && !this.filterRevolverBusy) {
          this.applyFilterRevolverStatus(status);
        }
      },
      error: () => undefined
    });
  }

  get activeFilter(): FilterDefinition | undefined {
    const position = this.filterRevolverStatus.position;
    if (!position) return undefined;
    const filterId = this.filterSettings.slots[position - 1];
    return this.filterSettings.filters.find(filter => filter.id === filterId);
  }

  get filterRevolverControlsDisabled(): boolean {
    return this.controlsDisabled
      || !this.filterRevolverStatus.homed
      || this.filterRevolverBusy
      || this.filterRevolverStatus.busy;
  }

  get activeHeightOffset(): number | null {
    const channel = this.lightStatus.active_channel;
    const position = this.filterRevolverStatus.position;
    if (!channel || !position) return null;
    const filterKey = this.filterSettings.slots[position - 1] || 'empty';
    const value = this.filterSettings.height_offsets_mm[filterKey]?.[channel];
    return typeof value === 'number' && Number.isFinite(value) ? value : null;
  }

  openHomeContextMenu(event: MouseEvent): void {
    if (this.isHoming || !this.isConnected || this.measurementActive || this.isAutofocusing) {
      return;
    }

    event.preventDefault();
    event.stopPropagation();
    const menuWidth = 210;
    const menuHeight = 80;
    this.homeContextMenuX = Math.max(4, Math.min(event.clientX, window.innerWidth - menuWidth - 4));
    this.homeContextMenuY = Math.max(4, Math.min(event.clientY, window.innerHeight - menuHeight - 4));
    this.homeContextMenuVisible = true;
  }

  closeHomeContextMenu(): void {
    this.homeContextMenuVisible = false;
  }

  @HostListener('document:click')
  @HostListener('document:contextmenu')
  closeHomeContextMenuFromDocument(): void {
    this.closeHomeContextMenu();
  }

  @HostListener('document:keydown.escape')
  closeHomeContextMenuOnEscape(): void {
    this.closeHomeContextMenu();
  }

  rotateFilterRevolver(direction: FilterRevolverDirection): void {
    if (this.filterRevolverControlsDisabled) return;
    this.filterStatusGeneration++;
    this.filterRevolverBusy = true;
    let commandSucceeded = false;
    this.filterRevolverService.rotate(direction).pipe(
      finalize(() => {
        this.filterRevolverBusy = false;
        if (!commandSucceeded) this.syncFilterRevolverStatus();
      })
    ).subscribe({
      next: status => {
        commandSucceeded = true;
        this.filterRevolverRotationDegrees += direction === 'up' ? -60 : 60;
        this.filterRevolverRotationInitialized = true;
        this.filterRevolverStatus = status;
        this.applyHeightOffsetPosition(status.height_offset);
      },
      error: error => {
        console.error('Filter revolver rotation failed:', error);
      }
    });
  }

  activateFilterPosition(position: number): void {
    if (
      this.filterRevolverControlsDisabled
      || position === this.filterRevolverStatus.position
    ) {
      return;
    }
    this.filterStatusGeneration++;
    this.filterRevolverBusy = true;
    let commandSucceeded = false;
    this.filterRevolverService.select(position).pipe(
      finalize(() => {
        this.filterRevolverBusy = false;
        if (!commandSucceeded) this.syncFilterRevolverStatus();
      })
    ).subscribe({
      next: response => {
        commandSucceeded = true;
        const visualDirection = response.direction === 'up' ? -1 : 1;
        this.filterRevolverRotationDegrees += visualDirection * response.steps * 60;
        this.filterRevolverRotationInitialized = true;
        this.filterRevolverStatus = response;
        this.applyHeightOffsetPosition(response.height_offset);
      },
      error: error => {
        console.error('Filter revolver position selection failed:', error);
      }
    });
  }

  isLampActive(channel: LightChannel): boolean {
    return this.lightStatus.active_channel === channel;
  }

  lampAsset(button: { channel: LightChannel; onAsset: string; offAsset: string }): string {
    return `assets/SVG/${this.isLampActive(button.channel) ? button.onAsset : button.offAsset}`;
  }

  onLampClick(channel: LightChannel): void {
    const existingTimer = this.lampClickTimers[channel];
    if (existingTimer) clearTimeout(existingTimer);
    this.lampClickTimers[channel] = setTimeout(() => {
      delete this.lampClickTimers[channel];
      this.toggleFourChannelLight(channel, channel === 'vis' ? undefined : 'dimmed');
    }, 250);
  }

  onLampDoubleClick(channel: LightChannel): void {
    const timer = this.lampClickTimers[channel];
    if (timer) clearTimeout(timer);
    delete this.lampClickTimers[channel];
    if (channel === 'vis') {
      // Browsers emit two click events before dblclick. Coalesce those events
      // into one normal VIS toggle; VIS has no separate full-power action.
      this.toggleFourChannelLight(channel);
      return;
    }
    this.toggleFourChannelLight(channel, 'full', true);
  }

  private toggleFourChannelLight(channel: LightChannel, mode?: 'dimmed' | 'full', forceOn = false): void {
    if (this.lightBusy || this.controlsDisabled) return;
    this.lightBusy = true;
    const request = this.isLampActive(channel) && !forceOn
      ? this.http.post<LightStatus>(`${BASE_URL}/lights/off`, { channel })
      : this.http.post<LightStatus>(`${BASE_URL}/lights/activate`, mode ? { channel, mode } : { channel });
    request.subscribe({
      next: status => {
        this.lightStatus = status;
        this.applyHeightOffsetPosition(status.height_offset);
        this.lightBusy = false;
      },
      error: error => { console.error('Four-channel lamp command failed:', error); this.lightBusy = false; this.syncFourChannelLightStatus(); }
    });
  }

  tryReconnectMotionPlatform(): void {
    this.http
      .post<{ message: string }>(`${BASE_URL}/connect-to-motionplatform`, {})
      .pipe(
        timeout(3000),
        catchError((err) => {
          console.warn('Motion platform reconnection attempt timed out or errored:', err);
          return of({ message: 'Reconnection failed' });
        })
      )
      .subscribe({
        next: (response) => {
          if (response.message !== 'Reconnection failed') {
            console.info('Motion platform reconnected:', response.message);
            this.isConnected = true;
            this.errorNotificationService.removeError('E1201');
            this.stopReconnectionPolling();
            this.startConnectionPolling();
            this.startPollingPosition();
          } else {
            console.warn('Motion platform reconnection attempt failed after fallback.');
          }
        },
        error: (error) => {
          console.warn('Motion platform reconnection attempt failed (unexpected).', error);
        },
      });
  }

  resetMotorOffState(): void {
    // Many firmwares auto-enable steppers on the first move; we clear the UI flag.
    this.motorOffState = false;
    this.updateMotionPlatformPosition();
  }

  private clamp(v: number, lo: number, hi: number): number {
    return Math.max(lo, Math.min(hi, v));
  }

  private applyHeightOffsetPosition(application?: HeightOffsetApplication): void {
    if (!application?.applied || typeof application.target_z !== 'number') return;
    this.zPosition = Math.round(application.target_z * 1000) / 1000;
  }

  private toNumberOrUndefined(v?: number | string): number | undefined {
    if (v === undefined || v === '?') return undefined;
    const n = typeof v === 'number' ? v : Number(v);
    return Number.isFinite(n) ? n : undefined;
  }

  submitOnEnter(axis: 'x' | 'y' | 'z') {
    this.skipNextBlurRevert = true;
    if (axis === 'x') this.moveToolHeadAbsolute(this.xPosition, undefined, undefined);
    else if (axis === 'y') this.moveToolHeadAbsolute(undefined, this.yPosition, undefined);
    else this.moveToolHeadAbsolute(undefined, undefined, this.zPosition);

    if (axis === 'x') this.isEditingX = false;
    else if (axis === 'y') this.isEditingY = false;
    else this.isEditingZ = false;
  }


  // ---------- Movements (existing API: *toolhead*) ----------

  moveToolHeadRelative(axis: string, value: number): void {
    if (this.motorOffState) { console.error('Cannot move toolhead while motors are off.'); return; }
    this.resetMotorOffState();
    this.autofocusDone = false;

    if ((axis === 'x' && !this.xHomed) || (axis === 'y' && !this.yHomed) || (axis === 'z' && !this.zHomed)) {
      console.error(`Cannot move ${axis.toUpperCase()} axis because it is not homed.`);
      return;
    }

    // Clamp instead of rejecting
    if (axis === 'x') {
      const cur = typeof this.xPosition === 'number' ? this.xPosition : 0;
      const target = cur + value;
      const clamped = this.clamp(target, this.xMin, this.xMax);
      const adj = clamped - cur;
      if (Math.abs(adj) < 1e-6) {
        console.warn(`X already at limit (${clamped}).`);
        return;
      }
      value = adj; // send adjusted delta
    } else if (axis === 'y') {
      const cur = typeof this.yPosition === 'number' ? this.yPosition : 0;
      const target = cur + value;
      const clamped = this.clamp(target, this.yMin, this.yMax);
      const adj = clamped - cur;
      if (Math.abs(adj) < 1e-6) {
        console.warn(`Y already at limit (${clamped}).`);
        return;
      }
      value = adj;
    }
    // Z: only clamp if you also maintain zMin/zMax in the UI; otherwise leave as-is.

    const payload = { axis, value };
    this.http.post(`${BASE_URL}/move_toolhead_relative`, payload).subscribe({
      next: (response: any) => { /* optionally toast if value was clamped */ },
    });
  }


  moveToolHeadAbsolute(x?: number | string, y?: number | string, z?: number | string): void {
    if (this.motorOffState) {
      console.error('Cannot move toolhead while motors are off.');
      return;
    }
    this.resetMotorOffState();
    this.autofocusDone = false;

    const xNum = this.toNumberOrUndefined(x);
    const yNum = this.toNumberOrUndefined(y);
    const zNum = this.toNumberOrUndefined(z);

    // Require homing only for axes we are actually commanding
    if (xNum !== undefined && !this.xHomed) {
      console.error('Cannot move on X axis because it is not homed.');
      return;
    }
    if (yNum !== undefined && !this.yHomed) {
      console.error('Cannot move on Y axis because it is not homed.');
      return;
    }
    if (zNum !== undefined && !this.zHomed) {
      console.error('Cannot move on Z axis because it is not homed.');
      return;
    }

    // Clamp X/Y to limits instead of rejecting. Z clamped only if limits exist.
    let xSend = xNum;
    let ySend = yNum;
    let zSend = zNum;

    let clampedX = false;
    let clampedY = false;
    let clampedZ = false;

    if (xSend !== undefined) {
      const lo = this.xMin;
      const hi = this.xMax;
      const c = Math.max(lo, Math.min(hi, xSend));
      clampedX = (c !== xSend);
      xSend = c;
    }

    if (ySend !== undefined) {
      const lo = this.yMin;
      const hi = this.yMax;
      const c = Math.max(lo, Math.min(hi, ySend));
      clampedY = (c !== ySend);
      ySend = c;
    }

    // Optional Z clamp if you maintain zMin/zMax in the component
    if (zSend !== undefined && typeof (this as any).zMin === 'number' && typeof (this as any).zMax === 'number') {
      const lo = (this as any).zMin as number;
      const hi = (this as any).zMax as number;
      const c = Math.max(lo, Math.min(hi, zSend));
      clampedZ = (c !== zSend);
      zSend = c;
    }

    // Build payload only with provided axes
    const payload: any = {};
    if (xSend !== undefined) payload.x = xSend;
    if (ySend !== undefined) payload.y = ySend;
    if (zSend !== undefined) payload.z = zSend;

    if (!('x' in payload) && !('y' in payload) && !('z' in payload)) {
      console.error('No axes specified.');
      return;
    }

    this.http.post(`${BASE_URL}/move_toolhead_absolute`, payload).subscribe({
      next: (response) => {
        if (clampedX || clampedY || clampedZ) {
          console.warn(
            `Position clamped` +
            `${clampedX ? ` X→${xSend}` : ''}` +
            `${clampedY ? ` Y→${ySend}` : ''}` +
            `${clampedZ ? ` Z→${zSend}` : ''}.`
          );
        }
        console.log('Toolhead moved to the specified position successfully!', response);
      },
      error: (error) => {
        console.error('Failed to move toolhead to the specified position!', error);
      },
    });
  }

  homeAxis(axis?: string): void {
    if (this.isHoming) return;

    this.closeHomeContextMenu();
    this.resetMotorOffState();
    this.autofocusDone = false;

    const ax = axis ? axis.toLowerCase() as 'x' | 'y' | 'z' | 'a' : undefined;
    const payload = { axes: ax ? [ax] : [] };

    // Prevent position polling during homing to avoid serial contention
    this.isHoming = true;
    this.stopPositionPolling();

    this.http.post(`${BASE_URL}/home_toolhead`, payload)
      .pipe(
        timeout(this.HOMING_TIMEOUT_MS), // G28 can take seconds
        finalize(() => {
          this.isHoming = false;
          // small settle before resuming position polling
          setTimeout(() => this.startPollingPosition(), 500);
        })
      )
      .subscribe({
        next: (response: any) => {
          console.log(`Motion platform ${ax ? ax.toUpperCase() : 'ALL'} homed successfully.`, response);
          const position = response?.position;
          const homedPosition = (axis: 'x' | 'y' | 'z'): number =>
            typeof position?.[axis] === 'number' && Number.isFinite(position[axis])
              ? position[axis]
              : 0;

          // Preserve your original side effects
          if (ax === 'a') {
            this.filterStatusGeneration++;
            this.filterRevolverStatus = {
              position: 1,
              homed: true,
              motion_platform_homed: this.xHomed && this.yHomed && this.zHomed,
              busy: false
            };
            this.filterRevolverRotationDegrees = 0;
            this.filterRevolverRotationInitialized = true;
            this.syncFilterRevolverStatus();
          } else if (ax) {
            if (ax === 'x') { this.xPosition = homedPosition('x'); this.xHomed = true; }
            else if (ax === 'y') { this.yPosition = homedPosition('y'); this.yHomed = true; }
            else if (ax === 'z') { this.zPosition = homedPosition('z'); this.zHomed = true; }
          } else {
            this.xPosition = homedPosition('x');
            this.yPosition = homedPosition('y');
            this.zPosition = homedPosition('z');
            this.xHomed = this.yHomed = this.zHomed = true;
          }
          // Publish homed state to SharedService
          this.updateSharedHomedState();
        },
        error: (error) => {
          console.error(`Failed to home Motion platform ${ax ? ax.toUpperCase() : 'ALL'}!`, error);
          // optional: surface a UI error message here
          // this.errorNotificationService.addError({ code: 'E13xx', message: 'Homing failed' });
        },
      });
  }

  async homeAllAxesInOrder(): Promise<void> {

    if (this.isHoming) return;

    this.closeHomeContextMenu();
    this.autofocusDone = false;
    this.isHoming = true;
    this.stopPositionPolling();

    try {
      this.filterStatusGeneration++;
      // The backend executes this exact order and acknowledges every axis.
      const response: any = await firstValueFrom(
        this.http.post(`${BASE_URL}/home_toolhead`, { axes: ['z', 'y', 'x', 'a'] })
      );
      const position = response?.position;
      const homedPosition = (axis: 'x' | 'y' | 'z'): number =>
        typeof position?.[axis] === 'number' && Number.isFinite(position[axis])
          ? position[axis]
          : 0;
      this.xHomed = this.yHomed = this.zHomed = true;
      this.xPosition = homedPosition('x');
      this.yPosition = homedPosition('y');
      this.zPosition = homedPosition('z');
      this.updateSharedHomedState();
      this.filterRevolverStatus = {
        position: 1,
        homed: true,
        motion_platform_homed: true,
        busy: false
      };
      this.filterRevolverRotationDegrees = 0;
      this.filterRevolverRotationInitialized = true;
      this.syncFilterRevolverStatus();

    } catch (err) {
      console.error("Homing error:", err);

    } finally {
      // ALWAYS executed — even when errors happened
      this.isHoming = false;

      // Restart polling safely
      setTimeout(() => {
        this.startPollingPosition();
      }, 500);
    }
  }

  private applyFilterRevolverStatus(status: FilterRevolverStatus): void {
    const externallyChanged = this.filterRevolverStatus.position !== status.position;
    if (
      status.position !== null
      && (!this.filterRevolverRotationInitialized || externallyChanged)
    ) {
      this.filterRevolverRotationDegrees = -(status.position - 1) * 60;
      this.filterRevolverRotationInitialized = true;
    } else if (status.position === null) {
      this.filterRevolverRotationDegrees = 0;
      this.filterRevolverRotationInitialized = false;
    }
    this.filterRevolverStatus = status;
  }


  setMovementAmount(amount: number): void {
    this.movementAmount = amount;
    console.log('Movement amount set to', this.movementAmount);
  }

  // ---------- Motor power ----------

  motorOff(): void {
    // If your backend uses a different endpoint, adjust here (e.g. /motors_off or /disable_steppers)
    this.http.post(`${BASE_URL}/disable_steppers`, {}).subscribe({
      next: () => {
        console.log('Motors have been turned off.');
        this.motorOffState = true;
        this.autofocusDone = false;
      },
      error: (error) => {
        console.error('Failed to turn off motors:', error);
      },
    });
  }


  async autoFocusCoarse(): Promise<void> {
    // If autofocus is already running, abort it
    if (this.isAutofocusing) {
      console.log('[MotionControl] Aborting autofocus...');
      this.autofocusAbortedByUser = true;
      this.http.post(`${BASE_URL}/abort-autofocus`, {}).subscribe({
        next: () => console.log('[MotionControl] Autofocus abort signaled'),
        error: (err) => console.error('Failed to signal autofocus abort', err),
      });
      return;
    }

    this.isAutofocusing = true;
    this.autofocusDone = false;
    this.autofocusAbortedByUser = false;
    this.sharedService.setAutofocusActive(true);

    this.sharedService.setAutofocusError(null);

    this.http.post(`${BASE_URL}/autofocus_coarse`, { skip_empty_check: true }).subscribe({
      next: (resp: any) => {
        console.log('Autofocus response:', resp);
        this.isAutofocusing = false;
        this.sharedService.setAutofocusActive(false);
        if (resp.status === 'ERROR' && resp.code) {
          this.autofocusDone = false;
          // Suppress error when the user manually aborted
          if (!(this.autofocusAbortedByUser && resp.code === 'E2007')) {
            this.sharedService.setAutofocusError(
              this.errorNotificationService.getMessage(resp.code)
            );
          }
        } else {
          this.autofocusDone = true;
          this.filterStatusGeneration++;
          this.syncFilterRevolverStatus();
          this.syncFourChannelLightStatus();
          this.sharedService.setAutofocusError(null);
        }
      },
      error: (error) => {
        console.error('Autofocus error:', error);
        this.isAutofocusing = false;
        this.sharedService.setAutofocusActive(false);
        this.autofocusDone = false;
      },
    });
  }


  // ---------- UI helpers ----------

  formatPosition(pos: number | string): string {
    if (pos === '?') return '?';
    const n = Number(pos);
    if (Number.isNaN(n)) {
      console.error('Invalid position value:', pos);
      return '?';
    }
    // Keep degrees if you really want it; typically mm is expected on a printer:
    return n.toFixed(1) + '°';
  }


  onFocus(axis: 'x' | 'y' | 'z') {
    if (axis === 'x') { this.originalOnFocus.x = this.xPosition; this.isEditingX = true; }
    else if (axis === 'y') { this.originalOnFocus.y = this.yPosition; this.isEditingY = true; }
    else { this.originalOnFocus.z = this.zPosition; this.isEditingZ = true; }
  }



  onBlur(axis: 'x' | 'y' | 'z') {
    if (this.skipNextBlurRevert) { this.skipNextBlurRevert = false; }
    else {
      if (axis === 'x') this.xPosition = this.originalOnFocus.x;
      else if (axis === 'y') this.yPosition = this.originalOnFocus.y;
      else this.zPosition = this.originalOnFocus.z;
    }
    if (axis === 'x') this.isEditingX = false;
    else if (axis === 'y') this.isEditingY = false;
    else this.isEditingZ = false;
  }

  applyCameraSettingsForLight(light: 'dome' | 'bar'): void {
    // Emit an event to notify the camera control component to apply the corresponding settings
    this.sharedService.applyCameraSettingsForLight(light);
  }

  /**
   * Publishes the current homed state to SharedService so other components can react.
   */
  private updateSharedHomedState(): void {
    const allHomed = this.xHomed && this.yHomed && this.zHomed;
    this.sharedService.setMotionHomed(allHomed);
  }
}
