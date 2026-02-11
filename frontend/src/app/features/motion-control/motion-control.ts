import { Component, OnInit, OnDestroy } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { interval, Subscription, of } from 'rxjs';
import { switchMap, catchError, timeout, finalize } from 'rxjs/operators';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ErrorNotificationService } from '../../services/error-notification.service';
import { SharedService } from '../../shared.service';
import { firstValueFrom } from 'rxjs';
import { BASE_URL } from '../../api-config';


@Component({
  selector: 'app-motion-control',
  // Important for Angular 15+ when using `imports` here:
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './motion-control.html',
  styleUrls: ['./motion-control.scss'], // fixed key (plural)
})
export class MotionControl implements OnInit, OnDestroy {
  movementAmount: number = 1;

  motorOffState: boolean = false;

  xPosition: number | string = '?';
  yPosition: number | string = '?';
  zPosition: number | string = '?';

  xMin: number = 0;
  xMax: number = 175;
  yMin: number = 0;
  yMax: number = 175;
  zMin: number = 0;
  zMax: number = 30;

  private originalOnFocus = { x: undefined as any, y: undefined as any, z: undefined as any };
  private skipNextBlurRevert = false;
  isEditing = { x: false, y: false, z: false };


  connectionPolling: Subscription | undefined;
  positionPolling: Subscription | undefined;
  reconnectionPolling: Subscription | undefined;
  private measurementActiveSub?: Subscription;
  private motionPositionSub?: Subscription;
  private lightsOffSub?: Subscription;
  isConnected: boolean = false;

  // Flag to lock controls during auto-measurement
  measurementActive: boolean = false;

  isEditingX: boolean = false;
  isEditingY: boolean = false;
  isEditingZ: boolean = false;

  ringLightOn: boolean = false;
  barLightOn: boolean = false;
  lightBusy = false;


  isHoming = false;
  private readonly HOMING_TIMEOUT_MS = 10000;
  xHomed: boolean = false;
  yHomed: boolean = false;
  zHomed: boolean = false;

  isAutofocusing = false;
  autofocusDone = false;

  constructor(
    private http: HttpClient,
    private errorNotificationService: ErrorNotificationService,
    private sharedService: SharedService
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
      this.ringLightOn = false;
      this.barLightOn = false;
    });
  }

  ngOnDestroy(): void {
    this.stopConnectionPolling();
    this.stopReconnectionPolling();
    this.stopPositionPolling();
    this.measurementActiveSub?.unsubscribe();
    this.motionPositionSub?.unsubscribe();
    this.lightsOffSub?.unsubscribe();
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

  // ---------- Helpers ----------

  async toggleBarLight(): Promise<void> {
    if (this.lightBusy) return;
    this.lightBusy = true;
    try {
      if (!this.barLightOn) {
        // Turning bar on: ensure dome is off first
        if (this.ringLightOn) {
          await this.sendGcode('M106 P0 S255');
          this.ringLightOn = false;
        }
        await this.sendGcode('M106 P1 S255');
        this.barLightOn = true;
        this.ringLightOn = false;
        console.log('[MotionControl] Bar light turned ON - setting active light to bar');
        this.sharedService.setActiveLight('bar');
        this.applyCameraSettingsForLight('bar');
      } else {
        // Turning bar off
        await this.sendGcode('M106 P1 S0');
        this.barLightOn = false;
        console.log('[MotionControl] Bar light turned OFF - setting active light to null');
        this.sharedService.setActiveLight(null);
      }
    } catch (err) {
      console.error('Failed to toggle bar light', err);
    } finally {
      this.lightBusy = false;
    }
  }


  async toggleDomeLight(): Promise<void> {
    if (this.lightBusy) return;
    this.lightBusy = true;
    try {
      if (!this.ringLightOn) {
        // Turning dome on: ensure bar is off first
        if (this.barLightOn) {
          await this.sendGcode('M106 P1 S0');
          this.barLightOn = false;
        }
        await this.sendGcode('M106 P0 S0');
        this.ringLightOn = true;
        this.barLightOn = false;
        console.log('[MotionControl] Dome light turned ON - setting active light to dome');
        this.sharedService.setActiveLight('dome');
        this.applyCameraSettingsForLight('dome');
      } else {
        // Turning dome off
        await this.sendGcode('M106 P0 S255');
        this.ringLightOn = false;
        console.log('[MotionControl] Dome light turned OFF - setting active light to null');
        this.sharedService.setActiveLight(null);
      }
    } catch (err) {
      console.error('Failed to toggle dome light', err);
    } finally {
      this.lightBusy = false;
    }
  }

  resetMotorOffState(): void {
    // Many firmwares auto-enable steppers on the first move; we clear the UI flag.
    this.motorOffState = false;
    this.updateMotionPlatformPosition();
  }

  private clamp(v: number, lo: number, hi: number): number {
    return Math.max(lo, Math.min(hi, v));
  }

  private toNumberOrUndefined(v?: number | string): number | undefined {
    if (v === undefined || v === '?') return undefined;
    const n = typeof v === 'number' ? v : Number(v);
    return Number.isFinite(n) ? n : undefined;
  }

  /**
   * Small helper to send a single G-code command sequentially.
   */
  private async sendGcode(command: string): Promise<void> {
    await firstValueFrom(this.http.post(`${BASE_URL}/send_gcode`, { command }));
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
    this.resetMotorOffState();

    const ax = axis ? axis.toLowerCase() as 'x' | 'y' | 'z' : undefined;
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

          // Preserve your original side effects
          if (ax) {
            if (ax === 'x') { this.xPosition = 0; this.xHomed = true; }
            else if (ax === 'y') { this.yPosition = 0; this.yHomed = true; }
            else if (ax === 'z') { this.zPosition = 0; this.zHomed = true; }
          } else {
            this.xPosition = 0; this.yPosition = 0; this.zPosition = 0;
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

    this.isHoming = true;
    this.stopPositionPolling();

    const homeAxis = (axis: 'x' | 'y' | 'z') =>
      firstValueFrom(this.http.post(`${BASE_URL}/home_toolhead`, { axes: [axis] }));

    try {
      // ---------------- Z ----------------
      await homeAxis('z');
      this.zHomed = true;
      this.zPosition = 0;
      this.updateSharedHomedState();

      // ---------------- Y ----------------
      await homeAxis('y');
      this.yHomed = true;
      this.yPosition = 0;
      this.updateSharedHomedState();

      // ---------------- X ----------------
      await homeAxis('x');
      this.xHomed = true;
      this.xPosition = 0;
      this.updateSharedHomedState();

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
      },
      error: (error) => {
        console.error('Failed to turn off motors:', error);
      },
    });
  }


  async autoFocusCoarse(): Promise<void> {
    this.isAutofocusing = true;
    this.autofocusDone = false;

    // If neither light is on, turn on the dome light before autofocus
    if (!this.ringLightOn && !this.barLightOn) {
      try {
        await this.sendGcode('M106 P0 S0');   // dome on
        this.ringLightOn = true;
        this.barLightOn = false;
        console.log('[MotionControl] Dome light auto-enabled for autofocus');
        this.sharedService.setActiveLight('dome');
        this.applyCameraSettingsForLight('dome');
      } catch (err) {
        console.error('Failed to turn on dome light before autofocus', err);
      }
    }

    this.http.post(`${BASE_URL}/autofocus_coarse`, {}).subscribe({
      next: (resp) => {
        console.log('Autofocus response:', resp);
        this.isAutofocusing = false;
        this.autofocusDone = true;
      },
      error: (error) => {
        console.error('Autofocus error:', error);
        this.isAutofocusing = false;
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
