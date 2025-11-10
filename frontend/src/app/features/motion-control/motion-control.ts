import { Component, OnInit, OnDestroy } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { interval, Subscription, of } from 'rxjs';
import { switchMap, catchError, timeout, finalize } from 'rxjs/operators';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ErrorNotificationService } from '../../services/error-notification.service';
import { SharedService } from '../../shared.service';

@Component({
  selector: 'app-motion-control',
  // Important for Angular 15+ when using `imports` here:
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './motion-control.html',
  styleUrls: ['./motion-control.scss'], // fixed key (plural)
})
export class MotionControl implements OnInit, OnDestroy {
  private readonly BASE_URL = 'http://localhost:5000/api';

  movementAmount: number = 1;

  motorOffState: boolean = false;

  xPosition: number | string = '?';
  yPosition: number | string = '?';
  zPosition: number | string = '?';

  xMin: number = 0;
  xMax: number = 200;
  yMin: number = 0;
  yMax: number = 200;

  connectionPolling: Subscription | undefined;
  positionPolling: Subscription | undefined;
  reconnectionPolling: Subscription | undefined;
  isConnected: boolean = false;

  isEditingX: boolean = false;
  isEditingY: boolean = false;
  isEditingZ: boolean = false;


  isHoming = false;
  private readonly HOMING_TIMEOUT_MS = 10000;
  xHomed: boolean = true;
  yHomed: boolean = true;
  zHomed: boolean = true;

  constructor(
    private http: HttpClient,
    private errorNotificationService: ErrorNotificationService,
    private sharedService: SharedService
  ) { }

  ngOnInit(): void {
    this.startConnectionPolling();
  }

  ngOnDestroy(): void {
    this.stopConnectionPolling();
    this.stopReconnectionPolling();
    this.stopPositionPolling();
  }

  // ---------- Polling ----------

  startPollingPosition(): void {
    if (this.positionPolling) return;
    this.positionPolling = interval(3000).subscribe(() => {
      this.updateMotionPlatformPosition();
    });
  }

  stopPositionPolling(): void {
    if (this.positionPolling) {
      this.positionPolling.unsubscribe();
      this.positionPolling = undefined;
    }
  }

  updateMotionPlatformPosition(): void {
    if (this.isConnected && !this.motorOffState) {
      this.http
        .get<{ x: number; y: number; z: number }>(`${this.BASE_URL}/get_motion_platform_position`)
        .subscribe({
          next: (position) => {
            if (!this.isEditingX && this.xHomed) this.xPosition = position.x;
            if (!this.isEditingY && this.yHomed) this.yPosition = position.y;
            if (!this.isEditingZ && this.zHomed) this.zPosition = position.z;
          },
          error: (error) => {
            console.error('Failed to get Motion platform position!', error);
          },
        });
    }
  }

  startConnectionPolling(): void {
    if (this.connectionPolling) return;

    this.connectionPolling = interval(5000)
      .pipe(
        switchMap(() =>
          this.http
            .get<{ connected: boolean }>(`${this.BASE_URL}/status/serial/motionplatform`)
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

          if (!this.isConnected && !this.reconnectionPolling) {
            console.warn('Motion platform disconnected – starting reconnection polling.');
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
      .post<{ message: string }>(`${this.BASE_URL}/connect-to-motionplatform`, {})
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

  resetMotorOffState(): void {
    // Many firmwares auto-enable steppers on the first move; we clear the UI flag.
    this.motorOffState = false;
    this.updateMotionPlatformPosition();
  }

  private toNumberOrUndefined(v?: number | string): number | undefined {
    if (v === undefined || v === '?') return undefined;
    const n = typeof v === 'number' ? v : Number(v);
    return Number.isFinite(n) ? n : undefined;
  }

  // ---------- Movements (existing API: *toolhead*) ----------

  moveToolHeadRelative(axis: string, value: number): void {
    if (this.motorOffState) {
      console.error('Cannot move toolhead while motors are off.');
      return;
    }
    this.resetMotorOffState();

    if ((axis === 'x' && !this.xHomed) || (axis === 'y' && !this.yHomed) || (axis === 'z' && !this.zHomed)) {
      console.error(`Cannot move ${axis.toUpperCase()} axis because it is not homed.`);
      return;
    }

    // simple UI bounds checking for X/Y (optional for Z)
    if (axis === 'x') {
      const cur = typeof this.xPosition === 'number' ? this.xPosition : 0;
      const next = cur + value;
      if (next < this.xMin || next > this.xMax) {
        console.error(`X position ${next} out of bounds!`);
        return;
      }
    } else if (axis === 'y') {
      const cur = typeof this.yPosition === 'number' ? this.yPosition : 0;
      const next = cur + value;
      if (next < this.yMin || next > this.yMax) {
        console.error(`Y position ${next} out of bounds!`);
        return;
      }
    }

    const payload = { axis, value };
    this.http.post(`${this.BASE_URL}/move_toolhead_relative`, payload).subscribe({
      next: (response: any) => {
        console.log('Toolhead moved successfully!', response);
      },
      error: (error: any) => {
        console.error('Failed to move Toolhead!', error);
      },
    });
  }

  moveToolHeadAbsolute(x?: number | string, y?: number | string, z?: number | string): void {
    if (this.motorOffState) {
      console.error('Cannot move toolhead while motors are off.');
      return;
    }
    this.resetMotorOffState();

    if ((x === undefined || x === '?') && !this.xHomed) {
      console.error('Cannot move on X axis because it is not homed.');
      return;
    }
    if ((y === undefined || y === '?') && !this.yHomed) {
      console.error('Cannot move on Y axis because it is not homed.');
      return;
    }
    if ((z === undefined || z === '?') && !this.zHomed) {
      console.error('Cannot move on Z axis because it is not homed.');
      return;
    }

    const xNum = this.toNumberOrUndefined(x);
    const yNum = this.toNumberOrUndefined(y);
    const zNum = this.toNumberOrUndefined(z);

    if ((xNum !== undefined && (xNum < this.xMin || xNum > this.xMax)) ||
      (yNum !== undefined && (yNum < this.yMin || yNum > this.yMax))) {
      console.error('New position out of bounds!');
      return;
    }

    const payload = { x: xNum, y: yNum, z: zNum };
    this.http.post(`${this.BASE_URL}/move_toolhead_absolute`, payload).subscribe({
      next: (response) => {
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

    this.http.post(`${this.BASE_URL}/home_toolhead`, payload)
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
        },
        error: (error) => {
          console.error(`Failed to home Motion platform ${ax ? ax.toUpperCase() : 'ALL'}!`, error);
          // optional: surface a UI error message here
          // this.errorNotificationService.addError({ code: 'E13xx', message: 'Homing failed' });
        },
      });
  }

  setMovementAmount(amount: number): void {
    this.movementAmount = amount;
    console.log('Movement amount set to', this.movementAmount);
  }

  // ---------- Motor power ----------

  motorOff(): void {
    // If your backend uses a different endpoint, adjust here (e.g. /motors_off or /disable_steppers)
    this.http.post(`${this.BASE_URL}/disable_steppers`, {}).subscribe({
      next: () => {
        console.log('Motors have been turned off.');
        this.motorOffState = true;
      },
      error: (error) => {
        console.error('Failed to turn off motors:', error);
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

  onFocus(input: string): void {
    if (input === 'x') this.isEditingX = true;
    else if (input === 'y') this.isEditingY = true;
    else if (input === 'z') this.isEditingZ = true;
  }

  onBlur(input: string): void {
    if (input === 'x') this.isEditingX = false;
    else if (input === 'y') this.isEditingY = false;
    else if (input === 'z') this.isEditingZ = false;
  }

  // ---------- Optional: wrappers to keep old template calls working ----------
  // If other templates still reference movePrinter* you can keep these.
  movePrinterRelative(axis: string, value: number) { this.moveToolHeadRelative(axis, value); }
  movePrinterAbsolute(x?: number | string, y?: number | string, z?: number | string) {
    this.moveToolHeadAbsolute(x, y, z);
  }
}
