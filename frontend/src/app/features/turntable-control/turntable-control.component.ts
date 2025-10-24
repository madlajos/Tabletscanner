import { Component, OnInit, OnDestroy } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { interval, Subscription, of } from 'rxjs';
import { switchMap, catchError, retryWhen, delay, take, timeout } from 'rxjs/operators';
import { MatIconModule } from '@angular/material/icon';
import { CommonModule } from '@angular/common';
import { ErrorNotificationService } from '../../services/error-notification.service';
import { SharedService } from '../../shared.service';

@Component({
  standalone: true,
  selector: 'app-turntable-control',
  templateUrl: './turntable-control.component.html',
  styleUrls: ['./turntable-control.component.css'],
  imports: [CommonModule, MatIconModule]
})
export class TurntableControlComponent implements OnInit, OnDestroy {
  private readonly BASE_URL = 'http://localhost:5000/api';
  movementAmount: number = 1;
  turntablePosition: number | string = '?';
  connectionPolling: Subscription | undefined;
  reconnectionPolling: Subscription | undefined;
  isConnected: boolean = false;
  isEditingPosition: boolean = false;
  isHomed: boolean = false;

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
  }

  // Start polling for turntable status every 5 seconds
  startConnectionPolling(): void {
    if (!this.connectionPolling) {
      this.connectionPolling = interval(5000)
        .pipe(
          switchMap(() =>
            this.http.get<{ connected: boolean }>(`${this.BASE_URL}/status/serial/turntable`)
              .pipe(
                timeout(5000),
                catchError(err => {
                  console.warn('Turntable connection polling timed out or failed.', err);
                  return of({ connected: false }); // fallback to disconnected
                })
              )
          )
        )
        .subscribe({
          next: (response) => {
            const wasConnected = this.isConnected;
            this.isConnected = response.connected;
  
            if (!this.isConnected && !this.reconnectionPolling) {
              console.warn("Turntable disconnected – starting reconnection polling.");
              this.errorNotificationService.addError({
                code: "E1201",
                message: this.errorNotificationService.getMessage("E1201")
              });
              this.stopConnectionPolling();
              this.startReconnectionPolling();
            } else if (this.isConnected && !wasConnected) {
              console.info("Turntable reconnected.");
              this.errorNotificationService.removeError("E1201");
              this.stopReconnectionPolling();
              this.startConnectionPolling();
            }
          },
          error: (error) => {
            // This is now mostly a safety net
            console.error('Unexpected polling error!', error);
            this.isConnected = false;
          }
        });
    }
  }

  stopConnectionPolling(): void {
    if (this.connectionPolling) {
      this.connectionPolling.unsubscribe();
      this.connectionPolling = undefined;
    }
  }

  // Start reconnection polling every 3 seconds
  startReconnectionPolling(): void {
    if (!this.reconnectionPolling) {
      this.reconnectionPolling = interval(3000).subscribe(() => {
        this.tryReconnectTurntable();
      });
    }
  }

  stopReconnectionPolling(): void {
    if (this.reconnectionPolling) {
      this.reconnectionPolling.unsubscribe();
      this.reconnectionPolling = undefined;
    }
  }

  // Attempt to reconnect the turntable
  tryReconnectTurntable(): void {
    this.http.post<{ message: string }>(`${this.BASE_URL}/connect-to-turntable`, {})
      .pipe(
        timeout(3000),
        catchError(err => {
          console.warn("Turntable reconnection attempt timed out or errored:", err);
          // Return a fallback so the subscribe block can proceed
          return of({ message: "Reconnection failed" });
        })
      )
      .subscribe({
        next: (response) => {
          if (response.message !== "Reconnection failed") {
            console.info("Turntable reconnected:", response.message);
            this.isConnected = true;
            this.errorNotificationService.removeError("E1201");
            this.stopReconnectionPolling();
            this.startConnectionPolling();
          } else {
            console.warn("Turntable reconnection attempt failed after fallback.");
          }
        },
        error: (error) => {
          // Should rarely occur because of catchError
          console.warn("Turntable reconnection attempt failed (unexpected).", error);
        }
      });
  }

  moveTurntableRelative(degrees: number): void {
    const payload = { degrees };
    this.http.post(`${this.BASE_URL}/move_turntable_relative`, payload).subscribe(
      (response: any) => {
        console.log('Turntable moved successfully!', response.message);
        this.turntablePosition = response.current_position !== '?' ? response.current_position : '?';
      },
      (error: any) => {
        console.error('Failed to move turntable relative!', error);
      }
    );
  }

  homeTurntable(): void {
    this.http.post(`${this.BASE_URL}/home_turntable_with_image`, {}).subscribe(
      (response: any) => {
        console.log('Homing successful:', response);
        this.turntablePosition = response.current_position;
      },
      (error) => {
        console.error('Failed to home turntable!', error);
      }
    );
  }

  setMovementAmount(amount: number): void {
    this.movementAmount = amount;
    console.log('Movement amount set to', this.movementAmount);
  }

  formatPosition(pos: number | string): string {
    if (pos === '?') {
      return '?';
    }
    const numericValue = Number(pos);
    if (isNaN(numericValue)) {
      console.error('Invalid position value:', pos);
      return '?';
    }
    return numericValue.toFixed(1) + '°';
  }

  onFocus(): void {
    this.isEditingPosition = true;
  }

  onBlur(): void {
    this.isEditingPosition = false;
  }
}