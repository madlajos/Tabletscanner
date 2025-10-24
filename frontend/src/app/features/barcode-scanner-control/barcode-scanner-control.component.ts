import { Component, OnInit, OnDestroy } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { interval, Subscription } from 'rxjs';
import { ErrorNotificationService } from '../../services/error-notification.service';
import { CommonModule } from '@angular/common';
import { MatIconModule } from '@angular/material/icon';

@Component({
  selector: 'app-barcode-scanner-control',
  standalone: true,
  templateUrl: './barcode-scanner-control.component.html',
  styleUrls: ['./barcode-scanner-control.component.css'],
  imports: [CommonModule, MatIconModule]
})
export class BarcodeScannerControlComponent implements OnInit, OnDestroy {
  private readonly BASE_URL = 'http://localhost:5000/api';
  isConnected: boolean = false;
  connectionPolling: Subscription | undefined;
  reconnectionPolling: Subscription | undefined;

  // Define a constant for the barcode error code:
  private readonly BARCODE_ERR_CODE = "E1301";

  constructor(
    private http: HttpClient,
    private errorNotificationService: ErrorNotificationService
  ) {}

  ngOnInit(): void {
      this.startConnectionPolling();
  }

  ngOnDestroy(): void {
    this.stopConnectionPolling();
    this.stopReconnectionPolling();
  }

  startConnectionPolling(): void {
    this.connectionPolling = interval(5000).subscribe(() => {
      this.checkBarcodeStatus();
    });
  }

  stopConnectionPolling(): void {
    this.connectionPolling?.unsubscribe();
    this.connectionPolling = undefined;
  }

  startReconnectionPolling(): void {
    if (!this.reconnectionPolling) {
      this.reconnectionPolling = interval(5000).subscribe(() => {
        this.tryReconnectBarcode();
      });
    }
  }

  stopReconnectionPolling(): void {
    this.reconnectionPolling?.unsubscribe();
    this.reconnectionPolling = undefined;
  }

  checkBarcodeStatus(): void {
    this.http.get<{ connected: boolean }>(`${this.BASE_URL}/status/serial/barcode`)
      .subscribe({
        next: (response) => {
          const wasConnected = this.isConnected;
          this.isConnected = response.connected;
          if (!this.isConnected && !this.reconnectionPolling) {
            console.warn("Barcode Scanner disconnected – starting reconnection polling.");
            this.errorNotificationService.addError({
              code: this.BARCODE_ERR_CODE,
              message: this.errorNotificationService.getMessage(this.BARCODE_ERR_CODE)
            });
            this.stopConnectionPolling();
            this.startReconnectionPolling();
          } else if (this.isConnected && !wasConnected) {
            console.info("Barcode Scanner reconnected.");
            this.errorNotificationService.removeError(this.BARCODE_ERR_CODE);
            this.stopReconnectionPolling();
            this.startConnectionPolling();
          }
        },
        error: (error) => {
          console.error('Failed to check barcode scanner connection!', error);
          if (!this.reconnectionPolling) {
            this.errorNotificationService.addError({
              code: this.BARCODE_ERR_CODE,
              message: this.errorNotificationService.getMessage(this.BARCODE_ERR_CODE)
            });
            this.stopConnectionPolling();
            this.startReconnectionPolling();
          }
          this.isConnected = false;
        }
      });
  }
  
  tryReconnectBarcode(): void {
    console.info('Attempting to reconnect Barcode Scanner...');
    this.http.post<{ message: string }>(`${this.BASE_URL}/connect-to-barcode`, {})
      .subscribe({
        next: (response) => {
          console.info('Barcode Scanner reconnected:', response.message);
          this.isConnected = true;
          this.errorNotificationService.removeError(this.BARCODE_ERR_CODE);
          this.stopReconnectionPolling();
          this.startConnectionPolling();
        },
        error: (error) => {
          console.warn('Barcode Scanner reconnection attempt failed.', error);
        }
      });
  }
}