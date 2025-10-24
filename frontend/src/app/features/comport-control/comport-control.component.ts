import { CommonModule } from '@angular/common';
import { Component, OnInit } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { of, interval } from 'rxjs';
import { catchError, switchMap, tap } from 'rxjs/operators';
import { MatIconModule } from '@angular/material/icon';
import { ErrorNotificationService } from '../../services/error-notification.service';

interface Device {
  name: string;    // Display name
  apiName: string; // Used in API endpoints (e.g., 'turntable' or 'barcode')
  status: string;
  action: string;
}

@Component({
  standalone: true,
  selector: 'app-comport-control',
  templateUrl: './comport-control.component.html',
  styleUrls: ['./comport-control.component.css'],
  imports: [CommonModule, MatIconModule]
})
export class ComportControlComponent implements OnInit {

  devices: Device[] = [
    { name: 'Turntable', apiName: 'turntable', status: 'Checking...', action: 'Connect' },
    { name: 'Barcode Scanner', apiName: 'barcode', status: 'Checking...', action: 'Connect' }
  ];

  private readonly BASE_URL = 'http://localhost:5000/api';

  constructor(private http: HttpClient, private errorNotificationService: ErrorNotificationService) {}


  ngOnInit(): void {
    this.initializeDevices();
    
    interval(5000).pipe(
      switchMap(() => this.checkStatus()),
      catchError(error => {
        console.error("Error in polling:", error);
        return of(null);
      })
    ).subscribe();
  }

  initializeDevices(): void {
    this.devices.forEach(device => {
      this.http
        .get<{ connected: boolean; port?: string }>(`${this.BASE_URL}/status/serial/${device.apiName}`)
        .pipe(
          tap((res) => {
            if (res && res.connected) {
              device.status = `Connected (${res.port || 'COM'})`;
              device.action = 'Disconnect';
            } else {
              device.status = 'Attempting to connect...';
              this.connectDevice(device);
            }
          }),
          catchError((error) => {
            console.error(`Error checking status for ${device.name}:`, error);
            device.status = 'Error';
            device.action = 'Connect';
            return of(null);
          })
        )
        .subscribe();
    });
  }

  checkStatus() {
    return of(null).pipe(
      tap(() => {
        this.devices.forEach(device => {
          this.http.get<{ connected: boolean; port?: string }>(`${this.BASE_URL}/status/serial/${device.apiName}`)
            .pipe(
              tap((res) => {
                if (res && res.connected) {
                  device.status = `Connected (${res.port || 'COM'})`;
                  device.action = 'Disconnect';
                } else {
                  device.status = 'Disconnected';
                  device.action = 'Connect';
                  console.warn(`${device.name} disconnected.`);
                }
              }),
              catchError(error => {
                console.error(`Error fetching status for ${device.name}:`, error);
                device.status = 'Error';
                device.action = 'Connect';
                return of(null);
              })
            )
            .subscribe();
        });
      })
    );
  }

  toggleConnection(device: Device): void {
    if (device.status.startsWith('Connected')) {
      this.disconnectDevice(device);
    } else {
      this.connectDevice(device);
    }
  }

  connectDevice(device: Device): void {
    console.log(`Connecting to ${device.name}...`);
    device.status = 'Connecting...';
    this.http.post(`${this.BASE_URL}/connect-to-${device.apiName}`, {})
      .pipe(
        tap(() => {
          console.log(`${device.name} connected successfully.`);
          if (device.apiName === 'turntable') {
            this.errorNotificationService.removeError("Turntable disconnected");
          } else if (device.apiName === 'barcode') {
            this.errorNotificationService.removeError("Barcode Scanner disconnected");
          }
          this.checkStatus(); // Refresh status to update all devices
        }),
        catchError((error) => {
          console.error(`Error connecting ${device.name}:`, error);
          device.status = 'Error';
          return of(null);
        })
      )
      .subscribe();
  }
  
  
  disconnectDevice(device: Device): void {
    console.log(`Disconnecting from ${device.name}...`);
    device.status = 'Disconnecting...';
    this.http
      .post(`${this.BASE_URL}/disconnect-${device.apiName}`, {})
      .pipe(
        tap(() => {
          console.log(`${device.name} disconnected successfully.`);
          this.checkStatus();  // Refresh status after disconnect
        }),
        catchError((error) => {
          console.error(`Error disconnecting ${device.name}:`, error);
          device.status = 'Error';
          return of(null);
        })
      )
      .subscribe();
  }
}
