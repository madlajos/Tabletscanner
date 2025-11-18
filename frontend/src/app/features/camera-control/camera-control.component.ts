import { CommonModule } from '@angular/common';
import { Component, OnInit, OnDestroy } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { SharedService } from '../../shared.service';
import { FormsModule } from '@angular/forms';
import { MatIconModule } from '@angular/material/icon';
import { interval, Subscription, switchMap, catchError, of, timeout } from 'rxjs';
import { ErrorNotificationService } from '../../services/error-notification.service';
import { SettingsUpdatesService, SizeLimits, SaveSettings, CameraSettings } from '../../services/settings-updates.service';


// TODO: Temporary placement; Relocate this to filesaver feature
declare global {
  interface Window {
    electronAPI?: {
      selectFolder: () => Promise<string>;
    };
  }
}


@Component({
  standalone: true,
  selector: 'app-camera-control',
  templateUrl: './camera-control.component.html',
  styleUrls: ['./camera-control.component.css'],
  imports: [CommonModule, FormsModule, MatIconModule]
})


export class CameraControlComponent implements OnInit, OnDestroy {

  // Camera state
  isConnected: boolean = false;
  isStreaming: boolean = false;

  // Polling handles
  connectionPolling: Subscription | undefined;
  reconnectionPolling: Subscription | undefined;
  unifiedPollingSub!: Subscription;

  autoReconnectEnabled = true; // controls auto-reconnect loop
  autoStreamEnabled = true;    // controls auto auto-start of stream when connected

  private statusPollSub?: Subscription;
  private reconnectPollSub?: Subscription;

  // Settings
  cameraSettings: any = {};

  // TODO: refactor to suit current settigns
  saveSettings: SaveSettings = {
    save_csv: false,
    save_images: false,
    csv_dir: ""
  };

  loadedFileName: string = '';
  saveDirectory: string = 'C:\\Users\\Public\\Pictures';

  // TODO: remove these after refactor
  mainCameraSettings: any = this.cameraSettings;
  sideCameraSettings: any = {};
  sizeLimits: SizeLimits = { class1: 0, class2: 0, ng_limit: 0 };

  private readonly CAMERA_ERR_CODE = "E1111";

  private readonly BASE_URL = 'http://localhost:5000/api';

  private settingsLoaded: boolean = false;

  settingOrder: string[] = [
    'Width',
    'Height',
    'OffsetX',
    'OffsetY',
    'ExposureTime',
    'Gain',
    'Gamma',
    'FrameRate'
  ];


  measurementActive: boolean = false;
  private measurementActiveSub!: Subscription;


  constructor(private http: HttpClient,
    public sharedService: SharedService,
    private errorNotificationService: ErrorNotificationService,
    private settingsUpdatesService: SettingsUpdatesService
  ) { }

  ngOnInit(): void {
    if (!this.settingsLoaded) {
      this.loadCameraSettings();
      this.settingsLoaded = true;
    }

    // Initial checks
    this.checkCameraStatus();

    // Periodically check both connection and stream status
    this.startConnectionPolling();

    // Set the shared save directory.
    this.sharedService.setSaveDirectory(this.saveDirectory);

    // Subscribe to shared observables.
    this.measurementActiveSub = this.sharedService.measurementActive$.subscribe(active => {
      this.measurementActive = active;
    });

    this.sharedService.cameraConnectionStatus$.subscribe(status => {
      this.isConnected = status;
    });

    this.sharedService.cameraStreamStatus$.subscribe(status => {
      this.isStreaming = status;
    });

    // TODO: Refactor to set current 'other' settings
    this.http.get<{ size_limits: SizeLimits }>(`${this.BASE_URL}/get-other-settings?category=size_limits`)
      .subscribe({
        next: response => {
          if (response && response.size_limits) {
            // this.sizeLimits = response.size_limits;
            // Publish the loaded settings to the shared service.
            // this.settingsUpdatesService.updateSizeLimits(this.sizeLimits);
            // console.log("Loaded size limits from backend:", this.sizeLimits);
          }
        },
        error: error => {
          console.error("Error loading size limits from backend:", error);
        }
      });

    this.http.get<{ save_settings: SaveSettings }>(`${this.BASE_URL}/get-other-settings?category=save_settings`)
      .subscribe({
        next: response => {
          if (response && response.save_settings) {
            this.saveSettings = response.save_settings;
            if (!this.saveSettings.csv_dir) this.saveSettings.csv_dir = "";
            this.settingsUpdatesService.updateSaveSettings(this.saveSettings);
            console.log("Loaded Save Settings from backend:");
          }
        },
        error: error => {
          console.error("Error loading Save Settings from backend:", error);
        }
      });
  }

  ngOnDestroy(): void {
    this.stopConnectionPolling();
    this.stopReconnectionPolling();

    if (this.unifiedPollingSub) {
      this.unifiedPollingSub.unsubscribe();
    }

    if (this.measurementActiveSub) {
      this.measurementActiveSub.unsubscribe();
    }
  }

  // Backend calls
  checkCameraStatus(): void {
    this.http
      .get<{ connected: boolean; streaming: boolean }>(`${this.BASE_URL}/status/camera`)
      .subscribe({
        next: (response) => {
          // Always sync connection state
          this.sharedService.setCameraConnectionStatus(response.connected);

          if (response.connected) {
            // We are connected → stop reconnection loop
            this.stopReconnectionPolling();

            // Only push "true" -> UI when backend actually reports it AND UI isn't already true.
            if (response.streaming && !this.isStreaming) {
              this.sharedService.setCameraStreamStatus(true);
            }

            // Only auto-start once: do NOT keep calling startVideoStream every poll.
            if (!response.streaming && !this.isStreaming && this.autoStreamEnabled) {
              this.startVideoStream(); // viewer attaches <img> once
            }

            // IMPORTANT: do NOT force this.sharedService.setCameraStreamStatus(false) here
            // when backend says false – that causes flapping and <img> re-creation.
          } else {
            // Disconnected → stop status polling, reflect stream=false, maybe auto-reconnect
            this.stopConnectionPolling();
            if (this.isStreaming) {
              this.sharedService.setCameraStreamStatus(false);
            }

            if (this.autoReconnectEnabled) {
              this.startReconnectionPolling();
            } else {
              this.stopReconnectionPolling();
            }
          }
        },
        error: () => {
          // Treat errors as disconnected
          this.sharedService.setCameraConnectionStatus(false);
          if (this.isStreaming) {
            this.sharedService.setCameraStreamStatus(false);
          }
          this.stopConnectionPolling();
          if (this.autoReconnectEnabled) {
            this.startReconnectionPolling();
          }
        },
      });
  }

  // TODO: Refactor to load current 'other' settings
  loadOtherSettings(): void {
    this.http.get<{ size_limits: SizeLimits }>(`${this.BASE_URL}/get-other-settings?category=size_limits`)
      .subscribe({
        next: response => {
          if (response && response.size_limits) {
            //this.sizeLimits = response.size_limits;
            //this.settingsUpdatesService.updateSizeLimits(this.sizeLimits);
            //console.log("Loaded size limits from backend:", this.sizeLimits);
          }
        },
        error: error => {
          console.error("Error loading size limits from backend:", error);
        }
      });

    this.http.get<{ save_settings: SaveSettings }>(`${this.BASE_URL}/get-other-settings?category=save_settings`)
      .subscribe({
        next: response => {
          if (response && response.save_settings) {
            this.saveSettings = response.save_settings;
            this.settingsUpdatesService.updateSaveSettings(this.saveSettings);
            console.log("Loaded save settings from backend:", this.saveSettings);
          }
        },
        error: error => {
          console.error("Error loading save settings from backend:", error);
        }
      });
  }

  startConnectionPolling(intervalMs: number = 1000): void {
    if (this.statusPollSub && !this.statusPollSub.closed) return;
    this.statusPollSub = interval(intervalMs).subscribe(() => this.checkCameraStatus());
    // kick one immediate check so UI reacts fast
    this.checkCameraStatus();
  }

  stopConnectionPolling(): void {
    if (this.statusPollSub) {
      this.statusPollSub.unsubscribe();
      this.statusPollSub = undefined;
    }
  }

  startReconnectionPolling(intervalMs: number = 3000): void {
    if (!this.autoReconnectEnabled) return; // safety
    if (this.reconnectPollSub && !this.reconnectPollSub.closed) return;

    this.reconnectPollSub = interval(intervalMs).subscribe(() => {
      this.http.post(`${this.BASE_URL}/connect-camera`, {}).subscribe({
        next: () => {
          // On success, stop reconnection and resume status polling
          this.stopReconnectionPolling();
          this.startConnectionPolling();
        },
        error: () => {
          // keep trying silently
        },
      });
    });
  }

  stopReconnectionPolling(): void {
    if (this.reconnectPollSub) {
      this.reconnectPollSub.unsubscribe();
      this.reconnectPollSub = undefined;
    }
  }

  tryReconnectCamera(): void {
    this.http.post(`${this.BASE_URL}/connect-camera`, {})
      .subscribe({
        next: (response: any) => {
          console.info(`Camera reconnected:`, response.message);
          this.sharedService.setCameraConnectionStatus(true);
          const errCode = this.CAMERA_ERR_CODE;
          this.errorNotificationService.removeError(errCode);
          this.stopReconnectionPolling();
          this.startConnectionPolling();
        },
        error: (error) => {
          console.warn(`Camera reconnection attempt failed.`, error);
        }
      });
  }

  checkCameraConnection(): void {
    this.http.get(`${this.BASE_URL}/status/camera`).subscribe(
      (response: any) => {
        this.sharedService.setCameraConnectionStatus(response.connected);
      },
      error => console.error(`Error checking camera status:`, error)
    );
  }

  toggleConnection(): void {
    if (this.isConnected) {
      this.disconnectCamera();
    } else {
      this.connectCamera();
    }
  }

  connectCamera(): void {
    // user wants to connect -> allow auto-reconnect going forward
    this.autoReconnectEnabled = true;

    this.http.post(`${this.BASE_URL}/connect-camera`, {}).subscribe({
      next: () => {
        // resume status polling; it'll flip the UI once /status is true
        this.startConnectionPolling();
        // Note: we do NOT auto-start stream here; user decides when to start.
      },
      error: () => {
        // failed connect – show disconnected and let auto-reconnect try (since user asked to connect)
        this.sharedService.setCameraConnectionStatus(false);
        this.sharedService.setCameraStreamStatus(false);
        this.stopConnectionPolling();
        if (this.autoReconnectEnabled) this.startReconnectionPolling();
      },
    });
  }

  fetchCameraSettings(): void {
    this.http.get(`${this.BASE_URL}/get-camera-settings`).subscribe(
      (settings: any) => {
        this.cameraSettings = settings.camera_params;
        console.log(`Camera settings loaded:`, settings);
      },
      error => console.error(`Error loading camera settings:`, error)
    );
  }

  disconnectCamera(): void {
    // user wants full stop -> pause automation
    this.autoReconnectEnabled = false;
    this.autoStreamEnabled = false;

    // do not let background loops fight user intent
    this.stopConnectionPolling();
    this.stopReconnectionPolling();

    this.http.post(`${this.BASE_URL}/disconnect-camera`, {}).subscribe({
      next: () => {
        // reflect immediately; don't call checkCameraStatus() here
        this.sharedService.setCameraConnectionStatus(false);
        this.sharedService.setCameraStreamStatus(false);
      },
      error: () => {
        // even on error, keep UI consistent with user's explicit intent
        this.sharedService.setCameraConnectionStatus(false);
        this.sharedService.setCameraStreamStatus(false);
      },
    });
  }

  toggleStream(): void {
    if (this.isStreaming) {
      // explicit user intent: pause auto-restart and stop
      this.autoStreamEnabled = false;
      this.stopVideoStream();
    } else {
      // explicit user intent: allow auto-restart and start
      this.autoStreamEnabled = true;
      this.startVideoStream();
    }
  }

  startVideoStream(): void {
    // Already streaming from the UI’s perspective? Don’t re-trigger.
    if (this.isStreaming) return;

    // Let the viewer attach <img src="..."> once.
    this.sharedService.setCameraStreamStatus(true);
    console.log(`Camera stream set to true in SharedService (UI only).`);
  }

  stopVideoStream(): void {
    this.http.post(`${this.BASE_URL}/stop-video-stream`, {}).subscribe(
      () => {
        this.sharedService.setCameraStreamStatus(false);
        console.log(`Camera stream stopped.`);
      },
      error => console.error(`Failed to stop camera stream:`, error)
    );
  }

  loadCameraSettings(): void {
    this.http.get<CameraSettings>(`${this.BASE_URL}/get-camera-settings`)
      .subscribe({
        next: (settings: CameraSettings) => {
          this.cameraSettings = settings;
          this.settingsUpdatesService.updateCameraSettings(settings);
          console.log(`Loaded main camera settings:`, settings);

        },
        error: error => console.error(`Error loading camera settings:`, error)
      });
  }

  applySetting(setting: string): void {
    const value = this.cameraSettings[setting];
    console.log(`Applying setting ${setting}: ${value}`);

    this.http.post(`${this.BASE_URL}/update-camera-settings`, {
      setting_name: setting,
      setting_value: value
    }).subscribe(
      (response: any) => {
        console.log(`Setting applied successfully for camera:`, response);

        const correctedValue = response?.updated_value;

        // Only update the input if the backend corrected it
        if (correctedValue !== undefined && correctedValue !== value) {
          this.cameraSettings[setting] = correctedValue;
          console.log(`Corrected ${setting}: ${value} → ${correctedValue}`);
        }
      },
      error => {
        console.error(`Error applying setting for camera:`, error);
      }
    );
  }


  applySizeLimit(limitName: 'class1' | 'class2' | 'ng_limit'): void {
    let value = Number(this.sizeLimits[limitName]); // convert to number explicitly
    console.log(`Applying size limit ${limitName}: ${value}`);

    this.http.post(`${this.BASE_URL}/update-other-settings`, {
      category: 'size_limits',
      setting_name: limitName,
      setting_value: value
    }).subscribe({
      next: (response: any) => {
        console.log(`Size limit applied successfully:`, response);
        // Update the local value with the response.
        this.sizeLimits[limitName] = Number(response.updated_value);

        // Optionally, reload all size limits from backend.
        this.http.get<{ size_limits: SizeLimits }>(`${this.BASE_URL}/get-other-settings?category=size_limits`)
          .subscribe({
            next: resp => {
              if (resp && resp.size_limits) {
                this.sizeLimits = resp.size_limits;
                console.log("Reloaded size limits:", this.sizeLimits);
                // Publish the updated limits to the shared service.
                this.settingsUpdatesService.updateSizeLimits(this.sizeLimits);
              }
            },
            error: err => console.error("Error reloading settings:", err)
          });
      },
      error: error => {
        console.error(`Error applying size limit ${limitName}:`, error);
      }
    });
  }

  applySaveSetting<K extends keyof SaveSettings>(settingName: K): void {
    // 1. Determine the outgoing value type
    let outgoingValue: SaveSettings[K];

    // If it's a boolean field, cast to Boolean so we never send "undefined"/"null"
    if (typeof this.saveSettings[settingName] === 'boolean') {
      outgoingValue = Boolean(this.saveSettings[settingName]) as SaveSettings[K];
    } else {
      // string (csv_dir) → send as-is
      outgoingValue = this.saveSettings[settingName];
    }

    console.log(`Applying save setting: ${settingName} →`, outgoingValue);

    // 2. Persist to backend
    this.http.post<{ updated_value: SaveSettings[K] }>(
      `${this.BASE_URL}/update-other-settings`,
      {
        category: 'save_settings',
        setting_name: settingName,
        setting_value: outgoingValue
      }
    ).subscribe({
      next: resp => {
        // 3. Mirror backend-confirmed value locally
        this.saveSettings[settingName] = resp.updated_value;

        // Broadcast the fresh copy so other components stay up-to-date
        this.settingsUpdatesService.updateSaveSettings(this.saveSettings);
        console.log('Save setting applied:', settingName, '→', resp.updated_value);
      },
      error: err => console.error(`Error applying ${settingName}:`, err)
    });
  }

  async openFolderBrowser(): Promise<void> {
    // If running in Electron (the preload script exposes an API)
    if (window.electronAPI?.selectFolder) {
      try {
        const folder = await window.electronAPI.selectFolder();
        if (!folder) {
          console.log('User cancelled folder selection');
          return;
        }
        this.updateCsvDirectory(folder);
      } catch (err) {
        console.error('Error selecting folder via Electron:', err);
      }
    } else {
      // Fallback for non-Electron: call backend to open a Tkinter dialog
      this.http.get<{ folder: string }>(`${this.BASE_URL}/select-folder`).subscribe({
        next: resp => {
          if (resp && resp.folder) {
            this.updateCsvDirectory(resp.folder);
          } else {
            console.log('User cancelled folder selection (Tkinter dialog)');
          }
        },
        error: err => {
          console.error('Error during folder selection (backend):', err);
        }
      });
    }
  }

  updateCsvDirectory(folderPath: string): void {
    // 1. Update the local model and UI
    this.saveSettings.csv_dir = folderPath;
    // 2. Persist the new setting to backend (settings.json)
    this.http.post(`${this.BASE_URL}/update-other-settings`, {
      category: 'save_settings',
      setting_name: 'csv_dir',
      setting_value: folderPath
    }).subscribe({
      next: (res: any) => {
        console.log('CSV directory updated in settings:', res);
        // Update shared service if needed so other components know about the change
        this.settingsUpdatesService.updateSaveSettings(this.saveSettings);
      },
      error: err => {
        console.error('Failed to update CSV directory setting:', err);
        // Optionally, revert this.saveSettings.csv_dir if save failed
      }
    });
  }

  objectKeys(obj: any): string[] {
    return Object.keys(obj);
  }
}