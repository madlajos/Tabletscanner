import { CommonModule } from '@angular/common';
import { Component, OnInit, OnDestroy } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { SharedService } from '../../shared.service';
import { FormsModule } from '@angular/forms';
import { MatIconModule } from '@angular/material/icon';
import { interval, Subscription, switchMap, catchError, of, timeout } from 'rxjs';
import { ErrorNotificationService } from '../../services/error-notification.service';
import { SettingsUpdatesService, SizeLimits, SaveSettings, CameraSettings } from '../../services/settings-updates.service';
import { BASE_URL } from '../../api-config';


// TODO: Temporary placement; Relocate this to filesaver feature
declare global {
  interface Window {
    electronAPI?: {
      selectFolder: () => Promise<string>;
      selectFile: () => Promise<string>;
    };
    showSaveFilePicker?: (options?: any) => Promise<any>;
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

  otherSettings = {
    objective: '50x',
    spacer_rings: '0',
    camera_settings_file: '',
    save_location: 'C:\\Users\\Public\\Pictures',  // default folder
    background_subtraction: false,
    save_settings: false,
    settings_preset_name: ''
  };

  loadedFileName: string = '';
  saveDirectory: string = 'C:\\Users\\Public\\Pictures';

  saveSettings: SaveSettings = {
    save_csv: false,
    save_images: false,
    csv_dir: 'C:\\Users\\Public\\Pictures'
  };

  // TODO: remove these after refactor
  mainCameraSettings: any = this.cameraSettings;
  sideCameraSettings: any = {};
  sizeLimits: SizeLimits = { class1: 0, class2: 0, ng_limit: 0 };

  private readonly CAMERA_ERR_CODE = "E1111";

  private settingsLoaded: boolean = false;

  measurementActive: boolean = false;
  private measurementActiveSub!: Subscription;
  private activeLightSub?: Subscription;
  currentActiveLight: 'dome' | 'bar' | null = null;

  // Track original values to detect actual changes
  private originalValues: { [key: string]: any } = {};


  constructor(private http: HttpClient,
    public sharedService: SharedService,
    private errorNotificationService: ErrorNotificationService,
    private settingsUpdatesService: SettingsUpdatesService
  ) { }

  // Preset name (Beállítások row)
  settingsPresetName: string = '';

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

    // Listen for light changes and apply corresponding camera settings
    this.sharedService.lightSettings$.subscribe(light => {
      if (light === 'dome') {
        this.applyLightSpecificSettings('Dome');
      } else if (light === 'bar') {
        this.applyLightSpecificSettings('Bar');
      }
    });

    // Subscribe to active light changes for debugging and state tracking
    this.activeLightSub = this.sharedService.activeLight$.subscribe(light => {
      this.currentActiveLight = light;
      console.log(`[CameraControl] Active light changed to: ${light || 'none'}`);
    });

    // TODO: Refactor to set current 'other' settings
    this.http.get<{ other_settings: any }>(`${BASE_URL}/get-other-settings?category=other_settings`)
      .subscribe({
        next: res => {
          if (res.other_settings) {
            // Merge stored settings into our object
            this.otherSettings = {
              ...this.otherSettings,
              ...res.other_settings
            };
            // Apply save location to shared service for consistency
            if (this.otherSettings.save_location) {
              this.sharedService.setSaveDirectory(this.otherSettings.save_location);
            }

            // Restore preset display name if persisted
            if (this.otherSettings.settings_preset_name) {
              this.settingsPresetName = this.stripExtension(this.otherSettings.settings_preset_name);
            }
          }
        },
        error: err => {
          console.warn('No saved other_settings (using defaults).', err);
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
    if (this.activeLightSub) {
      this.activeLightSub.unsubscribe();
    }
  }

  // Backend calls
  checkCameraStatus(): void {
    this.http
      .get<{ connected: boolean; streaming: boolean }>(`${BASE_URL}/status/camera`)
      .subscribe({
        next: (response) => {
          // Always sync connection state
          this.sharedService.setCameraConnectionStatus(response.connected);

          if (response.connected) {
            // We are connected → stop reconnection loop
            this.stopReconnectionPolling();

            // Clear the camera error since we're now connected
            this.errorNotificationService.removeError(this.CAMERA_ERR_CODE);

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
    this.http.get<{ size_limits: SizeLimits }>(`${BASE_URL}/get-other-settings?category=size_limits`)
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

    this.http.get<{ save_settings: SaveSettings }>(`${BASE_URL}/get-other-settings?category=save_settings`)
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

  applyOtherSetting(setting: string): void {
    // Determine value to send (cast booleans to actual boolean type)
    let value = this.otherSettings[setting as keyof typeof this.otherSettings];
    if (typeof value === 'boolean') {
      value = Boolean(value);
    }
    // Send update to backend
    this.http.post(`${BASE_URL}/update-other-settings`, {
      category: 'other_settings',
      setting_name: setting,
      setting_value: value
    }).subscribe({
      next: resp => {
        console.log(`Updated ${setting} = ${value} in settings.json`);
        // If the save location was changed, update shared service as well
        if (setting === 'save_location') {
          this.sharedService.setSaveDirectory(value as string);
        }
      },
      error: err => {
        console.error(`Failed to update setting ${setting}:`, err);
      }
    });
  }

  async openCameraFileBrowser(): Promise<void> {
    if (window.electronAPI?.selectFile) {
      // Electron environment: use a dialog to pick a .pfs file
      try {
        const filePath = await window.electronAPI.selectFile();  // hypothetical API
        if (filePath) {
          this.otherSettings.camera_settings_file = filePath;
          this.applyOtherSetting('camera_settings_file');
          this.invalidatePreset();
          // Load the profile onto the camera
          if (this.isConnected) {
            await this.loadCameraProfile(filePath);
          }
        }
      } catch (e) {
        console.error('File selection error:', e);
      }
    } else {
      // Fallback: (could implement a backend route similar to select-folder)
      this.http.get<{ file: string }>(`${BASE_URL}/select-file`).subscribe({
        next: async resp => {
          if (resp.file) {
            this.otherSettings.camera_settings_file = resp.file;
            this.applyOtherSetting('camera_settings_file');
            this.invalidatePreset();
            // Load the profile onto the camera
            if (this.isConnected) {
              await this.loadCameraProfile(resp.file);
            }
          }
        },
        error: err => console.error('File dialog failed:', err)
      });
    }
  }

  async openImageFolderBrowser(): Promise<void> {
    // Similar to openFolderBrowser, use existing backend dialog to choose folder
    if (window.electronAPI?.selectFolder) {
      try {
        const folder = await window.electronAPI.selectFolder();
        if (folder) {
          this.otherSettings.save_location = folder;
          this.applyOtherSetting('save_location');
        }
      } catch (e) {
        console.error('Folder selection error:', e);
      }
    } else {
      this.http.get<{ folder: string }>(`${BASE_URL}/select-folder`).subscribe({
        next: resp => {
          if (resp.folder) {
            this.otherSettings.save_location = resp.folder;
            this.applyOtherSetting('save_location');
          }
        },
        error: err => console.error('Folder dialog failed:', err)
      });
    }
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
      this.http.post(`${BASE_URL}/connect-camera`, {}).subscribe({
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
    this.http.post(`${BASE_URL}/connect-camera`, {})
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
    this.http.get(`${BASE_URL}/status/camera`).subscribe(
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

    this.http.post(`${BASE_URL}/connect-camera`, {}).subscribe({
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
    this.http.get(`${BASE_URL}/get-camera-settings`).subscribe(
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

    this.http.post(`${BASE_URL}/disconnect-camera`, {}).subscribe({
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
    this.http.post(`${BASE_URL}/stop-video-stream`, {}).subscribe(
      () => {
        this.sharedService.setCameraStreamStatus(false);
        console.log(`Camera stream stopped.`);
      },
      error => console.error(`Failed to stop camera stream:`, error)
    );
  }

  loadCameraSettings(): void {
    this.http.get<any>(`${BASE_URL}/get-camera-settings`)
      .subscribe({
        next: (response: any) => {
          // Load main settings
          if (response.camera_params) {
            this.cameraSettings = { ...this.cameraSettings, ...response.camera_params };
          }
          
          // Load light-specific settings
          if (response.camera_params_dome) {
            Object.keys(response.camera_params_dome).forEach(key => {
              this.cameraSettings[`${key}_Dome`] = response.camera_params_dome[key];
            });
          }
          
          if (response.camera_params_bar) {
            Object.keys(response.camera_params_bar).forEach(key => {
              this.cameraSettings[`${key}_Bar`] = response.camera_params_bar[key];
            });
          }

          this.settingsUpdatesService.updateCameraSettings(this.cameraSettings);
          console.log(`Loaded camera settings:`, this.cameraSettings);
        },
        error: error => console.error(`Error loading camera settings:`, error)
      });
  }

  applySetting(setting: string): void {
    const value = this.cameraSettings[setting];
    console.log(`Applying setting ${setting}: ${value}`);

    // Check if this is a light-specific setting (e.g., ExposureTime_Dome, Gamma_Dome)
    const lightMatch = setting.match(/(ExposureTime|Gamma)_(Dome|Bar)/);
    
    if (lightMatch) {
      // Light-specific setting
      const settingName = lightMatch[1];
      const lightSuffix = lightMatch[2];
      const lightName = lightSuffix.toLowerCase();

      // Check which light is currently active
      const activeLight = this.sharedService.getActiveLight();
      
      // Only apply to camera hardware if this setting matches the currently active light
      const shouldApplyToCamera = activeLight === lightName;
      
      console.log(`[CameraControl] Modifying ${lightName} setting ${settingName}. Active light: ${activeLight || 'none'}. Will apply to camera: ${shouldApplyToCamera}`);
      
      if (!shouldApplyToCamera) {
        console.log(`[CameraControl] ✓ Skipping camera hardware update for ${setting} - ${lightName} light is not active`);
      } else {
        console.log(`[CameraControl] → Applying ${setting} to camera hardware (${lightName} light is active)`);
      }

      this.http.post(`${BASE_URL}/update-camera-settings-light`, {
        light: lightName,
        setting_name: settingName,
        setting_value: value,
        apply_to_camera: shouldApplyToCamera
      }).subscribe(
        (response: any) => {
          if (shouldApplyToCamera) {
            console.log(`Light-specific setting applied to camera successfully:`, response);
          } else {
            console.log(`Light-specific setting saved to settings.json only:`, response);
          }

          const correctedValue = response?.updated_value;

          if (correctedValue !== undefined && correctedValue !== value) {
            this.cameraSettings[setting] = correctedValue;
            console.log(`Corrected ${setting}: ${value} → ${correctedValue}`);
          }
        },
        error => {
          console.error(`Error applying light-specific setting ${setting}:`, error);
        }
      );
    } else {
      // Non-light-specific setting (deprecated path - kept for backwards compatibility)
      this.http.post(`${BASE_URL}/update-camera-settings`, {
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
  }

  applyLightSpecificSettings(lightSuffix: 'Dome' | 'Bar'): void {
    // Apply the light-specific settings (e.g., ExposureTime_Dome, Gamma_Dome)
    const lightName = lightSuffix.toLowerCase();
    const settings = [
      { key: `ExposureTime_${lightSuffix}`, name: 'ExposureTime' },
      { key: `Gamma_${lightSuffix}`, name: 'Gamma' }
    ];

    settings.forEach(setting => {
      const value = this.cameraSettings[setting.key];
      if (value !== undefined) {
        console.log(`Applying light-specific setting ${setting.key}: ${value}`);

        this.http.post(`${BASE_URL}/update-camera-settings-light`, {
          light: lightName,
          setting_name: setting.name,
          setting_value: value
        }).subscribe(
          (response: any) => {
            console.log(`Setting applied successfully:`, response);

            const correctedValue = response?.updated_value;

            if (correctedValue !== undefined && correctedValue !== value) {
              this.cameraSettings[setting.key] = correctedValue;
              console.log(`Corrected ${setting.key}: ${value} → ${correctedValue}`);
            }
          },
          error => {
            console.error(`Error applying setting ${setting.key}:`, error);
          }
        );
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
      `${BASE_URL}/update-other-settings`,
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
      this.http.get<{ folder: string }>(`${BASE_URL}/select-folder`).subscribe({
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
    this.http.post(`${BASE_URL}/update-other-settings`, {
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

  // Store original value when user focuses on input
  handleNumericFocus(settingKey: string): void {
    this.originalValues[settingKey] = this.cameraSettings[settingKey];
  }

  // Numeric input handling for camera settings
  handleNumericBlur(settingKey: string, decimals: number): void {
    let raw = this.cameraSettings[settingKey];
    if (raw === undefined || raw === null || raw === '') {
      return; // do not apply empty
    }

    // Parse as number
    let num = Number(raw);
    if (!isFinite(num)) {
      // Revert to previous numeric value if invalid
      return;
    }

    // Round to requested decimals
    if (decimals <= 0) {
      num = Math.round(num);
    } else {
      const factor = Math.pow(10, decimals);
      num = Math.round(num * factor) / factor;
    }

    // Trim trailing zeros by storing as number (not string)
    this.cameraSettings[settingKey] = num;

    // Only invalidate preset if value actually changed
    const originalValue = this.originalValues[settingKey];
    const valueChanged = originalValue !== num;
    
    if (valueChanged) {
      console.log(`[CameraControl] Setting ${settingKey} changed from ${originalValue} to ${num} - invalidating preset`);
      this.invalidatePreset();
      // Apply to backend
      this.applySetting(settingKey);
    } else {
      console.log(`[CameraControl] Setting ${settingKey} unchanged (${num}) - keeping preset name`);
    }
  }

  invalidatePreset(): void {
    this.settingsPresetName = '';
  }

  sanitizeFileName(name: string): string {
    return (name || 'camera-settings-preset')
      .replace(/[^a-zA-Z0-9._-]/g, '_')
      .replace(/_+/g, '_')
      .replace(/^_+|_+$/g, '') || 'camera-settings-preset';
  }

  private stripExtension(filename: string): string {
    if (!filename) return '';
    const idx = filename.lastIndexOf('.');
    if (idx <= 0) return filename;
    return filename.substring(0, idx);
  }

  async saveSettingsPreset(): Promise<void> {
    const payload: any = {
      camera_profile_path: this.otherSettings.camera_settings_file || '',
      exposure_time_dome: Number(this.cameraSettings.ExposureTime_Dome),
      exposure_time_bar: Number(this.cameraSettings.ExposureTime_Bar),
      gamma_dome: Number(this.cameraSettings.Gamma_Dome),
      gamma_bar: Number(this.cameraSettings.Gamma_Bar)
    };

    // Basic validation
    const keysToCheck = [
      'exposure_time_dome', 'exposure_time_bar',
      'gamma_dome', 'gamma_bar'
    ];
    const invalid = keysToCheck.find(k => !isFinite(payload[k]));
    if (invalid) {
      console.error(`Cannot save preset, invalid value for ${invalid}`);
      return;
    }

    const fileNameBase = this.sanitizeFileName(this.settingsPresetName || 'camera-settings-preset');
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
    const suggestedName = fileNameBase; // show textbox without extension
    const downloadName = `${fileNameBase}.tss`;

    try {
      // Use native file save picker when available (Edge/Chrome/Win11) so user can choose path/name.
      if (typeof window.showSaveFilePicker === 'function') {
        const handle = await window.showSaveFilePicker({
          suggestedName,
          types: [
            {
              description: 'Tabletscanner Preset',
              accept: { 'application/json': ['.tss'] }
            }
          ]
        });
        const writable = await handle.createWritable();
        await writable.write(blob);
        await writable.close();

        const pickedName = this.stripExtension(handle?.name || suggestedName);
        this.settingsPresetName = pickedName;
        this.otherSettings.settings_preset_name = pickedName;
        this.applyOtherSetting('settings_preset_name');
        return;
      }
    } catch (err) {
      console.error('Save file picker failed, falling back to download:', err);
    }

    // Fallback: browser download (user can still pick location via standard download prompt).
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = downloadName;
    link.click();
    URL.revokeObjectURL(link.href);

    // Persist chosen preset name (without extension) to settings
    this.settingsPresetName = fileNameBase;
    this.otherSettings.settings_preset_name = fileNameBase;
    this.applyOtherSetting('settings_preset_name');
  }

  async onPresetFileSelected(event: Event): Promise<void> {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0];
    if (!file) return;

    try {
      const text = await file.text();
      const preset = JSON.parse(text);
      await this.applyPresetData(preset, file.name);
    } catch (err: any) {
      console.error('Failed to load preset:', err);
      this.errorNotificationService.addError({
        code: 'E1312',
        message: '',
        popupStyle: 'center'
      });
    } finally {
      input.value = '';
    }
  }

  async applyPresetData(preset: any, fileName?: string): Promise<void> {
    const requiredKeys = [
      'exposure_time_dome', 'exposure_time_bar',
      'gamma_dome', 'gamma_bar',
      'camera_profile_path'
    ];

    for (const key of requiredKeys) {
      if (!(key in preset)) {
        throw new Error(`Preset missing key: ${key}`);
      }
    }

    const numericKeys = [
      'exposure_time_dome', 'exposure_time_bar',
      'gamma_dome', 'gamma_bar'
    ];
    for (const nk of numericKeys) {
      if (!isFinite(Number(preset[nk]))) {
        throw new Error(`Invalid numeric value for ${nk}`);
      }
    }

    const profilePath = String(preset.camera_profile_path || '').trim();
    if (!profilePath) {
      throw new Error('camera_profile_path is empty');
    }

    const exists = await this.verifyFileExists(profilePath);
    if (!exists) {
      throw new Error('Kamera Profil path not found');
    }

    // Apply values
    this.cameraSettings.ExposureTime_Dome = Number(preset.exposure_time_dome);
    this.cameraSettings.ExposureTime_Bar = Number(preset.exposure_time_bar);
    this.cameraSettings.Gamma_Dome = Number(preset.gamma_dome);
    this.cameraSettings.Gamma_Bar = Number(preset.gamma_bar);
    this.otherSettings.camera_settings_file = profilePath;

    this.settingsPresetName = fileName ? this.stripExtension(fileName) : '';
    this.otherSettings.settings_preset_name = this.settingsPresetName;
    this.applyOtherSetting('settings_preset_name');

    // Persist to backend
    const keysToApply = [
      'ExposureTime_Dome', 'ExposureTime_Bar',
      'Gamma_Dome', 'Gamma_Bar'
    ];
    keysToApply.forEach(k => this.applySetting(k));
    this.applyOtherSetting('camera_settings_file');

    // Load .pfs profile onto camera
    await this.loadCameraProfile(profilePath);
  }

  async loadCameraProfile(profilePath: string): Promise<void> {
    if (!profilePath) return;
    if (!this.isConnected) {
      return;
    }
    try {
      const resp = await this.http.post<any>(`${BASE_URL}/load-camera-profile`, { path: profilePath }).toPromise();
      if (resp?.error) {
        this.errorNotificationService.addError({
          code: resp.code || 'E1311',
          message: '',
          popupStyle: 'center'
        });
      } else {
        console.log('Camera profile loaded successfully:', profilePath);
      }
    } catch (err: any) {
      console.error('Failed to load camera profile:', err);
      this.errorNotificationService.addError({
        code: 'E1311',
        message: '',
        popupStyle: 'center'
      });
    }
  }

  verifyFileExists(path: string): Promise<boolean> {
    return this.http.post<{ exists: boolean }>(`${BASE_URL}/check-file-exists`, { path })
      .pipe(
        timeout(2000),
        catchError(() => of({ exists: false }))
      )
      .toPromise()
      .then(res => !!res?.exists);
  }
}
