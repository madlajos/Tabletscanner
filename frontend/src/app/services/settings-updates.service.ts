import { Injectable } from '@angular/core';
import { BehaviorSubject } from 'rxjs';

export interface SizeLimits {
  class1: number;
  class2: number;
  ng_limit: number;
}

export interface SaveSettings {
  save_csv: boolean;
  save_images: boolean;
  csv_dir: string;
}

export interface CameraSettings {
  Width: number;
  Height: number;
  OffsetX: number;
  OffsetY: number;
  ExposureTime: number;
  Gain: number;
  Gamma: number;
  FrameRate: number;
  [key: string]: number;
}

const DEFAULT_CAMERA_SETTINGS: CameraSettings = {
  Width: 4200,
  Height: 2160,
  OffsetX: 0,
  OffsetY: 0,
  ExposureTime: 20000,
  Gain: 0,
  Gamma: 1,
  FrameRate: 20
};


@Injectable({
  providedIn: 'root'
})
export class SettingsUpdatesService {
  // Initialize with default/fallback values. These may be overwritten by the backend.
  private sizeLimitsSubject = new BehaviorSubject<SizeLimits>({
    class1: 5,
    class2: 95,
    ng_limit: 15
  });

  private saveSettingsSubject = new BehaviorSubject<SaveSettings>({
    save_csv: false,
    save_images: false,
    csv_dir: ""
  });

  private mainCameraSettingsSubject = new BehaviorSubject<CameraSettings>(DEFAULT_CAMERA_SETTINGS);
  private sideCameraSettingsSubject = new BehaviorSubject<CameraSettings>(DEFAULT_CAMERA_SETTINGS);
  
  // Expose the observable for components to subscribe to.
  sizeLimits$ = this.sizeLimitsSubject.asObservable();
  saveSettings$ = this.saveSettingsSubject.asObservable();
  mainCameraSettings$ = this.mainCameraSettingsSubject.asObservable();
  sideCameraSettings$ = this.sideCameraSettingsSubject.asObservable();

  // Call this method to update the current size limits.
  updateSizeLimits(newLimits: SizeLimits): void {
    this.sizeLimitsSubject.next(newLimits);
  }

  updateSaveSettings(newSettings: SaveSettings): void {
    this.saveSettingsSubject.next(newSettings);
  }

  updateMainCameraSettings(newSettings: CameraSettings): void {
    this.mainCameraSettingsSubject.next(newSettings);
  }

  updateSideCameraSettings(newSettings: CameraSettings): void {
    this.sideCameraSettingsSubject.next(newSettings);
  }
}