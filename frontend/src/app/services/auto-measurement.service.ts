import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { BASE_URL } from '../api-config';

export interface AutoMeasurementSettings {
  save_location: string;
  first_tablet_x: number;
  first_tablet_y: number;
  first_tablet_z: number;
  tablet_spacing: number;
}

// Interface for step-by-step measurement
export interface TabletStepRequest {
  tablet_index: number;
  x: number;
  y: number;
  z?: number;  // Optional: Z is controlled by autofocus or manual setting during automated measurement
  measurement_folder: string;
  measurement_name: string;
  autofocus: boolean;
  lamp_top: boolean;
  lamp_side: boolean;
  is_first_tablet?: boolean;
}

export interface TabletStepResponse {
  status: 'success' | 'error';
  message?: string;
  tablet_index?: number;
  saved_images?: string[];
  af_error_code?: string;
  af_error_message?: string;
}

@Injectable({
  providedIn: 'root'
})
export class AutoMeasurementService {
  constructor(private http: HttpClient) {}

  /**
   * Process a single tablet measurement step.
   * Called repeatedly by the frontend for each tablet in the queue.
   */
  measureSingleTablet(req: TabletStepRequest): Observable<TabletStepResponse> {
    return this.http.post<TabletStepResponse>(
      `${BASE_URL}/auto_measurement/step`,
      req
    );
  }

  getSettings(): Observable<{ auto_measurement_settings: AutoMeasurementSettings }> {
    return this.http.get<{ auto_measurement_settings: AutoMeasurementSettings }>(
      `${BASE_URL}/get-other-settings?category=auto_measurement_settings`
    );
  }

  updateSettings(setting_name: string, setting_value: any): Observable<any> {
    return this.http.post(`${BASE_URL}/update-other-settings`, {
      category: 'auto_measurement_settings',
      setting_name,
      setting_value
    });
  }

  selectFolder(): Observable<{ folder: string }> {
    return this.http.get<{ folder: string }>(`${BASE_URL}/select-folder`);
  }

  /**
   * Attempt to reconnect to the motion platform.
   * Returns an observable that succeeds if connected, errors if not.
   */
  reconnectMotionPlatform(): Observable<{ message: string; port?: string }> {
    return this.http.post<{ message: string; port?: string }>(
      `${BASE_URL}/connect-to-motionplatform`,
      {}
    );
  }

  /**
   * Attempt to reconnect to the camera.
   * Returns an observable that succeeds if connected, errors if not.
   */
  reconnectCamera(): Observable<any> {
    return this.http.post(`${BASE_URL}/connect-camera`, {});
  }
}
