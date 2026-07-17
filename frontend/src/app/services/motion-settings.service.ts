import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';
import { BASE_URL } from '../api-config';
import { AdvancedMotionSettings, MotionConnectionStatus } from '../models/motion.models';

@Injectable({ providedIn: 'root' })
export class MotionSettingsService {
  constructor(private readonly http: HttpClient) {}

  getAdvanced(): Observable<{ advanced_motion_settings: AdvancedMotionSettings }> {
    return this.http.get<{ advanced_motion_settings: AdvancedMotionSettings }>(
      `${BASE_URL}/settings/motion/advanced`
    );
  }

  updateAdvanced(settings: AdvancedMotionSettings): Observable<{
    advanced_motion_settings: AdvancedMotionSettings;
    connection: MotionConnectionStatus;
  }> {
    return this.http.put<{
      advanced_motion_settings: AdvancedMotionSettings;
      connection: MotionConnectionStatus;
    }>(`${BASE_URL}/settings/motion/advanced`, settings);
  }
}
