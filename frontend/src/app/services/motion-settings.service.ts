import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { BehaviorSubject, Observable, tap } from 'rxjs';
import { BASE_URL } from '../api-config';
import { AdvancedMotionSettings, MotionConnectionStatus } from '../models/motion.models';

@Injectable({ providedIn: 'root' })
export class MotionSettingsService {
  private readonly advancedSubject = new BehaviorSubject<AdvancedMotionSettings | null>(null);
  readonly advanced$ = this.advancedSubject.asObservable();

  constructor(private readonly http: HttpClient) {}

  getAdvanced(): Observable<{ advanced_motion_settings: AdvancedMotionSettings }> {
    return this.http.get<{ advanced_motion_settings: AdvancedMotionSettings }>(
      `${BASE_URL}/settings/motion/advanced`
    ).pipe(tap(response => this.advancedSubject.next(response.advanced_motion_settings)));
  }

  updateAdvanced(settings: AdvancedMotionSettings): Observable<{
    advanced_motion_settings: AdvancedMotionSettings;
    connection: MotionConnectionStatus;
  }> {
    return this.http.put<{
      advanced_motion_settings: AdvancedMotionSettings;
      connection: MotionConnectionStatus;
    }>(`${BASE_URL}/settings/motion/advanced`, settings).pipe(
      tap(response => this.advancedSubject.next(response.advanced_motion_settings))
    );
  }
}
