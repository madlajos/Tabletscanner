import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { BehaviorSubject, filter, tap } from 'rxjs';
import { BASE_URL } from '../api-config';
import { CameraCombinationSettings, CameraCombinationSettingsResponse } from '../models/camera-combination-settings.models';

@Injectable({ providedIn: 'root' })
export class CameraCombinationSettingsService {
  private readonly settingsSubject = new BehaviorSubject<CameraCombinationSettings | null>(null);
  readonly settings$ = this.settingsSubject.pipe(
    filter((settings): settings is CameraCombinationSettings => settings !== null)
  );

  constructor(private readonly http: HttpClient) {}

  get() {
    return this.http.get<CameraCombinationSettingsResponse>(`${BASE_URL}/settings/camera/combinations`).pipe(
      tap(response => this.settingsSubject.next(response.camera_combination_settings))
    );
  }

  update(settings: CameraCombinationSettings) {
    return this.http.put<CameraCombinationSettingsResponse>(`${BASE_URL}/settings/camera/combinations`, settings).pipe(
      tap(response => this.settingsSubject.next(response.camera_combination_settings))
    );
  }
}
