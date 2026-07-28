import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';
import { BASE_URL } from '../api-config';
import {
  CameraImageSettings,
  CameraImageSettingsResponse
} from '../models/camera-image-settings.models';

@Injectable({ providedIn: 'root' })
export class CameraImageSettingsService {
  constructor(private readonly http: HttpClient) {}

  get(): Observable<CameraImageSettingsResponse> {
    return this.http.get<CameraImageSettingsResponse>(`${BASE_URL}/settings/camera/image-size`);
  }

  update(settings: CameraImageSettings): Observable<CameraImageSettingsResponse> {
    return this.http.put<CameraImageSettingsResponse>(`${BASE_URL}/settings/camera/image-size`, settings);
  }

  center(axis: 'x' | 'y'): Observable<CameraImageSettingsResponse> {
    return this.http.post<CameraImageSettingsResponse>(
      `${BASE_URL}/settings/camera/image-size/center`,
      { axis }
    );
  }
}
