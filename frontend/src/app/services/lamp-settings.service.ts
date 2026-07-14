import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { BASE_URL } from '../api-config';
import { AdvancedLampSettings, LampSettings } from '../models/light.models';

@Injectable({ providedIn: 'root' })
export class LampSettingsService {
  constructor(private readonly http: HttpClient) {}

  get(): Observable<{ lamp_settings: LampSettings }> {
    return this.http.get<{ lamp_settings: LampSettings }>(`${BASE_URL}/settings/lamp`);
  }

  update(lampSettings: LampSettings): Observable<{ lamp_settings: LampSettings }> {
    return this.http.put<{ lamp_settings: LampSettings }>(`${BASE_URL}/settings/lamp`, lampSettings);
  }

  getAdvanced(): Observable<{ advanced_lamp_settings: Partial<AdvancedLampSettings> }> {
    return this.http.get<{ advanced_lamp_settings: Partial<AdvancedLampSettings> }>(`${BASE_URL}/settings/lamp/advanced`);
  }

  updateAdvanced(settings: AdvancedLampSettings): Observable<{ advanced_lamp_settings: AdvancedLampSettings }> {
    return this.http.put<{ advanced_lamp_settings: AdvancedLampSettings }>(`${BASE_URL}/settings/lamp/advanced`, settings);
  }
}
