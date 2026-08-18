import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { BASE_URL } from '../api-config';
import { AutofocusSettings } from '../models/autofocus-settings.models';

@Injectable({ providedIn: 'root' })
export class AutofocusSettingsService {
  constructor(private readonly http: HttpClient) {}

  get(): Observable<{ autofocus_settings: AutofocusSettings }> {
    return this.http.get<{ autofocus_settings: AutofocusSettings }>(
      `${BASE_URL}/settings/autofocus`
    );
  }

  update(settings: AutofocusSettings): Observable<{ autofocus_settings: AutofocusSettings }> {
    return this.http.put<{ autofocus_settings: AutofocusSettings }>(
      `${BASE_URL}/settings/autofocus`,
      settings
    );
  }
}
