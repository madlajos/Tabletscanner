import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { BehaviorSubject, tap } from 'rxjs';
import { BASE_URL } from '../api-config';
import { FilterSettings } from '../models/filter-settings.models';

@Injectable({ providedIn: 'root' })
export class FilterSettingsService {
  private readonly settingsSubject = new BehaviorSubject<FilterSettings | null>(null);
  readonly settings$ = this.settingsSubject.asObservable();

  constructor(private readonly http: HttpClient) {}

  get(): Observable<{ filter_settings: FilterSettings }> {
    return this.http.get<{ filter_settings: FilterSettings }>(`${BASE_URL}/settings/filter`).pipe(
      tap(response => this.settingsSubject.next(response.filter_settings))
    );
  }

  update(settings: FilterSettings): Observable<{ filter_settings: FilterSettings }> {
    return this.http.put<{ filter_settings: FilterSettings }>(`${BASE_URL}/settings/filter`, settings).pipe(
      tap(response => this.settingsSubject.next(response.filter_settings))
    );
  }
}
