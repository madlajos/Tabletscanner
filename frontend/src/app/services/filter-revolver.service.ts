import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';
import { BASE_URL } from '../api-config';
import {
  FilterRevolverDirection,
  FilterRevolverMoveResponse,
  FilterRevolverStatus
} from '../models/filter-settings.models';

@Injectable({ providedIn: 'root' })
export class FilterRevolverService {
  constructor(private readonly http: HttpClient) {}

  getStatus(): Observable<FilterRevolverStatus> {
    return this.http.get<FilterRevolverStatus>(`${BASE_URL}/filter-revolver/status`);
  }

  rotate(direction: FilterRevolverDirection): Observable<FilterRevolverStatus> {
    return this.http.post<FilterRevolverStatus>(
      `${BASE_URL}/filter-revolver/rotate`,
      { direction }
    );
  }

  select(position: number): Observable<FilterRevolverMoveResponse> {
    return this.http.post<FilterRevolverMoveResponse>(
      `${BASE_URL}/filter-revolver/select`,
      { position }
    );
  }
}
