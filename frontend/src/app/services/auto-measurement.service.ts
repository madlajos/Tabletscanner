import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface AutoMeasurementRequest {
  indices: number[];      // NEW: all selected tablet IDs
  grid_size: number;
  autofocus: boolean;
  lamp_top: boolean;
  lamp_side: boolean;
}
export interface AutoMeasurementResponse {
  status: 'success' | 'error';
  message?: string;
  positions?: { index: number; x: number; y: number }[];
  move_response?: any;
}

@Injectable({
  providedIn: 'root'
})
export class AutoMeasurementService {
  private readonly baseUrl = '/api'; // adjust if you use env.apiUrl

  constructor(private http: HttpClient) {}

  startMeasurement(req: AutoMeasurementRequest): Observable<AutoMeasurementResponse> {
    return this.http.post<AutoMeasurementResponse>(
      `${this.baseUrl}/auto_measurement`,
      req
    );
  }
}
