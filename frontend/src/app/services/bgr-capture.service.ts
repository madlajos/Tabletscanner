import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';
import { BASE_URL } from '../api-config';
import { HeightOffsetApplication } from '../models/filter-settings.models';

export interface BgrCapturedImage {
  filter_name: string;
  suffix: 'b' | 'g' | 'r';
  filter_position: number;
  path: string;
  height_offset: HeightOffsetApplication;
}

export interface BgrCaptureSeriesResponse {
  status: 'completed' | 'cancelled';
  series_index: number;
  saved_images: BgrCapturedImage[];
}

export interface BgrCaptureCancelResponse {
  status: 'cancellation_requested' | 'idle';
}

@Injectable({ providedIn: 'root' })
export class BgrCaptureService {
  constructor(private readonly http: HttpClient) {}

  start(targetFolder: string): Observable<BgrCaptureSeriesResponse> {
    return this.http.post<BgrCaptureSeriesResponse>(`${BASE_URL}/bgr-capture-series`, {
      target_folder: targetFolder
    });
  }

  cancel(): Observable<BgrCaptureCancelResponse> {
    return this.http.post<BgrCaptureCancelResponse>(
      `${BASE_URL}/bgr-capture-series/cancel`,
      {}
    );
  }
}
