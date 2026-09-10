import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { BehaviorSubject, Observable, finalize, timer, exhaustMap, catchError, EMPTY } from 'rxjs';
import { CaptureMetadata } from '../models/capture-metadata.models';
import { BASE_URL } from '../api-config';
import { HeightOffsetApplication } from '../models/filter-settings.models';
import { LightChannel } from '../models/light.models';
import { SharedService } from '../shared.service';
import { SettingsUpdatesService } from './settings-updates.service';

export interface FilterCaptureState { running: boolean; cancelling: boolean; }

export interface BgrCapturedImage {
  metadata?: CaptureMetadata;
  filter_name: string;
  suffix: 'b' | 'g' | 'r' | 'uv';
  filter_position: number;
  path: string;
  height_offset: HeightOffsetApplication;
  wavelength: LightChannel;
}

export interface BgrCaptureSeriesResponse {
  status: 'completed' | 'cancelled';
  series_index: number;
  saved_images: BgrCapturedImage[];
  camera_params?: Record<string, number> | null;
}

export interface BgrCaptureCancelResponse {
  status: 'cancellation_requested' | 'idle';
}

@Injectable({ providedIn: 'root' })
export class BgrCaptureService {
  private readonly stateSubject = new BehaviorSubject<FilterCaptureState>({ running: false, cancelling: false });
  readonly state$ = this.stateSubject.asObservable();
  private runGeneration = 0;

  constructor(
    private readonly http: HttpClient,
    private readonly shared: SharedService,
    private readonly settingsUpdates: SettingsUpdatesService,
  ) {}

  /** Own the request until backend completion, even if the scanner view is destroyed. */
  run(targetFolder: string, wavelengths: readonly LightChannel[]): void {
    if (this.stateSubject.value.running || this.shared.getMeasurementActive() || wavelengths.length === 0) return;
    this.runGeneration++;
    this.stateSubject.next({ running: true, cancelling: false });
    this.shared.setMeasurementActive(true);
    const captureId = crypto.randomUUID();
    const published = new Set<string>();
    const publish = (images: BgrCapturedImage[] = []) => images.forEach(image => {
      if (published.has(image.path)) return;
      published.add(image.path);
      this.shared.emitSavedImage({ path: image.path, tabletIndex: 0, ...(image.metadata ? { metadata: image.metadata } : {}) });
    });
    const progress = timer(500, 500).pipe(exhaustMap(() =>
      this.http.get<{ capture_id: string; saved_images: BgrCapturedImage[]; camera_params?: Record<string, number> | null }>(`${BASE_URL}/bgr-capture-series/status`)
        .pipe(catchError(() => EMPTY))
    )).subscribe(status => {
      if (status.capture_id !== captureId) return;
      publish(status.saved_images);
      if (status.camera_params) this.settingsUpdates.updateCameraSettings(status.camera_params);
    });
    this.start(targetFolder, wavelengths, captureId).pipe(finalize(() => {
      progress.unsubscribe();
      this.stateSubject.next({ running: false, cancelling: false });
      this.shared.setMeasurementActive(false);
    })).subscribe({
      next: response => {
        publish(response.saved_images);
        if (response.camera_params) this.settingsUpdates.updateCameraSettings(response.camera_params);
      },
      error: error => publish(error.error?.saved_images) // Keep successfully saved frames after a partial failure.
    });
  }

  requestCancel(): void {
    const state = this.stateSubject.value;
    const generation = this.runGeneration;
    if (!state.running || state.cancelling) return;
    this.stateSubject.next({ ...state, cancelling: true });
    this.cancel().subscribe({
      error: () => {
        if (generation === this.runGeneration && this.stateSubject.value.running) {
          this.stateSubject.next({ ...state, cancelling: false });
        }
      }
    });
  }

  start(targetFolder: string, wavelengths: readonly LightChannel[], captureId?: string): Observable<BgrCaptureSeriesResponse> {
    return this.http.post<BgrCaptureSeriesResponse>(`${BASE_URL}/bgr-capture-series`, {
      target_folder: targetFolder,
      wavelengths,
      ...(captureId ? { capture_id: captureId } : {})
    });
  }

  cancel(): Observable<BgrCaptureCancelResponse> {
    return this.http.post<BgrCaptureCancelResponse>(
      `${BASE_URL}/bgr-capture-series/cancel`,
      {}
    );
  }
}
