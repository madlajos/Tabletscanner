import { Injectable } from '@angular/core';
import { BehaviorSubject, Subject, interval, of } from 'rxjs';
import { HttpClient } from '@angular/common/http';
import { map, switchMap, tap, catchError  } from 'rxjs/operators';
import { BASE_URL } from './api-config';

//Older interface to store MeasurementResults, which are displayed on the control panel
export interface MeasurementResult {
  label: string;
  value: number;
}

//New interface to contain measurement records to display in the table and save to DB
export interface MeasurementRecord {
  date: string;
  time: string;
  id: number | string;
  operator: string;
  clogged: number;
  partiallyClogged: number;
  clean: number;
  result: string;
}

// Interface for saved images from auto-measurement
export interface SavedImageInfo {
  path: string;
  tabletIndex: number;
  lightType: 'dome' | 'bar';
}


@Injectable({
  providedIn: 'root'
})
export class SharedService {
  private measurementResultsSubject = new BehaviorSubject<MeasurementResult[]>([
    { label: 'Eldugult', value: 0 },
    { label: 'Részleges', value: 0 },
    { label: 'Tiszta', value: 0 }
  ]);

  private measurementActiveSubject = new BehaviorSubject<boolean>(false);
  measurementActive$ = this.measurementActiveSubject.asObservable();

  // Subject for emitting newly saved images (auto-measurement adds images here)
  private newSavedImageSubject = new Subject<SavedImageInfo>();
  newSavedImage$ = this.newSavedImageSubject.asObservable();

  setMeasurementActive(active: boolean): void {
    this.measurementActiveSubject.next(active);
  }

  getMeasurementActive(): boolean {
    return this.measurementActiveSubject.value;
  }

  // Emit a newly saved image to notify gallery components
  emitSavedImage(imageInfo: SavedImageInfo): void {
    this.newSavedImageSubject.next(imageInfo);
  }

  private measurementHistorySubject = new BehaviorSubject<MeasurementRecord[]>([]);
  public measurementHistory$ = this.measurementHistorySubject.asObservable();

  public measurementResults$ = this.measurementResultsSubject.asObservable();

  private cameraConnectionStatus = new BehaviorSubject<boolean>(false);

  cameraConnectionStatus$ = this.cameraConnectionStatus.asObservable();

  private cameraStreamStatus = new BehaviorSubject<boolean>(false);
  cameraStreamStatus$ = this.cameraStreamStatus.asObservable();

  private motionPlatformConnectionStatus = new BehaviorSubject<boolean>(false);
  motionPlatformConnectionStatus$ = this.motionPlatformConnectionStatus.asObservable();

  private lightSettingsSubject = new BehaviorSubject<'dome' | 'bar' | null>(null);
  lightSettings$ = this.lightSettingsSubject.asObservable();
  
  // Track which light is currently ON (null means both off)
  private activeLightSubject = new BehaviorSubject<'dome' | 'bar' | null>(null);
  activeLight$ = this.activeLightSubject.asObservable();
  
  // Track light state (on/off)
  private lightsOffSubject = new Subject<void>();
  lightsOff$ = this.lightsOffSubject.asObservable();

  // Signal to invalidate autofocus state (e.g., when another component moves the platform)
  private autofocusInvalidateSubject = new Subject<void>();
  autofocusInvalidate$ = this.autofocusInvalidateSubject.asObservable();

  invalidateAutofocus(): void {
    this.autofocusInvalidateSubject.next();
  }

  // Autofocus error from manual autofocus (motion-control → auto-measurement)
  private autofocusErrorSubject = new Subject<string | null>();
  autofocusError$ = this.autofocusErrorSubject.asObservable();

  setAutofocusError(message: string | null): void {
    this.autofocusErrorSubject.next(message);
  }
  
  private motionHomingStatus = new BehaviorSubject<boolean>(false);
  motionHomingStatus$ = this.motionHomingStatus.asObservable();
  private motionPositionSubject = new BehaviorSubject<{ x: number | null; y: number | null; z: number | null } | null>(null);
  motionPosition$ = this.motionPositionSubject.asObservable();

  // Track whether all axes are homed (xHomed && yHomed && zHomed)
  private motionHomedSubject = new BehaviorSubject<boolean>(false);
  motionHomed$ = this.motionHomedSubject.asObservable();

  setMotionHomed(homed: boolean): void {
    this.motionHomedSubject.next(homed);
  }

  getMotionHomed(): boolean {
    return this.motionHomedSubject.value;
  }

  setMotionHoming(isHoming: boolean): void {
    this.motionHomingStatus.next(isHoming);
  }

  getMotionHoming(): boolean {
    return this.motionHomingStatus.value;
  }

  setMotionPosition(position: { x: number | null; y: number | null; z: number | null } | null): void {
    this.motionPositionSubject.next(position);
  }
  isCameraConnected$ = this.cameraConnectionStatus.asObservable().pipe();
  isCameraStreaming$ = this.cameraStreamStatus.asObservable().pipe();

  private saveDirectory: string = '';
  private saveDirectorySubject = new BehaviorSubject<string>('');
  saveDirectory$ = this.saveDirectorySubject.asObservable();

  constructor(private http: HttpClient) {}

  addMeasurementResult(record: MeasurementRecord): void {
    const current = this.measurementHistorySubject.getValue();
    // append the new record and emit the updated list
    this.measurementHistorySubject.next([record, ...current]);
  }

  setSaveDirectory(directory: string): void {
    this.saveDirectory = directory;
    this.saveDirectorySubject.next(directory || '');
    console.log(`Save directory set to: ${directory}`);
  }

  getSaveDirectory(): string {
    console.log(`Retrieving save directory: ${this.saveDirectory}`);
    return this.saveDirectory;
  }

  setCameraConnectionStatus(status: boolean): void {
    this.cameraConnectionStatus.next(!!status);
    console.log(`Updated camera connection status to: ${status}`);
  }

  setCameraStreamStatus(isStreaming: boolean): void {
    this.cameraStreamStatus.next(!!isStreaming);
    console.log(`Updated camera stream status to: ${isStreaming}`);
  }

  getCameraConnectionStatus(): boolean {
    return this.cameraConnectionStatus.value;
  }

  getCameraStreamStatus(): boolean {
    return this.cameraStreamStatus.value;
  }

  setMotionPlatformConnectionStatus(status: boolean): void {
    this.motionPlatformConnectionStatus.next(!!status);
    console.log(`Updated motion platform connection status to: ${status}`);
  }

  getMotionPlatformConnectionStatus(): boolean {
    return this.motionPlatformConnectionStatus.value;
  }

  toggleStream(): void {
    const isStreaming = this.getCameraStreamStatus();
    
    console.log(`Toggling camera stream. Current status: ${isStreaming}`);
    
    this.setCameraStreamStatus(!isStreaming);
    
    if (isStreaming) {
      this.stopStream();
    } else {
      this.startStream();
    }
  }

  applyCameraSettingsForLight(light: 'dome' | 'bar'): void {
    this.lightSettingsSubject.next(light);
  }

  setActiveLight(light: 'dome' | 'bar' | null): void {
    this.activeLightSubject.next(light);
  }

  getActiveLight(): 'dome' | 'bar' | null {
    return this.activeLightSubject.value;
  }

  emitLightsOff(): void {
    this.lightsOffSubject.next();
    this.activeLightSubject.next(null);
  }
  
  startStream(): void {
    if (this.getCameraStreamStatus()) {
      console.warn(`Camera stream is already running. Preventing duplicate start.`);
      return;  //Prevent multiple start requests
    }
  
    console.log(`Starting camera stream...`);
  
    this.http.get(`${BASE_URL}/start-video-stream`).subscribe(
      () => {
        console.log(`Camera stream started.`);
        this.setCameraStreamStatus(true);
      },
      error => {
        console.error(`Failed to start camera stream:`, error);
        this.setCameraStreamStatus(false);
      }
    );
  }
  
  
  stopStream(): void {
    console.log(`Stopping camera stream...`);
    this.http.post(`${BASE_URL}/stop-video-stream`, {}).subscribe(
      () => {
        console.log(`Camera camera stream stopped.`);
        this.setCameraStreamStatus(false);
      },
      error => {
        console.error(`Failed to stop camera stream:`, error);
        this.setCameraStreamStatus(true);
      }
    );
  }

  toggleConnection(): void {
    const isConnected = this.getCameraConnectionStatus();

    if (isConnected) {
      this.disconnectCamera();
    } else {
      this.connectCamera();
    }
  }

  connectCamera(): void {
    this.http.post(`${BASE_URL}/connect-camera`, {}).subscribe(
      () => {
        this.setCameraConnectionStatus(true);
        console.log(`Camera connected.`);
      },
      error => {
        console.error(`Failed to connect camera:`, error);
      }
    );
  }

  disconnectCamera(): void {
    // Stop stream before disconnecting
    this.stopStream();
  
    this.http.post(`${BASE_URL}/disconnect-camera`, {}).subscribe(
      () => {
        this.setCameraConnectionStatus(false);
        this.setCameraStreamStatus(false);  // Reset stream status
        console.log(`Disconnected camera.`);
      },
      error => {
        console.error(`Failed to disconnect camera:`, error);
      }
    );
  }

  updateResults(response: any): void {
    if (response?.result_counts) {
      const currentResults = this.measurementResultsSubject.getValue();
      let newResults: MeasurementResult[];
      if (!currentResults || currentResults.length !== response.result_counts.length) {
        newResults = response.result_counts.map((count: number, index: number) => ({
          label: `Result ${index + 1}`,
          value: count
        }));
      } else {
        newResults = currentResults.map((result, index) => ({
          label: result.label,
          value: response.result_counts[index]
        }));
      }
      this.measurementResultsSubject.next(newResults);
    }
  }
}