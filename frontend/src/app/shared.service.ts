import { Injectable } from '@angular/core';
import { BehaviorSubject, interval, of } from 'rxjs';
import { HttpClient } from '@angular/common/http';
import { map, switchMap, tap, catchError  } from 'rxjs/operators';

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
  barcode: string;
  operator: string;
  clogged: number;
  partiallyClogged: number;
  clean: number;
  result: string;
}


@Injectable({
  providedIn: 'root'
})
export class SharedService {
  private readonly BASE_URL = 'http://localhost:5000/api';

  private measurementResultsSubject = new BehaviorSubject<MeasurementResult[]>([
    { label: 'Eldugult', value: 0 },
    { label: 'Részleges', value: 0 },
    { label: 'Tiszta', value: 0 }
  ]);

  private measurementActiveSubject = new BehaviorSubject<boolean>(false);
  measurementActive$ = this.measurementActiveSubject.asObservable();

  setMeasurementActive(active: boolean): void {
    this.measurementActiveSubject.next(active);
  }

  private measurementHistorySubject = new BehaviorSubject<MeasurementRecord[]>([]);
  public measurementHistory$ = this.measurementHistorySubject.asObservable();

  public measurementResults$ = this.measurementResultsSubject.asObservable();

  private cameraConnectionStatus = new BehaviorSubject<{ main: boolean; side: boolean }>({
    main: false,
    side: false
  });
  cameraConnectionStatus$ = this.cameraConnectionStatus.asObservable();

  private cameraStreamStatus = new BehaviorSubject<{ main: boolean; side: boolean }>({
    main: false,
    side: false
  });
  cameraStreamStatus$ = this.cameraStreamStatus.asObservable();

  isMainConnected$ = this.cameraConnectionStatus.asObservable().pipe(map(status => status.main));
  isSideConnected$ = this.cameraConnectionStatus.asObservable().pipe(map(status => status.side));

  isMainStreaming$ = this.cameraStreamStatus.asObservable().pipe(map(status => status.main));
  isSideStreaming$ = this.cameraStreamStatus.asObservable().pipe(map(status => status.side));

  private saveDirectory: string = '';

  constructor(private http: HttpClient) {}

  addMeasurementResult(record: MeasurementRecord): void {
    const current = this.measurementHistorySubject.getValue();
    // append the new record and emit the updated list
    this.measurementHistorySubject.next([record, ...current]);
  }

  setSaveDirectory(directory: string): void {
    this.saveDirectory = directory;
    console.log(`Save directory set to: ${directory}`);
  }

  getSaveDirectory(): string {
    console.log(`Retrieving save directory: ${this.saveDirectory}`);
    return this.saveDirectory;
  }

  setCameraConnectionStatus(cameraType: 'main' | 'side', status: boolean): void {
    const currentStatus = this.cameraConnectionStatus.getValue();
    const updatedStatus = { ...currentStatus, [cameraType]: status };
    this.cameraConnectionStatus.next(updatedStatus);

    console.log(`Updated ${cameraType} camera connection status to: ${status}`);
  }

  setCameraStreamStatus(cameraType: 'main' | 'side', isStreaming: boolean): void {
    const currentStatus = this.cameraStreamStatus.getValue();
    this.cameraStreamStatus.next({ ...currentStatus, [cameraType]: isStreaming });
    console.log(`Updated ${cameraType} camera stream status to: ${isStreaming}`);
  }

  getCameraConnectionStatus(cameraType: 'main' | 'side'): boolean {
    return this.cameraConnectionStatus.value[cameraType];
  }

  getCameraStreamStatus(cameraType: 'main' | 'side'): boolean {
    return this.cameraStreamStatus.value[cameraType];
  }

  toggleStream(cameraType: 'main' | 'side'): void {
    const isStreaming = this.getCameraStreamStatus(cameraType);  // Make sure this is defined first
  
    console.log(`Toggling ${cameraType} stream. Current status: ${isStreaming}`);  // Now it works
  
    this.setCameraStreamStatus(cameraType, !isStreaming);
  
    if (isStreaming) {
      this.stopStream(cameraType);
    } else {
      this.startStream(cameraType);
    }
  }
  
  startStream(cameraType: 'main' | 'side'): void {
    if (this.getCameraStreamStatus(cameraType)) {
      console.warn(`${cameraType} stream is already running. Preventing duplicate start.`);
      return;  //Prevent multiple start requests
    }
  
    console.log(`Starting ${cameraType} stream...`);
  
    this.http.get(`${this.BASE_URL}/start-video-stream?type=${cameraType}`).subscribe(
      () => {
        console.log(`${cameraType} camera stream started.`);
        this.setCameraStreamStatus(cameraType, true);
      },
      error => {
        console.error(`Failed to start ${cameraType} camera stream:`, error);
        this.setCameraStreamStatus(cameraType, false);
      }
    );
  }
  
  
  stopStream(cameraType: 'main' | 'side'): void {
    console.log(`Stopping ${cameraType} stream...`);
    this.http.post(`${this.BASE_URL}/stop-video-stream?type=${cameraType}`, {}).subscribe(
      () => {
        console.log(`${cameraType} camera stream stopped.`);
        this.setCameraStreamStatus(cameraType, false);
      },
      error => {
        console.error(`Failed to stop ${cameraType} camera stream:`, error);
        this.setCameraStreamStatus(cameraType, true);
      }
    );
  }

  toggleConnection(cameraType: 'main' | 'side'): void {
    const isConnected = this.getCameraConnectionStatus(cameraType);

    if (isConnected) {
      this.disconnectCamera(cameraType);
    } else {
      this.connectCamera(cameraType);
    }
  }

  connectCamera(cameraType: 'main' | 'side'): void {
    this.http.post(`${this.BASE_URL}/connect-camera?type=${cameraType}`, {}).subscribe(
      () => {
        this.setCameraConnectionStatus(cameraType, true);
        console.log(`Connected ${cameraType} camera.`);
      },
      error => {
        console.error(`Failed to connect ${cameraType} camera:`, error);
      }
    );
  }

  disconnectCamera(cameraType: 'main' | 'side'): void {
    // Stop stream before disconnecting
    this.stopStream(cameraType);
  
    this.http.post(`${this.BASE_URL}/disconnect-camera?type=${cameraType}`, {}).subscribe(
      () => {
        this.setCameraConnectionStatus(cameraType, false);
        this.setCameraStreamStatus(cameraType, false);  // Reset stream status
        console.log(`Disconnected ${cameraType} camera.`);
      },
      error => {
        console.error(`Failed to disconnect ${cameraType} camera:`, error);
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

  analyzeCenterCircle(): void {
    console.log("Analyzing Center Circle via SharedService...");
    this.http.post(`${this.BASE_URL}/analyze_center_circle`, {}).pipe(
      tap(response => console.log("analyze_center_circle Response:", response)),
      switchMap(() => {
        console.log("Calling update_results for center circle...");
        return this.http.post(`${this.BASE_URL}/update_results`, { mode: "center_circle" }, { headers: { 'Content-Type': 'application/json' } });
      }),
      tap((response: any) => {
        console.log("Update results response for center circle:", response);
        this.updateResults(response);
      })
    ).subscribe({
      error: err => console.error("Error in analyzeCenterCircle:", err)
    });
  }

  analyzeInnerSlice(): void {
    console.log("Analyzing Inner Slice via SharedService...");
    this.http.post(`${this.BASE_URL}/analyze_center_slice`, {}).pipe(
      switchMap(() => this.http.post(`${this.BASE_URL}/update_results`, { mode: "center_slice" })),
      tap((response: any) => {
        console.log("Update results response for center slice:", response);
        this.updateResults(response);
      })
    ).subscribe({
      error: err => console.error("Error in analyzeInnerSlice:", err)
    });
  }

  analyzeOuterSlice(): void {
    console.log("Analyzing Outer Slice via SharedService...");
    this.http.post(`${this.BASE_URL}/analyze_outer_slice`, {}).pipe(
      switchMap(() => this.http.post(`${this.BASE_URL}/update_results`, { mode: "outer_slice" })),
      tap((response: any) => {
        console.log("Update results response for outer slice:", response);
        this.updateResults(response);
      })
    ).subscribe({
      error: err => console.error("Error in analyzeOuterSlice:", err)
    });
  }
}
