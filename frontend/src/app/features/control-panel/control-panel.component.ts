import { CommonModule } from '@angular/common';
import { Component, OnInit, OnDestroy, ViewChild, ChangeDetectorRef } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { TurntableControlComponent } from '../turntable-control/turntable-control.component';
import { Observable, Subject, throwError, map, of, delay } from 'rxjs';
import { catchError, switchMap, tap, takeUntil, finalize } from 'rxjs/operators';
import { FormsModule } from '@angular/forms';
import { MatIconModule } from '@angular/material/icon';
import { SharedService, MeasurementResult } from '../../shared.service';
import { MeasurementResultsPopupComponent } from '../../components/measurement-results-popup/measurement-results-popup.component';
import { SettingsUpdatesService } from '../../services/settings-updates.service';


declare global {
  interface Window {
    electronAPI?: {
      selectFolder: () => Promise<string>;
    };
  }
}


@Component({
  standalone: true,
  selector: 'app-control-panel',
  templateUrl: './control-panel.component.html',
  styleUrls: ['./control-panel.component.css'],
  imports: [CommonModule, FormsModule, MatIconModule, TurntableControlComponent, MeasurementResultsPopupComponent]
})
export class ControlPanelComponent implements OnInit, OnDestroy {
  private readonly BASE_URL = 'http://localhost:5000/api';

  relayState: boolean = false;
  nozzleId: string = "";
  operatorId: string = "";
  ng_limit: number = 0;
  save_csv: boolean = false;
  save_images: boolean = false;
  csv_dir: string = "";

  isConnected: boolean = false;
  
  measurementValidationTriggered: boolean = false;

  measurementActive: boolean = false;
  isResultsPopupVisible: boolean = false;

  currentMeasurement: number = 0;
  totalMeasurements: number = 18;
  turntablePosition: number | string = '?';
  
  private measurementStop$ = new Subject<void>();

  private lookupCancel$ = new Subject<void>();


  @ViewChild(TurntableControlComponent) turntableControl!: TurntableControlComponent;

  get progressPercentage(): number {
    return (this.currentMeasurement / this.totalMeasurements) * 100;
  }

  results: { label: string, value: number }[] = [
    { label: 'Eldugult', value: 0 },
    { label: 'Részleges', value: 0 },
    { label: 'Tiszta', value: 0 }
  ];

  constructor(
    private http: HttpClient,
    private cdr: ChangeDetectorRef,
    private sharedService: SharedService,
    private settingsUpdatesService: SettingsUpdatesService,
  ) { }

  ngOnInit(): void {
    this.sharedService.cameraConnectionStatus$.subscribe(status => {
      this.isConnected = status;
    });

    // Load initial size limits from backend.
    this.http.get<{ size_limits: { [key: string]: number } }>(`${this.BASE_URL}/get-other-settings?category=size_limits`)
      .subscribe({
        next: response => {
          if (response && response.size_limits) {
            // Access ng_limit with bracket notation.
            this.ng_limit = response.size_limits['ng_limit'];
            console.log("ng_limit loaded initially:", this.ng_limit);
          }
        },
        error: error => {
          console.error("Error loading settings:", error);
        }
      });

    // Subscribe to size limits updates via the shared service.
    this.settingsUpdatesService.sizeLimits$.subscribe(limits => {
      // Use bracket notation to access ng_limit.
      this.ng_limit = limits['ng_limit'];
      console.log("ng_limit updated via shared service:", this.ng_limit);
      this.cdr.detectChanges();
    });


    // Load initial save settings
    this.http.get<{ save_settings: { [key: string]: boolean } }>(`${this.BASE_URL}/get-other-settings?category=save_settings`)
      .subscribe({
        next: response => {
          if (response && response.save_settings) {
            // Access save_csv and save_images with bracket notation.
            this.save_csv = response.save_settings['save_csv'];
            this.save_images = response.save_settings['save_images'];
            console.log("Save settings loaded:", this.save_csv, this.save_images);
          }
        },
        error: error => {
          console.error("Error loading settings:", error);
        }
      });

    // Subscribe to size limits updates via the shared service.
    this.settingsUpdatesService.saveSettings$.subscribe(settings => {
      this.save_csv = settings['save_csv'];
      this.save_images = settings['save_images'];
      this.csv_dir = settings['csv_dir'];
      console.log("saveSettings updated via shared service:", this.save_csv, this.save_images);
      this.cdr.detectChanges();
    });
  }

  ngOnDestroy(): void {
    this.measurementStop$.next();
    this.measurementStop$.complete();
  }

  ngAfterViewInit(): void {
    if (!this.turntableControl) {
      console.error("TurntableControlComponent is not available!");
    }
  }

  showResultsPopup(): void {
    this.isResultsPopupVisible = true;
  }

  get resultsValues(): number[] {
    return this.results.map(r => r.value);
  }

  canStartMeasurement(): boolean {
    if (!this.isConnected) {
      console.error("Cannot start measurement: Main camera is not connected.");
      return false;
    }
    if (!this.turntableControl) {
      console.error("Cannot start measurement: Turntable controller is not available.");
      return false;
    }
    if (!this.operatorId) {
      console.error("Cannot start measurement: Operator ID is missing.");
      return false;
    }
    return true;
  }
  
  // Function to save raw image using the selected folder.
  async saveRawImage(): Promise<void> {
    // 1) ask Electron to open a folder‐picker
    const folder = await (window as any).electronAPI.selectFolder();
    if (!folder) {
      console.log('User cancelled folder selection');
      return;
    }

    // 2) POST it to the Flask endpoint
    this.http.post(
      `${this.BASE_URL}/save_raw_image`,
      { target_folder: folder }    // <-- send the path here
    ).subscribe({
      next: (response: any) => {
        console.log('Raw images saved successfully:', response);
      },
      error: (error: any) => {
        console.error('Error saving raw images:', error);
      }
    });
  }

  // Rotate Up/Down Functions (unchanged)
  moveTurntable(degrees: number): void {
    this.http.post(`${this.BASE_URL}/move_turntable_relative`, { degrees }).subscribe(
      () => console.log(`Turntable moved ${degrees} degrees`),
      error => console.error(`Error moving turntable by ${degrees}°:`, error)
    );
  }

  
  homeTurntable(): void {
    if (this.turntableControl) {
      this.turntableControl.homeTurntable();
      this.turntablePosition = this.turntableControl.turntablePosition; // Fetch position
    } else {
      console.error("TurntableControlComponent is not available.");
    }
  }

  toggleMeasurement(): void {
    if (!this.measurementActive) {
      if (!this.canStartMeasurement()) {
        //Check if conditions are met to start measurement
        this.measurementValidationTriggered = true;
        this.cdr.detectChanges();
        return;
      }
      // If validations pass, reset the flag.
      this.measurementValidationTriggered = false;
      this.measurementActive = true;
      this.sharedService.setMeasurementActive(true);
      console.log("Measurement cycle started.");
      this.startMeasurement();
    } else { // Stop measurement.
      this.measurementActive = false;
      this.sharedService.setMeasurementActive(false);
      console.log("Measurement cycle stopping...");
      this.stopMeasurement();
    }
  }

  stopMeasurement(): void {
    this.measurementActive = false;
    this.currentMeasurement = 0;
    this.lookupCancel$.next();
    this.lookupCancel$ = new Subject<void>();

    // Reset results while keeping original labels.
    this.results = this.results.map(res => ({ ...res, value: 0 }));
  
    
    console.log("Results cleared, progress bar reset");
  
    // Emit a value to cancel the measurement chain.
    this.measurementStop$.next();
  
    this.http.post(`${this.BASE_URL}/reset_results`, {}).subscribe({
      next: (response: any) => {
        console.log("Backend results reset:", response);
        if (response?.result_counts && this.results.length === response.result_counts.length) {
          this.results.forEach((result, index) => {
            result.value = response.result_counts[index];
          });
        }
      },
      error: (error) => {
        console.error("Failed to reset backend results:", error);
      }
    });
  }

  private prepareAnnotatedFolder(): Observable<any> {
    const spinneretId = this.nozzleId;
    return this.http.post(`${this.BASE_URL}/start-annotated-save`, {
      spinneret_id: spinneretId
    });
  }

  startMeasurement(): void {
    console.log("Starting measurement cycle…");
  
    // 1) Reset backend results immediately
    this.resetBackendResults();
  
    // 2) Build an Observable that, if saving images, first calls start-annotated-save,
    //    otherwise simply emits `null`.
    const prep$ = this.save_images
      ? this.prepareAnnotatedFolder().pipe(
          tap(() => console.log("Annotated‐images folder ready")),
        )
      : of(null);
  
    // 3) Once that prep step completes (or immediately if save_images===false),
    //    kick off the real measurement chain
    prep$.pipe(
      switchMap(() => {
        // This is exactly your “kickoff” logic:
        // — toggle relay if needed, then fire executeCycle('full')
        if (!this.relayState) {
          // wait for relay to change
          return of(null).pipe(
            delay(500),
            switchMap(() => this.executeCycle("full"))
          );
        } else {
          return this.executeCycle("full");
        }
      })
    ).subscribe({
      next: () => {/* nothing */},
      error: err => console.error("Cycle startup error:", err)
    });
  }
  

  private executeCycle(cycleMode: 'full' | 'slices'): Observable<any> {
    let init$: Observable<any>;
    
    if (cycleMode === 'full') {
      init$ = this.http.post(`${this.BASE_URL}/home_turntable_with_image`, {}).pipe(
        tap(() => console.log("Turntable homed successfully")),
        tap(() => this.currentMeasurement++)
      );
    } else {
      init$ = this.http.post(`${this.BASE_URL}/move_turntable_relative`, { degrees: 20 }).pipe(
        tap(() => console.log("Turntable rotated 20°")),
        tap(() => this.currentMeasurement++)
      );
    }
    
    return init$.pipe(
      switchMap(() => this.waitForTurntableDone()),
      switchMap(() => of(null).pipe(delay(200))),
      // switchMap(() => this.runAnalysis(cycleMode)),
      // Optionally, update overall statistics & UI after the analysis chain.
      switchMap(() => this.http.post(`${this.BASE_URL}/calculate-statistics?mode=${cycleMode}`, {})),
      switchMap(() => this.http.post(`${this.BASE_URL}/update_results`, { mode: cycleMode })),
      tap((response: any) => this.updateResultsUI(response)),
      takeUntil(this.measurementStop$),
      catchError((err: any) => {
        console.error("Measurement cycle error:", err);
        this.stopMeasurement();
        this.measurementActive = false;
        this.sharedService.setMeasurementActive(false);
        
        return throwError(() => err);
      }),
      finalize(() => {
        if (!this.measurementActive) {
          console.log("Measurement cycle cancelled");
        } else if (this.currentMeasurement < this.totalMeasurements) {
          // Continue with the next measurement using 'slices' mode.
          this.executeCycle('slices').subscribe();
        } else {
          this.completeMeasurementCycle();
        }
      })
    );
  }

  // TODO:
  // Keep & refactor this function to run the automated measurement
  //private runAnalysis(cycleMode: 'full' | 'slices'): Observable<any> {

    //return this.http.post(`${this.BASE_URL}/analyze_center_circle`, {}).pipe( 
    //  switchMap(result => this.conditionalSave(result, 'center_circle')),
    //  switchMap(() => this.http.post(`${this.BASE_URL}/analyze_center_slice`, {})),
    //);
  //}
  
  // TODO:
  // Keep & refactor this function for the end of an automated measurement
  // Called when msmnt finishes
  private completeMeasurementCycle(): void {
    console.log("Measurement cycle completed.");
    this.measurementActive = false;
    this.sharedService.setMeasurementActive(false);
    this.showResultsPopup();
  }


  // TODO:
  // Keep & refactor this function for the end of an automated measurement
  // Reset msmnt data to null
  resetBackendResults(): void {
    this.http.post(`${this.BASE_URL}/reset_results`, {}).subscribe({
      next: (response: any) => {
        console.log("Backend results reset:", response);
        // If the response contains an array of counts, update UI.
        if (response?.result_counts && this.results.length === response.result_counts.length) {
          this.results.forEach((result, index) => result.value = response.result_counts[index]);
        }
      },
      error: (error) => console.error("Failed to reset backend results:", error)
    });
  }
  
  
  // TODO:
  // Refactor this function to support MotionPlatform instead
  // Check if motion platform is responsive
  waitForTurntableDone(): Observable<any> {
    return new Observable(observer => {
      const checkStatus = () => {
        this.http.get<{ connected: boolean; port?: string }>(`${this.BASE_URL}/status/serial/turntable`).subscribe(
          statusResponse => {
            if (statusResponse.connected) {
              console.log("Turntable movement completed.");
              observer.complete();
            } else {
              setTimeout(checkStatus, 500);
            }
          },
          error => {
            console.error("Error checking turntable status!", error);
            setTimeout(checkStatus, 500);
          }
        );
      };
      checkStatus();
    });
  }

  // TODO:
  // Keep & refactor this function for the end of an automated measurement
  // Update UI during measurement
  updateResultsUI(response: any): void {
    if (response?.result_counts) {
      if (!this.results || this.results.length !== response.result_counts.length) {
        // Initialize with default labels if results are empty or different in length
        this.results = response.result_counts.map((count: number, index: number) => ({
          label: `Result ${index + 1}`,
          value: count
        }));
      } else {
        // Update values but retain original labels
        this.results.forEach((result, index) => {
          result.value = response.result_counts[index];
        });
      }
      this.cdr.detectChanges();
    }
  }

  // TODO:
  // Keep & refactor this function for the end of an automated measurement
  // Aknowledge end of measurement 
  onPopupClosed(): void {
    // Reset the measurement cycle.
    this.stopMeasurement();
    // Hide the popup.
    this.isResultsPopupVisible = false;
  }
}