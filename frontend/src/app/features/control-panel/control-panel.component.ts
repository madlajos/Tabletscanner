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
import { BarcodeService } from '../../services/barcode.service';


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
  nozzleBarcode: string = "";
  operatorId: string = "";
  ng_limit: number = 0;
  save_csv: boolean = false;
  save_images: boolean = false;
  csv_dir: string = "";

  isMainConnected: boolean = false;
  isSideConnected: boolean = false;

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
    private barcodeService: BarcodeService
  ) { }

  ngOnInit(): void {
  this.barcodeService.barcode$.subscribe(scannedBarcode => {
    // Ignore any scans while a measurement is running OR while the results popup is on-screen
    if (this.measurementActive || this.isResultsPopupVisible) {
        if (scannedBarcode) {
          this.barcodeService.clearBarcode();   // flush stray scan
        }
        return;
    }

    if (scannedBarcode) {
      console.log('New barcode:', scannedBarcode);

      // Remember which barcode this lookup belongs to
      const requestToken = scannedBarcode;

      this.nozzleBarcode = requestToken;
      this.nozzleId = '';               // clear any previous ID
      this.cdr.detectChanges();         // update UI immediately

      this.http
        .get<{ spinneret_id: string | null }>(`${this.BASE_URL}/lookup-nozzle`,
              { params: { barcode: requestToken } })
        .pipe(takeUntil(this.lookupCancel$))
        .subscribe({
          next: resp => {
            // Discard late/stale responses that no longer match the current barcode
            if (this.nozzleBarcode !== requestToken) {
              return;
            }

            if (resp.spinneret_id) {
              this.nozzleId = resp.spinneret_id;
              console.log(`Filled nozzleId: ${resp.spinneret_id}`);
            } else {
              console.warn(`No ID mapping for ${requestToken}`);
            }
            this.cdr.detectChanges();
          },
          error: err => {
            console.error('Lookup failed:', err);
            this.cdr.detectChanges();
          }
        });
    }
  });


    this.sharedService.cameraConnectionStatus$.subscribe(status => {
      this.isMainConnected = status.main;
      this.isSideConnected = status.side;
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

    setTimeout(() => {
      this.getRelayState();
    }, 3000);
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
    if (!this.isMainConnected) {
      console.error("Cannot start measurement: Main camera is not connected.");
      return false;
    }
    if (!this.isSideConnected) {
      console.error("Cannot start measurement: Side camera is not connected.");
      return false;
    }
    if (!this.turntableControl) {
      console.error("Cannot start measurement: Turntable controller is not available.");
      return false;
    }
    // Check that either nozzleId or nozzleBarcode is provided.
    if (!this.nozzleId && !this.nozzleBarcode) {
      console.error("Cannot start measurement: Nozzle identifier is missing.");
      return false;
    }
    if (!this.operatorId) {
      console.error("Cannot start measurement: Operator ID is missing.");
      return false;
    }
    return true;
  }


  getRelayState(): void {
    this.http.get(`${this.BASE_URL}/get-relay`)
      .pipe(
        catchError((error: any) => {
          console.error("Error getting relay state:", error);
          // Optionally, you can return a default response so the error doesn't propagate.
          // For example, consider the relay off.
          return of({ state: 0 });
        })
      )
      .subscribe((response: any) => {
        // Expecting response.state to be 1 or 0.
        this.relayState = response.state === 1;
        console.log("Relay state obtained:", response.state);
      });
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
      this.toggleRelay();
    }
  }

  stopMeasurement(): void {
    this.measurementActive = false;
    this.currentMeasurement = 0;
    this.lookupCancel$.next();
    this.lookupCancel$ = new Subject<void>();

    // Reset results while keeping original labels.
    this.results = this.results.map(res => ({ ...res, value: 0 }));
    
    this.nozzleId = "";
    this.nozzleBarcode = "";
    this.barcodeService.clearBarcode();
    this.nozzleBarcode = "";
    
    console.log("Results cleared, progress bar reset, and barcode fields cleared.");
  
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
    const spinneretId = this.nozzleBarcode || this.nozzleId;
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
          this.toggleRelay();
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
      switchMap(() => this.runAnalysis(cycleMode)),
      // Optionally, update overall statistics & UI after the analysis chain.
      switchMap(() => this.http.post(`${this.BASE_URL}/calculate-statistics?mode=${cycleMode}`, {})),
      switchMap(() => this.http.post(`${this.BASE_URL}/update_results`, { mode: cycleMode })),
      tap((response: any) => this.updateResultsUI(response)),
      takeUntil(this.measurementStop$),
      catchError((err: any) => {
        console.error("Measurement cycle error:", err);
        this.stopMeasurement();
        this.toggleRelay();
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

  private runAnalysis(cycleMode: 'full' | 'slices'): Observable<any> {
    if (cycleMode === 'full') {
      // For a full cycle, include center_circle analysis.
      return this.http.post(`${this.BASE_URL}/analyze_center_circle`, {}).pipe(
        switchMap(result => this.conditionalSave(result, 'center_circle')),
        switchMap(() => this.http.post(`${this.BASE_URL}/analyze_center_slice`, {})),
        switchMap(result => this.conditionalSave(result, 'center_slice')),
        switchMap(() => this.http.post(`${this.BASE_URL}/analyze_outer_slice`, {})),
        switchMap(result => this.conditionalSave(result, 'outer_slice'))
      );
    } else {
      // For subsequent cycles, only run the slice analyses.
      return this.http.post(`${this.BASE_URL}/analyze_center_slice`, {}).pipe(
        switchMap(result => this.conditionalSave(result, 'center_slice')),
        switchMap(() => this.http.post(`${this.BASE_URL}/analyze_outer_slice`, {})),
        switchMap(result => this.conditionalSave(result, 'outer_slice'))
      );
    }
  }

  private conditionalSave<T>(result: T, analysisMode: string): Observable<T> {
    if (this.save_images) {
      return this.http.post(`${this.BASE_URL}/calculate-statistics?mode=${analysisMode}`, {}).pipe(
        switchMap(() => this.http.post(`${this.BASE_URL}/save-annotated-image`, {})),
        map(() => result)
      );
    } else {
      return of(result);
    }
  }
  
  private completeMeasurementCycle(): void {
    console.log("Measurement cycle completed.");
    this.measurementActive = false;
    this.sharedService.setMeasurementActive(false);
    if (this.save_csv) {
      this.saveResultsToCsv();
    }
    this.toggleRelay();
    this.showResultsPopup();
  }

  toggleRelay(): void {
    this.relayState = !this.relayState;
    const payload = { state: this.relayState ? 1 : 0 };
    this.http.post(`${this.BASE_URL}/toggle-relay`, payload).subscribe(
      (response: any) => {
        console.log(`Relay ${this.relayState ? 'ON' : 'OFF'}`, response);
      },
      (error: any) => {
        console.error('Error toggling relay:', error);
        this.relayState = !this.relayState;
      }
    );
  }
  

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

  onPopupClosed(): void {
    // Reset the measurement cycle.
    this.stopMeasurement();
    // Hide the popup.
    this.isResultsPopupVisible = false;
  }

  saveResultsToCsv(): void {
    const spinneretId = this.nozzleBarcode || this.nozzleId;
    if (!spinneretId) {
      console.error("Cannot save CSV: No spinneret ID provided.");
      return;
    }
    const payload = {
      spinneret_id: spinneretId
    };
  
    this.http.post(`${this.BASE_URL}/save_results_to_csv`, payload).subscribe({
      next: (resp: any) => {
        console.log("CSV saved successfully:", resp);
      },
      error: (err: any) => {
        console.error("CSV saving failed:", err);
      }
    });
  }

  // Tester Functions to analyse only parts of the images
  analyzeCenterCircle(): void {
    console.log("Analyzing Center Circle...");
    this.http.post(`${this.BASE_URL}/analyze_center_circle`, {}).pipe(
      tap(response => console.log("analyze_center_circle Response:", response)),
      switchMap(() => {
        console.log("➡️ Calling update_results...");
        return this.http.post(`${this.BASE_URL}/update_results`, { mode: "center_circle" }, { headers: { 'Content-Type': 'application/json' } });
      }),
      tap((response: any) => {
        console.log("update_results Response:", response);
        this.updateResultsUI(response);
      })
    ).subscribe({
      error: (err) => console.error("Error in analyzeCenterCircle:", err)
    });
  }

  analyzeInnerSlice(): void {
    console.log("Analyzing Center Slice...");
    this.http.post(`${this.BASE_URL}/analyze_center_slice`, {}).pipe(
      switchMap(() => this.http.post(`${this.BASE_URL}/update_results`, { mode: "center_slice" })),
      tap((response: any) => this.updateResultsUI(response))
    ).subscribe();
  }

  analyzeOuterSlice(): void {
    console.log("Analyzing Outer Slice...");
    this.http.post(`${this.BASE_URL}/analyze_outer_slice`, {}).pipe(
      switchMap(() => this.http.post(`${this.BASE_URL}/update_results`, { mode: "outer_slice" })),
      tap((response: any) => this.updateResultsUI(response))
    ).subscribe();
  }
}
