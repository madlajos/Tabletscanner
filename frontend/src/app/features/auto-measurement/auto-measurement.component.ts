import { Component, signal, OnInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Subscription, from } from 'rxjs';
import { concatMap, last, finalize, switchMap } from 'rxjs/operators';
import { HttpClient } from '@angular/common/http';
import {
  AutoMeasurementService,
  TabletStepRequest
} from '../../services/auto-measurement.service';
import { SharedService } from '../../shared.service';
import { BASE_URL } from '../../api-config';

// Interface for tablet position calculation
interface TabletPosition {
  index: number;
  x: number;
  y: number;
}

@Component({
  selector: 'app-auto-measurement',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './auto-measurement.component.html',
  styleUrls: ['./auto-measurement.component.css']
})
export class AutoMeasurementComponent implements OnInit, OnDestroy {

  readonly gridSize = 10; // change to 9 later if needed

  // Connection status (from shared service)
  cameraConnected = false;
  motionConnected = false;
  private cameraSub?: Subscription;
  private motionSub?: Subscription;

  // Tablet IDs, bottom-left = 1, top-right = gridSize^2
  readonly tablets = Array.from(
    { length: this.gridSize * this.gridSize },
    (_, i) => {
      const rowFromTop = Math.floor(i / this.gridSize); // 0..gridSize-1
      const col = i % this.gridSize;                    // 0..gridSize-1
      const rowFromBottom = this.gridSize - 1 - rowFromTop;
      return rowFromBottom * this.gridSize + col + 1;
    }
  );

  // Measurement settings
  autofocus = false;
  lampTop = false;
  lampSide = false;

  // Save location and measurement name
  saveLocation = '';
  measurementName = '';

  // First tablet position and spacing (from settings)
  firstTabletX = 3.0;
  firstTabletY = 7.0;
  firstTabletZ = 20.0;
  tabletSpacing = 18.3;

  // Set of selected tablet IDs (supports multiple ranges + gaps)
  private selectedSignal = signal<Set<number>>(new Set<number>());

  // Range anchor for adding ranges (first click)
  private rangeAnchorSignal = signal<number | null>(null);

  // Hover index for range preview
  private hoverIndexSignal = signal<number | null>(null);

  // Is measurement currently running (toggled "on")
  measurementActive = false;

  // Flag to signal stop request
  private stopRequested = false;

  // Progress tracking
  currentTabletIndex = 0;  // 1-based index of current tablet being processed
  private completedTablets = new Set<number>();  // Set of completed tablet IDs
  private currentTabletId: number | null = null;  // ID of tablet currently being processed

  // Measurement folder path (created at start)
  private measurementFolder = '';

  // Subscription to current tablet measurement (for immediate cancellation on stop)
  private currentTabletSubscription: Subscription | null = null;

  // Subscription to homing operation (for immediate cancellation on stop)
  private homingSubscription: Subscription | null = null;

  errorMessage: string | null = null;
  successMessage: string | null = null;
  validationMessage: string | null = null;

  constructor(
    private autoService: AutoMeasurementService,
    private sharedService: SharedService,
    private http: HttpClient
  ) {}

  ngOnInit(): void {
    // Subscribe to connection status from shared service
    this.cameraSub = this.sharedService.cameraConnectionStatus$.subscribe(status => {
      this.cameraConnected = status;
    });
    this.motionSub = this.sharedService.motionPlatformConnectionStatus$.subscribe(status => {
      this.motionConnected = status;
    });

    // Load saved settings from backend
    this.autoService.getSettings().subscribe({
      next: (res) => {
        if (res.auto_measurement_settings) {
          const settings = res.auto_measurement_settings;
          this.saveLocation = settings.save_location || '';
          this.firstTabletX = settings.first_tablet_x ?? 3.0;
          this.firstTabletY = settings.first_tablet_y ?? 7.0;
          this.firstTabletZ = settings.first_tablet_z ?? 20.0;
          this.tabletSpacing = settings.tablet_spacing ?? 18.3;
        }
      },
      error: (err) => {
        console.warn('Failed to load auto measurement settings:', err);
      }
    });
  }

  ngOnDestroy(): void {
    this.cameraSub?.unsubscribe();
    this.motionSub?.unsubscribe();
    
    // Ensure measurement is marked inactive on destroy
    if (this.measurementActive) {
      this.sharedService.setMeasurementActive(false);
    }
  }

  // ===== Derived properties =====

  get selectedCount(): number {
    return this.selectedSignal().size;
  }

  hasSelection(): boolean {
    return this.selectedSignal().size > 0;
  }

  // Get validation message based on current state
  getValidationMessage(): string | null {
    if (!this.cameraConnected && !this.motionConnected) {
      return 'Kamera és mozgásplatform nincs csatlakoztatva.';
    }
    if (!this.cameraConnected) {
      return 'Kamera nincs csatlakoztatva.';
    }
    if (!this.motionConnected) {
      return 'Mozgásplatform nincs csatlakoztatva.';
    }
    if (!this.saveLocation || this.saveLocation.trim() === '') {
      return 'Válasszon mentési helyet.';
    }
    if (!this.measurementName || this.measurementName.trim() === '') {
      return 'Adja meg a mérés nevét.';
    }
    if (!this.lampTop && !this.lampSide) {
      return 'Válasszon legalább egy világítási módot.';
    }
    if (this.selectedSignal().size === 0) {
      return 'Válasszon legalább egy tablettát.';
    }
    return null;
  }

  // Start/Stop button enabled state
  canStart(): boolean {
    if (this.measurementActive) {
      return true; // allow stopping
    }

    // Update validation message
    this.validationMessage = this.getValidationMessage();

    // Check all conditions
    const hasSelected = this.selectedSignal().size > 0;
    const connected = this.cameraConnected && this.motionConnected;
    const hasSaveLocation = !!(this.saveLocation && this.saveLocation.trim() !== '');
    const hasMeasurementName = !!(this.measurementName && this.measurementName.trim() !== '');
    const hasLightSelected = this.lampTop || this.lampSide;

    return hasSelected && connected && hasSaveLocation && hasMeasurementName && hasLightSelected;
  }

  // ===== Folder selection =====

  selectSaveFolder(): void {
    if (this.measurementActive) return;
    
    this.autoService.selectFolder().subscribe({
      next: (res) => {
        if (res.folder) {
          this.saveLocation = res.folder;
          // Persist to settings
          this.autoService.updateSettings('save_location', res.folder).subscribe({
            error: (err) => console.warn('Failed to save location setting:', err)
          });
        }
      },
      error: (err) => {
        console.error('Failed to select folder:', err);
      }
    });
  }

  // ===== DOT STATE METHODS =====

  isDotSelected(id: number): boolean {
    const selected = this.selectedSignal();
    if (selected.has(id)) {
      return true;
    }

    const anchor = this.rangeAnchorSignal();
    const hover = this.hoverIndexSignal();
    if (anchor != null && hover != null) {
      const start = Math.min(anchor, hover);
      const end = Math.max(anchor, hover);
      return id >= start && id <= end;
    }

    return false;
  }

  isDotCompleted(id: number): boolean {
    return this.completedTablets.has(id);
  }

  isDotInProgress(id: number): boolean {
    return this.currentTabletId === id;
  }

  // Determine dot state for CSS class binding
  getDotState(id: number): 'completed' | 'in-progress' | 'pending' | 'none' {
    if (this.isDotCompleted(id)) {
      return 'completed';
    }
    if (this.isDotInProgress(id)) {
      return 'in-progress';
    }
    if (this.isDotSelected(id)) {
      return 'pending';
    }
    return 'none';
  }

  // ===== CLICK SELECTION LOGIC =====

  onDotClick(id: number): void {
    if (this.measurementActive) {
      // Selection is locked while measurement is running
      return;
    }

    const selected = new Set(this.selectedSignal());
    const anchor = this.rangeAnchorSignal();

    if (anchor === null) {
      if (selected.has(id)) {
        selected.delete(id);
        this.selectedSignal.set(selected);
        this.hoverIndexSignal.set(null);
      } else {
        selected.add(id);
        this.selectedSignal.set(selected);
        this.rangeAnchorSignal.set(id);
        this.hoverIndexSignal.set(null);
      }
      return;
    }

    if (id === anchor) {
      this.rangeAnchorSignal.set(null);
      this.hoverIndexSignal.set(null);
      return;
    }

    const start = Math.min(anchor, id);
    const end = Math.max(anchor, id);
    for (let v = start; v <= end; v++) {
      selected.add(v);
    }

    this.selectedSignal.set(selected);
    this.rangeAnchorSignal.set(null);
    this.hoverIndexSignal.set(null);
  }

  onDotMouseEnter(id: number): void {
    if (this.measurementActive) {
      this.hoverIndexSignal.set(null);
      return;
    }
    if (this.rangeAnchorSignal() !== null) {
      this.hoverIndexSignal.set(id);
    } else {
      this.hoverIndexSignal.set(null);
    }
  }

  onGridMouseLeave(): void {
    this.hoverIndexSignal.set(null);
  }

  clearSelection(): void {
    if (this.measurementActive) {
      return;
    }
    this.selectedSignal.set(new Set<number>());
    this.rangeAnchorSignal.set(null);
    this.hoverIndexSignal.set(null);
    this.completedTablets.clear();
    this.currentTabletId = null;
    this.currentTabletIndex = 0;
    this.errorMessage = null;
    this.successMessage = null;
    this.validationMessage = null;
  }

  // ===== Position calculation =====

  private calculateTabletPosition(tabletIndex: number): TabletPosition {
    const zeroBasedIndex = tabletIndex - 1;
    const rowFromBottom = Math.floor(zeroBasedIndex / this.gridSize);
    const col = zeroBasedIndex % this.gridSize;
    
    return {
      index: tabletIndex,
      x: this.firstTabletX + col * this.tabletSpacing,
      y: this.firstTabletY + rowFromBottom * this.tabletSpacing
    };
  }

  // ===== Start / Stop measurement =====

  startMeasurement(): void {
    // Toggle behavior - if running, stop
    if (this.measurementActive) {
      this.stopMeasurement();
      return;
    }

    const indices = Array.from(this.selectedSignal()).sort((a, b) => a - b);
    if (indices.length === 0) {
      return;
    }

    // Clear messages and reset state
    this.errorMessage = null;
    this.successMessage = null;
    this.validationMessage = null;
    this.stopRequested = false;
    this.completedTablets.clear();
    this.currentTabletId = null;
    this.currentTabletIndex = 0;

    // Create measurement folder path
    const datePrefix = new Date().toISOString().slice(0, 10).replace(/-/g, '');
    this.measurementFolder = `${this.saveLocation}/${datePrefix}_${this.measurementName.trim()}`;

    // Set measurement active (locks UI)
    this.measurementActive = true;
    this.sharedService.setMeasurementActive(true);

    // Check if already homed, then home if needed or proceed directly
    this.checkHomedThenProceed(indices);
  }

  private checkHomedThenProceed(indices: number[]): void {
    this.http.get<{ x: boolean; y: boolean; z: boolean }>(`${BASE_URL}/check_axes_homed`)
      .subscribe({
        next: (homedStatus) => {
          // If all axes are homed, skip homing and proceed directly
          if (homedStatus.x && homedStatus.y && homedStatus.z) {
            console.log('All axes already homed, skipping home');
            this.processTabletQueue(indices, 0);
          } else {
            // Not all axes homed, perform homing
            console.log('Axes not fully homed, performing home');
            this.homeMotionPlatformThenProceed(indices);
          }
        },
        error: (err) => {
          console.warn('Could not check homed status, proceeding with home:', err);
          this.homeMotionPlatformThenProceed(indices);
        }
      });
  }

  private homeMotionPlatformThenProceed(indices: number[]): void {
    this.validationMessage = 'Mozgásplatform pozicionálása...';

    // Signal motion-control component that homing is in progress
    this.sharedService.setMotionHoming(true);

    const axesOrder: Array<'z' | 'y' | 'x'> = ['z', 'y', 'x']; // match manual home order

    const homing$ = from(axesOrder).pipe(
      concatMap((axis) => this.http.post(`${BASE_URL}/home_toolhead`, { axes: [axis] })),
      last(),
      switchMap(() => this.http.get<{ x?: number | null; y?: number | null; z?: number | null }>(`${BASE_URL}/get_motion_platform_position`)),
      finalize(() => {
        this.sharedService.setMotionHoming(false);
        this.homingSubscription = null;
      })
    );

    this.homingSubscription = homing$.subscribe({
      next: (position) => {
        console.log('Motion platform homed successfully (Z→Y→X). Position:', position);
        this.validationMessage = null;
        this.sharedService.setMotionPosition({
          x: position?.x ?? null,
          y: position?.y ?? null,
          z: position?.z ?? null
        });
        this.processTabletQueue(indices, 0);
      },
      error: (err) => {
        console.error('Failed to home motion platform:', err);
        this.errorMessage = 'Hiba: Nem sikerült pozicionálni a mozgásplatformot.';
        this.finishMeasurement(false);
      }
    });
  }

  stopMeasurement(): void {
    this.stopRequested = true;

    // Signal backend to abort autofocus immediately
    this.http.post(`${BASE_URL}/abort-autofocus`, {}).subscribe({
      next: () => console.log('Autofocus abort signal sent to backend'),
      error: (err) => console.warn('Could not send abort signal:', err)
    });

    // Cancel any active operations immediately
    this.cancelActiveOperations();

    // Clear homing flag so UI unlocks promptly
    this.sharedService.setMotionHoming(false);

    // Finish measurement immediately so UI resets
    this.finishMeasurement(false);
  }

  private cancelActiveOperations(): void {
    // Cancel current tablet measurement
    if (this.currentTabletSubscription) {
      this.currentTabletSubscription.unsubscribe();
      this.currentTabletSubscription = null;
    }

    // Cancel homing operation
    if (this.homingSubscription) {
      this.homingSubscription.unsubscribe();
      this.homingSubscription = null;
    }
  }

  private async processTabletQueue(indices: number[], queueIndex: number): Promise<void> {
    // Check if stopped or finished
    if (this.stopRequested || queueIndex >= indices.length) {
      this.finishMeasurement(queueIndex >= indices.length);
      return;
    }

    const tabletId = indices[queueIndex];
    const position = this.calculateTabletPosition(tabletId);
    
    // Update progress
    this.currentTabletIndex = queueIndex + 1;
    this.currentTabletId = tabletId;
    
    // Determine if this is the first tablet (for coarse vs fine autofocus)
    const isFirstTablet = queueIndex === 0;

    const req: TabletStepRequest = {
      tablet_index: tabletId,
      x: position.x,
      y: position.y,
      z: this.firstTabletZ,
      measurement_folder: this.measurementFolder,
      measurement_name: this.measurementName.trim(),
      autofocus: this.autofocus,
      lamp_top: this.lampTop,
      lamp_side: this.lampSide,
      is_first_tablet: isFirstTablet
    };

    // If stop was requested before starting this tablet, exit early
    if (this.stopRequested) {
      this.finishMeasurement(false);
      return;
    }

    this.currentTabletSubscription = this.autoService.measureSingleTablet(req).subscribe({
      next: (resp) => {
        if (this.stopRequested) {
          this.currentTabletSubscription?.unsubscribe();
          this.currentTabletSubscription = null;
          this.finishMeasurement(false);
          return;
        }

        if (resp.status === 'success') {
          // Mark tablet as completed
          this.completedTablets.add(tabletId);
          
          // Emit saved images to gallery
          if (resp.saved_images && resp.saved_images.length > 0) {
            for (const imagePath of resp.saved_images) {
              const lightType = imagePath.includes('_dome_') ? 'dome' : 'bar';
              this.sharedService.emitSavedImage({
                path: imagePath,
                tabletIndex: tabletId,
                lightType: lightType as 'dome' | 'bar'
              });
            }
          }

          // Process next tablet
          this.currentTabletSubscription?.unsubscribe();
          this.currentTabletSubscription = null;
          this.processTabletQueue(indices, queueIndex + 1);
        } else {
          // Error during measurement
          this.errorMessage = resp.message ?? `Hiba a ${tabletId}. tabletta mérésekor.`;
          this.currentTabletSubscription?.unsubscribe();
          this.currentTabletSubscription = null;
          this.finishMeasurement(false);
        }
      },
      error: (err) => {
        if (!this.stopRequested) {
          this.errorMessage = err?.error?.message ?? `Szerver hiba a ${tabletId}. tabletta mérésekor.`;
        }
        this.currentTabletSubscription?.unsubscribe();
        this.currentTabletSubscription = null;
        this.finishMeasurement(false);
      }
    });
  }

  private finishMeasurement(success: boolean): void {
    this.measurementActive = false;
    this.sharedService.setMeasurementActive(false);
    this.currentTabletId = null;
    
    if (this.stopRequested) {
      this.successMessage = `Mérés leállítva. ${this.completedTablets.size} tabletta mérése kész.`;
    } else if (success) {
      this.successMessage = `Mérés sikeresen befejezve. ${this.completedTablets.size} tabletta mérése kész.`;
    }
    // Error message is set in processTabletQueue if there was an error
    
    this.stopRequested = false;
  }
}
