import { Component, signal, OnInit, OnDestroy, AfterViewInit } from '@angular/core';
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
import { ErrorNotificationService } from '../../services/error-notification.service';
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
export class AutoMeasurementComponent implements OnInit, AfterViewInit, OnDestroy {

  readonly gridSize = 10; // change to 9 later if needed

  // Connection status (from shared service)
  cameraConnected = false;
  motionConnected = false;
  private cameraSub?: Subscription;
  private motionSub?: Subscription;
  private autofocusErrorSub?: Subscription;

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
  backgroundSubtraction = false;

  // Save location and measurement name
  saveLocation = '';
  measurementName = '';

  // First tablet position and spacing (from settings)
  firstTabletX = 6.0;
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
  private failedTablets = new Set<number>();      // Set of tablets with AF errors (E2000/E2002/E2003)
  private tabletErrors = new Map<number, string>(); // Tablet ID -> error message
  private tabletImages = new Map<number, string[]>(); // Tablet ID -> saved image paths (non-masked)
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

  // Reconnection state during auto-measurement
  reconnecting = false;
  reconnectMessage: string | null = null;
  private reconnectTimer: ReturnType<typeof setInterval> | null = null;
  private reconnectAttemptCount = 0;
  private static readonly MAX_RECONNECT_CYCLES = 3;

  // Context menu state for tablet grid
  tabletContextMenuVisible = false;
  tabletContextMenuX = 0;
  tabletContextMenuY = 0;
  tabletContextMenuId: number | null = null;
  tabletHomed = false;
  private homedSub?: Subscription;

  constructor(
    private autoService: AutoMeasurementService,
    private sharedService: SharedService,
    private errorNotificationService: ErrorNotificationService,
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

    // Subscribe to homed state from SharedService (published by motion-control component)
    this.homedSub = this.sharedService.motionHomed$.subscribe(homed => {
      this.tabletHomed = homed;
      console.log('Auto-measurement received homed state:', homed);
    });

    // Subscribe to manual autofocus errors from motion-control
    this.autofocusErrorSub = this.sharedService.autofocusError$.subscribe(msg => {
      if (!this.measurementActive) {
        this.errorMessage = msg;
      }
    });

    // Load saved settings from backend
    this.autoService.getSettings().subscribe({
      next: (res) => {
        if (res.auto_measurement_settings) {
          const settings = res.auto_measurement_settings;
          this.saveLocation = settings.save_location || '';
          this.firstTabletX = settings.first_tablet_x ?? 2.9;
          this.firstTabletY = settings.first_tablet_y ?? 10.6;
          this.firstTabletZ = settings.first_tablet_z ?? 20.0;
          this.tabletSpacing = settings.tablet_spacing ?? 18.3;
        }
      },
      error: (err) => {
        console.warn('Failed to load auto measurement settings:', err);
      }
    });
  }

  ngAfterViewInit(): void {
    // Hide context menu when clicking outside
    document.addEventListener('click', () => this.hideTabletContextMenu());
  }

  ngOnDestroy(): void {
    this.cameraSub?.unsubscribe();
    this.motionSub?.unsubscribe();
    this.homedSub?.unsubscribe();
    this.autofocusErrorSub?.unsubscribe();
    
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

  // Get info message to show when button is enabled
  getInfoMessage(): string | null {
    // Only show info message when:
    // 1. Not currently measuring
    // 2. Button is enabled (canStart returns true)
    // 3. No validation message is being shown
    if (this.measurementActive || this.validationMessage) {
      return null;
    }
    
    if (this.canStart()) {
      return 'Ellenőrizze a mentési beállításokat!';
    }
    
    return null;
  }

  // ===== Tablet context menu =====

  onTabletContextMenu(event: MouseEvent, id: number): void {
    // Prevent context menu if measurement is active
    if (this.measurementActive) {
      event.preventDefault();
      return;
    }

    event.preventDefault();

    // Estimated menu dimensions
    const menuWidth = 180;
    const menuHeight = 80;

    // Get viewport dimensions
    const viewportWidth = window.innerWidth;
    const viewportHeight = window.innerHeight;

    // Calculate position, adjusting if menu would go off-screen
    let menuX = event.clientX;
    let menuY = event.clientY;

    // Adjust horizontal position if menu would overflow right edge
    if (menuX + menuWidth > viewportWidth) {
      menuX = viewportWidth - menuWidth - 5;
    }

    // Adjust vertical position if menu would overflow bottom edge
    if (menuY + menuHeight > viewportHeight) {
      menuY = viewportHeight - menuHeight - 5;
    }

    // Ensure menu doesn't go off left/top edges
    menuX = Math.max(5, menuX);
    menuY = Math.max(5, menuY);

    this.tabletContextMenuVisible = true;
    this.tabletContextMenuX = menuX;
    this.tabletContextMenuY = menuY;
    this.tabletContextMenuId = id;
  }

  hideTabletContextMenu(): void {
    this.tabletContextMenuVisible = false;
    this.tabletContextMenuId = null;
  }

  selectTablet(): void {
    if (this.tabletContextMenuId !== null) {
      this.onDotClick(this.tabletContextMenuId);
    }
    this.hideTabletContextMenu();
  }

  moveToTablet(): void {
    if (this.tabletContextMenuId === null || !this.tabletHomed) {
      console.warn('moveToTablet blocked: tabletHomed =', this.tabletHomed);
      return;
    }

    const pos = this.getTabletPosition(this.tabletContextMenuId);
    if (pos) {
      this.sharedService.invalidateAutofocus();
      this.http.post(`${BASE_URL}/move_toolhead_absolute`, {
        x: pos.x,
        y: pos.y
      }).subscribe({
        next: () => {
          console.log(`Moved to tablet ${this.tabletContextMenuId}`);
          this.hideTabletContextMenu();
        },
        error: (err) => {
          console.error('Failed to move toolhead:', err);
        }
      });
    }
  }

  private getTabletPosition(tabletId: number): { x: number; y: number } | null {
    // Tablet IDs: 1-10 = bottom row, 11-20 = next row up, etc.
    // For gridSize=10: tablet 1 is at (row=0, col=0), tablet 2 at (row=0, col=1), etc.
    
    const tabletIndex = tabletId - 1; // Convert 1-based to 0-based
    if (tabletIndex < 0 || tabletIndex >= this.gridSize * this.gridSize) {
      return null;
    }

    // Calculate row from bottom and column from left
    const rowFromBottom = Math.floor(tabletIndex / this.gridSize);
    const col = tabletIndex % this.gridSize;

    // Calculate position
    const x = this.firstTabletX + col * this.tabletSpacing;
    const y = this.firstTabletY + rowFromBottom * this.tabletSpacing;

    return { x, y };
  }

  /**
   * Convert tablet ID to label format (A1, B1, C1, etc.)
   * Columns are letters (A, B, C...), rows are numbers (1, 2, 3...)
   * ID 1 = A1, ID 2 = B1, ID 10 = J1, ID 11 = A2, etc. (for 10x10 grid)
   */
  getTabletLabel(id: number): string {
    const index = id - 1; // Convert to 0-based index
    const col = index % this.gridSize; // Column determines letter
    const row = Math.floor(index / this.gridSize) + 1; // Row determines number (1-based)
    const letter = String.fromCharCode(65 + col); // 65 = 'A'
    return letter + row;
  }

  /**
   * Returns the tooltip text for a tablet based on its measurement state.
   */
  getTabletTooltip(id: number): string {
    const label = this.getTabletLabel(id);
    if (this.completedTablets.has(id)) {
      return `${label} - Sikeres mérés`;
    }
    if (this.currentTabletId === id) {
      return `${label} - Mérés folyamatban`;
    }
    if (this.failedTablets.has(id)) {
      const error = this.tabletErrors.get(id);
      return error ? `${label} - ${error}` : `${label} - Hiba`;
    }
    return label;
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

  isDotFailed(id: number): boolean {
    return this.failedTablets.has(id);
  }

  isDotInProgress(id: number): boolean {
    return this.currentTabletId === id;
  }

  // Determine dot state for CSS class binding
  getDotState(id: number): 'completed' | 'failed' | 'in-progress' | 'pending' | 'none' {
    if (this.isDotCompleted(id)) {
      return 'completed';
    }
    if (this.isDotFailed(id)) {
      return 'failed';
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
    // If the tablet was successfully measured, open its saved images (works during and after measurement)
    if (this.completedTablets.has(id)) {
      this.openTabletImages(id);
      return;
    }

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
    this.failedTablets.clear();
    this.tabletErrors.clear();
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
    this.reconnectAttemptCount = 0;
    this.completedTablets.clear();
    this.failedTablets.clear();
    this.tabletErrors.clear();
    this.currentTabletId = null;
    this.currentTabletIndex = 0;

    // Create measurement folder path
    this.measurementFolder = `${this.saveLocation}/${this.measurementName.trim()}`;

    // Set measurement active (locks UI)
    this.measurementActive = true;
    this.sharedService.setMeasurementActive(true);

    // Check if already homed, then home if needed or proceed directly
    this.checkHomedThenProceed(indices);
  }

  private checkHomedThenProceed(indices: number[]): void {
    // Always home at the start of auto measurement to ensure a known reference.
    this.homeMotionPlatformThenProceed(indices);
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
        
        // Update motion position and homed state via SharedService
        this.sharedService.setMotionPosition({
          x: position?.x ?? null,
          y: position?.y ?? null,
          z: position?.z ?? null
        });
        
        // Explicitly set homed state to true after successful homing
        this.sharedService.setMotionHomed(true);
        
        this.processTabletQueue(indices, 0);
      },
      error: (err) => {
        console.error('Failed to home motion platform:', err);

        if (this.stopRequested) {
          this.finishMeasurement(false);
          return;
        }

        // Detect USB disconnect during homing
        if (this.isDeviceDisconnectError(err)) {
          const device = this.detectDisconnectedDevice(err);
          this.attemptDeviceReconnect(
            device,
            'homing',
            () => this.homeMotionPlatformThenProceed(indices)
          );
          return;
        }

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

    // Note: Lights will be turned off in finishMeasurement()
    // No need to duplicate the call here

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

    // Cancel reconnection timer
    if (this.reconnectTimer) {
      clearInterval(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    this.reconnecting = false;
    this.reconnectMessage = null;
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
      measurement_folder: this.measurementFolder,
      measurement_name: this.measurementName.trim(),
      autofocus: this.autofocus,
      lamp_top: this.lampTop,
      lamp_side: this.lampSide,
      is_first_tablet: isFirstTablet,
      background_subtraction: this.backgroundSubtraction
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
          // Check if any E2xxx error was flagged (tablet missing, exposure, position, quality, etc.)
          if (resp.af_error_code && resp.af_error_code.startsWith('E2')) {
            // Mark tablet as failed (red) and store the error message
            this.failedTablets.add(tabletId);
            this.tabletErrors.set(tabletId, resp.af_error_message ?? resp.af_error_code);
            this.errorMessage = `${this.getTabletLabel(tabletId)} tabletta: ${resp.af_error_message ?? resp.af_error_code}`;
          } else {
            // Mark tablet as completed (green)
            this.completedTablets.add(tabletId);
          }
          
          // Emit saved images to gallery and store paths (exclude background-subtracted _masked images)
          if (resp.saved_images && resp.saved_images.length > 0) {
            const nonMaskedPaths: string[] = [];
            for (const imagePath of resp.saved_images) {
              if (imagePath.includes('_masked')) {
                continue;
              }
              nonMaskedPaths.push(imagePath);
              const lightType = imagePath.includes('_dome_') ? 'dome' : 'bar';
              this.sharedService.emitSavedImage({
                path: imagePath,
                tabletIndex: tabletId,
                lightType: lightType as 'dome' | 'bar'
              });
            }
            if (nonMaskedPaths.length > 0) {
              this.tabletImages.set(tabletId, nonMaskedPaths);
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
        this.currentTabletSubscription?.unsubscribe();
        this.currentTabletSubscription = null;

        if (this.stopRequested) {
          this.finishMeasurement(false);
          return;
        }

        // Check if this is a device disconnect
        if (this.isDeviceDisconnectError(err)) {
          const device = this.detectDisconnectedDevice(err);
          this.attemptDeviceReconnect(
            device,
            `tabletta ${tabletId}`,
            () => this.processTabletQueue(indices, queueIndex)
          );
          return;
        }

        this.errorMessage = err?.error?.message ?? `Szerver hiba a ${tabletId}. tabletta mérésekor.`;
        this.finishMeasurement(false);
      }
    });
  }

  // ===== Device disconnect detection helpers =====

  /**
   * Check if an HTTP error indicates a device (motion platform or camera) disconnect.
   */
  private isDeviceDisconnectError(err: any): boolean {
    const code = err?.error?.code;
    const details = err?.error?.details || err?.error?.message || err?.error?.error || '';
    const status = err?.status;

    // Motion platform: E1201 with 503
    if (code === 'E1201' && status === 503) return true;

    // Camera: E1111 with 503
    if (code === 'E1111' && status === 503) return true;

    // Serial exception wrapped in 500
    if (status === 500 && (
      details.includes('SerialException') ||
      details.includes('PermissionError') ||
      details.includes('WriteFile failed') ||
      details.includes('ClearCommError')
    )) return true;

    // Camera disconnect in 500
    if (status === 500 && (
      details.includes('Camera not ready') ||
      details.includes('Camera disconnected') ||
      details.includes('Grab failed') ||
      details.includes('Failed to grab') ||
      details.includes('physically removed') ||
      details.includes('not open')
    )) return true;

    return false;
  }

  /**
   * Determine which device is disconnected based on the error.
   * Returns 'motion' or 'camera'.
   */
  private detectDisconnectedDevice(err: any): 'motion' | 'camera' {
    const code = err?.error?.code;
    if (code === 'E1111') return 'camera';

    const details = err?.error?.details || err?.error?.message || err?.error?.error || '';
    if (
      details.includes('Camera not ready') ||
      details.includes('Camera disconnected') ||
      details.includes('Grab failed') ||
      details.includes('Failed to grab') ||
      details.includes('physically removed') ||
      details.includes('not open')
    ) return 'camera';

    return 'motion';
  }

  /**
   * Attempt to reconnect to the specified device for up to 30 seconds.
   * If reconnection succeeds, call resumeCallback to continue the measurement.
   * If it fails after 30s, show a center-error-popup and stop.
   *
   * @param device 'motion' or 'camera'
   * @param context Human-readable context (e.g. 'homing' or 'tabletta 3')
   * @param resumeCallback Function to call after successful reconnection
   */
  private attemptDeviceReconnect(
    device: 'motion' | 'camera',
    context: string,
    resumeCallback: () => void
  ): void {
    const RECONNECT_TIMEOUT_S = 30;
    const RECONNECT_INTERVAL_MS = 3000;
    const startTime = Date.now();

    const deviceName = device === 'motion' ? 'Mozgásplatform' : 'Kamera';
    const errorCode = device === 'motion' ? 'E1201' : 'E1111';

    // Track how many full reconnect cycles have been attempted.
    // If we keep cycling (connect returns 200 but device is still dead),
    // stop after MAX_RECONNECT_CYCLES to prevent an infinite loop.
    this.reconnectAttemptCount++;
    if (this.reconnectAttemptCount > AutoMeasurementComponent.MAX_RECONNECT_CYCLES) {
      console.error(`${deviceName} reconnect cycle limit reached (${AutoMeasurementComponent.MAX_RECONNECT_CYCLES}). Stopping measurement.`);
      this.errorNotificationService.addError({
        code: errorCode,
        message: `${deviceName} kapcsolat megszakadt (${context}). Többszöri újracsatlakozás sikertelen.`,
        popupStyle: 'center'
      });
      this.errorMessage = `${deviceName} újracsatlakozás többszöri sikertelen próbálkozás után. Mérés megszakítva.`;
      this.reconnectAttemptCount = 0;
      this.finishMeasurement(false);
      return;
    }

    this.reconnecting = true;
    this.reconnectMessage = `${deviceName} kapcsolat megszakadt (${context}). Újracsatlakozás... (0/${RECONNECT_TIMEOUT_S}s) [${this.reconnectAttemptCount}/${AutoMeasurementComponent.MAX_RECONNECT_CYCLES}]`;

    this.reconnectTimer = setInterval(() => {
      const elapsedMs = Date.now() - startTime;
      const elapsedS = Math.round(elapsedMs / 1000);

      // If stop was requested, cancel reconnection
      if (this.stopRequested) {
        this.clearReconnectState();
        this.finishMeasurement(false);
        return;
      }

      // Timeout reached — give up
      if (elapsedMs >= RECONNECT_TIMEOUT_S * 1000) {
        this.clearReconnectState();

        // Show center-error-popup
        this.errorNotificationService.addError({
          code: errorCode,
          message: `${deviceName} kapcsolat megszakadt (${context}). Újracsatlakozás sikertelen (${RECONNECT_TIMEOUT_S}s).`,
          popupStyle: 'center'
        });

        this.errorMessage = `${deviceName} újracsatlakozás sikertelen (${RECONNECT_TIMEOUT_S}s). Mérés megszakítva.`;
        this.finishMeasurement(false);
        return;
      }

      // Update message with countdown
      this.reconnectMessage = `${deviceName} kapcsolat megszakadt (${context}). Újracsatlakozás... (${elapsedS}/${RECONNECT_TIMEOUT_S}s) [${this.reconnectAttemptCount}/${AutoMeasurementComponent.MAX_RECONNECT_CYCLES}]`;

      // Try to reconnect to the appropriate device
      const reconnect$ = device === 'motion'
        ? this.autoService.reconnectMotionPlatform()
        : this.autoService.reconnectCamera();

      reconnect$.subscribe({
        next: (resp: any) => {
          const msg = resp?.message || '';
          if (msg.includes('failed')) return; // not actually connected

          console.info(`${deviceName} reconnected during auto-measurement:`, msg);
          this.clearReconnectState();
          // Reset cycle counter on successful reconnection
          this.reconnectAttemptCount = 0;

          // Update shared service so UI reflects the reconnected state
          if (device === 'motion') {
            this.sharedService.setMotionPlatformConnectionStatus(true);
          } else {
            this.sharedService.setCameraConnectionStatus(true);
          }

          // Resume the operation
          this.reconnectMessage = `Újracsatlakozás sikeres. Mérés folytatása...`;
          setTimeout(() => {
            this.reconnectMessage = null;
            resumeCallback();
          }, 500);
        },
        error: () => {
          // Reconnection attempt failed — timer will try again
          console.warn(`${deviceName} reconnect attempt failed (${elapsedS}s elapsed)`);
        }
      });
    }, RECONNECT_INTERVAL_MS);
  }

  private clearReconnectState(): void {
    if (this.reconnectTimer) {
      clearInterval(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    this.reconnecting = false;
  }

  private finishMeasurement(success: boolean): void {
    // Clean up reconnection state
    this.clearReconnectState();
    this.reconnectMessage = null;

    // Turn off all lights whenever measurement finishes (success or stopped)
    this.http.post(`${BASE_URL}/turn-off-all-lights`, {}).subscribe({
      next: () => {
        console.log('All lights turned off after measurement completion');
        // Emit event to notify other UI components that lights are off
        this.sharedService.emitLightsOff();
      },
      error: (err) => console.warn('Could not turn off lights after measurement:', err)
    });

    this.measurementActive = false;
    this.sharedService.setMeasurementActive(false);
    this.currentTabletId = null;
    
    if (this.stopRequested) {
      const failedNote = this.failedTablets.size > 0 ? ` ${this.failedTablets.size} hibás.` : '';
      this.successMessage = `Mérés leállítva. ${this.completedTablets.size} tabletta mérése kész.${failedNote}`;
    } else if (success) {
      const failedNote = this.failedTablets.size > 0 ? ` ${this.failedTablets.size} hibás.` : '';
      this.successMessage = `Mérés sikeresen befejezve. ${this.completedTablets.size} tabletta mérése kész.${failedNote}`;
      this.scheduleBedMoveToZero();
    }
    // Error message is set in processTabletQueue if there was an error
    
    this.stopRequested = false;
  }

  private openTabletImages(id: number): void {
    const paths = this.tabletImages.get(id);
    if (!paths || paths.length === 0) {
      return;
    }
    for (const path of paths) {
      this.http.post(`${BASE_URL}/open_image`, { path }).subscribe({
        error: (err) => console.error('Failed to open image:', err)
      });
    }
  }

  private scheduleBedMoveToZero(): void {
    if (!this.motionConnected) {
      return;
    }

    setTimeout(() => {
      if (!this.motionConnected) {
        return;
      }

      this.http.post(`${BASE_URL}/move_toolhead_absolute`, { z: 0 }).subscribe({
        next: () => console.log('Auto-measurement complete: moved bed to Z=0.'),
        error: (err) => console.warn('Failed to move bed to Z=0 after auto-measurement:', err)
      });
    }, 2000);
  }
}
