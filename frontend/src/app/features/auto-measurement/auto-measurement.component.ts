import { Component, signal, Input } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import {
  AutoMeasurementService,
  AutoMeasurementRequest,
  AutoMeasurementResponse
} from '../../services/auto-measurement.service';

@Component({
  selector: 'app-auto-measurement',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './auto-measurement.component.html',
  styleUrls: ['./auto-measurement.component.css']
})
export class AutoMeasurementComponent {

  readonly gridSize = 10; // change to 9 later if needed

  // External connection status (parent can bind real values)
  @Input() cameraConnected = true;
  @Input() motionConnected = true;

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

  // Measurement name
  measurementName = '';

  // Set of selected tablet IDs (supports multiple ranges + gaps)
  private selectedSignal = signal<Set<number>>(new Set<number>());

  // Range anchor for adding ranges (first click)
  private rangeAnchorSignal = signal<number | null>(null);

  // Hover index for range preview
  private hoverIndexSignal = signal<number | null>(null);

  // Is measurement currently running (toggled "on")
  measurementActive = false;

  errorMessage: string | null = null;
  successMessage: string | null = null;

  constructor(private autoService: AutoMeasurementService) {}

  // ===== Derived properties =====

  get selectedCount(): number {
    return this.selectedSignal().size;
  }

  hasSelection(): boolean {
    return this.selectedSignal().size > 0;
  }

  // Start/Stop button enabled state
  // - when OFF: require selection + both devices connected
  // - when ON: always allow clicking to stop
  canStart(): boolean {
    if (this.measurementActive) {
      return true; // allow stopping
    }
    const hasSelected = this.selectedSignal().size > 0;
    const connected = this.cameraConnected && this.motionConnected;
    return hasSelected && connected;
  }

  // ===== CLICK SELECTION LOGIC =====
  //
  // Click behavior:
  // - No anchor:
  //    - if id already selected → toggle OFF
  //    - else → select id, set as anchor
  // - With anchor:
  //    - if clicking anchor again → cancel range mode
  //    - else → add [anchor, id] to selection, clear anchor

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

  // Dots considered "selected" if:
  // - truly selected, OR
  // - part of the current preview range [anchor, hover]
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

  clearSelection(): void {
    if (this.measurementActive) {
      return;
    }
    this.selectedSignal.set(new Set<number>());
    this.rangeAnchorSignal.set(null);
    this.hoverIndexSignal.set(null);
    this.errorMessage = null;
    this.successMessage = null;
  }

  // ===== Start / Stop measurement =====

  startMeasurement(): void {
    // Toggle behavior
    if (this.measurementActive) {
      // TODO: call backend stop endpoint when available
      this.measurementActive = false;
      this.successMessage = 'Mérés leállítva.';
      return;
    }

    const indices = Array.from(this.selectedSignal()).sort((a, b) => a - b);
    if (indices.length === 0) {
      return;
    }

    this.errorMessage = null;
    this.successMessage = null;
    this.measurementActive = true;

    const req: AutoMeasurementRequest = {
      indices,
      grid_size: this.gridSize,
      autofocus: this.autofocus,
      lamp_top: this.lampTop,
      lamp_side: this.lampSide,
      // measurementName could be added to the backend payload later:
      // name: this.measurementName
    };

    this.autoService.startMeasurement(req).subscribe({
      next: (resp: AutoMeasurementResponse) => {
        if (resp.status === 'success') {
          this.successMessage = resp.message ?? 'Mérés sikeresen lefutott.';
        } else {
          this.errorMessage = resp.message ?? 'Hiba történt az automata mérés közben.';
        }
      },
      error: (err) => {
        this.errorMessage = err?.error?.message ?? 'Szerver hiba az automata mérés közben.';
      },
      complete: () => {
        this.measurementActive = false;
      }
    });
  }
}
