import { Component, signal } from '@angular/core';
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

  // Tablet IDs with bottom-left = 1, top-right = gridSize^2
  // Visual grid is still top row first, left to right,
  // but IDs are numbered from bottom-left upwards.
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

  // Set of selected tablet IDs (multi-range + holes)
  private selectedSignal = signal<Set<number>>(new Set<number>());

  // Anchor for range-adding mode (first click)
  private rangeAnchorSignal = signal<number | null>(null);

  // Hover index used only for preview of the range from anchor
  private hoverIndexSignal = signal<number | null>(null);

  isRunning = false;
  errorMessage: string | null = null;
  successMessage: string | null = null;

  constructor(private autoService: AutoMeasurementService) {}

  // ==== CLICK SELECTION LOGIC ====

  /**
   * Click semantics:
   * - If no anchor:
   *    - if id is already selected -> toggle OFF (remove from measurement)
   *    - else -> select it and set it as range anchor (so next click defines a range)
   * - If anchor exists:
   *    - if clicking the same anchor again -> just clear anchor (no change)
   *    - else -> add the full range [anchor, id] to the selection and clear anchor
   */
  onDotClick(id: number): void {
    const selected = new Set(this.selectedSignal());
    const anchor = this.rangeAnchorSignal();

    // No active range anchor yet
    if (anchor === null) {
      if (selected.has(id)) {
        // Toggle off single tablet (e.g. click 16 to deselect from 11–21)
        selected.delete(id);
        this.selectedSignal.set(selected);
        this.hoverIndexSignal.set(null);
      } else {
        // Start a new range and immediately select this tablet
        // (e.g. click 26 -> 26 is added; anchor = 26)
        selected.add(id);
        this.selectedSignal.set(selected);
        this.rangeAnchorSignal.set(id);
        this.hoverIndexSignal.set(null);
      }
      return;
    }

    // There is an active anchor
    if (id === anchor) {
      // Second click on the same anchor: cancel range mode
      this.rangeAnchorSignal.set(null);
      this.hoverIndexSignal.set(null);
      return;
    }

    // Add whole range [anchor, id] to selection (e.g. 26–36)
    const start = Math.min(anchor, id);
    const end = Math.max(anchor, id);
    for (let v = start; v <= end; v++) {
      selected.add(v);
    }

    this.selectedSignal.set(selected);
    this.rangeAnchorSignal.set(null);
    this.hoverIndexSignal.set(null);
  }

  // Hover preview: show would-be range [anchor, hover]
  onDotMouseEnter(id: number): void {
    if (this.rangeAnchorSignal() !== null) {
      this.hoverIndexSignal.set(id);
    } else {
      this.hoverIndexSignal.set(null);
    }
  }

  onGridMouseLeave(): void {
    this.hoverIndexSignal.set(null);
  }

  // ==== SELECTION QUERY (for CSS class .selected) ====

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

  // ==== BUTTON STATE / ACTION ====

  canStart(): boolean {
    return this.selectedSignal().size > 0 && !this.isRunning;
  }

   hasSelection(): boolean {
    // either something is selected or a range is being prepared
    return this.selectedSignal().size > 0 || this.rangeAnchorSignal() !== null;
  }

   clearSelection(): void {
    this.selectedSignal.set(new Set<number>());
    this.rangeAnchorSignal.set(null);
    this.hoverIndexSignal.set(null);
    this.errorMessage = null;
    this.successMessage = null;
  }

  startMeasurement(): void {
    const indices = Array.from(this.selectedSignal()).sort((a, b) => a - b);
    if (indices.length === 0) {
      return;
    }

    this.errorMessage = null;
    this.successMessage = null;
    this.isRunning = true;

    const req: AutoMeasurementRequest = {
      indices,
      grid_size: this.gridSize,
      autofocus: this.autofocus,
      lamp_top: this.lampTop,
      lamp_side: this.lampSide
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
        this.isRunning = false;
      }
    });
  }
}
