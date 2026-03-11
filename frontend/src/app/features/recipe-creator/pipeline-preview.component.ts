import { Component, OnInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Subscription } from 'rxjs';
import { PipelineStateService } from '../../services/pipeline-state.service';

@Component({
  selector: 'app-pipeline-preview',
  standalone: true,
  imports: [CommonModule, FormsModule],
  template: `
    <div class="preview-wrapper">
      @if (loading) {
        <div class="loading-overlay">
          <div class="spinner"></div>
          <span>Előnézet betöltése...</span>
        </div>
      }
      @if (imageSrc) {
        <img
          [src]="imageSrc"
          alt="Pipeline előnézet"
          class="preview-image"
          draggable="false"
        />
      } @else if (!loading) {
        <div class="no-preview">
          <div class="no-preview-icon">🖼</div>
          <span>Nincs előnézet</span>
          <span class="no-preview-hint">Adjon hozzá lépéseket és válasszon képet</span>
        </div>
      }

      @if (imageCount > 1) {
        <div class="pagination-bar">
          <button
            class="page-btn"
            (click)="prevImage()"
            [disabled]="currentIndex <= 0"
            title="Előző kép"
          >◀</button>
          <div class="page-indicator">
            <input
              type="number"
              class="page-input"
              [ngModel]="currentIndex + 1"
              (ngModelChange)="goToImage($event)"
              [min]="1"
              [max]="imageCount"
            />
            <span class="page-total">/ {{ imageCount }}</span>
          </div>
          <button
            class="page-btn"
            (click)="nextImage()"
            [disabled]="currentIndex >= imageCount - 1"
            title="Következő kép"
          >▶</button>
        </div>
      }
    </div>
  `,
  styles: [`
    :host {
      display: block;
      height: 100%;
      overflow: hidden;
    }

    .preview-wrapper {
      position: relative;
      width: 100%;
      height: 100%;
      display: flex;
      align-items: center;
      justify-content: center;
      background: #1a1a1a;
      border-radius: 4px;
    }

    .loading-overlay {
      position: absolute;
      inset: 0;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 12px;
      background: rgba(26, 26, 26, 0.85);
      z-index: 1;
      color: #999;
      font-size: 12px;
    }

    .spinner {
      width: 28px;
      height: 28px;
      border: 3px solid #333;
      border-top-color: #3b82f6;
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
    }

    @keyframes spin {
      to { transform: rotate(360deg); }
    }

    .preview-image {
      max-width: 100%;
      max-height: calc(100% - 40px);
      object-fit: contain;
      user-select: none;
    }

    .no-preview {
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 8px;
      color: #666;
      font-size: 13px;
    }

    .no-preview-icon {
      font-size: 36px;
      opacity: 0.4;
    }

    .no-preview-hint {
      font-size: 11px;
      color: #555;
    }

    /* Pagination bar */
    .pagination-bar {
      position: absolute;
      bottom: 0;
      left: 0;
      right: 0;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      padding: 6px 12px;
      background: rgba(30, 30, 30, 0.9);
      border-top: 1px solid #444;
    }

    .page-btn {
      background: #333;
      border: 1px solid #555;
      border-radius: 4px;
      color: #e0e0e0;
      cursor: pointer;
      padding: 4px 10px;
      font-size: 12px;
      line-height: 1;
    }

    .page-btn:hover:not(:disabled) {
      background: #3b82f6;
      border-color: #3b82f6;
    }

    .page-btn:disabled {
      opacity: 0.3;
      cursor: default;
    }

    .page-indicator {
      display: flex;
      align-items: center;
      gap: 4px;
    }

    .page-input {
      width: 44px;
      padding: 3px 4px;
      background: #2a2a2a;
      border: 1px solid #555;
      border-radius: 4px;
      color: #e0e0e0;
      font-size: 12px;
      text-align: center;
      -moz-appearance: textfield;
    }

    .page-input::-webkit-outer-spin-button,
    .page-input::-webkit-inner-spin-button {
      -webkit-appearance: none;
      margin: 0;
    }

    .page-total {
      color: #888;
      font-size: 12px;
      font-variant-numeric: tabular-nums;
    }
  `],
})
export class PipelinePreviewComponent implements OnInit, OnDestroy {
  imageSrc: string | null = null;
  loading = false;
  imageCount = 0;
  currentIndex = 0;

  private subs: Subscription[] = [];

  constructor(private pipelineState: PipelineStateService) {}

  ngOnInit(): void {
    this.subs.push(
      this.pipelineState.previewImage$.subscribe((img) => (this.imageSrc = img)),
      this.pipelineState.previewLoading$.subscribe((l) => (this.loading = l)),
      this.pipelineState.imageCount$.subscribe((c) => (this.imageCount = c)),
      this.pipelineState.previewImageIndex$.subscribe((i) => (this.currentIndex = i)),
    );
  }

  ngOnDestroy(): void {
    this.subs.forEach((s) => s.unsubscribe());
  }

  prevImage(): void {
    this.pipelineState.setPreviewImageIndex(this.currentIndex - 1);
  }

  nextImage(): void {
    this.pipelineState.setPreviewImageIndex(this.currentIndex + 1);
  }

  goToImage(oneBasedIndex: number): void {
    const idx = Math.round(oneBasedIndex) - 1;
    this.pipelineState.setPreviewImageIndex(idx);
  }
}
