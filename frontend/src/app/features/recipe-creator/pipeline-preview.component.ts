import { Component, OnInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Subscription } from 'rxjs';
import { PipelineStateService } from '../../services/pipeline-state.service';

@Component({
  selector: 'app-pipeline-preview',
  standalone: true,
  imports: [CommonModule],
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
      max-height: 100%;
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
  `],
})
export class PipelinePreviewComponent implements OnInit, OnDestroy {
  imageSrc: string | null = null;
  loading = false;

  private subs: Subscription[] = [];

  constructor(private pipelineState: PipelineStateService) {}

  ngOnInit(): void {
    this.subs.push(
      this.pipelineState.previewImage$.subscribe((img) => (this.imageSrc = img)),
      this.pipelineState.previewLoading$.subscribe((l) => (this.loading = l))
    );
  }

  ngOnDestroy(): void {
    this.subs.forEach((s) => s.unsubscribe());
  }
}
