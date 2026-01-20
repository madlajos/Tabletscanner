import {
  Component,
  ViewChild,
  ElementRef,
  AfterViewInit,
  OnDestroy
} from '@angular/core';
import {
  CommonModule,
  NgIf,
  NgForOf,
  NgClass
} from '@angular/common';
import { HttpClient } from '@angular/common/http';
import { Subscription } from 'rxjs';
import { MatIconModule } from '@angular/material/icon';
import { SharedService, SavedImageInfo } from '../../shared.service';
import { BASE_URL } from '../../api-config';

interface SavedImage {
  path: string;   // full-size image path on disk (used for openImage)
  url: string;    // thumbnail URL (served by backend)
}

@Component({
  standalone: true,
  selector: 'app-image-viewer',
  templateUrl: './image-viewer.component.html',
  styleUrls: ['./image-viewer.component.css'],
  imports: [CommonModule, NgIf, NgForOf, NgClass, MatIconModule]
})
export class ImageViewerComponent implements AfterViewInit, OnDestroy {
  @ViewChild('videoContainer', { static: false })
  videoContainer!: ElementRef<HTMLDivElement>;

  isStreaming = false;
  private streamSub?: Subscription;
  private savedImageSub?: Subscription;

  // Thumbnails of recently saved images
  savedImages: SavedImage[] = [];
  private readonly MAX_GALLERY_ITEMS = 8;  // smaller list = less work

  // Zoom and pan state
  zoomLevel = 1.0;
  panX = 0;
  panY = 0;
  private isDragging = false;
  private dragStart = { x: 0, y: 0 };
  private lastClickTime = 0;
  private readonly DOUBLE_CLICK_THRESHOLD = 300; // ms

  constructor(
    public http: HttpClient,
    public sharedService: SharedService
  ) { }

  ngAfterViewInit(): void {
    if (typeof (this.sharedService as any).getCameraStreamStatus === 'function') {
      this.isStreaming = (this.sharedService as any).getCameraStreamStatus();
      this.updateStreamDisplay();
    }

    const stream$ =
      (this.sharedService as any).isStreaming$ ||
      (this.sharedService as any).cameraStreamStatus$;

    if (stream$?.subscribe) {
      this.streamSub = stream$.subscribe((status: boolean) => {
        this.isStreaming = !!status;
        console.log(`Camera Streaming: ${this.isStreaming}`);
        this.updateStreamDisplay();
      });
    } else {
      console.warn(
        'No streaming observable found on SharedService (isStreaming$ / cameraStreamStatus$).'
      );
    }

    // Add zoom/pan event listeners to the video container
    if (this.videoContainer) {
      const container = this.videoContainer.nativeElement;
      container.addEventListener('wheel', (e) => this.onMouseWheel(e), false);
      container.addEventListener('mousedown', (e) => this.onMouseDown(e));
      container.addEventListener('mousemove', (e) => this.onMouseMove(e));
      container.addEventListener('mouseup', () => this.onMouseUp());
      container.addEventListener('mouseleave', () => this.onMouseUp());
      container.addEventListener('dblclick', () => this.resetZoomAndPan());
    }

    // Subscribe to saved images from auto-measurement
    this.savedImageSub = this.sharedService.newSavedImage$.subscribe(
      (imageInfo: SavedImageInfo) => {
        this.addImageToGallery(imageInfo.path);
      }
    );
  }

  /**
   * Add an image to the gallery given its file path.
   * Used by both manual capture and auto-measurement.
   */
  private addImageToGallery(imagePath: string): void {
    const img: SavedImage = {
      path: imagePath,
      url: `${BASE_URL}/get_thumbnail?path=${encodeURIComponent(imagePath)}`
    };

    this.savedImages.unshift(img);
    if (this.savedImages.length > this.MAX_GALLERY_ITEMS) {
      this.savedImages.pop();
    }
  }

  ngOnDestroy(): void {
    this.streamSub?.unsubscribe();
    this.savedImageSub?.unsubscribe();
  }

  updateStreamDisplay(): void {
    const container: HTMLElement | undefined = this.videoContainer?.nativeElement;
    if (!container) {
      console.warn('videoContainer not available.');
      return;
    }

    const streamUrl = `${BASE_URL}/start-video-stream?scale=0.25&ts=${Date.now()}`;
    let img = container.querySelector('img');

    if (this.isStreaming) {
      if (!img) {
        img = document.createElement('img');
        img.alt = 'camera stream';
        img.style.width = '100%';
        img.style.height = '100%';
        img.style.objectFit = 'contain';
        container.appendChild(img);
      }
      img.src = streamUrl;
      img.onload = () => console.log('Stream loaded.');
      img.onerror = (err) => console.error('Failed to load stream.', err);
    } else {
      if (img) {
        console.warn('Stream stopped. Clearing view.');
        img.remove();
      }
    }
  }

  captureImage(): void {
    const targetDir = (this.sharedService as any).getSaveDirectory
      ? (this.sharedService as any).getSaveDirectory()
      : null;

    if (!targetDir) {
      console.warn('No save directory set. Cannot capture image.');
      return;
    }

    // Fetch current "other_settings" from backend so we can embed metadata
    this.http.get<{ other_settings: any }>(`${BASE_URL}/get-other-settings?category=other_settings`)
      .subscribe({
        next: (resp) => {
          const metadata = resp?.other_settings || {};

          this.http.post<{
            message?: string;
            path?: string;
            error?: string;
          }>(
            `${BASE_URL}/save_raw_image`,
            { target_folder: targetDir, metadata }
          ).subscribe({
            next: (res) => {
              if (res?.path) {
                console.log(`Image saved to: ${res.path}`);
                this.addImageToGallery(res.path);
              } else if (res?.message) {
                console.log(`Save image response: ${res.message}`);
              } else {
                console.log('Save image request completed with no path.');
              }
            },
            error: (err) => {
              console.error('Failed to save image.', err);
            }
          });
        },
        error: (err) => {
          console.warn('Could not fetch other_settings; saving without metadata.', err);
          // fallback: save without metadata
          this.http.post(`${BASE_URL}/save_raw_image`, { target_folder: targetDir }).subscribe({
            next: () => console.log('Saved image without metadata'),
            error: (e) => console.error('Failed to save image.', e)
          });
        }
      });
  }


  openImage(img: SavedImage): void {
    this.http.post(
      `${BASE_URL}/open_image`,
      { path: img.path }
    ).subscribe({
      next: () => {
        console.log('Opened image:', img.path);
      },
      error: (err) => {
        console.error('Failed to open image.', err);
      }
    });
  }

  // For *ngFor to avoid re-rendering everything on each change
  trackByPath(index: number, img: SavedImage): string {
    return img.path;
  }

  // Zoom and pan event handlers
  private onMouseWheel(event: WheelEvent): void {
    // Only zoom if Ctrl key is held
    if (!event.ctrlKey) return;

    event.preventDefault();

    const zoomFactor = event.deltaY > 0 ? 0.9 : 1.1;
    const newZoom = this.zoomLevel * zoomFactor;

    // Clamp zoom between 1.0 (original) and 5.0 (max 5x zoom)
    this.zoomLevel = Math.max(1.0, Math.min(5.0, newZoom));
    this.applyTransform();
  }

  private onMouseDown(event: MouseEvent): void {
    // Record double-click for reset
    const now = Date.now();
    if (now - this.lastClickTime < this.DOUBLE_CLICK_THRESHOLD) {
      this.lastClickTime = 0;
      return; // Handled by dblclick event
    }
    this.lastClickTime = now;

    // Start dragging only if zoomed in
    if (this.zoomLevel > 1.0) {
      event.preventDefault(); // Prevent browser's default drag behavior
      this.isDragging = true;
      this.dragStart = { x: event.clientX, y: event.clientY };
    }
  }

  private onMouseMove(event: MouseEvent): void {
    if (!this.isDragging || this.zoomLevel <= 1.0) return;

    const container = this.videoContainer?.nativeElement;
    if (!container) return;

    // Calculate delta from initial drag position
    const deltaX = event.clientX - this.dragStart.x;
    const deltaY = event.clientY - this.dragStart.y;

    // Apply delta to current pan position
    this.panX += deltaX;
    this.panY += deltaY;

    // Update drag start for next move
    this.dragStart = { x: event.clientX, y: event.clientY };

    // Clamp pan to prevent dragging beyond image boundaries
    const containerWidth = container.clientWidth;
    const containerHeight = container.clientHeight;
    const maxPanX = (this.zoomLevel - 1) * containerWidth / 2;
    const maxPanY = (this.zoomLevel - 1) * containerHeight / 2;

    this.panX = Math.max(-maxPanX, Math.min(maxPanX, this.panX));
    this.panY = Math.max(-maxPanY, Math.min(maxPanY, this.panY));

    this.applyTransform();
  }

  private onMouseUp(): void {
    this.isDragging = false;
  }

  private resetZoomAndPan(): void {
    this.zoomLevel = 1.0;
    this.panX = 0;
    this.panY = 0;
    this.applyTransform();
  }

  private applyTransform(): void {
    const container = this.videoContainer?.nativeElement;
    if (!container) return;

    const img = container.querySelector('img');
    if (img) {
      const transform = `scale(${this.zoomLevel}) translate(${this.panX}px, ${this.panY}px)`;
      (img as HTMLImageElement).style.transform = transform;
      (img as HTMLImageElement).style.transformOrigin = 'center center';
      (img as HTMLImageElement).style.transition = this.isDragging ? 'none' : 'transform 0.2s ease-out';
    }
  }

  isZoomed(): boolean {
    return this.zoomLevel > 1.0;
  }

  getCursorStyle(): string {
    if (this.zoomLevel > 1.0) {
      return this.isDragging ? 'grabbing' : 'grab';
    }
    return 'default';
  }
}
