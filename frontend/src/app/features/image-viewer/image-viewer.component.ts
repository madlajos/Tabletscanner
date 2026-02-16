import {
  Component,
  ViewChild,
  ElementRef,
  AfterViewInit,
  OnDestroy,
  OnInit
} from '@angular/core';
import {
  CommonModule,
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
  imports: [CommonModule, NgClass, MatIconModule]
})
export class ImageViewerComponent implements OnInit, AfterViewInit, OnDestroy {
  @ViewChild('videoContainer', { static: false })
  videoContainer!: ElementRef<HTMLDivElement>;

  isStreaming = false;
  private streamSub?: Subscription;
  private savedImageSub?: Subscription;
  private saveDirSub?: Subscription;

  saveLocationValid = false;
  private saveDirectory = '';
  private currentLight: 'dome' | 'bar' | null = null;

  // Thumbnails of recently saved images
  savedImages: SavedImage[] = [];
  private readonly MAX_GALLERY_ITEMS = 8;  // smaller list = less work

  // Zoom and pan state
  zoomLevel = 1.0;
  panX = 0;
  panY = 0;
  private isDragging = false;
  private dragStart = { x: 0, y: 0, scrollLeft: 0, scrollTop: 0 };
  private lastClickTime = 0;
  private readonly DOUBLE_CLICK_THRESHOLD = 300; // ms

  // Capture button cooldown
  isCaptureCooldown = false;
  private readonly CAPTURE_COOLDOWN_MS = 1000;

  // Live view display toggle (stream keeps running, just hides the view)
  liveViewVisible = true;

  // Context menu state
  contextMenuVisible = false;
  contextMenuX = 0;
  contextMenuY = 0;
  contextMenuImage: SavedImage | null = null;

  constructor(
    public http: HttpClient,
    public sharedService: SharedService
  ) { }

  ngOnInit(): void {
    const saveDir$ = (this.sharedService as any).saveDirectory$;
    if (saveDir$?.subscribe) {
      this.saveDirSub = saveDir$.subscribe((dir: string) => {
        this.saveDirectory = (dir || '').trim();
        this.validateSaveDirectory();
      });
    } else {
      this.saveDirectory = (this.sharedService.getSaveDirectory() || '').trim();
      this.validateSaveDirectory();
    }

    // Subscribe to active light changes
    const lightSettings$ = (this.sharedService as any).lightSettings$;
    if (lightSettings$?.subscribe) {
      lightSettings$.subscribe((light: 'dome' | 'bar' | null) => {
        this.currentLight = light;
      });
    }
  }

  ngAfterViewInit(): void {
    // Hide context menu on any click outside
    document.addEventListener('click', () => this.hideContextMenu());

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
    this.saveDirSub?.unsubscribe();
  }

  private validateSaveDirectory(): void {
    if (!this.saveDirectory) {
      this.saveLocationValid = false;
      return;
    }

    this.http.post<{ exists: boolean }>(`${BASE_URL}/check-folder-exists`, {
      path: this.saveDirectory
    }).subscribe({
      next: (res) => {
        this.saveLocationValid = !!res?.exists;
      },
      error: () => {
        this.saveLocationValid = false;
      }
    });
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
    if (this.isCaptureCooldown) {
      return;
    }

    if (!this.saveLocationValid) {
      console.warn('Save location is invalid. Cannot capture image.');
      return;
    }

    // Start cooldown
    this.isCaptureCooldown = true;
    setTimeout(() => {
      this.isCaptureCooldown = false;
    }, this.CAPTURE_COOLDOWN_MS);

    const targetDir = (this.sharedService as any).getSaveDirectory
      ? (this.sharedService as any).getSaveDirectory()
      : null;

    if (!targetDir) {
      console.warn('No save directory set. Cannot capture image.');
      return;
    }

    // Fetch current "other_settings" and camera settings from backend so we can embed metadata
    this.http.get<{ other_settings: any }>(`${BASE_URL}/get-other-settings?category=other_settings`)
      .subscribe({
        next: (resp) => {
          const metadata = resp?.other_settings || {};

          const activeLight = this.currentLight || 'dome';

          // Also fetch camera settings to include in metadata
          this.http.get<any>(`${BASE_URL}/get-camera-settings`)
            .subscribe({
              next: (camResp) => {
                // Add only the currently active light's camera settings to metadata
                const category = `camera_params_${activeLight}`;
                if (camResp?.[category]) {
                  metadata[`exposure_time`] = camResp[category]['ExposureTime'];
                  metadata[`gamma`] = camResp[category]['Gamma'];
                }

                this.http.post<{
                  message?: string;
                  path?: string;
                  error?: string;
                }>(
                  `${BASE_URL}/save_raw_image`,
                  { target_folder: targetDir, metadata, light_type: activeLight }
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
                console.warn('Could not fetch camera settings; saving with other_settings only.', err);
                // Fallback: save with only other_settings
                this.http.post<{ path?: string; message?: string }>(
                  `${BASE_URL}/save_raw_image`,
                  { target_folder: targetDir, metadata, light_type: activeLight }
                ).subscribe({
                  next: (res) => {
                    if (res?.path) {
                      console.log(`Image saved to: ${res.path}`);
                      this.addImageToGallery(res.path);
                    }
                  },
                  error: (e) => console.error('Failed to save image.', e)
                });
              }
            });
        },
        error: (err) => {
          console.warn('Could not fetch other_settings; saving without metadata.', err);
          // fallback: save without metadata
          this.http.post(`${BASE_URL}/save_raw_image`, { target_folder: targetDir, light_type: this.currentLight || 'dome' }).subscribe({
            next: () => console.log('Saved image without metadata'),
            error: (e) => console.error('Failed to save image.', e)
          });
        }
      });
  }


  openImage(img: SavedImage): void {
    this.verifyFileExists(img.path).then((exists: boolean) => {
      if (!exists) {
        console.warn('Image no longer exists. Removing from gallery:', img.path);
        this.removeImageFromGallery(img.path);
        return;
      }

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
    });
  }

  clearGallery(): void {
    this.savedImages = [];
    this.hideContextMenu();
  }

  private removeImageFromGallery(imagePath: string): void {
    this.savedImages = this.savedImages.filter(img => img.path !== imagePath);
    if (this.contextMenuImage?.path === imagePath) {
      this.hideContextMenu();
    }
  }

  private verifyFileExists(path: string): Promise<boolean> {
    return this.http.post<{ exists: boolean }>(`${BASE_URL}/check-file-exists`, { path })
      .toPromise()
      .then(res => !!res?.exists)
      .catch(() => false);
  }

  onImageContextMenu(event: MouseEvent, img: SavedImage): void {
    event.preventDefault();
    
    // Estimated menu dimensions
    const menuWidth = 180;
    const menuHeight = 120;
    
    // Get viewport dimensions
    const viewportWidth = window.innerWidth;
    const viewportHeight = window.innerHeight;
    
    // Calculate position, adjusting if menu would go off-screen
    let menuX = event.clientX;
    let menuY = event.clientY;
    
    // Adjust horizontal position if menu would overflow right edge
    if (menuX + menuWidth > viewportWidth) {
      menuX = viewportWidth - menuWidth - 5; // 5px padding from edge
    }
    
    // Adjust vertical position if menu would overflow bottom edge
    if (menuY + menuHeight > viewportHeight) {
      menuY = viewportHeight - menuHeight - 5; // 5px padding from edge
    }
    
    // Ensure menu doesn't go off left/top edges
    menuX = Math.max(5, menuX);
    menuY = Math.max(5, menuY);
    
    this.verifyFileExists(img.path).then((exists: boolean) => {
      if (!exists) {
        console.warn('Image no longer exists. Removing from gallery:', img.path);
        this.removeImageFromGallery(img.path);
        return;
      }

      this.contextMenuVisible = true;
      this.contextMenuX = menuX;
      this.contextMenuY = menuY;
      this.contextMenuImage = img;
    });
  }

  hideContextMenu(): void {
    this.contextMenuVisible = false;
    this.contextMenuImage = null;
  }

  contextMenuOpen(): void {
    if (this.contextMenuImage) {
      this.openImage(this.contextMenuImage);
    }
    this.hideContextMenu();
  }

  contextMenuOpenFolder(): void {
    if (this.contextMenuImage) {
      const path = this.contextMenuImage.path;
      this.verifyFileExists(path).then((exists: boolean) => {
        if (!exists) {
          console.warn('Image no longer exists. Removing from gallery:', path);
          this.removeImageFromGallery(path);
          return;
        }

        this.http.post(
          `${BASE_URL}/open_folder`,
          { path }
        ).subscribe({
          next: () => {
            console.log('Opened folder for:', path);
          },
          error: (err) => {
            console.error('Failed to open folder.', err);
          }
        });
      });
    }
    this.hideContextMenu();
  }

  contextMenuDelete(): void {
    const img = this.contextMenuImage;
    if (!img) {
      this.hideContextMenu();
      return;
    }

    const path = img.path;
    const confirmed = window.confirm('Biztosan törli a képet?');
    if (!confirmed) {
      this.hideContextMenu();
      return;
    }

    this.verifyFileExists(path).then((exists: boolean) => {
      if (!exists) {
        console.warn('Image no longer exists. Removing from gallery:', path);
        this.removeImageFromGallery(path);
        return;
      }

      this.http.post(`${BASE_URL}/delete-image`, { path }).subscribe({
        next: () => {
          console.log('Deleted image:', path);
          this.removeImageFromGallery(path);
        },
        error: (err) => {
          console.error('Failed to delete image.', err);
        }
      });
    });

    this.hideContextMenu();
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
    if (this.zoomLevel > 1.0 && this.videoContainer) {
      event.preventDefault(); // Prevent browser's default drag behavior
      const container = this.videoContainer.nativeElement;
      this.isDragging = true;
      this.dragStart = {
        x: event.clientX,
        y: event.clientY,
        scrollLeft: container.scrollLeft,
        scrollTop: container.scrollTop
      };
    }
  }

  private onMouseMove(event: MouseEvent): void {
    if (!this.isDragging || this.zoomLevel <= 1.0) return;

    const container = this.videoContainer?.nativeElement;
    if (!container) return;

    // Calculate delta from initial drag position
    const deltaX = event.clientX - this.dragStart.x;
    const deltaY = event.clientY - this.dragStart.y;

    // Scroll the container instead of translating the image
    container.scrollLeft = this.dragStart.scrollLeft - deltaX;
    container.scrollTop = this.dragStart.scrollTop - deltaY;
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

    const img = container.querySelector('img') as HTMLImageElement | null;
    if (img) {
      const scalePercent = this.zoomLevel * 100;
      img.style.width = `${scalePercent}%`;
      img.style.height = `${scalePercent}%`;
      img.style.transform = 'none';
      img.style.transition = this.isDragging ? 'none' : 'transform 0.1s ease-out';

      // Reset scroll when returning to 1x
      if (this.zoomLevel === 1.0) {
        container.scrollLeft = 0;
        container.scrollTop = 0;
      }
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
