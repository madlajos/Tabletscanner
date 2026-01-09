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
import { SharedService } from '../../shared.service';

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

  // Thumbnails of recently saved images
  savedImages: SavedImage[] = [];
  private readonly MAX_GALLERY_ITEMS = 8;  // smaller list = less work

  private readonly BASE_URL = 'http://localhost:5000/api';

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
  }

  ngOnDestroy(): void {
    this.streamSub?.unsubscribe();
  }

  updateStreamDisplay(): void {
    const container: HTMLElement | undefined = this.videoContainer?.nativeElement;
    if (!container) {
      console.warn('videoContainer not available.');
      return;
    }

    const streamUrl = `${this.BASE_URL}/start-video-stream?scale=0.25&ts=${Date.now()}`;
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
    this.http.get<{ other_settings: any }>(`${this.BASE_URL}/get-other-settings?category=other_settings`)
      .subscribe({
        next: (resp) => {
          const metadata = resp?.other_settings || {};

          this.http.post<{
            message?: string;
            path?: string;
            error?: string;
          }>(
            `${this.BASE_URL}/save_raw_image`,
            { target_folder: targetDir, metadata }
          ).subscribe({
            next: (res) => {
              if (res?.path) {
                console.log(`Image saved to: ${res.path}`);

                const img: SavedImage = {
                  path: res.path,
                  // use dynamic thumbnail endpoint; no thumb file on disk
                  url: `${this.BASE_URL}/get_thumbnail?path=${encodeURIComponent(res.path)}`
                };

                this.savedImages.unshift(img);
                if (this.savedImages.length > this.MAX_GALLERY_ITEMS) {
                  this.savedImages.pop();
                }
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
          this.http.post(`${this.BASE_URL}/save_raw_image`, { target_folder: targetDir }).subscribe({
            next: () => console.log('Saved image without metadata'),
            error: (e) => console.error('Failed to save image.', e)
          });
        }
      });
  }


  openImage(img: SavedImage): void {
    this.http.post(
      `${this.BASE_URL}/open_image`,
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
}
