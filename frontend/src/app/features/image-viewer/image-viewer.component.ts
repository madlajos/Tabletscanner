import { CommonModule } from '@angular/common';
import { Component, ViewChild, ElementRef, AfterViewInit, OnDestroy } from '@angular/core';
import { SharedService } from '../../shared.service';
import { HttpClient } from '@angular/common/http';
import { Subscription } from 'rxjs';

@Component({
  standalone: true,
  selector: 'app-image-viewer',
  templateUrl: './image-viewer.component.html',
  styleUrls: ['./image-viewer.component.css'],
  imports: [CommonModule]
})
export class ImageViewerComponent implements AfterViewInit, OnDestroy {
  @ViewChild('videoContainer', { static: false }) videoContainer!: ElementRef<HTMLDivElement>;

  isStreaming = false;
  private streamSub?: Subscription;

  private readonly BASE_URL = 'http://localhost:5000/api';

  constructor(public http: HttpClient, public sharedService: SharedService) {}

  ngAfterViewInit(): void {
    // 1) Initialize from the service getter so the UI reflects current state immediately
    if (typeof (this.sharedService as any).getCameraStreamStatus === 'function') {
      this.isStreaming = (this.sharedService as any).getCameraStreamStatus();
      this.updateStreamDisplay();
    }

    // 2) Subscribe to streaming changes (prefer isStreaming$, else fall back to cameraStreamStatus$)
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
      console.warn('No streaming observable found on SharedService (isStreaming$ / cameraStreamStatus$).');
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
      img.src = streamUrl; // <img> loads the MJPEG
      img.onload = () => console.log('Stream loaded.');
      img.onerror = (err) => console.error('Failed to load stream.', err);
    } else {
      if (img) {
        console.warn('Stream stopped. Clearing view.');
        img.remove();
      }
    }
  }
}
