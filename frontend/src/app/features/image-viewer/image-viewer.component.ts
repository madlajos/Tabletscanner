import { CommonModule } from '@angular/common';
import { Component, ViewChild, ElementRef, AfterViewInit } from '@angular/core';
import { SharedService } from '../../shared.service';
import { HttpClient } from '@angular/common/http';

@Component({
  standalone: true,
  selector: 'app-image-viewer',
  templateUrl: './image-viewer.component.html',
  styleUrls: ['./image-viewer.component.css'],
  imports: [CommonModule]
})
export class ImageViewerComponent implements AfterViewInit {
  @ViewChild('mainVideoContainer', { static: false }) mainVideoContainer!: ElementRef<HTMLDivElement>;
  @ViewChild('sideVideoContainer', { static: false }) sideVideoContainer!: ElementRef<HTMLDivElement>;

  isMainStreaming: boolean = false;
  isSideStreaming: boolean = false;

  constructor(public http: HttpClient, public sharedService: SharedService) {}

  ngAfterViewInit(): void {
    this.sharedService.cameraStreamStatus$.subscribe(status => {
      this.isMainStreaming = status.main;
      this.isSideStreaming = status.side;

      // Add logs to confirm stream status changes
      console.log(`Main Camera Streaming: ${this.isMainStreaming}`);
      console.log(`Side Camera Streaming: ${this.isSideStreaming}`);

      // Update the stream display based on streaming status
      this.updateStreamDisplay('main');
      this.updateStreamDisplay('side');
    });
  }

  updateStreamDisplay(cameraType: 'main' | 'side'): void {
    const videoContainer = cameraType === 'main'
      ? this.mainVideoContainer.nativeElement
      : this.sideVideoContainer.nativeElement;
  
    // If you want to scale down the image, just add &scale=0.25 or similar
    const streamUrl = `http://localhost:5000/api/start-video-stream?type=${cameraType}&scale=0.25&ts=${Date.now()}`;
  
    let img = videoContainer.querySelector('img');
  
    if ((cameraType === 'main' && this.isMainStreaming) || (cameraType === 'side' && this.isSideStreaming)) {
      console.log(`Displaying ${cameraType} stream in UI.`);
  
      if (!img) {
        img = document.createElement('img');
        img.alt = `${cameraType} camera stream`;
        img.style.width = '100%';
        img.style.height = '100%';
        img.style.objectFit = 'contain';
        videoContainer.appendChild(img);
      }
  
      img.src = streamUrl;  // This is the key: the <img> loads the MJPEG response
      img.onload = () => console.log(`${cameraType} stream loaded.`);
      img.onerror = (err) => console.error(`Failed to load ${cameraType} stream.`, err);
  
    } else {
      // If the user or the system toggled streaming off, remove the <img>
      console.warn(`${cameraType} stream stopped. Clearing view.`);
      if (img) img.remove();
    }
  }
}
