import { Component, ViewChild, ChangeDetectorRef, OnInit, AfterViewInit } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Lightbox } from 'ngx-lightbox';
import { MatSidenav } from '@angular/material/sidenav';
import { CommonModule } from '@angular/common';
import { MatSidenavModule } from '@angular/material/sidenav';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatExpansionModule } from '@angular/material/expansion';
import { FormsModule } from '@angular/forms';

// Import standalone components
import { ImageViewerComponent } from './features/image-viewer/image-viewer.component';
import { ComportControlComponent } from './features/comport-control/comport-control.component';
import { CameraControlComponent } from './features/camera-control/camera-control.component';
import { ErrorPopupListComponent } from './components/error-popup-list/error-popup-list.component';
import { BackendReadyService } from './services/backend-ready.service'; // Adjust the path as needed
import { MotionControl } from './features/motion-control/motion-control';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [
    CommonModule, 
    MatSidenavModule, 
    MatButtonModule, 
    MatIconModule, 
    MatExpansionModule,
    FormsModule,

    // Standalone components
    ImageViewerComponent,
    MotionControl,
    ComportControlComponent,
    CameraControlComponent,
    ErrorPopupListComponent
  ],
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent implements OnInit, AfterViewInit {
  backendReady = false;
  title = 'Untitled';  
  @ViewChild('sidenav') sidenav!: MatSidenav;

  constructor(
    private http: HttpClient,
    private backendReadyService: BackendReadyService,
    private lightbox: Lightbox,
    private cdRef: ChangeDetectorRef,
  ) {}

  async ngOnInit(): Promise<void> {
    // Wait for the backend readiness check.
    try {
      await this.backendReadyService.waitForBackendReady();
      console.log("Backend is ready.");
      this.backendReady = true;
      this.cdRef.detectChanges();
    } catch (error) {
      console.error("Error waiting for backend readiness:", error);
      
      this.backendReady = true;
    }
  }

  ngAfterViewInit(): void {
    // Remove the splash overlay once the view is initialized.
    const overlay = document.getElementById('splash-overlay');
    if (overlay) {
      overlay.style.transition = 'opacity 0.5s ease-out';
      overlay.style.opacity = '0';
      setTimeout(() => overlay.remove(), 500);
    }
  }

  toggleSettingsPanel(): void {
    this.sidenav.toggle();
  }
}