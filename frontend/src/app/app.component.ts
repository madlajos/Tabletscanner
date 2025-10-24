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
import { ControlPanelComponent } from './features/control-panel/control-panel.component';
import { TurntableControlComponent } from './features/turntable-control/turntable-control.component';
import { ComportControlComponent } from './features/comport-control/comport-control.component';
import { CameraControlComponent } from './features/camera-control/camera-control.component';
import { ResultsTableComponent } from './features/results-table/results-table.component';
import { ErrorPopupListComponent } from './components/error-popup-list/error-popup-list.component';
import { BarcodeScannerControlComponent } from './features/barcode-scanner-control/barcode-scanner-control.component';
import { LoginPopupComponent } from './components/login-popup/login-popup.component';
import { SQLDatabaseComponent } from './components/sql-database/sql-database.component';
import { BackendReadyService } from './services/backend-ready.service'; // Adjust the path as needed

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
    ControlPanelComponent,
    TurntableControlComponent,
    ComportControlComponent,
    CameraControlComponent,
    ResultsTableComponent,
    ErrorPopupListComponent,
    BarcodeScannerControlComponent,
    LoginPopupComponent,
    SQLDatabaseComponent
  ],
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent implements OnInit, AfterViewInit {
  backendReady = false;
  title = 'Untitled';  
  @ViewChild('sidenav') sidenav!: MatSidenav;

  isAuthenticated = false;
  isLoginPopupVisible = false;
  loginError = '';

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
    this.isAuthenticated = sessionStorage.getItem('isAuthenticated') === 'true';
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
    if (this.isAuthenticated) {
      this.sidenav.toggle();
    } else {
      this.isLoginPopupVisible = true;
    }
  }

  hideLoginPopup(): void {
    this.isLoginPopupVisible = false;
  }

  onLoginSuccess(credentials: { username: string; password: string }): void {
    // You might process credentials if necessary.
    this.isAuthenticated = true;
    sessionStorage.setItem('isAuthenticated', 'true');
    this.sidenav.open();
  }

  logout(): void {
    this.isAuthenticated = false;
    sessionStorage.removeItem('isAuthenticated');
    this.sidenav.close();
  }
}