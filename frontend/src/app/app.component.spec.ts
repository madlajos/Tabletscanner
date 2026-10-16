import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';
import { Lightbox } from 'ngx-lightbox';
import { AppComponent } from './app.component';
import { BackendReadyService } from './services/backend-ready.service';
import { InlineAnalysisService } from './services/inline-analysis.service';
import { LightStatusService } from './services/light-status.service';
import { DoorInterlockStatusComponent } from './components/door-interlock-status/door-interlock-status.component';
import { of } from 'rxjs';
import { SharedService } from './shared.service';

@Component({
  selector: 'app-motion-control, app-camera-control, app-image-viewer, app-auto-measurement, app-recipe-creator, app-recipe-applier, app-software-settings, mat-icon',
  standalone: true, template: '',
})
class FeatureStub {}

describe('Application navigation during inline analysis', () => {
  beforeEach(() => TestBed.configureTestingModule({
    imports: [AppComponent],
    providers: [provideHttpClient(), provideHttpClientTesting(),
      { provide: Lightbox, useValue: {} },
      { provide: LightStatusService, useValue: { status$: of({ door_interlock: {
        known: true, front_open: true, top_open: false, uv_blocked: true, trip_count: 1,
      } }) } },
      { provide: BackendReadyService, useValue: { waitForBackendReady: () => Promise.resolve() } }],
  }).overrideComponent(AppComponent, { set: { imports: [CommonModule, FeatureStub, DoorInterlockStatusComponent] } }));

  it('keeps the scanner subtree mounted when results are opened', () => {
    const fixture = TestBed.createComponent(AppComponent);
    TestBed.inject(SharedService).setMotionPlatformConnectionStatus(true);
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('.app-nav .door-badge')?.textContent).toContain('Ajtó nyitva');
    const scanner = fixture.nativeElement.querySelector('app-auto-measurement');
    fixture.componentInstance.switchSection('applier');
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('app-auto-measurement')).toBe(scanner);
    expect(fixture.nativeElement.querySelector('.app-grid.section-inactive')).not.toBeNull();
    expect(fixture.nativeElement.querySelector('app-recipe-applier')).not.toBeNull();
    fixture.destroy();
  });

  it('marks analysis failures in navigation and acknowledges them when results open', () => {
    const fixture = TestBed.createComponent(AppComponent);
    fixture.detectChanges();
    const analysis = TestBed.inject(InlineAnalysisService);
    analysis.reportError({ code: 'E4503', error: 'Elemzési teszthiba' });
    fixture.detectChanges();
    const tab = fixture.nativeElement.querySelector('.nav-tab.analysis-error') as HTMLButtonElement;
    expect(tab).not.toBeNull();
    expect(tab.querySelector('.analysis-notification')?.textContent).toBe('!');
    tab.click();
    fixture.detectChanges();
    expect(fixture.componentInstance.activeSection).toBe('applier');
    expect(fixture.nativeElement.querySelector('.nav-tab.analysis-error')).toBeNull();
    fixture.destroy();
  });
});
