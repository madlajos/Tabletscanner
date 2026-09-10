import { TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { FilterCaptureButtonsComponent } from './filter-capture-buttons.component';
import { SharedService } from '../../shared.service';
import { BASE_URL } from '../../api-config';

describe('Filter capture buttons', () => {
  beforeEach(() => TestBed.configureTestingModule({
    imports: [FilterCaptureButtonsComponent], providers: [provideHttpClient(), provideHttpClientTesting()]
  }));

  it('renders a seamless wavelength selector and keeps cancellation available during capture', () => {
    const fixture = TestBed.createComponent(FilterCaptureButtonsComponent);
    const shared = TestBed.inject(SharedService);
    const http = TestBed.inject(HttpTestingController);
    shared.setCameraConnectionStatus(true);
    shared.setSaveDirectory('C:/captures/sample');
    fixture.componentInstance.disabled = true;
    fixture.detectChanges();
    const buttons = fixture.nativeElement.querySelectorAll('button') as NodeListOf<HTMLButtonElement>;
    expect(Array.from(buttons).map(button => button.textContent?.trim())).toEqual(['255', '310', '365', 'VIS', 'RGB+UV']);
    expect(Array.from(buttons).every(button => button.disabled)).toBeTrue();
    fixture.componentInstance.disabled = false;
    fixture.detectChanges();
    buttons[0].click();
    buttons[1].click();
    buttons[4].click();
    const request = http.expectOne(`${BASE_URL}/bgr-capture-series`);
    expect(request.request.body).toEqual(jasmine.objectContaining({
      target_folder: 'C:/captures/sample', wavelengths: ['uv255', 'uv310'], capture_id: jasmine.any(String)
    }));
    expect(shared.getMeasurementActive()).toBeTrue();
    fixture.componentInstance.disabled = true; // Shared measurement state locks the other controls.
    fixture.detectChanges();
    expect(buttons[4].classList.contains('running')).toBeTrue();
    expect(buttons[4].disabled).toBeFalse();
    expect(buttons[0].disabled).toBeTrue();
    buttons[4].click();
    http.expectOne(`${BASE_URL}/bgr-capture-series/cancel`).flush({ status: 'cancellation_requested' });
    expect(shared.getMeasurementActive()).toBeTrue();
    request.flush({ status: 'cancelled', series_index: 1, saved_images: [] });
    fixture.detectChanges();
    expect(shared.getMeasurementActive()).toBeFalse();
    expect(buttons[4].classList.contains('running')).toBeFalse();
    fixture.destroy();
    http.verify();
  });

  it('owns the in-flight request across view destruction and publishes the images already saved', () => {
    const fixture = TestBed.createComponent(FilterCaptureButtonsComponent);
    const shared = TestBed.inject(SharedService);
    const http = TestBed.inject(HttpTestingController);
    shared.setCameraConnectionStatus(true);
    shared.setSaveDirectory('C:/captures/sample');
    const saved = jasmine.createSpy('saved');
    const sub = shared.newSavedImage$.subscribe(saved);
    fixture.detectChanges();
    fixture.componentInstance.toggle('vis');
    fixture.componentInstance.start();
    const request = http.expectOne(`${BASE_URL}/bgr-capture-series`);
    expect(request.request.body.wavelengths).toEqual(['vis']);
    fixture.destroy();
    http.expectOne(`${BASE_URL}/bgr-capture-series/cancel`).flush({ status: 'cancellation_requested' });
    expect(shared.getMeasurementActive()).toBeTrue();
    request.flush({ status: 'cancelled', series_index: 1, saved_images: [{ path: 'C:/captures/sample/sample_1_r.jpg' }] });
    expect(saved).toHaveBeenCalledWith({ path: 'C:/captures/sample/sample_1_r.jpg', tabletIndex: 0 });
    expect(shared.getMeasurementActive()).toBeFalse();
    sub.unsubscribe();
    http.verify();
  });
});
