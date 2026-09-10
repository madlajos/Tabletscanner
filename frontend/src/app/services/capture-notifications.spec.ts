import { fakeAsync, TestBed, tick } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting, HttpTestingController } from '@angular/common/http/testing';
import { BgrCaptureService } from './bgr-capture.service';
import { ErrorNotificationService, AppError } from './error-notification.service';
import { ErrorPopupComponent } from '../components/error-popup/error-popup.component';
import { ErrorPopupListComponent } from '../components/error-popup-list/error-popup-list.component';
import { SharedService } from '../shared.service';
import { BASE_URL } from '../api-config';
import { SettingsUpdatesService } from './settings-updates.service';

describe('Capture notifications', () => {
  beforeEach(() => TestBed.configureTestingModule({
    imports: [ErrorPopupComponent, ErrorPopupListComponent], providers: [provideHttpClient(), provideHttpClientTesting()]
  }));

  it('publishes progress once and retains saved frames after partial failure', fakeAsync(() => {
    const http = TestBed.inject(HttpTestingController);
    const shared = TestBed.inject(SharedService);
    const saved = jasmine.createSpy('saved');
    const sub = shared.newSavedImage$.subscribe(saved);
    TestBed.inject(BgrCaptureService).run('C:/sample', ['uv255']);
    const run = http.expectOne(`${BASE_URL}/bgr-capture-series`);
    const image = { path: 'C:/sample/r.jpg', metadata: { wavelength: 'uv255', Errors: ['ZOffset difference: 2 mm'] } };
    tick(500);
    http.expectOne(`${BASE_URL}/bgr-capture-series/status`).flush({
      capture_id: run.request.body.capture_id,
      saved_images: [image],
      camera_params: { ExposureTime: 50000, Gain: 2.5 },
    });
    expect(saved).toHaveBeenCalledOnceWith({ ...image, tabletIndex: 0 });
    let cameraSettings: any;
    TestBed.inject(SettingsUpdatesService).cameraSettings$.subscribe(value => cameraSettings = value).unsubscribe();
    expect(cameraSettings.ExposureTime).toBe(50000);
    expect(cameraSettings.Gain).toBe(2.5);
    run.flush({ saved_images: [image, { path: 'C:/sample/g.jpg' }] }, { status: 500, statusText: 'Capture failed' });
    expect(saved).toHaveBeenCalledTimes(2);
    expect(shared.getMeasurementActive()).toBeFalse();
    tick(1000);
    http.verify();
    sub.unsubscribe();
  }));

  it('makes warnings yellow and dismissible without redisplaying the same polling event', () => {
    const service = TestBed.inject(ErrorNotificationService);
    const warning = { id: 'capture-warning', code: 'W1205', target_z: 40, missing_offset_mm: 2 };
    let errors: AppError[] = [];
    const sub = service.errors$.subscribe(value => errors = value);
    service.addWarnings([warning]);
    expect(errors[0].severity).toBe('warning');
    expect(errors[0].abortMeasurement).toBeUndefined();
    const fixture = TestBed.createComponent(ErrorPopupComponent);
    fixture.componentInstance.severity = 'warning';
    fixture.componentInstance.index = warning.id;
    fixture.componentInstance.close.subscribe(id => service.removeError(id));
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('.error-popup.warning')).not.toBeNull();
    fixture.nativeElement.querySelector('button').click();
    service.addWarnings([warning]);
    expect(errors.length).toBe(0);
    fixture.destroy();
    sub.unsubscribe();
  });

  it('automatically dismisses warnings after five seconds and shows a countdown bar', fakeAsync(() => {
    const service = TestBed.inject(ErrorNotificationService);
    const warning = { id: 'timed-warning', code: 'W1205', target_z: 40, missing_offset_mm: 2 };
    let errors: AppError[] = [];
    const sub = service.errors$.subscribe(value => errors = value);
    service.addWarnings([warning]);

    const fixture = TestBed.createComponent(ErrorPopupComponent);
    fixture.componentRef.setInput('severity', 'warning');
    fixture.componentRef.setInput('index', warning.id);
    fixture.componentInstance.close.subscribe(id => service.removeError(id));
    fixture.detectChanges();

    const timer = fixture.nativeElement.querySelector('.warning-timer') as HTMLElement;
    expect(timer).not.toBeNull();
    expect(timer.style.animationDuration).toBe('5000ms');
    tick(4999);
    expect(errors.length).toBe(1);
    tick(1);
    expect(errors.length).toBe(0);

    fixture.destroy();
    sub.unsubscribe();
  }));

  it('shows one stacked notification with severity count badges', () => {
    const service = TestBed.inject(ErrorNotificationService);
    const shared = TestBed.inject(SharedService);
    service.addError({ code: 'first-error', message: 'Első hiba' });
    service.addError({ code: 'second-error', message: 'Második hiba' });
    shared.setToolbarNotice({ severity: 'success', message: 'Mérés kész' });

    const fixture = TestBed.createComponent(ErrorPopupListComponent);
    fixture.detectChanges();

    expect(fixture.nativeElement.querySelectorAll('app-error-popup').length).toBe(1);
    const badges = Array.from(fixture.nativeElement.querySelectorAll('.notification-badge'))
      .map((element: any) => element.textContent.trim());
    expect(badges).toEqual(['2', '1']);
    expect(fixture.nativeElement.querySelector('.notification-badge.error').textContent.trim()).toBe('2');
    expect(fixture.nativeElement.querySelector('.notification-badge.success').textContent.trim()).toBe('1');
    expect(fixture.nativeElement.querySelector('.notification-stack.has-multiple')).not.toBeNull();
    fixture.destroy();
  });
});
