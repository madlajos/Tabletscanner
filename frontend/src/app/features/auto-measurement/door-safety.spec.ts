import { TestBed, fakeAsync, tick } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { BehaviorSubject } from 'rxjs';
import { BASE_URL } from '../../api-config';
import { DoorInterlockStatus, LightStatus, UNKNOWN_DOOR_INTERLOCK } from '../../models/light.models';
import { LightStatusService } from '../../services/light-status.service';
import { ErrorNotificationService, AppError } from '../../services/error-notification.service';
import { AutoMeasurementComponent } from './auto-measurement.component';

const CLOSED: DoorInterlockStatus = {
  known: true, front_open: false, top_open: false, uv_blocked: false, trip_count: 0,
};

describe('Automatic measurement door safety', () => {
  let component: AutoMeasurementComponent;
  let http: HttpTestingController;
  let status: BehaviorSubject<LightStatus>;
  let errors: AppError[];
  let flow: {
    checkHomedThenProceed(indices: number[]): void;
    processTabletQueue(indices: number[], index: number): Promise<void>;
    startProgressPolling(request: string, tablet: number): void;
  };

  beforeEach(() => {
    status = new BehaviorSubject<LightStatus>({ active_channel: null, active_mode: null,
      channels: { uv255: false, uv310: false, uv365: false, vis: false }, auto_turned_off: [],
      door_interlock: CLOSED });
    TestBed.configureTestingModule({ imports: [AutoMeasurementComponent],
      providers: [provideHttpClient(), provideHttpClientTesting(),
        { provide: LightStatusService, useValue: { status$: status } }],
    }).overrideComponent(AutoMeasurementComponent, { set: { template: '', imports: [] } });
    component = TestBed.createComponent(AutoMeasurementComponent).componentInstance;
    http = TestBed.inject(HttpTestingController);
    TestBed.inject(ErrorNotificationService).errors$.subscribe(value => { errors = value; });
    flow = component as unknown as typeof flow;
    component.onDotClick(31);
    component.saveLocation = 'C:/test-only';
    component.measurementName = 'Door test';
    component.doorInterlock = CLOSED;
    spyOn(flow, 'checkHomedThenProceed');
  });

  afterEach(() => { component.ngOnDestroy(); http.verify(); });

  it('blocks unknown status and either open door even for VIS capture plans', () => {
    for (const doors of [UNKNOWN_DOOR_INTERLOCK,
      { ...CLOSED, front_open: true, uv_blocked: true },
      { ...CLOSED, top_open: true, uv_blocked: true }]) {
      component.doorInterlock = doors;
      component.startMeasurement();
      expect(component.canStart()).toBeFalse();
      expect(component.measurementActive).toBeFalse();
    }
    http.expectNone(`${BASE_URL}/auto_measurement/preflight`);
  });

  it('requires a successful fresh preflight before homing or analysis', () => {
    component.startMeasurement();
    expect(component.preflightActive).toBeTrue();
    expect(flow.checkHomedThenProceed).not.toHaveBeenCalled();
    http.expectOne(`${BASE_URL}/auto_measurement/preflight`).flush({ code: 'E1402', popup: true },
      { status: 409, statusText: 'Door open' });
    expect(component.measurementActive).toBeFalse();
    component.startMeasurement();
    http.expectOne(`${BASE_URL}/auto_measurement/preflight`).flush({ door_interlock: CLOSED });
    expect(component.measurementActive).toBeTrue();
    expect(flow.checkHomedThenProceed).toHaveBeenCalledOnceWith([31]);
  });

  it('shows a non-aborting popup while paused and automatically clears it on resume', fakeAsync(() => {
    component.measurementActive = true;
    flow.startProgressPolling('door-test', 31);
    tick(100);
    http.expectOne(request => request.url.endsWith('/auto_measurement/progress')).flush({
      images: [], warnings: [], active_plan_row_index: 1, door_paused: true, door_pause_count: 1,
    });
    expect(component.doorPaused).toBeTrue();
    expect(errors.find(error => error.code === 'E1404')?.abortMeasurement).toBeFalse();
    expect(component.measurementActive).toBeTrue();
    tick(250);
    http.expectOne(request => request.url.endsWith('/auto_measurement/progress')).flush({
      images: [], warnings: [], active_plan_row_index: 1, door_paused: false, door_pause_count: 1,
    });
    expect(component.doorPaused).toBeFalse();
    expect(errors.some(error => error.code === 'E1404')).toBeFalse();
    expect(component.measurementActive).toBeTrue();
    component.ngOnDestroy();
  }));

  it('reports a brief opening even when the progress poll only sees resumed state', fakeAsync(() => {
    component.measurementActive = true;
    flow.startProgressPolling('brief-trip', 31);
    tick(100);
    http.expectOne(request => request.url.endsWith('/auto_measurement/progress')).flush({
      images: [], warnings: [], active_plan_row_index: 1, door_paused: false, door_pause_count: 1,
    });
    expect(errors.find(error => error.code === 'E1405')?.popupStyle).toBe('center');
    expect(component.measurementActive).toBeTrue();
    component.ngOnDestroy();
  }));

  it('keeps ownership after stop until the paused backend request returns', () => {
    component.startMeasurement();
    http.expectOne(`${BASE_URL}/auto_measurement/preflight`).flush({ door_interlock: CLOSED });
    void flow.processTabletQueue([31], 0);
    const request = http.expectOne(`${BASE_URL}/auto_measurement/step`);
    component.stopMeasurement();
    const stop = http.expectOne(`${BASE_URL}/abort-autofocus`);
    expect(stop.request.body.request_id).toBe(request.request.body.request_id);
    stop.flush({ status: 'abort_signaled' });
    expect(component.measurementActive).toBeTrue();
    expect(component.canStart()).toBeFalse();
    request.flush({ status: 'success', cancelled: true, saved_images: [] });
    http.expectOne(`${BASE_URL}/turn-off-all-lights`).flush({});
    expect(component.measurementActive).toBeFalse();
  });
});
