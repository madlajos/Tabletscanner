import { TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { Subject, throwError } from 'rxjs';
import { BASE_URL } from '../../api-config';
import { createEmptyPipeline } from '../../models/pipeline.models';
import { InlineAnalysisSession } from '../../models/inline-analysis.models';
import { InlineAnalysisService } from '../../services/inline-analysis.service';
import { AutoMeasurementComponent } from './auto-measurement.component';

describe('Inline measurement coordination', () => {
  let component: AutoMeasurementComponent;
  let http: HttpTestingController;
  let created: Subject<InlineAnalysisSession>;
  let analysis: jasmine.SpyObj<InlineAnalysisService>;
  let proceed: jasmine.Spy;
  let flow: {
    checkHomedThenProceed(indices: number[]): void;
    processTabletQueue(indices: number[], index: number): Promise<void>;
    finishMeasurement(success: boolean): void;
  };

  beforeEach(() => {
    created = new Subject<InlineAnalysisSession>();
    analysis = jasmine.createSpyObj<InlineAnalysisService>('analysis', ['start', 'finish']);
    analysis.start.and.returnValue(created);
    TestBed.configureTestingModule({
      imports: [AutoMeasurementComponent],
      providers: [provideHttpClient(), provideHttpClientTesting(),
        { provide: InlineAnalysisService, useValue: analysis }],
    }).overrideComponent(AutoMeasurementComponent, { set: { template: '', imports: [] } });
    component = TestBed.createComponent(AutoMeasurementComponent).componentInstance;
    http = TestBed.inject(HttpTestingController);
    flow = component as unknown as typeof flow;
    proceed = spyOn(flow, 'checkHomedThenProceed');
    component.onDotClick(31);
    component.onDotClick(33);
    component.saveLocation = 'C:/test-only';
    component.measurementName = 'Inline test';
    component.selectedAnalysisRecipe = createEmptyPipeline('Test crop');
    component.doorInterlock = { known: true, front_open: false, top_open: false,
      uv_blocked: false, trip_count: 0 };
  });

  afterEach(() => {
    component.ngOnDestroy();
    http.verify();
  });

  function acceptSession(id = 'session-one'): void {
    created.next({ session_id: id } as InlineAnalysisSession);
    created.complete();
  }

  function start(): void {
    component.startMeasurement();
    http.expectOne(`${BASE_URL}/auto_measurement/preflight`).flush({ door_interlock: component.doorInterlock });
  }

  it('registers the recipe before imaging and forwards the owned session to A4', () => {
    start();
    expect(analysis.start).toHaveBeenCalledOnceWith(
      component.selectedAnalysisRecipe!, 'Inline test', 'C:/test-only/Inline test');
    expect(proceed).not.toHaveBeenCalled();
    acceptSession();
    expect(proceed).toHaveBeenCalledOnceWith([31, 32, 33]);
    void flow.processTabletQueue([31], 0);
    const step = http.expectOne(`${BASE_URL}/auto_measurement/step`);
    expect(step.request.body.inline_analysis_session_id).toBe('session-one');
    expect(step.request.body.tablet_index).toBe(31);
    // A terminal capture failure still closes analysis; it does not wait for CPU work.
    step.flush({ status: 'error' });
    expect(analysis.finish).toHaveBeenCalledOnceWith('session-one', true);
    http.expectOne(`${BASE_URL}/turn-off-all-lights`).flush({});
  });

  it('continues imaging when analysis session setup fails', () => {
    analysis.start.and.returnValue(throwError(() => new Error('Analysis unavailable')));
    start();
    expect(proceed).toHaveBeenCalledOnceWith([31, 32, 33]);
    expect(component.measurementActive).toBeTrue();
    expect(analysis.finish).not.toHaveBeenCalled();
  });

  it('closes a late-created analysis session after stop without starting homing', () => {
    start();
    component.stopMeasurement();
    http.expectOne(`${BASE_URL}/abort-autofocus`).flush({});
    http.expectOne(`${BASE_URL}/turn-off-all-lights`).flush({});
    acceptSession('late-session');
    expect(proceed).not.toHaveBeenCalled();
    expect(analysis.finish).toHaveBeenCalledOnceWith('late-session', true);
    expect(component.measurementActive).toBeFalse();
  });

  it('keeps measurements without a selected recipe independent of analysis', () => {
    component.selectedAnalysisRecipe = null;
    start();
    expect(analysis.start).not.toHaveBeenCalled();
    expect(proceed).toHaveBeenCalledOnceWith([31, 32, 33]);
  });
});
