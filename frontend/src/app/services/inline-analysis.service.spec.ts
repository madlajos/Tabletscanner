import { fakeAsync, TestBed, tick } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { InlineAnalysisService } from './inline-analysis.service';
import { InlineAnalysisError, InlineAnalysisResult, InlineAnalysisSession } from '../models/inline-analysis.models';
import { createEmptyPipeline } from '../models/pipeline.models';
import { BASE_URL } from '../api-config';
import { LOCAL_ERROR_HANDLING } from '../interceptors/popup.interceptor';

describe('Inline analysis ownership and result selection', () => {
  let service: InlineAnalysisService;
  let http: HttpTestingController;
  const base = `${BASE_URL}/inline-analysis/sessions`;
  const snapshot = (overrides: Partial<InlineAnalysisSession> = {}): InlineAnalysisSession => ({
    session_id: 'run-1', recipe_name: 'Crop', measurement_name: 'A4 B4 C4',
    mode: 'each_tablet', input_scope: 'tablet', state: 'collecting', pipeline: createEmptyPipeline('Crop'),
    tablets: [], results: [], errors: [], revision: 1, pipeline_revision: 1, ...overrides,
  });
  const result = (resultId: string, revision = 1): InlineAnalysisResult => ({
    result_id: resultId, kind: 'tablet', tablet_id: resultId, label: resultId, status: 'completed',
    image_count: 3, input_count: 3, revision,
  });
  const start = (session = snapshot()) => {
    service.start(session.pipeline, session.measurement_name, 'C:/measurement').subscribe();
    http.expectOne(base).flush(session);
  };
  beforeEach(() => {
    TestBed.configureTestingModule({ providers: [provideHttpClient(), provideHttpClientTesting()] });
    service = TestBed.inject(InlineAnalysisService);
    http = TestBed.inject(HttpTestingController);
  });
  afterEach(() => { service.ngOnDestroy(); http.verify(); });

  it('keeps polling off-tab and counts each analysis error once without scanner popups', fakeAsync(() => {
    let unread = -1;
    let errors: InlineAnalysisError[] = [];
    service.unreadErrors$.subscribe(value => unread = value);
    service.errors$.subscribe(value => errors = value);
    start();
    const error: InlineAnalysisError = { id: 'error-1', code: 'E4503', error: 'Hiányzó RGB csatorna', popup: true, tablet_id: 'A4', result_id: null };
    tick(1000);
    const poll = http.expectOne(`${base}/run-1`);
    expect(poll.request.context.get(LOCAL_ERROR_HANDLING)).toBeTrue();
    poll.flush(snapshot({ revision: 2, errors: [error] }));
    expect(unread).toBe(1);
    expect(errors.length).toBe(1);
    service.setTabVisible(true);
    service.dismissError(error.id);
    tick(1000);
    http.expectOne(`${base}/run-1`).flush(snapshot({ revision: 3, errors: [error] }));
    expect(unread).toBe(0);
    expect(errors).toEqual([]);
    service.ngOnDestroy();
  }));

  it('recovers polling after transport errors without duplicating a popup each second', fakeAsync(() => {
    let count = 0;
    service.errors$.subscribe(errors => count = errors.length);
    start();
    tick(1000);
    http.expectOne(`${base}/run-1`).error(new ProgressEvent('offline'));
    tick(1000);
    http.expectOne(`${base}/run-1`).error(new ProgressEvent('offline'));
    expect(count).toBe(1);
    tick(1000);
    http.expectOne(`${base}/run-1`).flush(snapshot({ state: 'completed', revision: 3 }));
    expect(service.session?.state).toBe('completed');
    tick(5000);
    http.expectNone(`${base}/run-1`);
    service.ngOnDestroy();
  }));

  it('ignores a late start response after a newer measurement has been accepted', () => {
    service.start(createEmptyPipeline(), 'old', 'C:/measurement').subscribe();
    const old = http.expectOne(base);
    start(snapshot({ session_id: 'new' }));
    old.flush(snapshot({ session_id: 'old' }));
    expect(service.activeSessionId).toBe('new');
  });

  it('does not switch polling back to an older session on a late configure response', fakeAsync(() => {
    start();
    service.configure({ mode: 'measurement_end' });
    const configure = http.expectOne(`${base}/run-1/configure`);
    start(snapshot({ session_id: 'new' }));
    configure.flush(snapshot({ mode: 'measurement_end', revision: 2 }));
    tick(1000);
    http.expectOne(`${base}/new`).flush(snapshot({ session_id: 'new', state: 'completed', revision: 2 }));
    http.expectNone(`${base}/run-1`);
    service.ngOnDestroy();
  }));

  it('cancels old preview reads when another tablet is selected', () => {
    start(snapshot({ results: [result('A4'), result('B4')] }));
    const previous = http.expectOne(`${base}/run-1/results/B4?image_index=0`);
    service.selectResult('A4', 2);
    expect(previous.cancelled).toBeTrue();
    http.expectOne(`${base}/run-1/results/A4?image_index=2`).flush({ success: true, executed_up_to: 1, image_base64: 'new', image_count: 3 });
    expect(service.preview.result_id).toBe('A4');
    expect(service.preview.response?.image_base64).toBe('new');
    expect(service.followLatest).toBeFalse();
  });

  it('refreshes cumulative outputs when execution revision changes without parameter changes', fakeAsync(() => {
    const cumulative = { ...result('summary'), kind: 'cumulative' as const, tablet_id: null };
    start(snapshot({ results: [cumulative] }));
    http.expectOne(`${base}/run-1/results/summary?image_index=0`).flush({ success: true, executed_up_to: 1 });
    tick(1000);
    http.expectOne(`${base}/run-1`).flush(snapshot({ revision: 2, results: [{ ...cumulative, revision: 2 }] }));
    http.expectOne(`${base}/run-1/results/summary?image_index=0`).flush({ success: true, executed_up_to: 1, side_outputs: { values: [1, 2] } });
    expect(service.preview.response?.side_outputs?.['values']).toEqual([1, 2]);
    service.ngOnDestroy();
  }));

  it('holds result polling after measurement finish while background analysis runs', fakeAsync(() => {
    start();
    service.finish('run-1');
    const finish = http.expectOne(`${base}/run-1/finish`);
    expect(finish.request.body).toEqual({ cancelled: false });
    finish.flush(snapshot({ state: 'finishing', revision: 2 }));
    tick(1000);
    http.expectOne(`${base}/run-1`).flush(snapshot({ state: 'finishing', revision: 3 }));
    expect(service.session?.state).toBe('finishing');
    service.ngOnDestroy();
  }));
});
