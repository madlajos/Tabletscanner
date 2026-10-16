import { Injectable, OnDestroy } from '@angular/core';
import { HttpClient, HttpContext, HttpErrorResponse } from '@angular/common/http';
import { BehaviorSubject, EMPTY, Observable, Subscription, catchError, defer, exhaustMap, tap, throwError, timer } from 'rxjs';
import { BASE_URL } from '../api-config';
import { LOCAL_ERROR_HANDLING } from '../interceptors/popup.interceptor';
import { PipelineDocument, PreviewResponse } from '../models/pipeline.models';
import {
  InlineAnalysisConfiguration, InlineAnalysisError, InlineAnalysisMode,
  InlineAnalysisPreview, InlineAnalysisSession, InlineInputScope,
} from '../models/inline-analysis.models';

/** Owns analysis independently of scanner/applier component lifetimes. */
@Injectable({ providedIn: 'root' })
export class InlineAnalysisService implements OnDestroy {
  private readonly sessionSubject = new BehaviorSubject<InlineAnalysisSession | null>(null);
  readonly session$ = this.sessionSubject.asObservable();
  private readonly errorsSubject = new BehaviorSubject<InlineAnalysisError[]>([]);
  readonly errors$ = this.errorsSubject.asObservable();
  private readonly unreadSubject = new BehaviorSubject(0);
  readonly unreadErrors$ = this.unreadSubject.asObservable();
  private readonly previewSubject = new BehaviorSubject<InlineAnalysisPreview>({
    result_id: null, image_index: 0, loading: false, response: null,
  });
  readonly preview$ = this.previewSubject.asObservable();
  private readonly seenErrors = new Set<string>();
  private pollSubscription?: Subscription;
  private previewSubscription?: Subscription;
  private readonly requests = new Subscription();
  private tabVisible = false;
  private previewKey = '';
  private transportFailed = false;
  private startGeneration = 0;
  mode: InlineAnalysisMode = 'each_tablet';
  inputScope: InlineInputScope | undefined;
  followLatest = true;

  constructor(private readonly http: HttpClient) {}

  get session(): InlineAnalysisSession | null { return this.sessionSubject.value; }
  get activeSessionId(): string | null { return this.session?.session_id ?? null; }
  get preview(): InlineAnalysisPreview { return this.previewSubject.value; }

  start(pipeline: PipelineDocument, name: string, folder: string): Observable<InlineAnalysisSession> {
    return defer(() => {
      const generation = ++this.startGeneration;
      return this.http.post<InlineAnalysisSession>(`${BASE_URL}/inline-analysis/sessions`, {
      pipeline, measurement_name: name, measurement_folder: folder,
      mode: this.mode, input_scope: this.inputScope,
    }, this.options()).pipe(
      tap(session => {
        if (generation !== this.startGeneration) return;
        this.pollSubscription?.unsubscribe();
        this.previewSubscription?.unsubscribe();
        this.sessionSubject.next(null);
        this.previewKey = '';
        this.previewSubject.next({ result_id: null, image_index: 0, loading: false, response: null });
        this.seenErrors.clear();
        this.errorsSubject.next([]);
        this.unreadSubject.next(0);
        this.followLatest = true;
        this.transportFailed = false;
        this.consume(session);
        this.poll(session.session_id);
      }),
      catchError(error => {
        this.reportError(error, 'Az elemzés nem indítható. A képrögzítés folytatódik.');
        return throwError(() => error);
      }),
      );
    });
  }

  finish(sessionId: string, cancelled = false): void {
    this.requests.add(this.http.post<InlineAnalysisSession>(
      `${BASE_URL}/inline-analysis/sessions/${encodeURIComponent(sessionId)}/finish`,
      { cancelled }, this.options(),
    ).subscribe({
      next: session => this.consume(session),
      error: error => this.reportError(error, 'Az elemzés lezárása sikertelen. A rögzített képek megmaradtak.'),
    }));
  }

  configure(configuration: InlineAnalysisConfiguration): void {
    if (configuration.mode) this.mode = configuration.mode;
    if (configuration.input_scope) this.inputScope = configuration.input_scope;
    const sessionId = this.activeSessionId;
    if (!sessionId) return;
    this.requests.add(this.http.post<InlineAnalysisSession>(
      `${BASE_URL}/inline-analysis/sessions/${encodeURIComponent(sessionId)}/configure`,
      configuration, this.options(),
    ).subscribe({
      next: session => { this.consume(session); if (sessionId === this.activeSessionId) this.poll(sessionId); },
      error: error => this.reportError(error, 'Az elemzés beállításainak módosítása sikertelen.'),
    }));
  }

  selectResult(resultId: string, imageIndex = 0, followLatest = false): void {
    const session = this.session;
    const result = session?.results.find(item => item.result_id === resultId);
    if (!session || !result) return;
    this.followLatest = followLatest;
    const index = Math.max(0, Math.min(imageIndex, Math.max(0, result.image_count - 1)));
    const key = `${session.session_id}:${resultId}:${result.revision}:${result.status}:${index}`;
    if (key === this.previewKey) return;
    this.previewKey = key;
    this.previewSubscription?.unsubscribe();
    this.previewSubject.next({ result_id: resultId, image_index: index, loading: result.status === 'completed', response: null });
    if (result.status !== 'completed') return;
    this.previewSubscription = this.http.get<PreviewResponse>(
      `${BASE_URL}/inline-analysis/sessions/${encodeURIComponent(session.session_id)}/results/${encodeURIComponent(resultId)}`,
      { ...this.options(), params: { image_index: index } },
    ).subscribe({
      next: response => {
        if (key !== this.previewKey) return;
        this.previewSubject.next({ result_id: resultId, image_index: index, loading: false, response });
      },
      error: error => {
        if (key !== this.previewKey) return;
        this.previewKey = '';
        this.previewSubject.next({ result_id: resultId, image_index: index, loading: false, response: null });
        this.reportError(error, 'Az elemzés eredménye nem tölthető be.');
      },
    });
  }

  setFollowLatest(enabled: boolean): void {
    this.followLatest = enabled;
    if (enabled && this.session) this.selectLatest(this.session);
  }

  setTabVisible(visible: boolean): void {
    this.tabVisible = visible;
    if (visible) this.unreadSubject.next(0);
  }

  dismissError(id: string): void {
    this.errorsSubject.next(this.errorsSubject.value.filter(error => error.id !== id));
  }

  reportError(error: unknown, fallback = 'Az elemzés sikertelen. A képrögzítés folytatódik.'): void {
    const body: unknown = error instanceof HttpErrorResponse ? error.error : error;
    const detail = body && typeof body === 'object' ? body as Record<string, unknown> : {};
    this.addError({
      id: crypto.randomUUID(), code: typeof detail['code'] === 'string' ? detail['code'] : 'E4503',
      error: typeof detail['error'] === 'string' ? detail['error'] : fallback,
      popup: true, tablet_id: null, result_id: null,
    });
  }

  private options(): { context: HttpContext } {
    return { context: new HttpContext().set(LOCAL_ERROR_HANDLING, true) };
  }

  private poll(sessionId: string): void {
    this.pollSubscription?.unsubscribe();
    this.pollSubscription = timer(1000, 1000).pipe(exhaustMap(() =>
      this.http.get<InlineAnalysisSession>(
        `${BASE_URL}/inline-analysis/sessions/${encodeURIComponent(sessionId)}`, this.options(),
      ).pipe(catchError(error => {
        if (!this.transportFailed) {
          this.transportFailed = true;
          this.reportError(error, 'Az elemzés állapota átmenetileg nem elérhető. Újrapróbálkozás folyamatban.');
        }
        return EMPTY;
      })),
    )).subscribe(session => {
      this.transportFailed = false;
      this.consume(session);
      if (session.state === 'completed' || session.state === 'cancelled') this.pollSubscription?.unsubscribe();
    });
  }

  private consume(session: InlineAnalysisSession): void {
    const previous = this.session;
    if (previous && (previous.session_id !== session.session_id || previous.revision > session.revision)) return;
    this.mode = session.mode;
    this.sessionSubject.next(session);
    for (const error of session.errors) this.addError(error);
    if (!session.results.length) {
      this.previewSubscription?.unsubscribe();
      this.previewKey = '';
      this.previewSubject.next({ result_id: null, image_index: 0, loading: false, response: null });
    } else if (this.followLatest || !this.preview.result_id || !session.results.some(result => result.result_id === this.preview.result_id)) this.selectLatest(session);
    else this.selectResult(this.preview.result_id, this.preview.image_index, false);
  }

  private selectLatest(session: InlineAnalysisSession): void {
    const completed = session.results.filter(result => result.status === 'completed');
    const latest = completed[completed.length - 1] ?? session.results[session.results.length - 1];
    if (latest) this.selectResult(latest.result_id, this.preview.result_id === latest.result_id ? this.preview.image_index : 0, this.followLatest);
  }

  private addError(error: InlineAnalysisError): void {
    if (this.seenErrors.has(error.id)) return;
    this.seenErrors.add(error.id);
    this.errorsSubject.next([...this.errorsSubject.value, error].slice(-100));
    if (!this.tabVisible) this.unreadSubject.next(this.unreadSubject.value + 1);
  }

  ngOnDestroy(): void {
    this.pollSubscription?.unsubscribe();
    this.previewSubscription?.unsubscribe();
    this.requests.unsubscribe();
  }
}
