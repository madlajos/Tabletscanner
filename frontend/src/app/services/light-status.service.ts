import { Injectable } from '@angular/core';
import { HttpClient, HttpContext } from '@angular/common/http';
import { catchError, exhaustMap, of, shareReplay, startWith, tap, timer, timeout } from 'rxjs';
import { BASE_URL } from '../api-config';
import { LightStatus, UNKNOWN_DOOR_INTERLOCK } from '../models/light.models';
import { LOCAL_ERROR_HANDLING } from '../interceptors/popup.interceptor';

const UNKNOWN_LIGHT_STATUS: LightStatus = {
  active_channel: null, active_mode: null,
  channels: { uv255: false, uv310: false, uv365: false, vis: false },
  auto_turned_off: [], door_interlock: UNKNOWN_DOOR_INTERLOCK,
};

/** Share cached backend telemetry across scanner controls and measurement. */
@Injectable({ providedIn: 'root' })
export class LightStatusService {
  private lastStatus = UNKNOWN_LIGHT_STATUS;
  readonly status$ = timer(0, 500).pipe(
    exhaustMap(() => this.getStatus().pipe(
      timeout(2000), tap(status => { this.lastStatus = status; }),
      catchError(() => of({ ...this.lastStatus, door_interlock: UNKNOWN_DOOR_INTERLOCK })),
    )),
    startWith(UNKNOWN_LIGHT_STATUS),
    shareReplay({ bufferSize: 1, refCount: true }),
  );

  constructor(private http: HttpClient) {}

  getStatus() {
    return this.http.get<LightStatus>(`${BASE_URL}/lights/status`, {
      context: new HttpContext().set(LOCAL_ERROR_HANDLING, true),
    });
  }
}
