import { TestBed, fakeAsync, tick } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { BASE_URL } from '../api-config';
import { LightStatus, doorsAllowUv } from '../models/light.models';
import { LightStatusService } from './light-status.service';

describe('Shared lamp and door telemetry', () => {
  it('shares polling, fails closed on errors, and stops after the last subscriber', fakeAsync(() => {
    TestBed.configureTestingModule({ providers: [provideHttpClient(), provideHttpClientTesting()] });
    const http = TestBed.inject(HttpTestingController);
    const service = TestBed.inject(LightStatusService);
    const values: LightStatus[] = [];
    const first = service.status$.subscribe(value => values.push(value));
    const second = service.status$.subscribe();
    expect(doorsAllowUv(values.at(-1)?.door_interlock)).toBeFalse();
    tick(0);
    const request = http.expectOne(`${BASE_URL}/lights/status`);
    expect(request.request.method).toBe('GET');
    request.flush({ active_channel: 'vis', active_mode: null,
      channels: { uv255: false, uv310: false, uv365: false, vis: true }, auto_turned_off: [],
      door_interlock: { known: true, front_open: false, top_open: false, uv_blocked: false, trip_count: 0 } });
    expect(doorsAllowUv(values.at(-1)?.door_interlock)).toBeTrue();
    tick(500);
    http.expectOne(`${BASE_URL}/lights/status`).flush({}, { status: 503, statusText: 'Unavailable' });
    expect(doorsAllowUv(values.at(-1)?.door_interlock)).toBeFalse();
    expect(values.at(-1)?.active_channel).toBe('vis');
    first.unsubscribe(); second.unsubscribe();
    tick(1000);
    http.expectNone(`${BASE_URL}/lights/status`);
    http.verify();
  }));
});
