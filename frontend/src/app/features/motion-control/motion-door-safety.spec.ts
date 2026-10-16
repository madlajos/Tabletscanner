import { TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { BehaviorSubject } from 'rxjs';
import { By } from '@angular/platform-browser';
import { MatTooltip } from '@angular/material/tooltip';
import { LightStatus, UNKNOWN_DOOR_INTERLOCK } from '../../models/light.models';
import { LightStatusService } from '../../services/light-status.service';
import { MotionControl } from './motion-control';

describe('Manual lamp door safety', () => {
  it('blocks every UV activation for open/unknown doors, permits VIS and reconciles a trip', () => {
    const status = new BehaviorSubject<LightStatus>({ active_channel: null, active_mode: null,
      channels: { uv255: false, uv310: false, uv365: false, vis: false }, auto_turned_off: [],
      door_interlock: UNKNOWN_DOOR_INTERLOCK });
    TestBed.configureTestingModule({ imports: [MotionControl],
      providers: [provideHttpClient(), provideHttpClientTesting(),
        { provide: LightStatusService, useValue: { status$: status } }],
    });
    spyOn(MotionControl.prototype, 'ngOnInit');
    const fixture = TestBed.createComponent(MotionControl);
    const component = fixture.componentInstance;
    component.isConnected = true;
    component.startFourChannelLightPolling();
    fixture.detectChanges();
    const buttons = (): HTMLButtonElement[] => Array.from(
      (fixture.nativeElement as HTMLElement).querySelectorAll<HTMLButtonElement>('.lamp-button'));
    expect(buttons().map(button => button.disabled)).toEqual([true, true, true, false]);
    const tooltips = () => fixture.debugElement.queryAll(By.directive(MatTooltip))
      .map(element => element.injector.get(MatTooltip).message);
    expect(tooltips().slice(0, 3).every(message => message.includes('ismeretlen'))).toBeTrue();
    expect(tooltips()[3]).toBe('');
    component.onLampDoubleClick('uv255');
    TestBed.inject(HttpTestingController).expectNone(request => request.method === 'POST');
    const closed = { known: true, front_open: false, top_open: false, uv_blocked: false, trip_count: 0 };
    status.next({ ...status.value, active_channel: 'uv255', active_mode: 'full',
      channels: { uv255: true, uv310: false, uv365: false, vis: false }, door_interlock: closed });
    fixture.detectChanges();
    expect(component.isLampActive('uv255')).toBeTrue();
    expect(tooltips()).toEqual(['', '', '', '']);
    status.next({ ...status.value, active_channel: null, active_mode: null,
      channels: { uv255: false, uv310: false, uv365: false, vis: false },
      door_interlock: { ...closed, top_open: true, uv_blocked: true, trip_count: 1 } });
    fixture.detectChanges();
    expect(component.isLampActive('uv255')).toBeFalse();
    expect(buttons().map(button => button.disabled)).toEqual([true, true, true, false]);
    expect(tooltips().slice(0, 3).every(message => message.includes('zárja be a fedelet'))).toBeTrue();
    expect(tooltips()[3]).toBe('');
    status.next({ ...status.value,
      door_interlock: { ...closed, front_open: true, uv_blocked: true, trip_count: 2 } });
    fixture.detectChanges();
    expect(tooltips()[0]).toContain('zárja be az ajtót');
    status.next({ ...status.value,
      door_interlock: { ...closed, front_open: true, top_open: true, uv_blocked: true, trip_count: 3 } });
    fixture.detectChanges();
    expect(tooltips()[0]).toContain('zárja be mindkettőt');
    expect((fixture.nativeElement as HTMLElement).querySelector('.lamp-button-wrapper')?.getAttribute('tabindex')).toBe('0');
    fixture.destroy();
    TestBed.inject(HttpTestingController).verify();
  });
});
