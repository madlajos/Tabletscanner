import { fakeAsync, TestBed, tick } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { SoftwareSettingsComponent } from './software-settings.component';
import { BASE_URL } from '../../api-config';

describe('Settings XY collision protection', () => {
  beforeEach(() => TestBed.configureTestingModule({
    imports: [SoftwareSettingsComponent],
    providers: [provideHttpClient(), provideHttpClientTesting()]
  }));

  it('renders and persists the enabled 35 mm Z limit', fakeAsync(() => {
    spyOn(SoftwareSettingsComponent.prototype, 'ngOnInit').and.stub();
    const fixture = TestBed.createComponent(SoftwareSettingsComponent);
    const component = fixture.componentInstance;
    component.selectedType = 'advanced';
    fixture.detectChanges();

    const checkbox = fixture.nativeElement.querySelector(
      'input[aria-label="X/Y mozgás előtti Z-süllyesztés"]'
    ) as HTMLInputElement | null;
    const limit = fixture.nativeElement.querySelector(
      'input[aria-label="X/Y mozgás biztonságos Z-határa"]'
    ) as HTMLInputElement;
    expect(checkbox?.checked).toBeTrue();
    expect(limit.value).toBe('35');

    (component as any).persistAdvancedSettings().subscribe();
    const http = TestBed.inject(HttpTestingController);
    const motion = http.expectOne(`${BASE_URL}/settings/motion/advanced`);
    expect(motion.request.method).toBe('PUT');
    expect(motion.request.body.lower_z_before_xy_move).toBeTrue();
    expect(motion.request.body.xy_move_z_limit_mm).toBe(35);
    motion.flush({
      advanced_motion_settings: motion.request.body,
      connection: { connected: false, port: null, virtual: false }
    });
    const lamp = http.expectOne(`${BASE_URL}/settings/lamp/advanced`);
    lamp.flush({ advanced_lamp_settings: lamp.request.body });
    tick();

    fixture.destroy();
    http.verify();
  }));

  it('rejects limits outside the physical Z range', () => {
    spyOn(SoftwareSettingsComponent.prototype, 'ngOnInit').and.stub();
    const fixture = TestBed.createComponent(SoftwareSettingsComponent);
    const component = fixture.componentInstance;
    component.xyMoveZLimitMm = 40.1;

    (component as any).persistAdvancedSettings().subscribe();

    expect(component.advancedError).toContain('0 és 40 mm');
    TestBed.inject(HttpTestingController).expectNone(`${BASE_URL}/settings/motion/advanced`);
    fixture.destroy();
  });
});
