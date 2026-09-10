import { fakeAsync, TestBed, tick } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { SoftwareSettingsComponent } from './software-settings.component';
import { SettingsUpdatesService } from '../../services/settings-updates.service';
import { BASE_URL } from '../../api-config';
import { CameraCombinationSettingsService } from '../../services/camera-combination-settings.service';

describe('Settings camera combination table', () => {
  beforeEach(() => TestBed.configureTestingModule({
    imports: [SoftwareSettingsComponent], providers: [provideHttpClient(), provideHttpClientTesting()]
  }));

  it('renders every filter/light cell and saves exposure and gain', fakeAsync(() => {
    spyOn(SoftwareSettingsComponent.prototype, 'ngOnInit').and.stub();
    const fixture = TestBed.createComponent(SoftwareSettingsComponent);
    const component = fixture.componentInstance;
    component.selectedType = 'camera';
    component.filterSettings = {
      filters: [{ id: 'red', name: 'Piros', wavelength_range: '600-650', color: '#ff0000' }],
      slots: [null, 'red', null, null, null, null],
      height_offsets_mm: {
        empty: { uv255: 0, uv310: 0, uv365: 0, vis: 0 },
        red: { uv255: 0, uv310: 0, uv365: 0, vis: 0 }
      }
    };
    const row = () => ({
      uv255: { exposure_time: 100, gain: 1 }, uv310: { exposure_time: 200, gain: 2 },
      uv365: { exposure_time: 300, gain: 3 }, vis: { exposure_time: 400, gain: 4 }
    });
    component.cameraCombinationSettings = {
      empty: row(), rgb: row(), filter_255nm: row(), filter_365nm: row()
    };
    component.cameraCombinationSettings['rgb'].uv255 = { exposure_time: 1000000, gain: 10 };
    fixture.detectChanges();
    tick();
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelectorAll('.camera-combination-cell').length).toBe(11);
    expect(component.isCameraCombinationUnavailable('filter_255nm', 'uv365')).toBeTrue();
    expect(fixture.nativeElement.querySelectorAll('.camera-combination-unavailable').length).toBe(5);
    expect(fixture.nativeElement.querySelectorAll('.camera-combination-unavailable input').length).toBe(0);
    expect(fixture.nativeElement.querySelectorAll('.camera-combination-na').length).toBe(5);
    const axisCorner = fixture.nativeElement.querySelector('.camera-axis-corner');
    expect(axisCorner.textContent).toContain('Lámpa');
    expect(axisCorner.textContent).toContain('Szűrő');
    const rgbLabel = fixture.nativeElement.querySelectorAll('.camera-filter-label')[1].textContent;
    expect(rgbLabel.trim().split(/\s+/).join(' ')).toBe('Piros Zöld Kék');
    const rgbInputs = fixture.nativeElement.querySelectorAll('.camera-combination-table tbody tr')[1]
      .querySelectorAll('input') as NodeListOf<HTMLInputElement>;
    expect(rgbInputs[0].value).toBe('1 000 000');
    expect(rgbInputs[1].value).toBe('10.0');
    component.cameraCombinationSettings['rgb'].uv365.gain = 7.5;
    (component as any).persistCameraCombinationSettings().subscribe();
    const request = TestBed.inject(HttpTestingController).expectOne(`${BASE_URL}/settings/camera/combinations`);
    expect(request.request.method).toBe('PUT');
    expect(request.request.body.rgb.uv365.gain).toBe(7.5);
    request.flush({ camera_combination_settings: request.request.body,
      camera_params: { ExposureTime: 300, Gain: 7.5 }, ranges: {} });
    tick();
    let current: any;
    TestBed.inject(SettingsUpdatesService).cameraSettings$.subscribe(value => current = value).unsubscribe();
    expect(current.ExposureTime).toBe(300);
    expect(current.Gain).toBe(7.5);
    expect(current.Gamma).toBe(1);
    let combinations: any;
    TestBed.inject(CameraCombinationSettingsService).settings$.subscribe(value => combinations = value).unsubscribe();
    expect(combinations.rgb.uv365.gain).toBe(7.5);
    fixture.destroy();
    TestBed.inject(HttpTestingController).verify();
  }));
});
