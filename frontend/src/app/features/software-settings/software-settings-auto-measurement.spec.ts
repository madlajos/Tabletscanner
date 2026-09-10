import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { TestBed } from '@angular/core/testing';
import { BASE_URL } from '../../api-config';
import { SoftwareSettingsComponent } from './software-settings.component';

describe('SoftwareSettingsComponent automatic measurement settings', () => {
  beforeEach(() => TestBed.configureTestingModule({
    imports: [SoftwareSettingsComponent],
    providers: [provideHttpClient(), provideHttpClientTesting()]
  }));

  it('renders and persists the autofocus-image and tablet-presence options', () => {
    spyOn(SoftwareSettingsComponent.prototype, 'ngOnInit').and.stub();
    const fixture = TestBed.createComponent(SoftwareSettingsComponent);
    const component = fixture.componentInstance;
    component.selectedType = 'autoMeasurement' as any;
    component.saveAutofocusImage = false;
    component.checkTabletPresence = false;
    fixture.detectChanges();

    const autofocusImage = fixture.nativeElement.querySelector(
      'input[aria-label="Autofókusz-kép mentése"]'
    ) as HTMLInputElement;
    const tabletPresence = fixture.nativeElement.querySelector(
      'input[aria-label="Hiányzó tabletták ellenőrzése"]'
    ) as HTMLInputElement;
    expect(autofocusImage.checked).toBeFalse();
    expect(tabletPresence.checked).toBeFalse();

    (component as any).persistAutoMeasurementSettings().subscribe();
    const http = TestBed.inject(HttpTestingController);
    const autofocusRequest = http.expectOne(`${BASE_URL}/update-other-settings`);
    expect(autofocusRequest.request.body).toEqual({
      category: 'auto_measurement_settings',
      setting_name: 'save_autofocus_image',
      setting_value: false
    });
    autofocusRequest.flush({ updated_value: false });
    const presenceRequest = http.expectOne(`${BASE_URL}/update-other-settings`);
    expect(presenceRequest.request.body).toEqual({
      category: 'auto_measurement_settings',
      setting_name: 'check_tablet_presence',
      setting_value: false
    });
    presenceRequest.flush({ updated_value: false });

    fixture.destroy();
    http.verify();
  });
});
