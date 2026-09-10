import { TestBed, fakeAsync, tick } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { SoftwareSettingsComponent } from './software-settings.component';
import { BASE_URL } from '../../api-config';

describe('Settings focus table', () => {
  beforeEach(() => TestBed.configureTestingModule({
    imports: [SoftwareSettingsComponent], providers: [provideHttpClient(), provideHttpClientTesting()]
  }));

  it('locks the selected zero, recalculates on selection and saves edits in master coordinates', fakeAsync(() => {
    spyOn(SoftwareSettingsComponent.prototype, 'ngOnInit').and.stub();
    const fixture = TestBed.createComponent(SoftwareSettingsComponent);
    const component = fixture.componentInstance;
    component.selectedType = 'focus';
    component.filterSettings = {
      filters: [
        { id: 'blue', name: 'Kék', wavelength_range: '450', color: '#0000ff' },
        { id: 'green', name: 'Zöld', wavelength_range: '550', color: '#00ff00' }
      ],
      slots: [null, 'blue', 'green', null, null, null],
      height_offsets_mm: {
        empty: { vis: 0, uv255: 0, uv310: 0, uv365: 0 },
        blue: { vis: 0, uv255: 1, uv310: 1, uv365: 1 },
        green: { vis: 2, uv255: 3, uv310: 3, uv365: 3 }
      }
    };
    component.autofocusSettings = { channel: 'vis', brightness: 'full', filter_position: 3 };
    fixture.detectChanges();
    tick();
    const cells = (): HTMLInputElement[] => Array.from(fixture.nativeElement.querySelectorAll('.height-offset-table input'));
    expect(cells()[11].disabled).toBeTrue();
    expect(cells()[11].value).toBe('0');
    expect(cells()[7].disabled).toBeFalse();
    expect(cells()[7].value).toBe('-2');
    component.onAutofocusLightChanged('uv255:dimmed');
    fixture.detectChanges();
    tick();
    expect(cells()[8].disabled).toBeTrue();
    expect(cells()[8].value).toBe('0');
    expect(cells()[7].value).toBe('-3');

    component.onRelativeHeightOffsetChanged('empty', 'vis', -1);
    component['persistFilterSettings']().subscribe();
    const http = TestBed.inject(HttpTestingController);
    const save = http.expectOne(`${BASE_URL}/settings/filter`);
    expect(save.request.body.height_offsets_mm.empty.vis).toBe(2);
    expect(save.request.body.height_offsets_mm.green.uv255).toBe(3);
    expect(save.request.body.height_offsets_mm.blue.vis).toBe(0);
    save.flush({ filter_settings: save.request.body });
    fixture.destroy();
    http.verify();
  }));
});
