import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';
import { TestBed } from '@angular/core/testing';
import { SoftwareSettingsComponent } from './software-settings.component';

describe('SoftwareSettingsComponent motion settings', () => {
  function createComponent(): SoftwareSettingsComponent {
    TestBed.configureTestingModule({
      imports: [SoftwareSettingsComponent],
      providers: [provideHttpClient(), provideHttpClientTesting()]
    });
    spyOn(SoftwareSettingsComponent.prototype, 'ngOnInit').and.stub();
    return TestBed.createComponent(SoftwareSettingsComponent).componentInstance;
  }

  it('derives the first-tablet coordinate limits from the tray spacing', () => {
    const component = createComponent();
    component.tabletSpacingMm = 18.3;

    expect(component.firstTabletXMaxMm).toBeCloseTo(10.8, 10);
    expect(component.firstTabletYMaxMm).toBeCloseTo(0.8, 10);
  });

  it('reports the affected axis and its actual limit', () => {
    const component = createComponent();
    component.tabletSpacingMm = 18.3;
    component.firstTabletYMm = 1;

    const error = (component as any).getAdvancedValidationError() as string;

    expect(error).toContain('Y koordinátája');
    expect(error).toContain('0,8 mm');
  });

  it('accepts a Y coordinate that keeps the last row within tolerance', () => {
    const component = createComponent();
    component.tabletSpacingMm = 18.3;
    component.firstTabletYMm = 0.8;

    expect((component as any).getAdvancedValidationError()).toBeNull();
  });
});
