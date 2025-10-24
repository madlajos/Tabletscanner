import { ComponentFixture, TestBed } from '@angular/core/testing';

import { MeasurementResultsPopupComponent } from './measurement-results-popup.component';

describe('MeasurementResultsPopupComponent', () => {
  let component: MeasurementResultsPopupComponent;
  let fixture: ComponentFixture<MeasurementResultsPopupComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [MeasurementResultsPopupComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(MeasurementResultsPopupComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
