import { ComponentFixture, TestBed } from '@angular/core/testing';

import { CenterErrorPopupComponent } from './center-error-popup.component';

describe('CenterErrorPopupComponent', () => {
  let component: CenterErrorPopupComponent;
  let fixture: ComponentFixture<CenterErrorPopupComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [CenterErrorPopupComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(CenterErrorPopupComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
