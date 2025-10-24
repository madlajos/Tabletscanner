import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ErrorPopupListComponent } from './error-popup-list.component';

describe('ErrorPopupListComponent', () => {
  let component: ErrorPopupListComponent;
  let fixture: ComponentFixture<ErrorPopupListComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ErrorPopupListComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(ErrorPopupListComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
