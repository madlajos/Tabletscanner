import { ComponentFixture, TestBed } from '@angular/core/testing';

import { MotionControl } from './motion-control';

describe('MotionControl', () => {
  let component: MotionControl;
  let fixture: ComponentFixture<MotionControl>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [MotionControl]
    })
    .compileComponents();

    fixture = TestBed.createComponent(MotionControl);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
