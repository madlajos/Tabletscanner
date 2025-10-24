import { ComponentFixture, TestBed } from '@angular/core/testing';

import { BarcodeScannerControlComponent } from './barcode-scanner-control.component';

describe('BarcodeScannerControlComponent', () => {
  let component: BarcodeScannerControlComponent;
  let fixture: ComponentFixture<BarcodeScannerControlComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [BarcodeScannerControlComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(BarcodeScannerControlComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
