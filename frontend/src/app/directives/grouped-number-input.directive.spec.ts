import { Component } from '@angular/core';
import { ComponentFixture, TestBed } from '@angular/core/testing';
import { FormsModule } from '@angular/forms';
import { GroupedNumberInputDirective } from './grouped-number-input.directive';

@Component({
  standalone: true,
  imports: [FormsModule, GroupedNumberInputDirective],
  template: `
    <input class="natural" appGroupedNumber [(ngModel)]="natural">
    <input class="fixed" appGroupedNumber [groupedNumberDecimals]="1" [(ngModel)]="fixed">
  `
})
class TestHostComponent {
  natural = 1000000;
  fixed = 0;
}

describe('GroupedNumberInputDirective', () => {
  let fixture: ComponentFixture<TestHostComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({ imports: [TestHostComponent] }).compileComponents();
    fixture = TestBed.createComponent(TestHostComponent);
    fixture.detectChanges();
    await fixture.whenStable();
    fixture.detectChanges();
  });

  it('groups thousands and only applies configured fixed decimals', () => {
    const natural = fixture.nativeElement.querySelector('.natural') as HTMLInputElement;
    const fixed = fixture.nativeElement.querySelector('.fixed') as HTMLInputElement;

    expect(natural.value).toBe('1 000 000');
    expect(fixed.value).toBe('0.0');

    natural.value = '12 345,67';
    natural.dispatchEvent(new Event('input'));
    expect(fixture.componentInstance.natural).toBe(12345.67);

    fixed.value = '2,55';
    fixed.dispatchEvent(new Event('input'));
    fixed.dispatchEvent(new Event('blur'));
    expect(fixture.componentInstance.fixed).toBe(2.6);
    expect(fixed.value).toBe('2.6');
  });
});
