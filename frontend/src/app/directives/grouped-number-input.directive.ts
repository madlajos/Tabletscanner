import { Directive, ElementRef, HostListener, Input, forwardRef } from '@angular/core';
import { ControlValueAccessor, NG_VALUE_ACCESSOR } from '@angular/forms';

@Directive({
  selector: 'input[appGroupedNumber]',
  standalone: true,
  providers: [{
    provide: NG_VALUE_ACCESSOR,
    useExisting: forwardRef(() => GroupedNumberInputDirective),
    multi: true
  }]
})
export class GroupedNumberInputDirective implements ControlValueAccessor {
  @Input() groupedNumberDecimals: number | null = null;

  private value: number | null = null;
  private onChange: (value: number | null) => void = () => undefined;
  private onTouched: () => void = () => undefined;

  constructor(private readonly element: ElementRef<HTMLInputElement>) {}

  writeValue(value: unknown): void {
    const numeric = this.toFiniteNumber(value);
    this.value = numeric;
    this.element.nativeElement.value = numeric === null ? '' : this.formatNumber(numeric);
  }

  registerOnChange(callback: (value: number | null) => void): void {
    this.onChange = callback;
  }

  registerOnTouched(callback: () => void): void {
    this.onTouched = callback;
  }

  setDisabledState(disabled: boolean): void {
    this.element.nativeElement.disabled = disabled;
  }

  @HostListener('input')
  handleInput(): void {
    const input = this.element.nativeElement;
    const cursor = input.selectionStart ?? input.value.length;
    const significantBeforeCursor = input.value.slice(0, cursor).replace(/\s/g, '').length;
    const raw = input.value.replace(/\s/g, '');

    if (!/^\d*(?:[.,]\d*)?$/.test(raw)) {
      input.value = this.value === null ? '' : this.formatNumber(this.value);
      return;
    }

    input.value = this.groupRawValue(raw);
    this.restoreCursor(input, significantBeforeCursor);
    const numeric = this.toFiniteNumber(raw.replace(',', '.'));
    this.value = numeric;
    this.onChange(numeric);
  }

  @HostListener('blur')
  handleBlur(): void {
    this.onTouched();
    if (this.value !== null) {
      if (this.groupedNumberDecimals !== null) {
        const factor = 10 ** this.groupedNumberDecimals;
        const rounded = Math.round((this.value + Number.EPSILON) * factor) / factor;
        if (rounded !== this.value) {
          this.value = rounded;
          this.onChange(rounded);
        }
      }
      this.element.nativeElement.value = this.formatNumber(this.value);
    }
  }

  private formatNumber(value: number): string {
    const raw = this.groupedNumberDecimals === null
      ? String(value)
      : value.toFixed(this.groupedNumberDecimals);
    return this.groupRawValue(raw);
  }

  private groupRawValue(value: string): string {
    const separatorIndex = value.search(/[.,]/);
    const integerPart = separatorIndex >= 0 ? value.slice(0, separatorIndex) : value;
    const decimalPart = separatorIndex >= 0 ? value.slice(separatorIndex) : '';
    return integerPart.replace(/\B(?=(\d{3})+(?!\d))/g, ' ') + decimalPart;
  }

  private toFiniteNumber(value: unknown): number | null {
    if (value === '' || value === null || value === undefined || typeof value === 'boolean') return null;
    const numeric = Number(value);
    return Number.isFinite(numeric) ? numeric : null;
  }

  private restoreCursor(input: HTMLInputElement, significantCharacters: number): void {
    let cursor = 0;
    let seen = 0;
    while (cursor < input.value.length && seen < significantCharacters) {
      if (input.value[cursor] !== ' ') seen++;
      cursor++;
    }
    input.setSelectionRange(cursor, cursor);
  }
}
