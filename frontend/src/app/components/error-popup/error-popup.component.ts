import { Component, Input, Output, EventEmitter, OnChanges, OnDestroy, SimpleChanges } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-error-popup',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './error-popup.component.html',
  styleUrls: ['./error-popup.component.css']
})
export class ErrorPopupComponent implements OnChanges, OnDestroy {
  @Input() severity: 'error' | 'warning' | 'info' | 'success' = 'error';
  @Input() message!: string;
  @Input() index!: string;  // Here we can use the error code as the identifier
  @Input() popupStyle: 'default' | 'center' = 'default';
  @Input() warningDurationMs = 5000;
  @Output() close = new EventEmitter<string>();

  private warningTimer?: ReturnType<typeof setTimeout>;

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['severity'] || changes['index'] || changes['warningDurationMs']) {
      this.startWarningTimer();
    }
  }

  ngOnDestroy(): void {
    this.clearWarningTimer();
  }

  dismiss(): void {
    this.clearWarningTimer();
    this.close.emit(this.index);
  }

  private startWarningTimer(): void {
    this.clearWarningTimer();
    if (this.severity !== 'warning' || !this.index || this.warningDurationMs <= 0) return;

    this.warningTimer = setTimeout(() => this.dismiss(), this.warningDurationMs);
  }

  private clearWarningTimer(): void {
    if (this.warningTimer !== undefined) {
      clearTimeout(this.warningTimer);
      this.warningTimer = undefined;
    }
  }
}
