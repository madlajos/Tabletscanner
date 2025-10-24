import { Component, Input, Output, EventEmitter } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-error-popup',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './error-popup.component.html',
  styleUrls: ['./error-popup.component.css']
})
export class ErrorPopupComponent {
  @Input() message!: string;
  @Input() index!: string;  // Here we can use the error code as the identifier
  @Input() popupStyle: 'default' | 'center' = 'default';
  @Output() close = new EventEmitter<string>();

  dismiss() {
    this.close.emit(this.index);
  }
}
