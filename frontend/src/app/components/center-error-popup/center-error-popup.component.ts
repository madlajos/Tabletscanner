import { Component, Input, Output, EventEmitter } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-center-error-popup',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './center-error-popup.component.html',
  styleUrls: ['./center-error-popup.component.css']
})
export class CenterErrorPopupComponent {
  @Input() message!: string;
  @Input() index!: string;
  @Output() close = new EventEmitter<string>();

  dismiss() {
    this.close.emit(this.index);
  }
}
