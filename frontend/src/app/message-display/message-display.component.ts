import { Component } from '@angular/core';
import { Observable } from 'rxjs';
import { CommonModule } from '@angular/common';
import { ErrorNotificationService, AppError } from '../services/error-notification.service';

@Component({
  selector: 'app-message-display',
  standalone: true,
  imports: [CommonModule], // <-- add CommonModule here
  templateUrl: './message-display.component.html',
  styleUrls: ['./message-display.component.css']
})
export class MessageDisplayComponent {
  // Assume we inject the error notification service and assign its errors observable.
  messages: Observable<AppError[]>;

  constructor(private errorNotificationService: ErrorNotificationService) {
    this.messages = this.errorNotificationService.errors$;
  }

  // Getter method so that you can call getMessages() in the template.
  get getMessages(): Observable<AppError[]> {
    return this.messages;
  }
}
