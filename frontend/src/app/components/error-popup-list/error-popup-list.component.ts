import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ErrorNotificationService, AppError } from '../../services/error-notification.service';
import { Observable } from 'rxjs';
import { map } from 'rxjs/operators';
import { ErrorPopupComponent } from '../error-popup/error-popup.component';
import { CenterErrorPopupComponent } from '../center-error-popup/center-error-popup.component';

@Component({
  selector: 'app-error-popup-list',
  standalone: true,
  imports: [CommonModule, ErrorPopupComponent, CenterErrorPopupComponent],
  templateUrl: './error-popup-list.component.html',
  styleUrls: ['./error-popup-list.component.css']
})
export class ErrorPopupListComponent {
  // Default errors that use the standard style.
  defaultErrors$: Observable<AppError[]>;
  // Center errors that should appear in a modal-like overlay.
  centerErrors$: Observable<AppError[]>;

  constructor(private errorNotificationService: ErrorNotificationService) {
    const allErrors$ = this.errorNotificationService.errors$;
    this.defaultErrors$ = allErrors$.pipe(
      map(errors => errors.filter(err => !err.popupStyle || err.popupStyle === 'default'))
    );
    this.centerErrors$ = allErrors$.pipe(
      map(errors => errors.filter(err => err.popupStyle === 'center'))
    );
  }

  dismissError(code: string): void {
    this.errorNotificationService.removeError(code);
  }
}
