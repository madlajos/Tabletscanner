import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ErrorNotificationService, AppError } from '../../services/error-notification.service';
import { combineLatest, Observable } from 'rxjs';
import { map } from 'rxjs/operators';
import { ErrorPopupComponent } from '../error-popup/error-popup.component';
import { CenterErrorPopupComponent } from '../center-error-popup/center-error-popup.component';
import { SharedService, ToolbarNotice } from '../../shared.service';

type NotificationSeverity = 'error' | 'warning' | 'info' | 'success';
interface StackNotification extends AppError { severity: NotificationSeverity; }
interface NotificationStackView {
  active: StackNotification | null;
  total: number;
  badges: Array<{ severity: NotificationSeverity; count: number; label: string }>;
}

const AUTO_NOTICE_CODE = 'AUTO_MEASUREMENT_NOTICE';

@Component({
  selector: 'app-error-popup-list',
  standalone: true,
  imports: [CommonModule, ErrorPopupComponent, CenterErrorPopupComponent],
  templateUrl: './error-popup-list.component.html',
  styleUrls: ['./error-popup-list.component.css']
})
export class ErrorPopupListComponent {
  stack$: Observable<NotificationStackView>;
  // Center errors that should appear in a modal-like overlay.
  centerErrors$: Observable<AppError[]>;

  constructor(
    private errorNotificationService: ErrorNotificationService,
    private sharedService: SharedService,
  ) {
    const allErrors$ = this.errorNotificationService.errors$;
    const defaultErrors$ = allErrors$.pipe(
      map(errors => errors.filter(err => !err.popupStyle || err.popupStyle === 'default'))
    );
    this.stack$ = combineLatest([defaultErrors$, this.sharedService.toolbarNotice$]).pipe(
      map(([errors, notice]) => this.createStackView(errors, notice))
    );
    this.centerErrors$ = allErrors$.pipe(
      map(errors => errors.filter(err => err.popupStyle === 'center'))
    );
  }

  dismissError(code: string): void {
    if (code === AUTO_NOTICE_CODE) {
      this.sharedService.clearToolbarNotice();
      return;
    }
    this.errorNotificationService.removeError(code);
  }

  private createStackView(errors: AppError[], notice: ToolbarNotice | null): NotificationStackView {
    const notifications: StackNotification[] = [
      ...(notice ? [{ code: AUTO_NOTICE_CODE, message: notice.message, severity: notice.severity }] : []),
      ...errors.map(error => ({ ...error, severity: error.severity ?? 'error' as NotificationSeverity })),
    ];
    const badgeLabels: Record<NotificationSeverity, string> = {
      error: 'hiba', warning: 'figyelmeztetés', info: 'információ', success: 'sikeres üzenet'
    };
    const severities: NotificationSeverity[] = ['error', 'warning', 'info', 'success'];
    const badges = notifications.length > 1
      ? severities.map(severity => ({
          severity,
          count: notifications.filter(item => item.severity === severity).length,
          label: badgeLabels[severity],
        })).filter(item => item.count > 0)
      : [];
    return {
      active: notifications.at(-1) ?? null,
      total: notifications.length,
      badges,
    };
  }
}
