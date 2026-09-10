// popup.interceptor.ts
import { HttpInterceptorFn } from '@angular/common/http';
import { inject } from '@angular/core';
import { catchError, tap, throwError } from 'rxjs';
import { HttpRequest, HttpHandlerFn, HttpEvent, HttpResponse, HttpErrorResponse } from '@angular/common/http';
import { Observable } from 'rxjs';
import { ErrorNotificationService, AppError } from '../services/error-notification.service';
import { CaptureWarning } from '../models/capture-metadata.models';

interface ApiResponse {
  warnings?: CaptureWarning[];
  popup?: boolean;
  error?: string;
  code?: string;
}

export const popupInterceptor: HttpInterceptorFn = (
  req: HttpRequest<any>,
  next: HttpHandlerFn
): Observable<HttpEvent<any>> => {
  const errorNotificationService = inject(ErrorNotificationService);

  return next(req).pipe(
    tap((event) => {
      if (event instanceof HttpResponse) {
        const body = event.body as ApiResponse;
        errorNotificationService.addWarnings(body?.warnings);
        if (body?.popup) {
          const code = body.code || 'GENERIC';
          const message = errorNotificationService.getMessage(code);
          console.debug("Interceptor tap: adding error", { code, message });
          errorNotificationService.addError({ code, message });
        }
      }
    }),
    catchError((error: HttpErrorResponse) => {
      console.debug("Interceptor caught error:", error);
      const errorBody = error.error as ApiResponse;
      errorNotificationService.addWarnings(errorBody?.warnings);
      if (errorBody?.popup === false) return throwError(() => error);
      const code = errorBody?.code || 'GENERIC';
      const message = errorNotificationService.getMessage(code);
      console.debug("Interceptor processing error with code:", code, "message:", message);
      let appError: AppError = { code, message };
      errorNotificationService.addError(appError);
      return throwError(() => error);
    })
    
  );
};
