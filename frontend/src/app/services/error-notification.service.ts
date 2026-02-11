// error-notification.service.ts
import { Injectable } from '@angular/core';
import { BehaviorSubject, of } from 'rxjs';
import { HttpClient } from '@angular/common/http';
import { catchError, tap } from 'rxjs/operators';

export interface AppError {
  code: string;
  message: string;
  popupStyle?: 'default' | 'center';
  abortMeasurement?: boolean;
}


@Injectable({ providedIn: 'root' })
export class ErrorNotificationService {
  private errorsSubject = new BehaviorSubject<AppError[]>([]);
  errors$ = this.errorsSubject.asObservable();

  private errorMapping: { [code: string]: string } = {};

  constructor(private http: HttpClient) {}

  loadErrorMapping(): Promise<void> {
    return this.http.get<{ [code: string]: string }>('assets/error_messages.json')
      .pipe(
        tap(mapping => { this.errorMapping = mapping; }),
        catchError((error) => {
          console.error('Failed to load error mapping:', error);
          // Even if loading fails, we use an empty mapping
          this.errorMapping = {};
          return of({});
        })
      ).toPromise().then(() => { });
  }

  getMessage(code: string): string {
    const msg = this.errorMapping[code] || this.errorMapping['GENERIC'] || 'An error occurred.';
    console.log(`getMessage('${code}') returns: ${msg}`);
    return msg;
  }
  

  addError(error: AppError): void {
    // Respect explicitly provided popupStyle. Only auto-set for measurement errors if not provided.
    if (!error.popupStyle && error.code && (error.code.startsWith("E2") || error.code.startsWith("E13"))) {
      error.popupStyle = 'center';
      error.abortMeasurement = true;
    }
    
    const currentErrors = this.errorsSubject.value;
    const existingIndex = currentErrors.findIndex(err => err.code === error.code);
    
    if (existingIndex === -1) {
      // New error — add it
      if (!error.message) {
        error.message = this.getMessage(error.code);
      }
      console.debug("Adding error to subject:", error);
      this.errorsSubject.next([...currentErrors, error]);
    } else if (error.popupStyle === 'center' && currentErrors[existingIndex].popupStyle !== 'center') {
      // Replace existing error with center-popup version (e.g., after 30s reconnection timeout)
      const updated = [...currentErrors];
      updated[existingIndex] = { ...currentErrors[existingIndex], ...error };
      console.debug("Updating error with center-error-popup:", updated[existingIndex]);
      this.errorsSubject.next(updated);
    }
  }
  
  
  removeError(code: string): void {
    const currentErrors = this.errorsSubject.value.filter(err => err.code !== code);
    this.errorsSubject.next(currentErrors);
  }
}
