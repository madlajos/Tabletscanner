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
    if (error.code && error.code.startsWith("E2")) {
      error.popupStyle = 'center';
      error.abortMeasurement = true;
    }
    const currentErrors = this.errorsSubject.value;
    if (!currentErrors.find(err => err.code === error.code)) {
      if (!error.message) {
        error.message = this.getMessage(error.code);
      }
      console.debug("Adding error to subject:", error);
      this.errorsSubject.next([...currentErrors, error]);
    }
  }
  
  
  removeError(code: string): void {
    const currentErrors = this.errorsSubject.value.filter(err => err.code !== code);
    this.errorsSubject.next(currentErrors);
  }
}
