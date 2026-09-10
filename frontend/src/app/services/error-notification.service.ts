// error-notification.service.ts
import { Injectable } from '@angular/core';
import { BehaviorSubject, of } from 'rxjs';
import { HttpClient } from '@angular/common/http';
import { catchError, tap } from 'rxjs/operators';
import { CaptureWarning } from '../models/capture-metadata.models';

export interface AppError {
  severity?: 'error' | 'warning' | 'info' | 'success';
  code: string;
  message: string;
  popupStyle?: 'default' | 'center';
  abortMeasurement?: boolean;
}

const CENTER_POPUP_CODES = new Set([
  'E1202', // Homing rejected: endstop / Hall sensor was not reached.
  'E1203', // Homing timed out.
]);

@Injectable({ providedIn: 'root' })
export class ErrorNotificationService {
  private readonly seenWarnings = new Set<string>();

  addWarnings(warnings: CaptureWarning[] = []): void {
    for (const warning of warnings) {
      if (this.seenWarnings.has(warning.id)) continue;
      this.seenWarnings.add(warning.id);
      if (this.seenWarnings.size > 1000) this.seenWarnings.delete(this.seenWarnings.values().next().value!);
      const message = this.getMessage(warning.code)
        .replace('{target_z}', String(warning.target_z))
        .replace('{missing_offset_mm}', String(warning.missing_offset_mm));
      this.addError({ code: warning.id, message, severity: 'warning' });
    }
  }
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
    if (
      !error.popupStyle
      && error.code
      && (
        error.code.startsWith('E2')
        || error.code.startsWith('E13')
        || CENTER_POPUP_CODES.has(error.code)
      )
    ) {
      error.popupStyle = 'center';
      // Homing errors use the same centered presentation as analysis errors,
      // but only measurement/profile error families own measurement abort.
      if (error.code.startsWith('E2') || error.code.startsWith('E13')) {
        error.abortMeasurement = true;
      }
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
