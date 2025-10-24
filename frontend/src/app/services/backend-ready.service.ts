import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { interval, of } from 'rxjs';
import { switchMap, catchError } from 'rxjs/operators';

@Injectable({ providedIn: 'root' })
export class BackendReadyService {

  private readonly BASE_URL = 'http://localhost:5000/api';

  constructor(private http: HttpClient) { }

  /**
   * Polls the health-check endpoint every 1000 ms until it returns { ready: true }.
   * Returns a Promise that resolves when backend is ready.
   */
  waitForBackendReady(): Promise<void> {
    return new Promise<void>((resolve, reject) => {
      const pollingSub = interval(1000).pipe(
        switchMap(() =>
          this.http.get<{ ready: boolean }>(`${this.BASE_URL}/health`)
            .pipe(
              catchError(err => {
                // Log error, but continue polling by returning ready: false.
                console.warn('Error polling /health:', err);
                return of({ ready: false });
              })
            )
        )
      ).subscribe(response => {
        console.log('Backend health response:', response);
        if (response.ready) {
          pollingSub.unsubscribe();
          resolve();
        }
      });
    });
  }
}
