import { Injectable } from '@angular/core';
import { BehaviorSubject, timer } from 'rxjs';
import { switchMap } from 'rxjs/operators';
import { HttpClient } from '@angular/common/http';

@Injectable({
  providedIn: 'root'
})
export class BarcodeService {
  private readonly BASE_URL = 'http://localhost:5000/api';
  private barcodeSubject = new BehaviorSubject<string>('');
  barcode$ = this.barcodeSubject.asObservable();

  constructor(private http: HttpClient) {
    // Start polling every 3 seconds.
    timer(0, 1000).pipe(
      switchMap(() => this.http.get<{ barcode: string }>(`${this.BASE_URL}/get-barcode`))
    ).subscribe(response => {
      // Only emit a new barcode if it's different from the current one.
      if (response?.barcode && response.barcode !== this.barcodeSubject.getValue()) {
        this.barcodeSubject.next(response.barcode);
      }
    });
  }

  // Optional: Clear the current barcode (and optionally inform the backend).
  clearBarcode(): void {
    this.http.post(`${this.BASE_URL}/clear-barcode`, {}).subscribe({
      next: () => {
        // Backend is definitely cleared; now update the stream
        this.barcodeSubject.next('');
      },
      error: (err) => {
        console.error('clear-barcode failed:', err);
        // Still clear the subject so the UI resets even if backend failed
        this.barcodeSubject.next('');
      }
    });
  }
}
