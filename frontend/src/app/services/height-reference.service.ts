import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { BASE_URL } from '../api-config';
import { HeightReferenceStatus } from '../models/height-reference.models';

@Injectable({ providedIn: 'root' })
export class HeightReferenceService {
  constructor(private readonly http: HttpClient) {}

  get(): Observable<HeightReferenceStatus> {
    return this.http.get<HeightReferenceStatus>(`${BASE_URL}/height-offset/reference`);
  }

  setEnabled(enabled: boolean): Observable<HeightReferenceStatus> {
    return this.http.post<HeightReferenceStatus>(`${BASE_URL}/height-offset/reference`, { enabled });
  }
}
