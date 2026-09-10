import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { BASE_URL } from '../api-config';
import { CaptureMetadata } from '../models/capture-metadata.models';
import { map } from 'rxjs';

@Injectable({ providedIn: 'root' })
export class ImageGalleryService {
  constructor(private readonly http: HttpClient) {}

  save(targetFolder: string) {
    return this.http.post<{ path: string; metadata: CaptureMetadata }>(`${BASE_URL}/save_raw_image`, {
      target_folder: targetFolder.replace(/\\/g, '/')
    });
  }

  metadata(path: string) {
    return this.http.get<{ metadata: CaptureMetadata }>(`${BASE_URL}/image-metadata`, { params: { path } }).pipe(map(response => response.metadata));
  }
}
