import { TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting, HttpTestingController } from '@angular/common/http/testing';
import { ImageViewerComponent } from './image-viewer.component';
import { SharedService } from '../../shared.service';
import { BASE_URL } from '../../api-config';

describe('ImageViewerComponent gallery', () => {
  beforeEach(() => TestBed.configureTestingModule({
    imports: [ImageViewerComponent], providers: [provideHttpClient(), provideHttpClientTesting()]
  }));

  it('replays captures after view changes and labels actual optics and grid coordinates', () => {
    const shared = TestBed.inject(SharedService);
    shared.emitSavedImage({ path: 'C:/sample.jpg', tabletIndex: 0,
      metadata: { wavelength: 'uv310', filter_name: 'Piros', tray_position: 'B1', Errors: [] } });
    const fixture = TestBed.createComponent(ImageViewerComponent);
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('.optics-badge').textContent).toContain('310 nm');
    expect(fixture.nativeElement.querySelector('.optics-badge').textContent).toContain('Piros');
    expect(fixture.nativeElement.querySelector('.tray-badge').textContent).toBe('B1');
    fixture.destroy();
    const second = TestBed.createComponent(ImageViewerComponent);
    second.detectChanges();
    expect(second.componentInstance.savedImages.length).toBe(1);
    second.componentInstance.clearGallery();
    expect(second.componentInstance.savedImages.length).toBe(0);
    second.destroy();
    TestBed.inject(HttpTestingController).verify();
  });

  it('loads EXIF metadata for legacy capture notifications and omits off-grid labels', () => {
    const shared = TestBed.inject(SharedService);
    const fixture = TestBed.createComponent(ImageViewerComponent);
    fixture.detectChanges();
    shared.emitSavedImage({ path: 'C:/legacy.jpg', tabletIndex: 5 });
    TestBed.inject(HttpTestingController).expectOne(request => request.url === `${BASE_URL}/image-metadata`)
      .flush({ metadata: { wavelength: 'uv255', filter_name: 'Kék', tray_position: null } });
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('.optics-badge').textContent).toContain('255 nm');
    expect(fixture.nativeElement.querySelector('.tray-badge')).toBeNull();
    fixture.destroy();
    TestBed.inject(HttpTestingController).verify();
  });

  it('keeps only the 24 most recent thumbnails', () => {
    const shared = TestBed.inject(SharedService);
    const fixture = TestBed.createComponent(ImageViewerComponent);
    fixture.nativeElement.style.display = 'block';
    fixture.nativeElement.style.width = '500px';
    fixture.detectChanges();

    for (let index = 0; index < 30; index += 1) {
      shared.emitSavedImage({
        path: `C:/capture-${index}.jpg`,
        tabletIndex: index,
        metadata: { wavelength: 'vis', Errors: [] }
      });
    }
    fixture.detectChanges();

    expect(fixture.componentInstance.savedImages.length).toBe(24);
    expect(fixture.componentInstance.savedImages[0].path).toBe('C:/capture-29.jpg');
    expect(fixture.componentInstance.savedImages[23].path).toBe('C:/capture-6.jpg');
    expect(fixture.nativeElement.querySelectorAll('.thumb').length).toBe(24);
    const galleryScroll = fixture.nativeElement.querySelector('.gallery-scroll') as HTMLElement;
    expect(galleryScroll.clientWidth).toBeLessThanOrEqual(fixture.nativeElement.clientWidth);
    expect(galleryScroll.scrollWidth).toBeGreaterThan(galleryScroll.clientWidth);
    expect(getComputedStyle(galleryScroll).overflowX).toBe('auto');

    fixture.destroy();
    TestBed.inject(HttpTestingController).verify();
  });
});
