import { fakeAsync, TestBed, tick } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { BASE_URL } from '../../api-config';
import { FocusAnchorComponent } from './focus-anchor.component';

describe('FocusAnchorComponent', () => {
  const url = `${BASE_URL}/height-offset/reference`;
  const status = (source: 'anchor' | 'autofocus' | null) => ({
    available: source !== null, source, reference_z: source ? 10 : null,
    applied_offset_mm: 2, baseline_offset_mm: 2
  });

  beforeEach(() => TestBed.configureTestingModule({
    imports: [FocusAnchorComponent], providers: [provideHttpClient(), provideHttpClientTesting()]
  }));

  it('anchors only after backend confirmation, ignores old polls and clears on invalidation', fakeAsync(() => {
    const fixture = TestBed.createComponent(FocusAnchorComponent);
    const http = TestBed.inject(HttpTestingController);
    fixture.componentInstance.canAnchor = true;
    fixture.detectChanges();
    tick(0);
    const oldPoll = http.expectOne(url);
    fixture.componentInstance.toggle();
    const command = http.expectOne(url);
    expect(command.request.method).toBe('POST');
    expect(command.request.body).toEqual({ enabled: true });
    expect(fixture.componentInstance.active).toBeFalse();
    command.flush(status('anchor'));
    oldPoll.flush(status(null));
    fixture.detectChanges();
    const button = fixture.nativeElement.querySelector('button') as HTMLButtonElement;
    expect(button.classList.contains('active')).toBeTrue();
    expect(button.getAttribute('aria-pressed')).toBe('true');
    tick(500);
    http.expectOne(url).flush(status(null));
    fixture.detectChanges();
    expect(button.classList.contains('active')).toBeFalse();
    fixture.destroy();
    tick(1000);
    http.verify();
  }));

  it('restores a saved runtime anchor and releases it with a second press', fakeAsync(() => {
    const fixture = TestBed.createComponent(FocusAnchorComponent);
    const http = TestBed.inject(HttpTestingController);
    fixture.detectChanges();
    tick(0);
    http.expectOne(url).flush(status('anchor'));
    fixture.componentInstance.toggle();
    const command = http.expectOne(url);
    expect(command.request.body).toEqual({ enabled: false });
    command.flush(status(null));
    expect(fixture.componentInstance.active).toBeFalse();
    fixture.destroy();
    http.verify();
  }));
});
