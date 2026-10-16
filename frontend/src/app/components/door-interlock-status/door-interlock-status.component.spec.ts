import { TestBed } from '@angular/core/testing';
import { DoorInterlockStatusComponent } from './door-interlock-status.component';

describe('Door status icons', () => {
  it('shows each open sensor separately and announces unknown status', () => {
    const fixture = TestBed.createComponent(DoorInterlockStatusComponent);
    fixture.detectChanges();
    const element: HTMLElement = fixture.nativeElement;
    expect(element.querySelector('[role="status"]')?.getAttribute('aria-live')).toBe('polite');
    expect(element.textContent).toContain('ismeretlen');
    expect(element.querySelectorAll('.door-badge').length).toBe(1);
    fixture.componentInstance.doors = { known: true, front_open: false, top_open: false,
      uv_blocked: false, trip_count: 0 };
    fixture.detectChanges();
    expect(element.querySelectorAll('.door-badge').length).toBe(0);
    expect(element.textContent?.trim()).toBe('');
    fixture.componentInstance.doors = { known: true, front_open: true, top_open: false,
      uv_blocked: true, trip_count: 1 };
    fixture.detectChanges();
    expect(element.querySelectorAll('img').length).toBe(1);
    expect(element.textContent).toContain('Ajtó nyitva');
    expect(element.textContent).not.toContain('zárva');
    expect(element.querySelector('img')?.getAttribute('src')).toContain('front-door-open.svg');
    fixture.componentInstance.doors = { known: true, front_open: false, top_open: true,
      uv_blocked: true, trip_count: 2 };
    fixture.detectChanges();
    expect(element.querySelectorAll('img').length).toBe(1);
    expect(element.querySelector('img')?.getAttribute('src')).toContain('top-cover-open.svg');
    expect(element.textContent).toContain('Fedél nyitva');
    fixture.componentInstance.doors = { known: true, front_open: true, top_open: true,
      uv_blocked: true, trip_count: 3 };
    fixture.detectChanges();
    expect(element.querySelectorAll('img').length).toBe(2);
  });
});
