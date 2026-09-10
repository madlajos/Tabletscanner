import { TestBed } from '@angular/core/testing';
import { ParticleSizeDistributionPreviewComponent } from './particle-size-distribution-preview.component';

describe('ParticleSizeDistributionPreviewComponent', () => {
  it('lists every particle, highlights the clicked contour and emits exclusion', async () => {
    await TestBed.configureTestingModule({ imports: [ParticleSizeDistributionPreviewComponent] }).compileComponents();
    const fixture = TestBed.createComponent(ParticleSizeDistributionPreviewComponent);
    const component = fixture.componentInstance;
    component.imageSrc = 'data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7';
    component.distribution = {
      mode: 'per_image', unit: 'px', size_label: 'Diameter (px)', groups: [],
      particles: Array.from({ length: 178 }, (_, index) => ({
        particle_id: `img3_label${index + 1}`, label: index + 1, image_index: 3,
        value: index + 0.5, excluded: false, polygon: [[1, 2], [3, 4], [5, 2]],
      })),
    };
    fixture.detectChanges();
    const rows = fixture.nativeElement.querySelectorAll('tbody tr') as NodeListOf<HTMLTableRowElement>;
    expect(rows.length).toBe(178);
    rows[100].click();
    fixture.detectChanges();
    expect(rows[100].classList.contains('selected')).toBeTrue();
    const polygon = fixture.nativeElement.querySelector('.particle-image polygon');
    expect(polygon.getAttribute('points')).toBe('1,2 3,4 5,2');
    expect(polygon.getAttribute('stroke')).toBe('#ffff00');
    const emit = spyOn(component.toggleExcluded, 'emit');
    rows[100].querySelector('button')!.click();
    expect(emit).toHaveBeenCalledWith('img3_label101');
    component.distribution.particles[100].excluded = true;
    fixture.detectChanges();
    expect(rows[100].classList.contains('excluded')).toBeTrue();
    expect(rows[100].querySelector('button')!.textContent).toContain('Visszavétel');
    expect(fixture.nativeElement.textContent).not.toContain('Dv10');
  });
});
