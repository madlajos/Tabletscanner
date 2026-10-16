import { TestBed } from '@angular/core/testing';
import { PCAChartComponent } from './pca-chart.component';

describe('PCA output chart', () => {
  it('selects PC2 on the Y axis and reports cumulative variance without double counting', async () => {
    TestBed.configureTestingModule({ imports: [PCAChartComponent] });
    const fixture = TestBed.createComponent(PCAChartComponent);
    fixture.componentRef.setInput('data', {
      scores: [[1, 2], [3, 4], [5, 6]], explained_ratio: [0.6, 0.3], cumulative_ratio: [0.6, 0.9],
    });
    fixture.detectChanges();
    await fixture.whenStable();
    expect((fixture.nativeElement.querySelector('[aria-label="PCA Y tengely"]') as HTMLSelectElement).value).toBe('1');
    expect(fixture.nativeElement.querySelector('.pca-info').textContent).toContain('90.0%');
    expect(fixture.nativeElement.querySelector('.pca-info').textContent).not.toContain('150.0%');
    fixture.destroy();
  });
});
