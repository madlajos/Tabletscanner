import { TestBed } from '@angular/core/testing';
import { HistogramPreviewComponent } from './histogram-preview.component';
import { HistogramSummary } from '../../models/histogram.models';

describe('Histogram split preview', () => {
  const summary: HistogramSummary = {
    mode: 'per_image',
    samples: [
      { label: '1', image_index: 0, histogram: [1, 2], stats: {} },
      { label: '2', image_index: 1, histogram: [3, 4], stats: {} },
    ],
    groups: [],
  };
  it('shows only the selected image histogram beside the image', () => {
    const fixture = TestBed.createComponent(HistogramPreviewComponent);
    fixture.componentRef.setInput('summary', summary);
    fixture.componentRef.setInput('imageIndex', 1);
    fixture.componentRef.setInput('imageSrc', 'data:image/png;base64,');
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('.image img')).not.toBeNull();
    expect(fixture.nativeElement.querySelector('.legend').textContent).toContain('Kép 2');
    expect(fixture.nativeElement.querySelectorAll('.hist-line').length).toBe(1);
  });
  it('shows one histogram per CSV group', () => {
    const fixture = TestBed.createComponent(HistogramPreviewComponent);
    fixture.componentRef.setInput('summary', { ...summary, mode: 'grouped', groups: [
      { label: 'A', indices: [0], sample_count: 1, histogram: [1, 2] },
      { label: 'B', indices: [1], sample_count: 1, histogram: [3, 4] },
    ] });
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelectorAll('.hist-line').length).toBe(2);
    expect(fixture.nativeElement.querySelector('.legend').textContent).toContain('A');
    expect(fixture.nativeElement.querySelector('.legend').textContent).toContain('B');
  });
  it('respects the diagram checkbox', () => {
    const fixture = TestBed.createComponent(HistogramPreviewComponent);
    fixture.componentRef.setInput('summary', summary);
    fixture.componentRef.setInput('chartEnabled', false);
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('.chart')).toBeNull();
  });
  it('overlays every image in pooled mode with different colors', () => {
    const fixture = TestBed.createComponent(HistogramPreviewComponent);
    fixture.componentRef.setInput('summary', { ...summary, mode: 'pooled' });
    fixture.detectChanges();
    const lines = fixture.nativeElement.querySelectorAll('.hist-line');
    expect(lines.length).toBe(2);
    expect(lines[0].getAttribute('stroke')).not.toBe(lines[1].getAttribute('stroke'));
    expect(lines[0].getAttribute('points')).toBeTruthy();
  });
});
