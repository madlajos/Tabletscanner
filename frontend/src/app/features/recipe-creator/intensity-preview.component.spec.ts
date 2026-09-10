import { TestBed } from '@angular/core/testing';
import { IntensityPreviewComponent } from './intensity-preview.component';
import { IntensitySettingsComponent } from './intensity-settings.component';
import { IntensityStatistics, IntensitySummary } from '../../models/intensity.models';

describe('Intensity split preview', () => {
  const stat: IntensityStatistics = { min: 10, max: 20, mean: 15, median: 15, std: 5, pixel_count: 2, dynamic_range: null };
  const summary: IntensitySummary = { mode: 'per_image', groups: [
    { label: 'Első', image_indices: [0], sample_count: 1, channels: [stat] },
    { label: 'Második', image_indices: [1], sample_count: 1, channels: [stat] },
  ], samples: [
    { label: '1', image_index: 0, channels: [stat] },
    { label: '2', image_index: 1, channels: [{ ...stat, mean: 18 }] },
  ] };
  it('renders only the selected sample next to its image', () => {
    const fixture = TestBed.createComponent(IntensityPreviewComponent);
    fixture.componentRef.setInput('summary', summary);
    fixture.componentRef.setInput('imageIndex', 1);
    fixture.componentRef.setInput('imageSrc', 'data:image/png;base64,');
    fixture.detectChanges();
    const element: HTMLElement = fixture.nativeElement;
    expect(element.querySelector('.image img')).not.toBeNull();
    expect(element.querySelector('.results')?.textContent).toContain('Második');
    expect(element.querySelector('.results')?.textContent).not.toContain('Első');
    expect(element.querySelector('table')?.textContent).toContain('Átlag');
  });
  it('renders all CSV groups regardless of the currently selected image', () => {
    const fixture = TestBed.createComponent(IntensityPreviewComponent);
    fixture.componentRef.setInput('summary', { ...summary, mode: 'grouped' });
    fixture.componentRef.setInput('imageIndex', 1);
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelectorAll('table').length).toBe(2);
  });
  it('imports the selected CSV column through the settings action', () => {
    const fixture = TestBed.createComponent(IntensitySettingsComponent);
    fixture.componentRef.setInput('mode', 'grouped');
    const component = fixture.componentInstance;
    component.rows = [['Kép', 'Csoport'], ['1', 'A'], ['2', 'B']];
    component.position = 2; component.skipFirst = true; component.refresh();
    const emit = spyOn(component.labelsChange, 'emit');
    fixture.detectChanges();
    fixture.nativeElement.querySelector('button').click();
    expect(emit).toHaveBeenCalledWith(['A', 'B']);
  });
  it('plots image numbers without CSV grouping and the selected metric on Y', () => {
    const fixture = TestBed.createComponent(IntensityPreviewComponent);
    fixture.componentRef.setInput('summary', summary);
    fixture.componentRef.setInput('chartEnabled', true);
    fixture.componentRef.setInput('chartMetric', 'mean');
    fixture.detectChanges();
    const element: HTMLElement = fixture.nativeElement;
    expect(element.querySelector('.chart')).not.toBeNull();
    expect(Array.from(element.querySelectorAll('.axis-label')).some(axis => axis.textContent?.includes('Átlag'))).toBeTrue();
    expect(element.querySelectorAll('.x-label').length).toBe(2);
    expect(element.querySelectorAll('circle').length).toBe(2);
    expect(element.querySelector('table')).toBeNull();
  });
  it('plots one point for each CSV group', () => {
    const fixture = TestBed.createComponent(IntensityPreviewComponent);
    fixture.componentRef.setInput('summary', { ...summary, mode: 'grouped' });
    fixture.componentRef.setInput('chartEnabled', true);
    fixture.componentRef.setInput('chartMetric', 'mean');
    fixture.detectChanges();
    const labels = Array.from(fixture.nativeElement.querySelectorAll('.x-label')).map((el: any) => el.textContent.trim());
    expect(labels).toEqual(['Első', 'Második']);
  });
  it('shows the chart controls only when charting is enabled', () => {
    const fixture = TestBed.createComponent(IntensitySettingsComponent);
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelectorAll('select').length).toBe(1);
    fixture.componentRef.setInput('chartEnabled', true);
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelectorAll('select').length).toBe(2);
    expect(fixture.componentInstance.metricOptions.some(option => option.value === 'p95')).toBeTrue();
  });
});
