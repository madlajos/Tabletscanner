import { of, Subject } from 'rxjs';
import { ComparisonPanelComponent } from './comparison-panel.component';
import { MontageResponse, RecipeService } from '../../services/recipe.service';
import { createEmptyPipeline, PreviewResponse } from '../../models/pipeline.models';

describe('Comparison montage', () => {
  function setup(count = 3) {
    const api = jasmine.createSpyObj<RecipeService>('api', ['getStepImagesMontage', 'previewStep']);
    const montage: MontageResponse = { success: true, montage_base64: 'grid', image_count: count,
      montage_width: 606, montage_height: 666, grid_rows: 2, grid_cols: 2,
      cell_width: 300, cell_height: 300, label_height: 30 };
    api.getStepImagesMontage.and.returnValue(of(montage));
    api.previewStep.and.returnValue(of({ success: true, executed_up_to: 2, image_base64: 'full' }));
    const component = new ComparisonPanelComponent(api);
    component.panel = { label: 'Node', imageSrc: 'initial', imageCount: count,
      context: { pipeline: createEmptyPipeline(), stepIndex: 2 } };
    component.ngOnChanges();
    return { api, component };
  }

  it('defaults to montage, opens the selected full image and returns to the cached grid', () => {
    const { api, component } = setup();
    expect(component.montageMode).toBeTrue();
    expect(component.cells).toEqual([0, 1, 2]);
    component.openImage(2);
    expect(api.previewStep).toHaveBeenCalledWith(component.panel.context!.pipeline, 2, 2, false);
    expect(component.imageSrc).toBe('data:image/jpeg;base64,full');
    expect(component.montageMode).toBeFalse();
    component.showMontage();
    expect(component.montageMode).toBeTrue();
    expect(api.getStepImagesMontage).toHaveBeenCalledTimes(1);
  });

  it('shows a single image without requesting a montage', () => {
    const { api, component } = setup(1);
    expect(component.montageMode).toBeFalse();
    expect(api.getStepImagesMontage).not.toHaveBeenCalled();
  });

  it('ignores a pending image after returning to montage or destroying the panel', () => {
    const { api, component } = setup();
    const response = new Subject<PreviewResponse>();
    api.previewStep.and.returnValue(response);
    component.openImage(1);
    component.showMontage();
    response.next({ success: true, executed_up_to: 2, image_base64: 'stale' });
    expect(component.montageMode).toBeTrue();
    expect(component.imageSrc).toBe('initial');
    component.openImage(1);
    component.ngOnDestroy();
    expect(response.observed).toBeFalse();
  });
});
