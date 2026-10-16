import { fakeAsync, tick } from '@angular/core/testing';
import { PipelineStateService } from './pipeline-state.service';
import { RecipeService } from './recipe.service';
import { createEmptyPipeline, createStepInstance } from '../models/pipeline.models';

describe('External pipeline preview integration', () => {
  it('selects the final enabled node and routes all preview changes through the asynchronous owner', fakeAsync(() => {
    const api = jasmine.createSpyObj<RecipeService>('RecipeService', ['previewStep', 'validatePipeline']);
    const state = new PipelineStateService(api);
    const changes: string[] = [];
    state.externalPreviewRequests$.subscribe(request => changes.push(request.reason));
    const pipeline = { ...createEmptyPipeline(), steps: [createStepInstance('load_image', 0), createStepInstance('fit_curve', 1), { ...createStepInstance('save_images', 2), enabled: false }] };
    let selected = -1;
    state.selectedStepIndex$.subscribe(index => selected = index);
    state.loadExternalPipeline(pipeline);
    expect(selected).toBe(1);
    state.applyExternalPreview({ success: true, executed_up_to: 1, image_count: 3, side_outputs: { values: [1, 2, 3] } });
    state.setPreviewImageIndex(2);
    state.updateParams(1, { model: 'linear' });
    tick(400);
    state.requestPreviewForStep(0);
    expect(changes).toEqual(['image', 'parameters', 'refresh']);
    expect(api.previewStep).not.toHaveBeenCalled();
    expect(api.validatePipeline).not.toHaveBeenCalled();
    state.ngOnDestroy();
  }));

  it('clears prior image and side outputs when a different result is pending', () => {
    const state = new PipelineStateService(jasmine.createSpyObj<RecipeService>('RecipeService', ['previewStep']));
    let image: string | null = null;
    let outputs: Record<string, unknown> = {};
    state.previewImage$.subscribe(value => image = value);
    state.sideOutputs$.subscribe(value => outputs = value);
    state.applyExternalPreview({ success: true, executed_up_to: 1, image_base64: 'old', side_outputs: { mean: 5 } });
    state.applyExternalPreview(null, 0, true);
    expect(image).toBeNull();
    expect(outputs).toEqual({});
    state.ngOnDestroy();
  });
});
