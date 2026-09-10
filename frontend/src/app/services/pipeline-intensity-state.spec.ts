import { fakeAsync, tick } from '@angular/core/testing';
import { of, throwError } from 'rxjs';
import { PipelineStateService } from './pipeline-state.service';
import { RecipeService } from './recipe.service';
import { createEmptyPipeline, createStepInstance, StepDefinition } from '../models/pipeline.models';

describe('Intensity preview state integration', () => {
  function setup() {
    const api = jasmine.createSpyObj<RecipeService>('RecipeService', ['previewStep', 'validatePipeline', 'getStepCatalog']);
    api.validatePipeline.and.returnValue(of({ valid: true, errors: [] }));
    api.previewStep.and.returnValue(of({ success: true, executed_up_to: 0 }));
    const definition = (id: string, input_type: any, output_type: any, secondary_inputs?: string[]): StepDefinition => ({
      id, name: id, category: 'test', description: '', icon: '', input_type, output_type,
      params: [], side_output_types: {}, secondary_inputs,
    });
    api.getStepCatalog.and.returnValue(of([
      definition('load_image', 'IMAGE', 'IMAGE'),
      definition('calculate_intensity_stats', 'GRAYSCALE', 'GRAYSCALE'),
      definition('calculate_histograms', 'GRAYSCALE', 'GRAYSCALE'),
      definition('add_sequence_values', 'IMAGE', 'IMAGE'),
      definition('fit_curve', 'IMAGE', 'IMAGE', ['add_sequence_values']),
    ]));
    const state = new PipelineStateService(api);
    state.loadCatalog();
    state.loadPipeline({ ...createEmptyPipeline(), steps: ['load_image', 'calculate_intensity_stats', 'fit_curve'].map((id, i) => createStepInstance(id, i)) });
    tick(400);
    return { api, state };
  }
  it('retains curve errors through upstream previews and clears them after a successful fit', fakeAsync(() => {
    const { api, state } = setup();
    api.previewStep.and.returnValue(of({ success: false, executed_up_to: 2,
      errors: [{ step_index: 2, step_def_id: 'fit_curve', error_code: 'E2706', message: 'Nincs elég adat.' }] }));
    state.requestPreviewForStep(2);
    expect(state.getStepErrors(2).length).toBe(1);
    api.previewStep.and.returnValue(of({ success: true, executed_up_to: 1 }));
    state.requestPreviewForStep(1);
    state.validate();
    expect(state.getStepErrors(2).length).toBe(1);
    state.requestPreviewForStep(2);
    expect(state.getStepErrors(2)).toEqual([]);
  }));
  it('requests the entire batch for pooled and grouped intensity views', fakeAsync(() => {
    const { api, state } = setup();
    for (const mode of ['per_image', 'pooled', 'grouped']) {
      state.updateParams(1, { display_mode: mode });
      tick(400);
      state.requestPreviewForStep(1);
      expect(api.previewStep.calls.mostRecent().args[3]).toBe(mode === 'per_image');
    }
  }));
  it('requests the entire batch for pooled and grouped histogram views', fakeAsync(() => {
    const { api, state } = setup();
    state.loadPipeline({ ...createEmptyPipeline(), steps: [
      createStepInstance('load_image', 0), createStepInstance('calculate_histograms', 1),
    ] });
    tick(400);
    for (const mode of ['per_image', 'pooled', 'grouped']) {
      state.updateParams(1, { display_mode: mode });
      tick(400);
      state.requestPreviewForStep(1);
      expect(api.previewStep.calls.mostRecent().args[3]).toBe(mode === 'per_image');
    }
  }));
  it('marks the requested node when the preview request fails', fakeAsync(() => {
    const { api, state } = setup();
    api.previewStep.and.returnValue(throwError(() => new Error('offline')));
    state.requestPreviewForStep(2);
    expect(state.getStepErrors(2).length).toBe(1);
  }));
  it('allows curve fitting directly after intensity statistics', fakeAsync(() => {
    const { state } = setup();
    state.loadPipeline({ ...createEmptyPipeline(), steps: [
      createStepInstance('load_image', 0),
      createStepInstance('calculate_intensity_stats', 1),
    ] });
    tick(400);
    expect(state.getStepOutputType(1)).toBe('GRAYSCALE');
    expect(state.canInsertStepAtFlatIndex('fit_curve', 2)).toBeTrue();
  }));
});
