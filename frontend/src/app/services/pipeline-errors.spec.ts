import { PipelineErrors } from './pipeline-errors';
import { createEmptyPipeline, createStepInstance, StepError } from '../models/pipeline.models';

describe('Pipeline error ownership', () => {
  const makeDoc = () => ({ ...createEmptyPipeline(), steps: ['load_image', 'calculate_intensity_stats', 'fit_curve'].map((id, i) => createStepInstance(id, i)) });
  const failure: StepError = { step_index: 2, step_def_id: 'fit_curve', error_code: 'E2706', message: 'Nincs elég adat.' };
  it('retains a curve failure when an earlier preview and validation succeed', () => {
    const errors = new PipelineErrors(), doc = makeDoc();
    errors.recordPreview(doc, 0, 2, [failure]);
    errors.recordPreview(doc, 0, 1, []);
    errors.setValidation([]);
    expect(errors.all(doc)).toEqual([failure]);
    errors.recordPreview(doc, 0, 2, []);
    expect(errors.all(doc)).toEqual([]);
  });
  it('keeps structural downstream errors through successful earlier previews', () => {
    const errors = new PipelineErrors(), doc = makeDoc();
    errors.setValidation([failure]);
    errors.recordPreview(doc, 0, 1, []);
    expect(errors.all(doc)).toEqual([failure]);
  });
  it('tracks runtime errors by instance across reorder and ignores removed or disabled steps', () => {
    const errors = new PipelineErrors(), doc = makeDoc();
    errors.recordPreview(doc, 0, 2, [failure]);
    [doc.steps[1], doc.steps[2]] = [doc.steps[2], doc.steps[1]];
    expect(errors.all(doc)[0].step_index).toBe(1);
    doc.steps[1].enabled = false;
    expect(errors.all(doc)).toEqual([]);
    doc.steps.splice(1, 1);
    expect(errors.all(doc)).toEqual([]);
  });
});
