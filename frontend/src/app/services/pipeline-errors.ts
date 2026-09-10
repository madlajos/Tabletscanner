import { PipelineDocument, StepError } from '../models/pipeline.models';

/** Runtime failures belong to step instances, independently of preview selection. */
export class PipelineErrors {
  private structural: StepError[] = [];
  private runtime = new Map<string, StepError[]>();
  reset(): void { this.structural = []; this.runtime.clear(); }
  setValidation(errors: StepError[]): void { this.structural = errors; }
  recordPreview(doc: PipelineDocument, start: number, end: number, errors: StepError[]): void {
    const failedAt = errors.length ? Math.min(...errors.map(error => error.step_index)) : end + 1;
    for (let index = start; index <= end && index < failedAt; index++) {
      const step = doc.steps[index];
      if (step?.enabled !== false) this.runtime.delete(step.instance_id);
    }
    const failedIds = new Set(errors.map(error => doc.steps[error.step_index]?.instance_id).filter(Boolean));
    for (const id of failedIds) this.runtime.delete(id);
    for (const error of errors) {
      const id = doc.steps[error.step_index]?.instance_id;
      if (id) this.runtime.set(id, [...(this.runtime.get(id) ?? []), error]);
    }
  }
  all(doc: PipelineDocument): StepError[] {
    const errors = [...this.structural];
    doc.steps.forEach((step, index) => {
      if (step.enabled !== false) {
        errors.push(...(this.runtime.get(step.instance_id) ?? []).map(error => ({ ...error, step_index: index })));
      }
    });
    return errors.filter((error, index) => errors.findIndex(other => other.step_index === error.step_index
      && other.error_code === error.error_code && other.message === error.message) === index);
  }
}
