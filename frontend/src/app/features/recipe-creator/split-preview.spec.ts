import { ChangeDetectorRef } from '@angular/core';
import { Subject } from 'rxjs';
import { PipelinePreviewComponent } from './pipeline-preview.component';
import { PipelineCanvasComponent } from './pipeline-canvas.component';
import { PipelineStateService } from '../../services/pipeline-state.service';
import { RecipeService } from '../../services/recipe.service';
import { createEmptyPipeline, createStepInstance, PreviewResponse } from '../../models/pipeline.models';

describe('Selected node comparison', () => {
  const pipeline = { ...createEmptyPipeline(), steps: [0, 1, 2].map(i => createStepInstance('test', i)) };

  it('compares the clicked node with the previous selection even when not adjacent', () => {
    const state = jasmine.createSpyObj<PipelineStateService>('state', ['getPipeline', 'selectStep', 'requestSplitPreview']);
    state.getPipeline.and.returnValue(pipeline);
    const canvas = new PipelineCanvasComponent(state);
    canvas.selectedIndex = 0;
    canvas.onSelect({ step: pipeline.steps[2], pipelineIndex: 2 }, new MouseEvent('click', { ctrlKey: true }));
    expect(state.requestSplitPreview).toHaveBeenCalledWith(2, 0);
    expect(canvas.selectedNodeIds.size).toBe(2);
  });

  it('renders both requested outputs and discards a cancelled comparison', () => {
    const state = jasmine.createSpyObj<PipelineStateService>('state', ['getPreviewContext', 'getStepDefinition']);
    state.getPreviewContext.and.callFake(index => ({ pipeline, stepIndex: index, startIndex: 0 }));
    const api = jasmine.createSpyObj<RecipeService>('api', ['previewStep']);
    const responses: Subject<PreviewResponse>[] = [];
    api.previewStep.and.callFake(() => {
      const response = new Subject<PreviewResponse>();
      responses.push(response);
      return response;
    });
    const preview = new PipelinePreviewComponent(state, {} as ChangeDetectorRef, api);
    preview['currentPipeline'] = pipeline;
    spyOn(preview, 'resetZoom');
    preview['loadNodeSplitPreview']([0, 2]);
    expect(api.previewStep.calls.allArgs().map(args => args[1])).toEqual([0, 2]);
    const complete = (index: number, image: string) => {
      responses[index].next({ success: true, executed_up_to: 2, image_base64: image });
      responses[index].complete();
    };
    complete(0, 'first');
    complete(1, 'second');
    expect(preview.branchMergePanels.map(panel => panel.imageSrc)).toEqual([
      'data:image/jpeg;base64,first', 'data:image/jpeg;base64,second',
    ]);
    expect(preview.splitPreviewActive).toBeTrue();
    preview['loadNodeSplitPreview']([0, 1]);
    preview['clearNodeSplitPreview']();
    complete(2, 'stale');
    complete(3, 'stale');
    expect(preview.splitPreviewActive).toBeFalse();
    expect(preview.branchMergePanels).toEqual([]);
  });
});
