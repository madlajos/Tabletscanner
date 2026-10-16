import { Component, OnDestroy, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Subscription } from 'rxjs';
import { PipelinePreviewComponent } from '../recipe-creator/pipeline-preview.component';
import { PipelineStateService } from '../../services/pipeline-state.service';
import { InlineAnalysisService } from '../../services/inline-analysis.service';
import { InlineAnalysisError, InlineAnalysisSession, InlineAnalysisTablet, InlineInputScope } from '../../models/inline-analysis.models';
import { PipelineDocument, StepDefinition, StepInstance } from '../../models/pipeline.models';
import { InlineOutputSettingsComponent } from './inline-output-settings.component';
import { BASE_URL } from '../../api-config';

@Component({
  selector: 'app-recipe-applier', standalone: true,
  imports: [CommonModule, FormsModule, PipelinePreviewComponent, InlineOutputSettingsComponent],
  providers: [PipelineStateService], templateUrl: './recipe-applier.component.html', styleUrls: ['./recipe-applier.component.css'],
})
export class RecipeApplierComponent implements OnInit, OnDestroy {
  session: InlineAnalysisSession | null = null;
  errors: InlineAnalysisError[] = [];
  selectedResultId: string | null = null;
  resultEntries: { key: string; text: string }[] = [];
  lastStep: StepInstance | undefined;
  lastDefinition: StepDefinition | undefined;
  private readonly subscriptions = new Subscription();
  private pipelineKey = '';
  private sideOutputs: Record<string, unknown> = {};
  constructor(public readonly analysis: InlineAnalysisService, public readonly pipelineState: PipelineStateService) {
    this.pipelineState.isExternalPreview = true;
  }
  ngOnInit(): void {
    this.analysis.setTabVisible(true);
    this.pipelineState.loadCatalog();
    this.subscriptions.add(this.analysis.session$.subscribe(session => {
      this.session = session;
      if (!session) return;
      const key = JSON.stringify(session.pipeline);
      if (key !== this.pipelineKey) {
        this.pipelineKey = key;
        this.pipelineState.loadExternalPipeline(session.pipeline);
        this.updateDefinition(session.pipeline);
      }
    }));
    this.subscriptions.add(this.pipelineState.stepCatalog$.subscribe(() => this.updateDefinition(this.pipelineState.getPipeline())));
    this.subscriptions.add(this.analysis.errors$.subscribe(errors => this.errors = errors));
    this.subscriptions.add(this.analysis.preview$.subscribe(preview => {
      this.selectedResultId = preview.result_id;
      this.pipelineState.applyExternalPreview(preview.response, preview.image_index, preview.loading);
      this.sideOutputs = preview.response?.side_outputs ?? {};
      this.resultEntries = Object.entries(this.sideOutputs).filter(([key]) => !/base64|image|mask|loaded_.*paths|meta$|^history$/.test(key))
        .map(([key, value]) => ({ key, text: this.formatResult(value) }));
    }));
    this.subscriptions.add(this.pipelineState.externalPreviewRequests$.subscribe(request => {
      if (request.reason === 'image') {
        if (this.selectedResultId) this.analysis.selectResult(this.selectedResultId, request.imageIndex, this.analysis.followLatest);
        return;
      }
      this.updateDefinition(request.pipeline);
      this.analysis.configure({ pipeline: request.pipeline });
    }));
  }
  private updateDefinition(pipeline: PipelineDocument): void {
    const steps = pipeline.steps.filter(step => step.enabled !== false);
    this.lastStep = steps[steps.length - 1];
    this.lastDefinition = this.lastStep ? this.pipelineState.getStepDefinition(this.lastStep.step_def_id) : undefined;
  }
  updateParameter(change: { name: string; value: string | number | boolean }): void {
    const pipeline = this.pipelineState.getPipeline();
    const index = pipeline.steps.findIndex(step => step.instance_id === this.lastStep?.instance_id);
    if (index < 0) return;
    this.pipelineState.updateParams(index, { ...pipeline.steps[index].param_values, [change.name]: change.value });
    this.updateDefinition(this.pipelineState.getPipeline());
  }
  setMode(atEnd: boolean): void { this.analysis.configure({ mode: atEnd ? 'measurement_end' : 'each_tablet' }); }
  setInputScope(scope: InlineInputScope): void { this.analysis.configure({ input_scope: scope }); }
  tabletImageIndex(tablet: InlineAnalysisTablet): number {
    const result = this.session?.results.find(item => item.result_id === tablet.result_id);
    return result?.output_images?.find(image => image.tablet_id === tablet.tablet_id)?.image_index ?? 0;
  }
  selectTablet(tablet: InlineAnalysisTablet): void {
    if (tablet.result_id) this.analysis.selectResult(tablet.result_id, this.tabletImageIndex(tablet));
  }
  isTabletSelected(tablet: InlineAnalysisTablet): boolean {
    if (!tablet.result_id || tablet.result_id !== this.selectedResultId) return false;
    const result = this.session?.results.find(item => item.result_id === tablet.result_id);
    if (result?.kind !== 'cumulative') return true;
    return result.output_images?.find(image => image.image_index === this.analysis.preview.image_index)?.tablet_id === tablet.tablet_id;
  }
  thumbnail(resultId: string | null, imageIndex = 0): string | null {
    const result = this.session?.results.find(item => item.result_id === resultId);
    if (result?.status === 'completed' && result.image_count && this.session) {
      return `${BASE_URL}/inline-analysis/sessions/${encodeURIComponent(this.session.session_id)}/results/${encodeURIComponent(result.result_id)}/thumbnail?image_index=${imageIndex}&revision=${result.revision}`;
    }
    return result?.thumbnail_base64 ? `data:image/jpeg;base64,${result.thumbnail_base64}` : null;
  }
  statusLabel(status: string): string {
    const labels: Record<string, string> = {
      collecting: 'Képek fogadása', finishing: 'Elemzés befejezése', cancelled: 'Mérés megszakítva',
      waiting: 'Képsorozatra vár', queued: 'Sorban áll', running: 'Elemzés folyamatban', completed: 'Elkészült', failed: 'Elemzési hiba',
    };
    return labels[status] ?? status;
  }
  get completedCount(): number { return this.session?.results.filter(result => result.status === 'completed').length ?? 0; }
  get pendingCount(): number { return this.session?.results.filter(result => result.status === 'queued' || result.status === 'running').length ?? 0; }
  get cumulativeResults() { return this.session?.results.filter(result => result.kind === 'cumulative') ?? []; }
  private formatResult(value: unknown): string {
    const text = typeof value === 'string' ? value : JSON.stringify(value, (_key, item: unknown) => {
      if (typeof item === 'string' && item.length > 1000) return `${item.slice(0, 1000)}…`;
      return item;
    }, 2) ?? '';
    return text.length > 12000 ? `${text.slice(0, 12000)}\n… A teljes adatsor letölthető.` : text;
  }
  downloadData(): void {
    const payload = Object.fromEntries(Object.entries(this.sideOutputs).filter(([key]) => !/base64/.test(key)));
    const url = URL.createObjectURL(new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' }));
    const link = document.createElement('a'); link.href = url; link.download = 'elemzesi-eredmeny.json'; link.click(); URL.revokeObjectURL(url);
  }
  ngOnDestroy(): void { this.analysis.setTabVisible(false); this.subscriptions.unsubscribe(); }
}
