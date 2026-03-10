import { Component, OnInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import {
  CdkDropList,
  CdkDrag,
  CdkDragDrop,
  CdkDragPlaceholder,
  moveItemInArray,
} from '@angular/cdk/drag-drop';
import { Subscription } from 'rxjs';
import { PipelineStateService } from '../../services/pipeline-state.service';
import { StepInstance, StepDefinition, StepError } from '../../models/pipeline.models';
import { StepCardComponent } from './step-card.component';

@Component({
  selector: 'app-pipeline-canvas',
  standalone: true,
  imports: [CommonModule, CdkDropList, CdkDrag, CdkDragPlaceholder, StepCardComponent],
  template: `
    <div class="canvas-wrapper">
      <div class="canvas-header">
        <span class="canvas-title">Feldolgozási lánc</span>
        <span class="step-count">{{ steps.length }} lépés</span>
      </div>
      <div
        class="pipeline-chain"
        cdkDropList
        id="pipeline-list"
        [cdkDropListData]="steps"
        [cdkDropListConnectedTo]="['toolbox-list']"
        (cdkDropListDropped)="onDrop($event)"
        cdkDropListOrientation="horizontal"
        [cdkDropListEnterPredicate]="allowDrop"
      >
        @if (steps.length === 0) {
          <div class="empty-hint">
            Húzzon elemeket az eszköztárból ide, vagy kattintson duplán rájuk
          </div>
        }
        @for (step of steps; track step.instance_id; let i = $index) {
          <div class="step-wrapper" cdkDrag [cdkDragData]="step">
            @if (i > 0) {
              <div class="connector">
                <svg width="24" height="20" viewBox="0 0 24 20">
                  <line x1="0" y1="10" x2="20" y2="10" stroke="#555" stroke-width="2"/>
                  <polygon points="18,5 24,10 18,15" fill="#555"/>
                </svg>
              </div>
            }
            <app-step-card
              [step]="step"
              [definition]="getDefinition(step.step_def_id)"
              [selected]="selectedIndex === i"
              [hasError]="hasStepError(i)"
              (select)="onSelect(i)"
              (remove)="onRemove(i)"
            ></app-step-card>
            <div *cdkDragPlaceholder class="step-placeholder"></div>
          </div>
        }
      </div>
    </div>
  `,
  styles: [`
    :host {
      display: block;
      height: 100%;
    }

    .canvas-wrapper {
      display: flex;
      flex-direction: column;
      height: 100%;
      padding: 8px;
    }

    .canvas-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 8px;
    }

    .canvas-title {
      font-size: 11px;
      font-weight: 600;
      color: #999;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }

    .step-count {
      font-size: 11px;
      color: #666;
    }

    .pipeline-chain {
      display: flex;
      align-items: center;
      gap: 0;
      overflow-x: auto;
      flex: 1;
      padding: 8px 0;
    }

    .step-wrapper {
      display: flex;
      align-items: center;
      flex-shrink: 0;
    }

    .connector {
      display: flex;
      align-items: center;
      margin: 0 2px;
    }

    .empty-hint {
      color: #666;
      font-size: 12px;
      text-align: center;
      width: 100%;
      padding: 20px;
    }

    .step-placeholder {
      width: 110px;
      height: 60px;
      background: rgba(59, 130, 246, 0.1);
      border: 2px dashed #3b82f6;
      border-radius: 8px;
    }

    .cdk-drag-preview {
      box-sizing: border-box;
    }

    .cdk-drag-animating {
      transition: transform 200ms ease;
    }
  `],
})
export class PipelineCanvasComponent implements OnInit, OnDestroy {
  steps: StepInstance[] = [];
  selectedIndex = -1;
  validationErrors: StepError[] = [];

  private subs: Subscription[] = [];

  constructor(private pipelineState: PipelineStateService) {}

  ngOnInit(): void {
    this.subs.push(
      this.pipelineState.pipeline$.subscribe((p) => (this.steps = p.steps)),
      this.pipelineState.selectedStepIndex$.subscribe((i) => (this.selectedIndex = i)),
      this.pipelineState.validationErrors$.subscribe((e) => (this.validationErrors = e))
    );
  }

  ngOnDestroy(): void {
    this.subs.forEach((s) => s.unsubscribe());
  }

  getDefinition(stepDefId: string): StepDefinition | undefined {
    return this.pipelineState.getStepDefinition(stepDefId);
  }

  hasStepError(index: number): boolean {
    return this.validationErrors.some((e) => e.step_index === index);
  }

  onSelect(index: number): void {
    this.pipelineState.selectStep(index);
  }

  onRemove(index: number): void {
    this.pipelineState.removeStep(index);
  }

  onDrop(event: CdkDragDrop<StepInstance[], any>): void {
    if (event.previousContainer === event.container) {
      // Reorder within pipeline
      if (event.previousIndex !== event.currentIndex) {
        this.pipelineState.moveStep(event.previousIndex, event.currentIndex);
      }
    } else {
      // Drop from toolbox
      const stepDef = event.item.data as StepDefinition;
      if (stepDef && stepDef.id) {
        this.pipelineState.addStep(stepDef.id, event.currentIndex);
      }
    }
  }

  allowDrop = (): boolean => true;
}
