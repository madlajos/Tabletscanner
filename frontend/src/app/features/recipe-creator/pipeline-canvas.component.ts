import { Component, OnInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import {
  CdkDropList,
  CdkDrag,
  CdkDragDrop,
  CdkDragPlaceholder,
} from '@angular/cdk/drag-drop';
import { MatIconModule } from '@angular/material/icon';
import { Subscription } from 'rxjs';
import { PipelineStateService } from '../../services/pipeline-state.service';
import { StepInstance, StepDefinition, StepError } from '../../models/pipeline.models';
import { StepCardComponent } from './step-card.component';

@Component({
  selector: 'app-pipeline-canvas',
  standalone: true,
  imports: [CommonModule, CdkDropList, CdkDrag, CdkDragPlaceholder, StepCardComponent, MatIconModule],
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
        [cdkDropListData]="mainSteps"
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
        @for (node of mainSteps; track node.step.instance_id; let i = $index) {
          <div class="step-wrapper" cdkDrag [cdkDragData]="node.step">
            @if (i > 0) {
              <div class="connector">
                <svg width="24" height="20" viewBox="0 0 24 20">
                  <line x1="0" y1="10" x2="20" y2="10" stroke="#555" stroke-width="2"/>
                  <polygon points="18,5 24,10 18,15" fill="#555"/>
                </svg>
              </div>
            }
            <div class="step-column">
              <app-step-card
                [step]="node.step"
                [definition]="node.definition"
                [selected]="selectedIndex === node.pipelineIndex"
                [hasError]="hasStepError(node.pipelineIndex)"
                (select)="onSelect(node.pipelineIndex)"
                (remove)="onRemove(node.pipelineIndex)"
              ></app-step-card>
              @for (sec of node.secondaries; track $index) {
                <div class="req-branch">
                  <div class="req-branch-line"></div>
                  @if (sec.step) {
                    <div class="sec-card"
                         [class.selected]="selectedIndex === sec.pipelineIndex"
                         (click)="onSelect(sec.pipelineIndex)">
                      <mat-icon class="req-icon">{{ sec.definition?.icon || 'extension' }}</mat-icon>
                      <span class="req-name">{{ sec.definition?.name || sec.step.step_def_id }}</span>
                    </div>
                  } @else {
                    <div class="req-box missing">
                      <mat-icon class="req-icon">{{ sec.definition?.icon || 'extension' }}</mat-icon>
                      <span class="req-name">{{ sec.definition?.name || 'Hiányzó bemenet' }}</span>
                    </div>
                  }
                </div>
              }
            </div>
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

    .step-column {
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 0;
    }

    .req-branch {
      display: flex;
      flex-direction: column;
      align-items: center;
    }

    .req-branch-line {
      width: 2px;
      height: 8px;
      background: #555;
    }

    .req-box {
      display: flex;
      align-items: center;
      gap: 4px;
      padding: 4px 8px;
      background: #2a2a2a;
      border: 2px solid #555;
      border-radius: 6px;
      font-size: 10px;
      color: #999;
      white-space: nowrap;
    }

    .req-box.missing {
      border-color: #ef4444;
      color: #ef4444;
    }

    .sec-card {
      display: flex;
      align-items: center;
      gap: 4px;
      padding: 4px 8px;
      background: #2a2a2a;
      border: 2px solid #555;
      border-radius: 6px;
      font-size: 10px;
      color: #999;
      cursor: pointer;
      white-space: nowrap;
      transition: border-color 0.15s, background 0.15s;
    }

    .sec-card:hover {
      border-color: #3b82f6;
      background: #333;
    }

    .sec-card.selected {
      border-color: #3b82f6;
      color: #ccc;
      background: #333;
    }

    .req-icon {
      font-size: 12px;
      width: 12px;
      height: 12px;
    }

    .req-name {
      max-width: 80px;
      overflow: hidden;
      text-overflow: ellipsis;
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
  mainSteps: MainChainNode[] = [];

  private subs: Subscription[] = [];

  constructor(private pipelineState: PipelineStateService) {}

  ngOnInit(): void {
    this.subs.push(
      this.pipelineState.pipeline$.subscribe((p) => {
        this.steps = p.steps;
        this.computeMainSteps();
      }),
      this.pipelineState.selectedStepIndex$.subscribe((i) => (this.selectedIndex = i)),
      this.pipelineState.validationErrors$.subscribe((e) => (this.validationErrors = e))
    );
  }

  ngOnDestroy(): void {
    this.subs.forEach((s) => s.unsubscribe());
  }

  /** Build the secondary-indices set: which flat-list indices are secondary inputs. */
  private getSecondaryIndices(): Set<number> {
    const secondary = new Set<number>();
    for (let i = 0; i < this.steps.length; i++) {
      const defn = this.getDefinition(this.steps[i].step_def_id);
      if (!defn?.secondary_inputs?.length) continue;
      for (const secId of defn.secondary_inputs) {
        for (let j = i - 1; j >= 0; j--) {
          if (this.steps[j].step_def_id === secId && !secondary.has(j)) {
            secondary.add(j);
            break;
          }
        }
      }
    }
    return secondary;
  }

  /** Compute the main chain nodes (excluding secondary steps, which are nested). */
  private computeMainSteps(): void {
    const secondaryIndices = this.getSecondaryIndices();
    this.mainSteps = [];

    for (let i = 0; i < this.steps.length; i++) {
      if (secondaryIndices.has(i)) continue;

      const step = this.steps[i];
      const defn = this.getDefinition(step.step_def_id);
      const secondaries: SecondaryNode[] = [];

      if (defn?.secondary_inputs?.length) {
        for (const secId of defn.secondary_inputs) {
          let found = false;
          for (let j = i - 1; j >= 0; j--) {
            if (this.steps[j].step_def_id === secId && secondaryIndices.has(j)) {
              secondaries.push({
                step: this.steps[j],
                definition: this.getDefinition(secId),
                pipelineIndex: j,
              });
              found = true;
              break;
            }
          }
          if (!found) {
            // Missing secondary — show placeholder
            const secDefn = this.getDefinition(secId);
            secondaries.push({
              step: null,
              definition: secDefn,
              pipelineIndex: -1,
            });
          }
        }
      }

      this.mainSteps.push({
        step,
        definition: defn,
        pipelineIndex: i,
        secondaries,
      });
    }
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

  onDrop(event: CdkDragDrop<MainChainNode[], any>): void {
    if (event.previousContainer === event.container) {
      // Reorder within pipeline — translate main-chain indices to flat indices
      if (event.previousIndex !== event.currentIndex) {
        if (!this.pipelineState.canMoveMainStep(event.previousIndex, event.currentIndex)) {
          return;
        }
        const fromFlat = this.mainSteps[event.previousIndex].pipelineIndex;
        const toFlat = this.mainSteps[event.currentIndex].pipelineIndex;
        this.pipelineState.moveStep(fromFlat, toFlat);
      }
    } else {
      // Drop from toolbox
      const stepDef = event.item.data as StepDefinition;
      if (stepDef?.id) {
        if (!this.pipelineState.canInsertStepAtMainIndex(stepDef.id, event.currentIndex)) {
          return;
        }
        const insertAt = event.currentIndex < this.mainSteps.length
          ? this.mainSteps[event.currentIndex].pipelineIndex
          : this.steps.length;
        this.pipelineState.addStep(stepDef.id, insertAt);
      }
    }
  }

  allowDrop = (): boolean => true;
}

interface SecondaryNode {
  step: StepInstance | null;
  definition?: StepDefinition;
  pipelineIndex: number; // -1 if missing
}

interface MainChainNode {
  step: StepInstance;
  definition?: StepDefinition;
  pipelineIndex: number;
  secondaries: SecondaryNode[];
}
