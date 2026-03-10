import { Component, OnInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { CdkDropList, CdkDrag, CdkDragPlaceholder, CdkDragPreview } from '@angular/cdk/drag-drop';
import { MatIconModule } from '@angular/material/icon';
import { Subscription } from 'rxjs';
import { PipelineStateService } from '../../services/pipeline-state.service';
import { StepDefinition } from '../../models/pipeline.models';

interface CategoryGroup {
  label: string;
  icon: string;
  steps: StepDefinition[];
}

const CATEGORY_LABELS: Record<string, string> = {
  io: 'Bemenet / Kimenet',
  adjustment: 'Beállítás',
  filter: 'Szűrők',
  analysis: 'Elemzés',
  detection: 'Detektálás',
};

const CATEGORY_ICONS: Record<string, string> = {
  io: 'swap_horiz',
  adjustment: 'tune',
  filter: 'filter',
  analysis: 'analytics',
  detection: 'search',
};

const CATEGORY_ORDER = ['io', 'adjustment', 'filter', 'analysis', 'detection'];

@Component({
  selector: 'app-step-toolbox',
  standalone: true,
  imports: [CommonModule, CdkDropList, CdkDrag, CdkDragPlaceholder, CdkDragPreview, MatIconModule],
  template: `
    <div
      class="toolbox-scroll"
      cdkDropList
      id="toolbox-list"
      [cdkDropListConnectedTo]="['pipeline-list']"
      [cdkDropListSortingDisabled]="true"
    >
      @for (group of categoryGroups; track group.label) {
        <div class="category-group">
          <div class="category-header">
            <mat-icon class="category-icon">{{ group.icon }}</mat-icon>
            <span>{{ group.label }}</span>
          </div>
          @for (step of group.steps; track step.id) {
            <div
              class="tool-item"
              cdkDrag
              [cdkDragData]="step"
              (cdkDragStarted)="onDragStarted(step)"
              (dblclick)="onDoubleClick(step)"
            >
              <div class="tool-item-content">
                <mat-icon class="tool-icon">{{ step.icon }}</mat-icon>
                <span class="tool-name">{{ step.name }}</span>
              </div>
              <div *cdkDragPlaceholder class="drag-placeholder"></div>
              <div *cdkDragPreview class="drag-preview">
                <mat-icon>{{ step.icon }}</mat-icon>
                <span>{{ step.name }}</span>
              </div>
            </div>
          }
        </div>
      }
    </div>
  `,
  styles: [`
    :host {
      display: flex;
      flex-direction: column;
      flex: 1;
      min-height: 0;
    }

    .toolbox-scroll {
      flex: 1;
      overflow-y: auto;
      padding: 8px;
    }

    .category-group {
      margin-bottom: 12px;
    }

    .category-header {
      display: flex;
      align-items: center;
      gap: 6px;
      color: #999;
      font-size: 11px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      padding: 4px 0;
      border-bottom: 1px solid #3a3a3a;
      margin-bottom: 4px;
    }

    .category-icon {
      font-size: 16px;
      width: 16px;
      height: 16px;
    }

    .tool-item {
      display: block;
      padding: 6px 8px;
      border-radius: 4px;
      cursor: grab;
      margin-bottom: 2px;
      transition: background 0.15s;
    }

    .tool-item:hover {
      background: #3a3a3a;
    }

    .tool-item:active {
      cursor: grabbing;
    }

    .tool-item-content {
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .tool-icon {
      font-size: 18px;
      width: 18px;
      height: 18px;
      color: #aaa;
    }

    .tool-name {
      font-size: 12px;
      color: #ddd;
    }

    .drag-placeholder {
      height: 36px;
      background: rgba(59, 130, 246, 0.1);
      border: 1px dashed #3b82f6;
      border-radius: 4px;
    }

    .drag-preview {
      display: flex;
      align-items: center;
      gap: 6px;
      padding: 6px 12px;
      background: #3b82f6;
      border-radius: 4px;
      color: white;
      font-size: 12px;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
    }
  `],
})
export class StepToolboxComponent implements OnInit, OnDestroy {
  categoryGroups: CategoryGroup[] = [];
  private sub?: Subscription;

  constructor(private pipelineState: PipelineStateService) {}

  ngOnInit(): void {
    this.sub = this.pipelineState.stepCatalog$.subscribe((catalog) => {
      this.categoryGroups = this.groupByCategory(catalog);
    });
  }

  ngOnDestroy(): void {
    this.sub?.unsubscribe();
  }

  onDragStarted(_step: StepDefinition): void {
    // future: could highlight compatible drop zones
  }

  onDoubleClick(step: StepDefinition): void {
    this.pipelineState.addStep(step.id);
  }

  private groupByCategory(catalog: StepDefinition[]): CategoryGroup[] {
    const map = new Map<string, StepDefinition[]>();
    for (const step of catalog) {
      const list = map.get(step.category) || [];
      list.push(step);
      map.set(step.category, list);
    }

    return CATEGORY_ORDER
      .filter((cat) => map.has(cat))
      .map((cat) => ({
        label: CATEGORY_LABELS[cat] || cat,
        icon: CATEGORY_ICONS[cat] || 'extension',
        steps: map.get(cat)!,
      }));
  }
}
