import { Component, Input, Output, EventEmitter } from '@angular/core';
import { CommonModule } from '@angular/common';
import { MatIconModule } from '@angular/material/icon';
import { StepInstance, StepDefinition } from '../../models/pipeline.models';

@Component({
  selector: 'app-step-card',
  standalone: true,
  imports: [CommonModule, MatIconModule],
  template: `
    <div
      class="step-card"
      [class.selected]="selected"
      [class.compare-selected]="compareSelected"
      [class.has-error]="hasError"
      [class.merge-connect-source]="mergeConnectSource"
      [class.merge-connect-target]="mergeConnectTarget"
      [class.disabled-step]="step.enabled === false"
      [attr.title]="step.enabled === false ? 'Kikapcsolva - jobb klikk a bekapcsoláshoz' : 'Jobb klikk a kikapcsoláshoz'"
      (click)="select.emit($event)"
      (dblclick)="compare.emit(); $event.stopPropagation()"
    >
      <div class="card-header">
        <mat-icon class="card-icon">{{ definition?.icon || 'extension' }}</mat-icon>
        <span class="card-name">{{ definition?.name || step.step_def_id }}</span>
        @if (step.enabled === false) {
          <mat-icon class="disabled-icon">power_off</mat-icon>
        }
        <button class="card-delete" (click)="remove.emit(); $event.stopPropagation()" title="Törlés">
          <mat-icon>close</mat-icon>
        </button>
      </div>
      <div class="card-order">{{ step.order + 1 }}</div>
    </div>
  `,
  styles: [`
    .step-card {
      display: flex;
      flex-direction: column;
      min-width: 110px;
      max-width: 140px;
      padding: 8px;
      background: #333;
      border: 2px solid #555;
      border-radius: 8px;
      cursor: pointer;
      transition: border-color 0.15s, box-shadow 0.15s;
      position: relative;
      user-select: none;
    }

    .step-card:hover {
      border-color: #777;
    }

    .step-card.selected {
      border-color: #3b82f6;
      box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.3);
    }

    .step-card.compare-selected {
      border-color: #a855f7;
      background: #3b2947;
      box-shadow:
        0 0 0 2px rgba(168, 85, 247, 0.4),
        0 0 14px rgba(168, 85, 247, 0.28);
    }

    .step-card.has-error {
      border-color: #ef4444;
    }

    .step-card.merge-connect-source {
      border-color: #22c55e;
      box-shadow: 0 0 0 2px rgba(34, 197, 94, 0.28);
    }

    .step-card.merge-connect-target {
      border-color: #f59e0b;
      box-shadow: 0 0 0 2px rgba(245, 158, 11, 0.28);
    }

    .step-card.disabled-step {
      opacity: 0.48;
      filter: grayscale(1);
      border-style: dashed;
      background: #292929;
    }

    .step-card.disabled-step .card-name {
      text-decoration: line-through;
    }

    .disabled-icon {
      color: #ef4444;
      font-size: 15px;
      width: 15px;
      height: 15px;
    }

    .card-header {
      display: flex;
      align-items: center;
      gap: 4px;
    }

    .card-icon {
      font-size: 16px;
      width: 16px;
      height: 16px;
      color: #aaa;
    }

    .card-name {
      flex: 1;
      font-size: 11px;
      color: #ddd;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .card-delete {
      background: none;
      border: none;
      padding: 0;
      cursor: pointer;
      opacity: 0;
      transition: opacity 0.15s;
      color: #888;
      display: flex;
    }

    .step-card:hover .card-delete {
      opacity: 1;
    }

    .card-delete:hover {
      color: #ef4444;
    }

    .card-delete mat-icon {
      font-size: 14px;
      width: 14px;
      height: 14px;
    }

    .card-order {
      font-size: 9px;
      color: #666;
      text-align: center;
      margin-top: 2px;
    }
  `],
})
export class StepCardComponent {
  @Input() step!: StepInstance;
  @Input() definition?: StepDefinition;
  @Input() selected = false;
  @Input() compareSelected = false;
  @Input() hasError = false;
  @Input() mergeConnectSource = false;
  @Input() mergeConnectTarget = false;
  @Output() select = new EventEmitter<MouseEvent>();
  @Output() compare = new EventEmitter<void>();
  @Output() remove = new EventEmitter<void>();
}
