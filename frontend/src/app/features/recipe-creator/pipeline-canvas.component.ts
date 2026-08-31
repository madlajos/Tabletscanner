import {
  AfterViewInit,
  Component,
  ElementRef,
  HostListener,
  OnDestroy,
  OnInit,
  QueryList,
  ViewChild,
  ViewChildren,
} from '@angular/core';
import { CommonModule } from '@angular/common';
import {
  CdkDrag,
  CdkDragDrop,
  CdkDragHandle,
  CdkDragPlaceholder,
  CdkDropList,
} from '@angular/cdk/drag-drop';
import { MatIconModule } from '@angular/material/icon';
import { Subscription } from 'rxjs';
import { PipelineStateService } from '../../services/pipeline-state.service';
import { PipelineConnection, StepDefinition, StepError, StepInstance } from '../../models/pipeline.models';
import { StepCardComponent } from './step-card.component';

@Component({
  selector: 'app-pipeline-canvas',
  standalone: true,
  imports: [CommonModule, CdkDropList, CdkDrag, CdkDragHandle, CdkDragPlaceholder, StepCardComponent, MatIconModule],
  template: `
    <div class="canvas-wrapper">
      <div class="canvas-header">
        <span class="canvas-title">Feldolgozasi lanc</span>
        <div class="canvas-header-actions">
          <span class="step-count" title="Shift + jobb klikk: agak osszekotese">
            {{ steps.length }} lepes · jobb klikk: ki/be
          </span>
          <button class="shortcuts-help" type="button" title="Utolsó áthelyezés vagy módosítás visszavonása" (click)="undo()">
            <mat-icon>undo</mat-icon>
          </button>
          <button class="shortcuts-help" type="button" title="Visszavont módosítás újraalkalmazása" (click)="redo()">
            <mat-icon>redo</mat-icon>
          </button>
          <button class="shortcuts-help" type="button" title="Gyorsbillentyűk és segítség" (click)="showShortcuts = true">
            <mat-icon>help_outline</mat-icon>
          </button>
        </div>
      </div>

      @if (showShortcuts) {
        <div class="shortcuts-backdrop" (click)="showShortcuts = false">
          <section class="shortcuts-dialog" role="dialog" aria-modal="true" aria-labelledby="shortcuts-title" (click)="$event.stopPropagation()">
            <div class="shortcuts-dialog-header">
              <h2 id="shortcuts-title">Feldolgozási lánc – gyorsbillentyűk</h2>
              <button type="button" class="shortcuts-close" title="Bezárás" (click)="showShortcuts = false">
                <mat-icon>close</mat-icon>
              </button>
            </div>
            <dl class="shortcuts-list">
              <div><dt>Shift/Ctrl + egérgörgő</dt><dd>A preview nagyítása vagy kicsinyítése az egérmutató körül (1×–5×)</dd></div>
              <div><dt>Bal egérgomb + húzás</dt><dd>A nagyított preview mozgatása</dd></div>
              <div><dt>Dupla kattintás</dt><dd>A preview nagyításának visszaállítása 1×-re</dd></div>
              <div><dt>Ctrl/Cmd + kattintás</dt><dd>Több node kijelölése vagy kivétele a kijelölésből</dd></div>
              <div><dt>Shift + kattintás</dt><dd>Node-tartomány kijelölése ugyanazon az ágon</dd></div>
              <div><dt>Ctrl + Z</dt><dd>Utolsó módosítás visszavonása</dd></div>
              <div><dt>Ctrl + Shift + Z</dt><dd>Visszavont módosítás megismétlése</dd></div>
              <div><dt>Ctrl + Y</dt><dd>Visszavont módosítás megismétlése</dd></div>
              <div><dt>Jobb klikk</dt><dd>Lépés ki- vagy bekapcsolása</dd></div>
              <div><dt>Shift + jobb klikk</dt><dd>Ág bekötése az „Ágak összevonása” lépésbe</dd></div>
              <div><dt>Esc</dt><dd>A súgóablak bezárása</dd></div>
            </dl>
          </section>
        </div>
      }

      <div
        class="branch-board"
        #branchBoard
        cdkDropList
        cdkDropListOrientation="vertical"
        [cdkDropListData]="branchRows"
        (cdkDropListDropped)="onBranchDrop($event)"
        (scroll)="scheduleMergeLineRefresh()"
      >
        <svg
          class="merge-lines-layer"
          [attr.width]="mergeLineSvgWidth"
          [attr.height]="mergeLineSvgHeight"
          [attr.viewBox]="'0 0 ' + mergeLineSvgWidth + ' ' + mergeLineSvgHeight"
        >
          <defs>
            <marker
              id="mergeArrowHead"
              markerWidth="8"
              markerHeight="8"
              refX="7"
              refY="4"
              orient="auto"
              markerUnits="strokeWidth"
            >
              <path d="M 0 0 L 8 4 L 0 8 z" class="merge-arrow-head"></path>
            </marker>
          </defs>
          @for (line of mergeLines; track line.key) {
            <path
              class="merge-line"
              [attr.d]="line.path"
              marker-end="url(#mergeArrowHead)"
            ></path>
          }
        </svg>
        @if (steps.length === 0) {
          <div
            class="branch-row empty-row"
            cdkDropList
            id="pipeline-list"
            [cdkDropListData]="emptyBranch.nodes"
            [cdkDropListConnectedTo]="connectedDropLists"
            (cdkDropListDropped)="onDrop($event, emptyBranch)"
            cdkDropListOrientation="horizontal"
            [cdkDropListEnterPredicate]="allowDrop"
          >
            <div class="empty-hint">
              Huzzon elemeket az eszkoztarbol ide, vagy kattintson duplan rajuk
            </div>
          </div>
        } @else {
          @for (branch of branchRows; track branch.id) {
            <div
              class="branch-lane"
              cdkDrag
              cdkDragLockAxis="y"
              [cdkDragData]="branch"
              [class.selected-branch]="selectedBranchId === branch.id"
              (cdkDragStarted)="selectBranch(branch)"
              (cdkDragEnded)="scheduleMergeLineRefresh()"
            >
              <div
                class="branch-label"
                title="Az egész ág mozgatása: húzza egy másik ág fölé vagy alá"
                (click)="selectBranch(branch)"
                (dblclick)="startBranchRename(branch, $event)"
              >
                <mat-icon cdkDragHandle>drag_indicator</mat-icon>
                @if (editingBranchKey === branch.key) {
                  <input
                    class="branch-name-input"
                    [value]="editingBranchName"
                    maxlength="40"
                    (input)="editingBranchName = $any($event.target).value"
                    (click)="$event.stopPropagation()"
                    (dblclick)="$event.stopPropagation()"
                    (keydown.enter)="finishBranchRename(branch)"
                    (keydown.escape)="cancelBranchRename()"
                    (blur)="finishBranchRename(branch)"
                  />
                } @else {
                  <span title="Dupla kattintás az átnevezéshez">{{ branch.name }}</span>
                }
                @if (canCopySelectionTo(branch)) {
                  <button
                    type="button"
                    class="copy-to-branch"
                    title="A kijelölt node-ok másolása erre az ágra, a beállításaikkal együtt"
                    (pointerdown)="$event.stopPropagation()"
                    (click)="copySelectionTo(branch, $event)"
                  >
                    <mat-icon>content_copy</mat-icon>
                    Ide másol
                  </button>
                }
              </div>
              <div
                class="branch-row"
                cdkDropList
                [id]="branch.id"
                [cdkDropListData]="branch.nodes"
                [cdkDropListConnectedTo]="connectedDropLists"
                (cdkDropListDropped)="onDrop($event, branch)"
                cdkDropListOrientation="horizontal"
                [cdkDropListEnterPredicate]="allowDrop"
              >
                @for (node of branch.nodes; track node.step.instance_id; let i = $index) {
                  <div
                    class="step-wrapper"
                    #stepNodeEl
                    [attr.data-instance-id]="node.step.instance_id"
                    cdkDrag
                    [cdkDragData]="node.step"
                    [class.merge-connect-source]="pendingMergeSourceId === node.step.instance_id"
                    [class.merge-connect-target]="!!pendingMergeSourceId && node.step.step_def_id === 'branch_merge'"
                    [class.multi-selected]="selectedNodeIds.has(node.step.instance_id)"
                    (contextmenu)="onNodeContextMenu($event, node)"
                  >
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
                        [compareSelected]="splitPreviewStepIndex === node.pipelineIndex"
                        [hasError]="hasStepError(node.pipelineIndex)"
                        [mergeConnectSource]="pendingMergeSourceId === node.step.instance_id"
                        [mergeConnectTarget]="!!pendingMergeSourceId && node.step.step_def_id === 'branch_merge'"
                        (select)="onSelect(node, $event)"
                        (compare)="onCompare(node.pipelineIndex)"
                        (remove)="onRemove(node.pipelineIndex)"
                      ></app-step-card>
                      @for (sec of node.secondaries; track $index) {
                        <div class="req-branch">
                          <div class="req-branch-line"></div>
                          @if (sec.step) {
                            <div
                              class="sec-card"
                              [class.selected]="selectedIndex === sec.pipelineIndex"
                              [class.compare-selected]="splitPreviewStepIndex === sec.pipelineIndex"
                              [class.disabled-step]="sec.step.enabled === false"
                              (click)="onSelectSecondary(sec.pipelineIndex)"
                              (dblclick)="onCompare(sec.pipelineIndex); $event.stopPropagation()"
                              (contextmenu)="onSecondaryContextMenu($event, sec.pipelineIndex)"
                            >
                              <mat-icon class="req-icon">{{ sec.definition?.icon || 'extension' }}</mat-icon>
                              <span class="req-name">{{ sec.definition?.name || sec.step.step_def_id }}</span>
                            </div>
                          } @else {
                            <div class="req-box missing">
                              <mat-icon class="req-icon">{{ sec.definition?.icon || 'extension' }}</mat-icon>
                              <span class="req-name">{{ sec.definition?.name || 'Hianyzo bemenet' }}</span>
                            </div>
                          }
                        </div>
                      }
                    </div>
                    <div *cdkDragPlaceholder class="step-placeholder"></div>
                  </div>
                }
              </div>
              <div *cdkDragPlaceholder class="branch-placeholder"></div>
            </div>
          }
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

    .canvas-header-actions { display: flex; align-items: center; gap: 8px; }

    .shortcuts-help, .shortcuts-close {
      display: flex;
      align-items: center;
      justify-content: center;
      border: 0;
      background: transparent;
      color: #aaa;
      cursor: pointer;
      padding: 2px;
    }

    .shortcuts-help:hover, .shortcuts-close:hover { color: #fff; }
    .shortcuts-help mat-icon { width: 19px; height: 19px; font-size: 19px; }

    .shortcuts-backdrop {
      position: fixed;
      inset: 0;
      z-index: 1000;
      display: flex;
      align-items: center;
      justify-content: center;
      background: rgba(0, 0, 0, 0.65);
    }

    .shortcuts-dialog {
      width: min(520px, calc(100vw - 32px));
      border: 1px solid #555;
      border-radius: 10px;
      background: #292929;
      color: #eee;
      box-shadow: 0 18px 60px rgba(0, 0, 0, 0.5);
      padding: 18px;
    }

    .shortcuts-dialog-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 14px;
    }

    .shortcuts-dialog h2 { margin: 0; font-size: 17px; }
    .shortcuts-close mat-icon { width: 20px; height: 20px; font-size: 20px; }
    .shortcuts-list { margin: 0; }
    .shortcuts-list > div {
      display: grid;
      grid-template-columns: 145px 1fr;
      gap: 14px;
      padding: 9px 0;
      border-top: 1px solid #414141;
    }
    .shortcuts-list dt { color: #93c5fd; font-weight: 600; }
    .shortcuts-list dd { margin: 0; color: #ccc; }

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

    .branch-board {
      position: relative;
      display: flex;
      flex-direction: column;
      gap: 10px;
      overflow: auto;
      flex: 1;
      padding: 8px 4px 8px 0;
      scrollbar-width: thin;
      scrollbar-color: #444 #242424;
    }

    .branch-board::-webkit-scrollbar {
      width: 10px;
      height: 10px;
    }

    .branch-board::-webkit-scrollbar-track {
      background: #242424;
      border-radius: 8px;
    }

    .branch-board::-webkit-scrollbar-thumb {
      background: #444;
      border-radius: 8px;
      border: 2px solid #242424;
    }

    .branch-board::-webkit-scrollbar-thumb:hover {
      background: #5a5a5a;
    }

    .branch-board::-webkit-scrollbar-corner {
      background: #242424;
    }

    .branch-lane {
      display: grid;
      grid-template-columns: 112px minmax(0, 1fr);
      align-items: stretch;
      min-height: 92px;
    }

    .branch-label {
      display: flex;
      align-items: center;
      justify-content: center;
      color: #7aa2d6;
      font-size: 10px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      border-right: 1px solid #345170;
      margin-right: 8px;
      white-space: nowrap;
      cursor: grab;
      user-select: none;
      flex-direction: column;
      gap: 2px;
      border-radius: 5px 0 0 5px;
    }

    .branch-label:active {
      cursor: grabbing;
    }

    .branch-label mat-icon {
      width: 16px;
      height: 16px;
      font-size: 16px;
    }

    .branch-name-input {
      width: 98px;
      box-sizing: border-box;
      border: 1px solid #60a5fa;
      border-radius: 3px;
      background: #171717;
      color: #e5e7eb;
      font: inherit;
      text-align: center;
      text-transform: none;
      outline: none;
      padding: 3px 4px;
    }

    .copy-to-branch {
      display: flex;
      align-items: center;
      gap: 3px;
      margin-top: 4px;
      padding: 3px 6px;
      border: 1px solid #3b82f6;
      border-radius: 4px;
      background: rgba(59, 130, 246, 0.16);
      color: #bfdbfe;
      font-size: 9px;
      cursor: pointer;
      text-transform: none;
    }

    .copy-to-branch:hover {
      background: rgba(59, 130, 246, 0.32);
      color: #fff;
    }

    .copy-to-branch mat-icon {
      width: 12px;
      height: 12px;
      font-size: 12px;
    }

    .branch-lane.selected-branch .branch-label {
      color: #bfdbfe;
      background: rgba(59, 130, 246, 0.14);
      border-right-color: #3b82f6;
    }

    .branch-placeholder {
      grid-column: 1 / -1;
      min-height: 92px;
      border: 2px dashed #3b82f6;
      border-radius: 8px;
      background: rgba(59, 130, 246, 0.08);
    }

    .branch-row {
      display: flex;
      align-items: center;
      gap: 0;
      min-width: 0;
      min-height: 92px;
      overflow-x: auto;
      padding: 6px 0;
      scrollbar-width: thin;
      scrollbar-color: #444 transparent;
    }

    .branch-row::-webkit-scrollbar {
      height: 8px;
    }

    .branch-row::-webkit-scrollbar-track {
      background: transparent;
    }

    .branch-row::-webkit-scrollbar-thumb {
      background: #444;
      border-radius: 8px;
      border: 2px solid #292929;
    }

    .branch-row::-webkit-scrollbar-thumb:hover {
      background: #5a5a5a;
    }

    .empty-row {
      justify-content: center;
      min-height: 100%;
    }

    .step-wrapper {
      position: relative;
      z-index: 1;
      display: flex;
      align-items: center;
      flex-shrink: 0;
    }

    .step-wrapper.multi-selected .step-column {
      border-radius: 10px;
      outline: 3px solid #22c55e;
      outline-offset: 2px;
      box-shadow: 0 0 12px rgba(34, 197, 94, 0.32);
    }

    .merge-lines-layer {
      position: absolute;
      inset: 0;
      overflow: visible;
      pointer-events: none;
      z-index: 0;
    }

    .merge-line {
      fill: none;
      stroke: #f59e0b;
      stroke-width: 2;
      stroke-linecap: square;
      stroke-linejoin: miter;
    }

    .merge-arrow-head {
      fill: #f59e0b;
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

    .sec-card.compare-selected {
      border-color: #a855f7;
      color: #f3e8ff;
      background: #3b2947;
      box-shadow: 0 0 0 2px rgba(168, 85, 247, 0.35);
    }

    .sec-card.disabled-step {
      opacity: 0.48;
      filter: grayscale(1);
      border-style: dashed;
      text-decoration: line-through;
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
export class PipelineCanvasComponent implements OnInit, OnDestroy, AfterViewInit {
  showShortcuts = false;
  @ViewChild('branchBoard') branchBoard?: ElementRef<HTMLDivElement>;
  @ViewChildren('stepNodeEl') stepNodeElements?: QueryList<ElementRef<HTMLDivElement>>;

  steps: StepInstance[] = [];
  connections: PipelineConnection[] = [];
  selectedIndex = -1;
  splitPreviewStepIndex = -1;
  validationErrors: StepError[] = [];
  mainSteps: MainChainNode[] = [];
  branchRows: BranchRow[] = [];
  emptyBranch: BranchRow = {
    id: 'pipeline-list',
    key: '',
    name: 'Ág 1',
    index: 0,
    nodes: [],
    startFlatIndex: 0,
    endFlatIndex: 0,
    startMainIndex: 0,
    endMainIndex: 0,
  };
  connectedDropLists = ['toolbox-list', 'pipeline-list'];
  pendingMergeSourceId: string | null = null;
  mergeLines: MergeLine[] = [];
  mergeLineSvgWidth = 0;
  mergeLineSvgHeight = 0;
  selectedBranchId: string | null = null;
  editingBranchKey: string | null = null;
  editingBranchName = '';
  selectedNodeIds = new Set<string>();

  private subs: Subscription[] = [];
  private handledShiftRightClick: { instanceId: string; at: number } | null = null;
  private selectionAnchorId: string | null = null;

  constructor(private pipelineState: PipelineStateService) {}

  @HostListener('document:keydown', ['$event'])
  onDocumentKeydown(event: KeyboardEvent): void {
    if (event.key === 'Escape') {
      if (this.showShortcuts) {
        event.preventDefault();
        this.showShortcuts = false;
        return;
      }
      if (this.selectedNodeIds.size) {
        event.preventDefault();
        this.clearNodeSelection();
        return;
      }
    }

    const target = event.target as HTMLElement | null;
    const isEditable = target instanceof HTMLInputElement ||
      target instanceof HTMLTextAreaElement ||
      target instanceof HTMLSelectElement ||
      target?.isContentEditable;
    if (isEditable || !(event.ctrlKey || event.metaKey)) return;

    const key = event.key.toLowerCase();
    if (key === 'z') {
      event.preventDefault();
      event.shiftKey ? this.pipelineState.redo() : this.pipelineState.undo();
    } else if (key === 'y') {
      event.preventDefault();
      this.pipelineState.redo();
    }
  }

  ngOnInit(): void {
    this.subs.push(
      this.pipelineState.pipeline$.subscribe((p) => {
        this.steps = p.steps;
        this.connections = p.connections ?? [];
        const existingIds = new Set(this.steps.map((step) => step.instance_id));
        this.selectedNodeIds = new Set(
          [...this.selectedNodeIds].filter((instanceId) => existingIds.has(instanceId))
        );
        if (this.selectionAnchorId && !existingIds.has(this.selectionAnchorId)) {
          this.selectionAnchorId = null;
        }
        if (this.pendingMergeSourceId && !this.steps.some((step) => step.instance_id === this.pendingMergeSourceId)) {
          this.pendingMergeSourceId = null;
        }
        this.computeMainSteps();
      }),
      this.pipelineState.selectedStepIndex$.subscribe((i) => (this.selectedIndex = i)),
      this.pipelineState.splitPreviewStepIndex$.subscribe((i) => (this.splitPreviewStepIndex = i)),
      this.pipelineState.validationErrors$.subscribe((e) => (this.validationErrors = e))
    );
  }

  ngOnDestroy(): void {
    this.branchBoard?.nativeElement.removeEventListener(
      'pointerdown',
      this.onBranchBoardPointerDown,
      true
    );
    this.subs.forEach((s) => s.unsubscribe());
  }

  ngAfterViewInit(): void {
    // Capture Shift + right click before the nested CDK drag directives see it.
    // Depending on the browser/CDK event order, a bubbling mousedown handler on
    // the draggable node can otherwise be skipped.
    this.branchBoard!.nativeElement.addEventListener(
      'pointerdown',
      this.onBranchBoardPointerDown,
      true
    );
    this.subs.push(
      this.stepNodeElements!.changes.subscribe(() => this.scheduleMergeLineRefresh())
    );
    this.scheduleMergeLineRefresh();
  }

  @HostListener('window:resize')
  onWindowResize(): void {
    this.scheduleMergeLineRefresh();
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

  /** Compute main nodes and split them into horizontal branch rows. */
  private computeMainSteps(): void {
    const secondaryIndices = this.getSecondaryIndices();
    this.mainSteps = [];
    this.branchRows = [];

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
            secondaries.push({
              step: null,
              definition: this.getDefinition(secId),
              pipelineIndex: -1,
            });
          }
        }
      }

      const node: MainChainNode = {
        step,
        definition: defn,
        pipelineIndex: i,
        mainIndex: this.mainSteps.length,
        secondaries,
        isBranchStart: defn?.id === 'load_image',
      };
      this.mainSteps.push(node);

      if (!this.branchRows.length || (node.isBranchStart && this.branchRows[this.branchRows.length - 1].nodes.length)) {
        const branchIndex = this.branchRows.length;
        this.branchRows.push({
          id: branchIndex === 0 ? 'pipeline-list' : `pipeline-branch-${branchIndex}`,
          key: step.instance_id,
          name: this.pipelineState.getPipeline().branch_names?.[step.instance_id] ?? `Ág ${branchIndex + 1}`,
          index: branchIndex,
          nodes: [],
          startFlatIndex: i,
          endFlatIndex: this.steps.length,
          startMainIndex: node.mainIndex,
          endMainIndex: this.mainSteps.length,
        });
      }

      const branch = this.branchRows[this.branchRows.length - 1];
      branch.nodes.push(node);
      branch.endFlatIndex = i + 1;
      branch.endMainIndex = node.mainIndex + 1;
    }

    for (let i = 0; i < this.branchRows.length; i++) {
      const next = this.branchRows[i + 1];
      this.branchRows[i].endFlatIndex = next ? next.startFlatIndex : this.steps.length;
      this.branchRows[i].endMainIndex = next ? next.startMainIndex : this.mainSteps.length;
    }

    this.connectedDropLists = [
      'toolbox-list',
      ...this.branchRows.map((branch) => branch.id),
    ];
    this.scheduleMergeLineRefresh();
  }

  getDefinition(stepDefId: string): StepDefinition | undefined {
    return this.pipelineState.getStepDefinition(stepDefId);
  }

  hasStepError(index: number): boolean {
    return this.validationErrors.some((e) => e.step_index === index);
  }

  onSelect(node: MainChainNode, event: MouseEvent): void {
    const additive = event.ctrlKey || event.metaKey;
    const range = event.shiftKey;

    if ((additive || range) && node.step.step_def_id !== 'load_image') {
      event.preventDefault();
      event.stopPropagation();

      if (range && this.selectionAnchorId) {
        const branch = this.branchRows.find((candidate) =>
          candidate.nodes.some((item) => item.step.instance_id === this.selectionAnchorId)
        );
        const anchorIndex = branch?.nodes.findIndex(
          (item) => item.step.instance_id === this.selectionAnchorId
        ) ?? -1;
        const nodeIndex = branch?.nodes.findIndex(
          (item) => item.step.instance_id === node.step.instance_id
        ) ?? -1;
        if (branch && anchorIndex >= 0 && nodeIndex >= 0) {
          const selected = additive ? new Set(this.selectedNodeIds) : new Set<string>();
          const from = Math.min(anchorIndex, nodeIndex);
          const to = Math.max(anchorIndex, nodeIndex);
          for (const item of branch.nodes.slice(from, to + 1)) {
            if (item.step.step_def_id !== 'load_image') selected.add(item.step.instance_id);
          }
          this.selectedNodeIds = selected;
        } else {
          this.selectedNodeIds = new Set([node.step.instance_id]);
        }
      } else {
        const selectedBranch = this.getSelectionSourceBranch();
        const nodeBranch = this.branchRows.find((candidate) =>
          candidate.nodes.some((item) => item.step.instance_id === node.step.instance_id)
        );
        const selected = selectedBranch && selectedBranch.id !== nodeBranch?.id
          ? new Set<string>()
          : new Set(this.selectedNodeIds);
        selected.has(node.step.instance_id)
          ? selected.delete(node.step.instance_id)
          : selected.add(node.step.instance_id);
        this.selectedNodeIds = selected;
      }

      this.selectionAnchorId = node.step.instance_id;
      this.pipelineState.selectStep(node.pipelineIndex);
      return;
    }

    this.clearNodeSelection();
    this.pipelineState.selectStep(node.pipelineIndex);
  }

  onSelectSecondary(index: number): void {
    this.clearNodeSelection();
    this.pipelineState.selectStep(index);
  }

  canCopySelectionTo(branch: BranchRow): boolean {
    const sourceBranch = this.getSelectionSourceBranch();
    return this.selectedNodeIds.size > 0 && !!sourceBranch && sourceBranch.id !== branch.id;
  }

  copySelectionTo(branch: BranchRow, event: MouseEvent): void {
    event.preventDefault();
    event.stopPropagation();
    if (!this.canCopySelectionTo(branch)) return;
    this.pipelineState.copyStepsTo([...this.selectedNodeIds], branch.endFlatIndex);
    this.clearNodeSelection();
  }

  private getSelectionSourceBranch(): BranchRow | null {
    if (!this.selectedNodeIds.size) return null;
    return this.branchRows.find((branch) =>
      branch.nodes.some((node) => this.selectedNodeIds.has(node.step.instance_id))
    ) ?? null;
  }

  private clearNodeSelection(): void {
    this.selectedNodeIds = new Set<string>();
    this.selectionAnchorId = null;
  }

  onCompare(index: number): void {
    this.pipelineState.selectStep(index);
    this.pipelineState.requestSplitPreview(index);
  }

  onRemove(index: number): void {
    this.pipelineState.removeStep(index);
  }

  undo(): void {
    this.pipelineState.undo();
  }

  redo(): void {
    this.pipelineState.redo();
  }

  onNodeContextMenu(event: MouseEvent, node: MainChainNode): void {
    event.preventDefault();
    event.stopPropagation();

    if (
      event.shiftKey &&
      this.handledShiftRightClick?.instanceId === node.step.instance_id &&
      performance.now() - this.handledShiftRightClick.at < 1000
    ) {
      this.handledShiftRightClick = null;
      return;
    }

    if (!event.shiftKey) {
      this.pendingMergeSourceId = null;
      this.pipelineState.toggleStepEnabled(node.pipelineIndex);
      return;
    }

    this.handleMergeConnection(node);
  }

  private readonly onBranchBoardPointerDown = (event: PointerEvent): void => {
    if (event.button !== 2 || !event.shiftKey) return;

    const target = event.target as Element | null;
    const wrapper = target?.closest<HTMLElement>('.step-wrapper[data-instance-id]');
    const instanceId = wrapper?.dataset['instanceId'];
    if (!instanceId || !this.branchBoard?.nativeElement.contains(wrapper)) return;

    const node = this.mainSteps.find((candidate) => candidate.step.instance_id === instanceId);
    if (!node) return;

    event.preventDefault();
    event.stopPropagation();
    this.handledShiftRightClick = {
      instanceId: node.step.instance_id,
      at: performance.now(),
    };
    this.handleMergeConnection(node);
  };

  private handleMergeConnection(node: MainChainNode): void {
    const selectedMerge = this.getSelectedMergeNode();
    if (
      selectedMerge &&
      node.step.step_def_id !== 'branch_merge' &&
      node.step.instance_id !== selectedMerge.step.instance_id
    ) {
      if (this.findBranchStartIndex(node.pipelineIndex) === this.findBranchStartIndex(selectedMerge.pipelineIndex)) {
        this.pendingMergeSourceId = null;
        this.pipelineState.selectStep(selectedMerge.pipelineIndex);
        return;
      }
      this.pipelineState.connectSteps(node.step.instance_id, selectedMerge.step.instance_id, 'merge');
      this.pendingMergeSourceId = null;
      this.pipelineState.selectStep(selectedMerge.pipelineIndex);
      return;
    }

    if (this.pendingMergeSourceId) {
      if (node.step.step_def_id === 'branch_merge' && node.step.instance_id !== this.pendingMergeSourceId) {
        this.pipelineState.connectSteps(this.pendingMergeSourceId, node.step.instance_id, 'merge');
        this.pendingMergeSourceId = null;
        this.pipelineState.selectStep(node.pipelineIndex);
        return;
      }

      if (node.step.instance_id === this.pendingMergeSourceId) {
        this.pendingMergeSourceId = null;
        this.pipelineState.selectStep(node.pipelineIndex);
        return;
      }
    }

    if (node.step.step_def_id !== 'branch_merge') {
      this.pendingMergeSourceId = node.step.instance_id;
    }
    this.pipelineState.selectStep(node.pipelineIndex);
  }

  onSecondaryContextMenu(event: MouseEvent, pipelineIndex: number): void {
    event.preventDefault();
    event.stopPropagation();
    this.pipelineState.toggleStepEnabled(pipelineIndex);
  }

  private getSelectedMergeNode(): MainChainNode | null {
    if (this.selectedIndex < 0 || this.selectedIndex >= this.steps.length) return null;
    const selected = this.steps[this.selectedIndex];
    if (selected?.step_def_id !== 'branch_merge') return null;
    return this.mainSteps.find((node) => node.step.instance_id === selected.instance_id) ?? null;
  }

  scheduleMergeLineRefresh(): void {
    requestAnimationFrame(() => this.refreshMergeLines());
  }

  private refreshMergeLines(): void {
    const board = this.branchBoard?.nativeElement;
    const nodeElements = this.stepNodeElements?.toArray() ?? [];
    if (!board || nodeElements.length === 0) {
      this.mergeLines = [];
      return;
    }

    this.mergeLineSvgWidth = Math.max(board.scrollWidth, board.clientWidth);
    this.mergeLineSvgHeight = Math.max(board.scrollHeight, board.clientHeight);

    const boardRect = board.getBoundingClientRect();
    const elementById = new Map<string, HTMLElement>();
    for (const ref of nodeElements) {
      const id = ref.nativeElement.dataset['instanceId'];
      if (id) elementById.set(id, ref.nativeElement);
    }

    const lines: MergeLine[] = [];
    for (const connection of this.connections) {
      if (connection.kind !== 'merge') continue;
      if (this.isCurrentBranchImplicitMergeConnection(connection)) continue;
      // A merge connection represents the whole source branch. Its persisted
      // source id identifies that branch, but the arrow must always originate
      // from the branch's current last node after inserts or reordering.
      const sourceNode = this.getMergeConnectionSourceNode(connection);
      const fromEl = sourceNode ? elementById.get(sourceNode.step.instance_id) : undefined;
      const toEl = elementById.get(connection.to_instance_id);
      if (!fromEl || !toEl) continue;

      const fromRect = (fromEl.querySelector('.step-card') ?? fromEl).getBoundingClientRect();
      const toRect = (toEl.querySelector('.step-card') ?? toEl).getBoundingClientRect();
      const x1 = fromRect.right - boardRect.left + board.scrollLeft;
      const y1 = fromRect.top - boardRect.top + board.scrollTop + fromRect.height / 2;
      const x2 = toRect.left - boardRect.left + board.scrollLeft;
      const y2 = toRect.top - boardRect.top + board.scrollTop + toRect.height / 2;
      const elbowX = x2 > x1 ? x2 - 18 : x1 + 18;

      lines.push({
        key: `${connection.from_instance_id}-${connection.to_instance_id}`,
        path: `M ${x1} ${y1} L ${elbowX} ${y1} L ${elbowX} ${y2} L ${x2} ${y2}`,
      });
    }

    this.mergeLines = lines;
  }

  private getMergeConnectionSourceNode(connection: PipelineConnection): MainChainNode | null {
    const sourceBranch = this.branchRows.find((branch) =>
      branch.nodes.some((node) => node.step.instance_id === connection.from_instance_id)
    );
    return sourceBranch?.nodes[sourceBranch.nodes.length - 1] ?? null;
  }

  onDrop(event: CdkDragDrop<MainChainNode[], any>, branch: BranchRow): void {
    if (event.previousContainer === event.container) {
      if (event.previousIndex !== event.currentIndex) {
        const fromMain = branch.nodes[event.previousIndex]?.mainIndex;
        if (fromMain === undefined) return;
        const fromFlat = this.mainSteps[fromMain].pipelineIndex;
        const toFlat = this.getBranchInsertFlatIndex(branch, event.currentIndex);
        this.pipelineState.moveStep(fromFlat, toFlat);
      }
      return;
    }

    if (event.previousContainer.id !== 'toolbox-list') {
      const fromNode = event.previousContainer.data[event.previousIndex] as MainChainNode | undefined;
      if (!fromNode) return;
      this.pipelineState.moveStep(fromNode.pipelineIndex, this.getBranchInsertFlatIndex(branch, event.currentIndex));
      return;
    }

    const stepDef = event.item.data as StepDefinition;
    if (stepDef?.id) {
      const insertMainIndex = this.getBranchInsertMainIndex(branch, event.currentIndex);
      if (!this.pipelineState.canInsertStepAtMainIndex(stepDef.id, insertMainIndex)) {
        return;
      }
      this.pipelineState.addStep(stepDef.id, this.getBranchInsertFlatIndex(branch, event.currentIndex));
    }
  }

  selectBranch(branch: BranchRow): void {
    this.selectedBranchId = branch.id;
  }

  startBranchRename(branch: BranchRow, event: MouseEvent): void {
    event.preventDefault();
    event.stopPropagation();
    this.editingBranchKey = branch.key;
    this.editingBranchName = branch.name;
    setTimeout(() => {
      const input = this.branchBoard?.nativeElement.querySelector(
        '.branch-name-input'
      ) as HTMLInputElement | null;
      input?.focus();
      input?.select();
    });
  }

  finishBranchRename(branch: BranchRow): void {
    if (this.editingBranchKey !== branch.key) return;
    const name = this.editingBranchName.trim();
    this.editingBranchKey = null;
    this.pipelineState.renameBranch(branch.key, name);
  }

  cancelBranchRename(): void {
    this.editingBranchKey = null;
  }

  onBranchDrop(event: CdkDragDrop<BranchRow[]>): void {
    if (event.previousContainer !== event.container || event.previousIndex === event.currentIndex) return;

    const branch = this.branchRows[event.previousIndex];
    const target = this.branchRows[event.currentIndex];
    if (!branch || !target) return;

    this.pipelineState.moveStepRange(
      branch.startFlatIndex,
      branch.endFlatIndex,
      event.currentIndex > event.previousIndex ? target.endFlatIndex : target.startFlatIndex
    );
  }

  allowDrop = (): boolean => true;

  private isCurrentBranchImplicitMergeConnection(connection: PipelineConnection): boolean {
    const fromIndex = this.steps.findIndex((step) => step.instance_id === connection.from_instance_id);
    const toIndex = this.steps.findIndex((step) => step.instance_id === connection.to_instance_id);
    if (fromIndex < 0 || toIndex < 0 || fromIndex >= toIndex) return false;
    return this.findBranchStartIndex(fromIndex) === this.findBranchStartIndex(toIndex);
  }

  private findBranchStartIndex(stepIndex: number): number {
    for (let i = Math.min(stepIndex, this.steps.length - 1); i >= 0; i--) {
      if (this.steps[i].step_def_id === 'load_image') return i;
    }
    return 0;
  }

  private getBranchInsertMainIndex(branch: BranchRow, currentIndex: number): number {
    const target = branch.nodes[currentIndex];
    return target ? target.mainIndex : branch.endMainIndex;
  }

  private getBranchInsertFlatIndex(branch: BranchRow, currentIndex: number): number {
    const target = branch.nodes[currentIndex];
    return target ? target.pipelineIndex : branch.endFlatIndex;
  }
}

interface SecondaryNode {
  step: StepInstance | null;
  definition?: StepDefinition;
  pipelineIndex: number;
}

interface MainChainNode {
  step: StepInstance;
  definition?: StepDefinition;
  pipelineIndex: number;
  mainIndex: number;
  secondaries: SecondaryNode[];
  isBranchStart: boolean;
}

interface MergeLine {
  key: string;
  path: string;
}

interface BranchRow {
  id: string;
  key: string;
  name: string;
  index: number;
  nodes: MainChainNode[];
  startFlatIndex: number;
  endFlatIndex: number;
  startMainIndex: number;
  endMainIndex: number;
}
