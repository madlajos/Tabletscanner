import { Component, Input, OnChanges, OnDestroy } from '@angular/core';
import { Subscription } from 'rxjs';
import { PipelineDocument } from '../../models/pipeline.models';
import { MontageResponse, RecipeService } from '../../services/recipe.service';

export interface ComparisonPanel {
  label: string;
  imageSrc: string;
  imageCount: number;
  context?: { pipeline: PipelineDocument; stepIndex: number };
  imageIndex?: number;
}

@Component({
  selector: 'app-comparison-panel',
  standalone: true,
  template: `
    <header>
      <strong>{{ panel.label }}</strong>
      @if (panel.context && panel.imageCount > 1) {
        <button type="button" (click)="showMontage()" [attr.aria-pressed]="montageMode">Montázs</button>
        <button type="button" (click)="openImage(selectedIndex)" [attr.aria-pressed]="!montageMode">Egy kép</button>
      }
    </header>
    @if (loading) { <p role="status">Előnézet betöltése…</p> }
    @if (error) { <p role="status">{{ error }}</p> }
    @if (montageMode && montage) {
      <div class="montage">
        <img [src]="'data:image/jpeg;base64,' + montage.montage_base64" [alt]="panel.label + ' – montázs'" draggable="false" />
        @for (cell of cells; track cell) {
          <button type="button" class="cell" [attr.aria-label]="(cell + 1) + '. kép nagyítása'"
            [title]="(cell + 1) + '. kép nagyítása'" (click)="openImage(cell)"
            [style.left.%]="100 * (2 + (cell % montage.grid_cols) * (montage.cell_width + 2)) / montage.montage_width"
            [style.top.%]="100 * (2 + row(cell) * (montage.cell_height + montage.label_height + 2)) / montage.montage_height"
            [style.width.%]="100 * montage.cell_width / montage.montage_width"
            [style.height.%]="100 * (montage.cell_height + montage.label_height) / montage.montage_height"></button>
        }
      </div>
    } @else if (!montageMode) {
      <img class="single" [src]="imageSrc" [alt]="panel.label + ' – ' + (selectedIndex + 1) + '. kép'" draggable="false" />
      @if (panel.context && panel.imageCount > 1) {
        <nav aria-label="Képek lapozása">
          <button type="button" (click)="openImage(selectedIndex - 1)" [disabled]="selectedIndex === 0">Előző</button>
          <span>{{ selectedIndex + 1 }} / {{ panel.imageCount }}</span>
          <button type="button" (click)="openImage(selectedIndex + 1)" [disabled]="selectedIndex >= panel.imageCount - 1">Következő</button>
        </nav>
      }
    }
  `,
  styles: [`
    :host { display: flex; flex-direction: column; min-width: 0; min-height: 0; width: 100%; height: 100%; max-width: 100%; overflow: hidden; box-sizing: border-box; }
    header, nav { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; padding: 8px; }
    strong { flex: 1; }
    button { color: inherit; background: #30343b; border: 1px solid #78818e; border-radius: 4px; padding: 6px 10px; cursor: pointer; }
    button[aria-pressed="true"] { border-color: #60a5fa; background: #234468; }
    button:focus-visible { outline: 3px solid #60a5fa; outline-offset: 2px; }
    button:disabled { opacity: .45; cursor: default; }
    .montage { position: relative; flex: 1 1 auto; min-height: 0; width: 100%; max-width: 100%; overflow: hidden; box-sizing: border-box; }
    .montage > img { display: block; width: 100%; height: 100%; max-width: 100%; object-fit: contain; }
    img { display: block; max-width: 100%; height: auto; }
    .single { flex: 1 1 auto; min-height: 0; width: 100%; object-fit: contain; }
    .cell { position: absolute; background: transparent; border: 0; border-radius: 0; padding: 0; }
    .cell:hover, .cell:focus-visible { outline: 3px solid #60a5fa; outline-offset: -3px; }
  `],
})
export class ComparisonPanelComponent implements OnChanges, OnDestroy {
  @Input({ required: true }) panel!: ComparisonPanel;
  montageMode = true;
  montage: MontageResponse | null = null;
  cells: number[] = [];
  imageSrc = '';
  selectedIndex = 0;
  loading = false;
  error = '';
  private request?: Subscription;

  constructor(private api: RecipeService) {}

  ngOnChanges(): void {
    this.request?.unsubscribe();
    this.montage = null;
    this.cells = [];
    this.imageSrc = this.panel.imageSrc;
    this.selectedIndex = this.panel.imageIndex ?? 0;
    this.error = '';
    this.loading = false;
    this.montageMode = !!this.panel.context && this.panel.imageCount > 1;
    if (this.montageMode) this.showMontage();
  }

  row(index: number): number { return Math.floor(index / this.montage!.grid_cols); }

  showMontage(): void {
    const context = this.panel.context;
    if (!context) return;
    this.request?.unsubscribe();
    this.montageMode = true;
    this.error = '';
    this.loading = false;
    if (this.montage) return;
    this.loading = true;
    this.request = this.api.getStepImagesMontage(context.pipeline, context.stepIndex).subscribe({
      next: result => {
        this.loading = false;
        if (!result.success || !result.montage_base64) { this.fail(); return; }
        this.montage = result;
        this.cells = Array.from({ length: result.image_count }, (_, i) => i);
      },
      error: () => this.fail(),
    });
  }

  openImage(index: number): void {
    const context = this.panel.context;
    if (!context || index < 0 || index >= this.panel.imageCount) return;
    this.request?.unsubscribe();
    this.error = '';
    this.loading = true;
    this.request = this.api.previewStep(context.pipeline, context.stepIndex, index, false).subscribe({
      next: result => {
        this.loading = false;
        if (!result.success || !result.image_base64) { this.fail(); return; }
        this.selectedIndex = index;
        this.imageSrc = `data:image/jpeg;base64,${result.image_base64}`;
        this.montageMode = false;
      },
      error: () => this.fail(),
    });
  }

  private fail(): void {
    this.loading = false;
    this.error = 'Az előnézet betöltése sikertelen. Próbáld újra a nézet kiválasztásával.';
  }

  ngOnDestroy(): void { this.request?.unsubscribe(); }
}
