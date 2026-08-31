import { Component, OnInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { Subscription } from 'rxjs';
import { PipelineStateService } from '../../services/pipeline-state.service';
import { RecipeService } from '../../services/recipe.service';
import { PipelineDocument, RecipeSummary } from '../../models/pipeline.models';
import { StepToolboxComponent } from './step-toolbox.component';
import { PipelineCanvasComponent } from './pipeline-canvas.component';
import { StepInspectorComponent } from './step-inspector.component';
import { PipelinePreviewComponent } from './pipeline-preview.component';

@Component({
  selector: 'app-recipe-creator',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    MatButtonModule,
    MatIconModule,
    StepToolboxComponent,
    PipelineCanvasComponent,
    StepInspectorComponent,
    PipelinePreviewComponent,
  ],
  templateUrl: './recipe-creator.component.html',
  styleUrls: ['./recipe-creator.component.css'],
})
export class RecipeCreatorComponent implements OnInit, OnDestroy {
  private static readonly CANVAS_HEIGHT_KEY = 'recipeCreatorCanvasHeight';

  recipeName = '';
  isDirty = false;
  showLoadDialog = false;
  savedRecipes: RecipeSummary[] = [];
  showSaveInput = false;
  saveInputName = '';
  showNewRecipeConfirm = false;
  showOverwriteConfirm = false;
  editingDescriptionFor: string | null = null;
  editingDescriptionText = '';
  showDeleteConfirm: string | null = null;
  canvasHeight = 220;

  private subs: Subscription[] = [];
  private resizeState: {
    startY: number;
    startHeight: number;
    centerHeight: number;
    onMove: (event: PointerEvent) => void;
    onUp: () => void;
  } | null = null;

  constructor(
    public pipelineState: PipelineStateService,
    private recipeService: RecipeService
  ) {}

  ngOnInit(): void {
    this.pipelineState.loadCatalog();
    this.restoreCanvasHeight();

    this.subs.push(
      this.pipelineState.recipeName$.subscribe((n) => (this.recipeName = n)),
      this.pipelineState.dirty$.subscribe((d) => (this.isDirty = d))
    );
  }

  ngOnDestroy(): void {
    this.subs.forEach((s) => s.unsubscribe());
    this.stopCanvasResize();
  }

  startCanvasResize(event: PointerEvent): void {
    event.preventDefault();
    const center = (event.currentTarget as HTMLElement).closest('.creator-center') as HTMLElement | null;
    if (!center) return;

    this.stopCanvasResize();
    const state = {
      startY: event.clientY,
      startHeight: this.canvasHeight,
      centerHeight: center.getBoundingClientRect().height,
      onMove: (moveEvent: PointerEvent) => this.resizeCanvas(moveEvent),
      onUp: () => this.stopCanvasResize(),
    };
    this.resizeState = state;
    document.body.classList.add('resizing-canvas');
    window.addEventListener('pointermove', state.onMove);
    window.addEventListener('pointerup', state.onUp, { once: true });
  }

  private resizeCanvas(event: PointerEvent): void {
    if (!this.resizeState) return;
    const delta = this.resizeState.startY - event.clientY;
    const maxHeight = Math.max(160, this.resizeState.centerHeight - 160);
    this.canvasHeight = this.clamp(this.resizeState.startHeight + delta, 120, maxHeight);
  }

  private stopCanvasResize(): void {
    if (!this.resizeState) return;
    window.removeEventListener('pointermove', this.resizeState.onMove);
    window.removeEventListener('pointerup', this.resizeState.onUp);
    document.body.classList.remove('resizing-canvas');
    localStorage.setItem(RecipeCreatorComponent.CANVAS_HEIGHT_KEY, String(Math.round(this.canvasHeight)));
    this.resizeState = null;
  }

  private restoreCanvasHeight(): void {
    const saved = Number(localStorage.getItem(RecipeCreatorComponent.CANVAS_HEIGHT_KEY));
    if (Number.isFinite(saved) && saved > 0) {
      this.canvasHeight = this.clamp(saved, 120, 600);
    }
  }

  private clamp(value: number, min: number, max: number): number {
    return Math.min(Math.max(value, min), max);
  }

  onNew(): void {
    this.showNewRecipeConfirm = true;
  }

  confirmNewRecipe(): void {
    this.showNewRecipeConfirm = false;
    this.pipelineState.newPipeline();
  }

  cancelNewRecipe(): void {
    this.showNewRecipeConfirm = false;
  }

  onSave(): void {
    if (!this.recipeName) {
      this.showSaveInput = true;
      this.saveInputName = '';
      return;
    }
    this.showOverwriteConfirm = true;
  }

  confirmOverwrite(): void {
    this.showOverwriteConfirm = false;
    if (this.recipeName) {
      this.doSave(this.recipeName);
    }
  }

  cancelOverwrite(): void {
    this.showOverwriteConfirm = false;
  }

  onSaveAs(): void {
    this.showSaveInput = true;
    this.saveInputName = this.recipeName || '';
  }

  confirmSave(): void {
    const name = this.saveInputName.trim();
    if (!name) return;
    this.showSaveInput = false;
    this.doSave(name);
  }

  cancelSave(): void {
    this.showSaveInput = false;
  }

  private doSave(name: string): void {
    const pipeline = this.pipelineState.getPipeline();
    const doc: PipelineDocument = { ...pipeline, name };
    this.recipeService.saveRecipe(doc).subscribe({
      next: () => {
        this.pipelineState.markSaved(name);
      },
      error: (err) => console.error('Save failed:', err),
    });
  }

  onLoad(): void {
    this.showLoadDialog = !this.showLoadDialog;
    if (this.showLoadDialog) {
      this.recipeService.listRecipes().subscribe({
        next: (recipes) => (this.savedRecipes = recipes),
        error: (err) => console.error('List recipes failed:', err),
      });
    }
  }

  loadRecipe(name: string): void {
    this.showLoadDialog = false;
    this.recipeService.loadRecipe(name).subscribe({
      next: (doc) => this.pipelineState.loadPipeline(doc),
      error: (err) => console.error('Load failed:', err),
    });
  }

  appendRecipe(name: string, event: Event): void {
    event.stopPropagation();
    this.recipeService.loadRecipe(name).subscribe({
      next: (doc) => this.pipelineState.appendPipeline(doc),
      error: (err) => console.error('Append failed:', err),
    });
  }

  deleteRecipe(name: string): void {
    this.showDeleteConfirm = name;
  }

  confirmDelete(): void {
    const name = this.showDeleteConfirm;
    this.showDeleteConfirm = null;
    if (!name) return;
    this.recipeService.deleteRecipe(name).subscribe({
      next: () => {
        this.savedRecipes = this.savedRecipes.filter((r) => r.name !== name);
      },
      error: (err) => console.error('Delete failed:', err),
    });
  }

  cancelDelete(): void {
    this.showDeleteConfirm = null;
  }

  duplicateRecipe(name: string, event: Event): void {
    event.stopPropagation();
    this.recipeService.duplicateRecipe(name).subscribe({
      next: (res) => {
        this.recipeService.listRecipes().subscribe({
          next: (recipes) => (this.savedRecipes = recipes),
        });
      },
      error: (err) => console.error('Duplicate failed:', err),
    });
  }

  startEditDescription(recipe: RecipeSummary, event: Event): void {
    event.stopPropagation();
    this.editingDescriptionFor = recipe.name;
    this.editingDescriptionText = recipe.description || '';
  }

  saveDescription(recipe: RecipeSummary): void {
    const newDesc = this.editingDescriptionText.trim();
    this.editingDescriptionFor = null;
    this.recipeService.updateRecipeDescription(recipe.name, newDesc).subscribe({
      next: () => {
        recipe.description = newDesc;
      },
      error: (err) => console.error('Description update failed:', err),
    });
  }

  cancelEditDescription(): void {
    this.editingDescriptionFor = null;
  }
}
