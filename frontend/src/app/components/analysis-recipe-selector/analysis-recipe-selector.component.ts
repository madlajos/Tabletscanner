import { CommonModule } from '@angular/common';
import { Component, EventEmitter, Input, Output } from '@angular/core';
import { MatIconModule } from '@angular/material/icon';
import { finalize } from 'rxjs/operators';
import { PipelineDocument } from '../../models/pipeline.models';
import { RecipeService } from '../../services/recipe.service';
import { RecipeBrowserComponent } from '../recipe-browser/recipe-browser.component';

@Component({
  selector: 'app-analysis-recipe-selector',
  standalone: true,
  imports: [CommonModule, MatIconModule, RecipeBrowserComponent],
  templateUrl: './analysis-recipe-selector.component.html',
  styleUrls: ['./analysis-recipe-selector.component.css'],
})
export class AnalysisRecipeSelectorComponent {
  @Input() disabled = false;
  @Output() recipeSelected = new EventEmitter<PipelineDocument>();

  selectedRecipeName = '';
  dialogOpen = false;
  loadingSelection = false;
  errorMessage = '';

  constructor(private recipeService: RecipeService) {}

  openDialog(): void {
    if (this.disabled || this.loadingSelection) return;
    this.dialogOpen = true;
    this.errorMessage = '';
  }

  selectRecipe(name: string): void {
    if (this.loadingSelection) return;

    this.loadingSelection = true;
    this.errorMessage = '';
    this.recipeService.loadRecipe(name).pipe(
      finalize(() => (this.loadingSelection = false)),
    ).subscribe({
      next: recipe => {
        this.selectedRecipeName = recipe.name;
        this.dialogOpen = false;
        this.recipeSelected.emit(recipe);
      },
      error: () => (this.errorMessage = 'A recept betöltése sikertelen.'),
    });
  }
}
