import { CommonModule } from '@angular/common';
import { Component, EventEmitter, Input, OnInit, Output } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { MatIconModule } from '@angular/material/icon';
import { forkJoin } from 'rxjs';
import { RecipeFolder, RecipeSummary } from '../../models/pipeline.models';
import { RecipeService } from '../../services/recipe.service';

type FolderFilter = 'all' | 'unfiled' | string;

@Component({
  selector: 'app-recipe-browser',
  standalone: true,
  imports: [CommonModule, FormsModule, MatIconModule],
  templateUrl: './recipe-browser.component.html',
  styleUrls: ['./recipe-browser.component.css'],
})
export class RecipeBrowserComponent implements OnInit {
  @Input() editorActions = false;
  @Output() closed = new EventEmitter<void>();
  @Output() recipeChosen = new EventEmitter<string>();
  @Output() recipeAppended = new EventEmitter<string>();

  recipes: RecipeSummary[] = [];
  folders: RecipeFolder[] = [];
  selectedFolderId: FolderFilter = 'all';
  loading = true;
  errorMessage = '';
  newFolderName = '';
  creatingFolder = false;
  renamingFolderId: string | null = null;
  renameFolderName = '';
  pendingFolderDelete: RecipeFolder | null = null;
  pendingRecipeDelete: RecipeSummary | null = null;
  editingDescriptionFor: string | null = null;
  editingDescriptionText = '';

  constructor(private recipeService: RecipeService) {}

  ngOnInit(): void {
    this.refresh();
  }

  get visibleRecipes(): RecipeSummary[] {
    if (this.selectedFolderId === 'all') return this.recipes;
    if (this.selectedFolderId === 'unfiled') return this.recipes.filter(recipe => !recipe.folder_id);
    return this.recipes.filter(recipe => recipe.folder_id === this.selectedFolderId);
  }

  get selectedFolderName(): string {
    if (this.selectedFolderId === 'all') return 'Összes recept';
    if (this.selectedFolderId === 'unfiled') return 'Mappán kívül';
    return this.folders.find(folder => folder.id === this.selectedFolderId)?.name ?? 'Receptek';
  }

  recipeCount(folderId: FolderFilter): number {
    if (folderId === 'all') return this.recipes.length;
    if (folderId === 'unfiled') return this.recipes.filter(recipe => !recipe.folder_id).length;
    return this.recipes.filter(recipe => recipe.folder_id === folderId).length;
  }

  refresh(): void {
    this.loading = true;
    this.errorMessage = '';
    forkJoin({
      recipes: this.recipeService.listRecipes(),
      folders: this.recipeService.listRecipeFolders(),
    }).subscribe({
      next: result => {
        this.recipes = [...result.recipes].sort((left, right) =>
          left.name.localeCompare(right.name, 'hu', { sensitivity: 'base' }));
        this.folders = [...result.folders].sort((left, right) =>
          left.name.localeCompare(right.name, 'hu', { sensitivity: 'base' }));
        this.loading = false;
      },
      error: () => {
        this.errorMessage = 'A receptek betöltése sikertelen.';
        this.loading = false;
      },
    });
  }

  createFolder(): void {
    const name = this.newFolderName.trim();
    if (!name) return;
    this.recipeService.createRecipeFolder(name).subscribe({
      next: ({ folder }) => {
        this.folders = [...this.folders, folder].sort((left, right) =>
          left.name.localeCompare(right.name, 'hu', { sensitivity: 'base' }));
        this.selectedFolderId = folder.id;
        this.newFolderName = '';
        this.creatingFolder = false;
      },
      error: () => (this.errorMessage = 'A mappa létrehozása sikertelen.'),
    });
  }

  startRename(folder: RecipeFolder, event: Event): void {
    event.stopPropagation();
    this.renamingFolderId = folder.id;
    this.renameFolderName = folder.name;
  }

  renameFolder(folder: RecipeFolder): void {
    const name = this.renameFolderName.trim();
    if (!name) return;
    this.recipeService.renameRecipeFolder(folder.id, name).subscribe({
      next: ({ folder: updated }) => {
        this.folders = this.folders.map(item => item.id === updated.id ? updated : item);
        this.renamingFolderId = null;
      },
      error: () => (this.errorMessage = 'A mappa átnevezése sikertelen.'),
    });
  }

  requestFolderDelete(folder: RecipeFolder, event: Event): void {
    event.stopPropagation();
    this.pendingFolderDelete = folder;
  }

  confirmFolderDelete(): void {
    const folder = this.pendingFolderDelete;
    if (!folder) return;
    this.recipeService.deleteRecipeFolder(folder.id).subscribe({
      next: () => {
        this.recipes = this.recipes.map(recipe => recipe.folder_id === folder.id
          ? { ...recipe, folder_id: null }
          : recipe);
        this.folders = this.folders.filter(item => item.id !== folder.id);
        if (this.selectedFolderId === folder.id) this.selectedFolderId = 'unfiled';
        this.pendingFolderDelete = null;
      },
      error: () => (this.errorMessage = 'A mappa törlése sikertelen.'),
    });
  }

  moveRecipe(recipe: RecipeSummary, folderId: string): void {
    const targetId = folderId || null;
    this.recipeService.assignRecipeFolder(recipe.name, targetId).subscribe({
      next: () => (recipe.folder_id = targetId),
      error: () => (this.errorMessage = 'A recept áthelyezése sikertelen.'),
    });
  }

  duplicateRecipe(recipe: RecipeSummary, event: Event): void {
    event.stopPropagation();
    this.recipeService.duplicateRecipe(recipe.name).subscribe({
      next: () => this.refresh(),
      error: () => (this.errorMessage = 'A recept másolása sikertelen.'),
    });
  }

  startEditDescription(recipe: RecipeSummary, event: Event): void {
    event.stopPropagation();
    this.editingDescriptionFor = recipe.name;
    this.editingDescriptionText = recipe.description || '';
  }

  saveDescription(recipe: RecipeSummary): void {
    const description = this.editingDescriptionText.trim();
    this.recipeService.updateRecipeDescription(recipe.name, description).subscribe({
      next: () => {
        recipe.description = description;
        this.editingDescriptionFor = null;
      },
      error: () => (this.errorMessage = 'A megjegyzés mentése sikertelen.'),
    });
  }

  confirmRecipeDelete(): void {
    const recipe = this.pendingRecipeDelete;
    if (!recipe) return;
    this.recipeService.deleteRecipe(recipe.name).subscribe({
      next: () => {
        this.recipes = this.recipes.filter(item => item.name !== recipe.name);
        this.pendingRecipeDelete = null;
      },
      error: () => (this.errorMessage = 'A recept törlése sikertelen.'),
    });
  }
}
