import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-recipe-applier',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="applier-placeholder">
      <div class="placeholder-icon">⚗️</div>
      <h3>Recept alkalmazása</h3>
      <p>Ez a funkció hamarosan elérhető lesz.</p>
      <p class="hint">Itt fogja tudni a mentett recepteket képekre alkalmazni kötegelt feldolgozással.</p>
    </div>
  `,
  styles: [`
    :host {
      display: flex;
      align-items: center;
      justify-content: center;
      height: 100%;
    }

    .applier-placeholder {
      text-align: center;
      color: #888;
      padding: 40px;
    }

    .placeholder-icon {
      font-size: 48px;
      margin-bottom: 16px;
    }

    h3 {
      color: #ccc;
      font-size: 18px;
      margin: 0 0 8px;
    }

    p {
      font-size: 13px;
      margin: 4px 0;
    }

    .hint {
      font-size: 11px;
      color: #666;
      margin-top: 12px;
    }
  `],
})
export class RecipeApplierComponent {}
