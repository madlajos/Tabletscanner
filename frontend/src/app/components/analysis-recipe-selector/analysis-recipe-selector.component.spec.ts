import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { TestBed } from '@angular/core/testing';
import { BASE_URL } from '../../api-config';
import { AnalysisRecipeSelectorComponent } from './analysis-recipe-selector.component';

describe('AnalysisRecipeSelectorComponent', () => {
  beforeEach(() => TestBed.configureTestingModule({
    imports: [AnalysisRecipeSelectorComponent],
    providers: [provideHttpClient(), provideHttpClientTesting()],
  }));

  it('lists saved recipes and displays the selected recipe name', () => {
    const fixture = TestBed.createComponent(AnalysisRecipeSelectorComponent);
    const http = TestBed.inject(HttpTestingController);
    const selected = jasmine.createSpy('selected');
    fixture.componentInstance.recipeSelected.subscribe(selected);
    fixture.detectChanges();

    const browse = fixture.nativeElement.querySelector('.browse-recipe-button') as HTMLButtonElement;
    browse.click();
    http.expectOne(`${BASE_URL}/recipes`).flush({
      recipes: [{ name: 'Élő mérés', description: 'Teszt recept', step_count: 3 }],
    });
    http.expectOne(`${BASE_URL}/recipe-folders`).flush({ folders: [] });
    fixture.detectChanges();

    const recipeButton = fixture.nativeElement.querySelector('.recipe-card') as HTMLElement;
    expect(recipeButton.textContent).toContain('Élő mérés');
    recipeButton.click();
    const document = {
      schema_version: 1,
      name: 'Élő mérés',
      description: 'Teszt recept',
      steps: [],
      connections: [],
    };
    http.expectOne(`${BASE_URL}/recipes/${encodeURIComponent('Élő mérés')}`).flush(document);
    fixture.detectChanges();

    const input = fixture.nativeElement.querySelector('#analysis-recipe-name') as HTMLInputElement;
    expect(input.value).toBe('Élő mérés');
    expect(selected).toHaveBeenCalledWith(document);
    expect(fixture.nativeElement.querySelector('.recipe-dialog')).toBeNull();
    http.verify();
  });
});
