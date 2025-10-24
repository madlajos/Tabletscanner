import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ResultsTableComponent } from './results-table.component';

@NgModule({
  imports: [
    CommonModule,
    ResultsTableComponent
  ],
  exports: [ResultsTableComponent]
})
export class ResultsTableModule {}
