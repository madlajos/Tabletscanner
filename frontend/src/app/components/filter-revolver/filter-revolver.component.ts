import { CommonModule } from '@angular/common';
import { Component, EventEmitter, Input, Output } from '@angular/core';
import { FilterDefinition, FilterSettings } from '../../models/filter-settings.models';

@Component({
  selector: 'app-filter-revolver',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './filter-revolver.component.html',
  styleUrls: ['./filter-revolver.component.scss']
})
export class FilterRevolverComponent {
  @Input() settings: FilterSettings = {
    filters: [],
    slots: [null, null, null, null, null, null]
  };
  @Input() activePosition: number | null = null;
  @Input() size = 270;
  @Input() rotationDegrees: number | null = null;
  @Input() interactive = false;
  @Output() positionActivate = new EventEmitter<number>();

  get displayedRotationDegrees(): number {
    return this.rotationDegrees ?? (this.activePosition ? -(this.activePosition - 1) * 60 : 0);
  }

  filterForSlot(filterId: string | null): FilterDefinition | undefined {
    return this.settings.filters.find(filter => filter.id === filterId);
  }

  displayNameForSlot(filterId: string | null): string {
    return (this.filterForSlot(filterId)?.name || 'Üres').slice(0, 6);
  }

  trackSlot(index: number): number {
    return index;
  }

  activatePosition(index: number): void {
    if (this.interactive) this.positionActivate.emit(index + 1);
  }
}
