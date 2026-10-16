import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';
import { DoorInterlockStatus, UNKNOWN_DOOR_INTERLOCK } from '../../models/light.models';

@Component({
  selector: 'app-door-interlock-status', standalone: true, imports: [CommonModule],
  template: `
    <div class="doors" role="status" aria-live="polite" aria-atomic="true">
      <span *ngIf="doors.known && doors.front_open" class="door-badge">
        <img src="assets/SVG/front-door-open.svg" alt="" />
        Ajtó nyitva
      </span>
      <span *ngIf="doors.known && doors.top_open" class="door-badge">
        <img src="assets/SVG/top-cover-open.svg" alt="" />
        Fedél nyitva
      </span>
      <span *ngIf="!doors.known" class="door-badge">
        <img src="assets/SVG/front-door-open.svg" alt="" />
        Ajtóállapot ismeretlen
      </span>
    </div>
  `,
  styles: [`
    :host { display: block; }
    .doors { display: flex; flex-wrap: wrap; justify-content: flex-end; align-items: center; gap: 6px;
      font-size: 12px; }
    .door-badge { display: inline-flex; align-items: center; gap: 6px; padding: 3px 9px;
      border: 1px solid #e9bd43; border-radius: 4px; background: #756020;
      color: #fff; white-space: nowrap; line-height: 20px; }
    img { width: 24px; height: 24px; }
  `],
})
export class DoorInterlockStatusComponent {
  @Input() doors: DoorInterlockStatus = UNKNOWN_DOOR_INTERLOCK;
}
