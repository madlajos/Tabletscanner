import { AsyncPipe } from '@angular/common';
import { ChangeDetectionStrategy, Component, inject } from '@angular/core';
import { combineLatest, map } from 'rxjs';
import { SharedService } from '../../shared.service';
import { MotionSettingsService } from '../../services/motion-settings.service';
import { AdvancedMotionSettings } from '../../models/motion.models';

export interface ToolheadMarker {
  left: number;
  top: number;
  outsideTray: boolean;
  label: string;
}

export function mapToolheadMarker(
  position: { x: number; y: number },
  settings: AdvancedMotionSettings
): ToolheadMarker | null {
  const spacing = settings.tablet_spacing_mm;
  if (!Number.isFinite(spacing) || spacing <= 0) return null;

  const column = (position.x - settings.first_tablet_x_mm) / spacing;
  const row = (position.y - settings.first_tablet_y_mm) / spacing;
  if (![column, row].every(Number.isFinite)) return null;

  const outsideTray = column < 0 || column > 9 || row < 0 || row > 9;
  const clamp = (value: number) => Math.max(0, Math.min(9, value));
  const location = outsideTray ? ', a tálcatartományon kívül' : '';
  return {
    left: clamp(column) / 9 * 100,
    top: (9 - clamp(row)) / 9 * 100,
    outsideTray,
    label: `Kamera utolsó ismert helyzete${location}: X ${position.x.toFixed(2)}, Y ${position.y.toFixed(2)} mm`
  };
}

@Component({
  selector: 'app-toolhead-position',
  standalone: true,
  imports: [AsyncPipe],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    @if (marker$ | async; as marker) {
      <span class="position-marker" [class.outside-tray]="marker.outsideTray"
        [style.left.%]="marker.left" [style.top.%]="marker.top"
        role="img" [attr.aria-label]="marker.label" [attr.title]="marker.label"></span>
    }
  `,
  styles: `
    :host { position: absolute; inset: 0; pointer-events: none; z-index: 1; }
    .position-marker {
      position: absolute;
      width: 13px;
      height: 13px;
      box-sizing: border-box;
      border: 2px solid #71c3ff;
      border-radius: 50%;
      transform: translate(-50%, -50%);
      background: rgba(32, 43, 52, .92);
      box-shadow: 0 0 0 2px rgba(20, 28, 34, .72), 0 0 7px rgba(87, 185, 255, .5);
    }
    .position-marker::after {
      content: '';
      position: absolute;
      inset: 3px;
      border-radius: 50%;
      background: #9dd8ff;
    }
    .position-marker.outside-tray {
      border-color: #ffbf69;
      border-style: dashed;
      box-shadow: 0 0 0 2px rgba(20, 28, 34, .72), 0 0 7px rgba(255, 191, 105, .55);
    }
    .position-marker.outside-tray::after { background: #ffd39a; }
  `
})
export class ToolheadPositionComponent {
  private readonly shared = inject(SharedService);
  private readonly settings = inject(MotionSettingsService);

  readonly marker$ = combineLatest([
    this.shared.reportedToolheadPosition$, this.settings.advanced$,
    this.shared.motionPlatformConnectionStatus$, this.shared.motionHomingStatus$
  ]).pipe(map(([position, settings, connected, homing]) => {
    if (!position || !settings || !connected || homing) return null;
    return mapToolheadMarker(position, settings);
  }));
}
