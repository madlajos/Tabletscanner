import { CommonModule } from '@angular/common';
import { Component, Input, OnDestroy } from '@angular/core';
import { LightChannel, LIGHT_CHANNEL_LABELS, LIGHT_CHANNELS } from '../../models/light.models';
import { BgrCaptureService } from '../../services/bgr-capture.service';
import { SharedService } from '../../shared.service';

@Component({
  selector: 'app-filter-capture-buttons',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="capture-buttons" *ngIf="capture.state$ | async as state"
      role="group" aria-label="RGB+UV képsorozat hullámhosszai">
      <button *ngFor="let channel of channels" type="button" class="wavelength-button"
        [class.selected]="isSelected(channel)"
        [attr.aria-pressed]="isSelected(channel)"
        [attr.aria-label]="labels[channel] + ' megvilágítás kiválasztása'"
        [disabled]="state.running || disabled"
        (click)="toggle(channel)">{{ labels[channel].replace(' nm', '') }}</button>
      <button type="button" class="start-button"
        [class.running]="state.running"
        [attr.aria-label]="state.running ? 'RGB+UV képsorozat megszakítása' : 'RGB+UV képsorozat indítása'"
        [title]="state.running ? 'Kattintson a megszakításhoz' : 'Képsorozat a kiválasztott hullámhosszakkal, szükség esetén autofókusszal'"
        [disabled]="state.cancelling || (!state.running && (disabled || selected.size === 0 || !(shared.cameraConnectionStatus$ | async) || !(shared.saveDirectory$ | async)))"
        (click)="state.running ? capture.requestCancel() : start()">RGB+UV</button>
    </div>
  `,
  styles: [`
    .capture-buttons {
      display: inline-flex;
      justify-content: center;
      overflow: hidden;
      width: 192px;
      border-radius: 4px;
      background: #565555;
      box-shadow: 0 1px 2px rgba(0, 0, 0, .4);
    }
    button {
      box-sizing: border-box;
      width: 32px;
      min-width: 0;
      height: 28px;
      padding: 0;
      border: 0;
      border-right: 1px solid rgba(35, 35, 35, .35);
      border-radius: 0;
      background: #565555;
      color: #fff;
      cursor: pointer;
      font-size: 11px;
      font-weight: 500;
      line-height: 28px;
      transition: background-color .25s ease-out, box-shadow .2s ease-out;
    }
    button:last-child { border-right: 0; }
    button:hover:not(:disabled) { background: #224477; }
    .wavelength-button.selected {
      background: var(--main-blue, #224477);
      color: #fff;
      box-shadow: inset 0 1px 3px rgba(0, 0, 0, .45);
    }
    .start-button {
      width: 64px;
      color: #fff;
      font-size: 10px;
      font-weight: 600;
    }
    button:disabled { opacity: .45; cursor: default; }
    button:focus-visible { outline: 2px solid #90caf9; outline-offset: -3px; z-index: 1; }
    button.running { background: #cc8800; animation: capture-pulse 1.2s ease-in-out infinite; opacity: 1; }
    @keyframes capture-pulse {
      0%, 100% { box-shadow: 0 0 12px rgba(204,136,0,.6), inset 3px 3px 8px rgba(0,0,0,.6); }
      50% { box-shadow: 0 0 20px rgba(204,136,0,.9), inset 3px 3px 8px rgba(0,0,0,.6); }
    }
    @media (prefers-reduced-motion: reduce) { button.running { animation: none; box-shadow: 0 0 12px #cc8800; } }
  `]
})
export class FilterCaptureButtonsComponent implements OnDestroy {
  @Input() disabled = false;
  readonly channels = LIGHT_CHANNELS;
  readonly labels = LIGHT_CHANNEL_LABELS;
  readonly selected = new Set<LightChannel>();

  constructor(public readonly capture: BgrCaptureService, public readonly shared: SharedService) {}

  isSelected(channel: LightChannel): boolean { return this.selected.has(channel); }

  toggle(channel: LightChannel): void {
    this.selected.has(channel) ? this.selected.delete(channel) : this.selected.add(channel);
  }

  start(): void {
    const wavelengths = this.channels.filter(channel => this.selected.has(channel));
    if (!this.disabled && wavelengths.length) this.capture.run(this.shared.getSaveDirectory(), wavelengths);
  }

  ngOnDestroy(): void {
    this.capture.requestCancel();
  }
}
