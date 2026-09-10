import { Component, Input, OnDestroy, OnInit } from '@angular/core';
import { MatIconModule } from '@angular/material/icon';
import { EMPTY, Subject, catchError, exhaustMap, finalize, takeUntil, tap, timer } from 'rxjs';
import { HeightReferenceService } from '../../services/height-reference.service';

@Component({
  selector: 'app-focus-anchor',
  standalone: true,
  imports: [MatIconModule],
  template: `
    <button type="button" class="anchor-button" [class.active]="active"
      [attr.aria-pressed]="active" [disabled]="disabled || busy || (!active && !canAnchor)"
      [attr.aria-label]="active ? 'Z-alaphelyzet feloldása' : 'Aktuális Z rögzítése alaphelyzetként'"
      [title]="active ? 'Z-alaphelyzet rögzítve – kattintson a feloldáshoz' : 'Aktuális Z rögzítése az aktív megvilágítás–szűrő párhoz'"
      (click)="toggle()"><mat-icon aria-hidden="true">anchor</mat-icon></button>
  `,
  styles: [`
    :host { display: flex; }
    .anchor-button { display: grid; place-items: center; box-sizing: border-box; width: 28px; height: 30px;
      padding: 0; border: 1px solid rgba(255,255,255,.25); border-radius: 0 4px 4px 0;
      background: #1f2933; color: #eee; cursor: pointer; }
    .anchor-button mat-icon { width: 18px; height: 18px; font-size: 18px; }
    .anchor-button.active { background: var(--main-blue, #224477); color: white; }
    .anchor-button:disabled { opacity: .45; cursor: default; }
    .anchor-button:focus-visible { outline: 2px solid #90caf9; outline-offset: 3px; }
  `]
})
export class FocusAnchorComponent implements OnInit, OnDestroy {
  @Input() disabled = false;
  @Input() canAnchor = false;
  active = false;
  busy = false;
  private generation = 0;
  private readonly destroy$ = new Subject<void>();

  constructor(private readonly reference: HeightReferenceService) {}

  ngOnInit(): void {
    timer(0, 500).pipe(
      exhaustMap(() => {
        if (this.busy) return EMPTY;
        const generation = this.generation;
        return this.reference.get().pipe(
          // A poll started before a toggle must not overwrite its response.
          tap(status => {
            if (generation === this.generation) this.active = status.available && status.source === 'anchor';
          }),
          catchError(() => EMPTY)
        );
      }),
      takeUntil(this.destroy$)
    ).subscribe();
  }

  toggle(): void {
    if (this.disabled || this.busy || (!this.active && !this.canAnchor)) return;
    this.generation++;
    this.busy = true;
    this.reference.setEnabled(!this.active).pipe(
      finalize(() => this.busy = false),
      takeUntil(this.destroy$)
    ).subscribe({
      next: status => this.active = status.available && status.source === 'anchor',
      error: () => undefined // The shared interceptor presents actionable backend errors.
    });
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }
}
