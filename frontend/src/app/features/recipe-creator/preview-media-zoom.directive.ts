import { Directive, ElementRef, HostListener } from '@angular/core';

@Directive({ selector: '[appPreviewMediaZoom]', standalone: true })
export class PreviewMediaZoomDirective {
  private scale = 1;

  constructor(private readonly elementRef: ElementRef<HTMLElement>) {}

  @HostListener('wheel', ['$event'])
  onWheel(event: WheelEvent): void {
    if (!event.ctrlKey || (event.target as HTMLElement | null)?.closest('table')) return;
    event.preventDefault();
    event.stopPropagation();
    this.scale = Math.max(1, Math.min(5, this.scale * (event.deltaY > 0 ? 0.9 : 1.1)));
    this.elementRef.nativeElement.style.setProperty('zoom', String(this.scale));
  }

  @HostListener('dblclick', ['$event'])
  reset(event: MouseEvent): void {
    event.stopPropagation();
    this.scale = 1;
    this.elementRef.nativeElement.style.removeProperty('zoom');
  }
}
