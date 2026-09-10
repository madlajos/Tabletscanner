import { AutofocusSettings } from './autofocus-settings.models';
import { FilterSettings, HeightOffsetChannel } from './filter-settings.models';

export function focusReferenceKey(settings: FilterSettings, focus: AutofocusSettings): string {
  return settings.slots[focus.filter_position - 1] || 'empty';
}

export function isMasterOffset(settings: FilterSettings, key: string, channel: HeightOffsetChannel): boolean {
  const name = settings.filters.find(filter => filter.id === key)?.name.trim().toLocaleLowerCase('hu-HU');
  return channel === 'vis' && (name === 'kék' || name === 'blue');
}

export function isUnavailableOffset(settings: FilterSettings, key: string, channel: HeightOffsetChannel): boolean {
  const name = settings.filters.find(filter => filter.id === key)?.name
    .trim().toLocaleLowerCase('hu-HU').replace(/[\s_-]+/g, '');
  return channel === 'vis' && ['255nm', '265nm', '365nm'].includes(name || '');
}

export function focusReferenceOffset(settings: FilterSettings, focus: AutofocusSettings): number {
  return settings.height_offsets_mm[focusReferenceKey(settings, focus)]?.[focus.channel] ?? 0;
}

export function relativeFocusOffset(
  settings: FilterSettings, focus: AutofocusSettings, key: string, channel: HeightOffsetChannel
): number {
  return Number((settings.height_offsets_mm[key][channel] - focusReferenceOffset(settings, focus)).toFixed(10));
}

/** Convert an edited relative cell back into the existing blue/VIS calibration. */
export function editRelativeFocusOffset(
  settings: FilterSettings, focus: AutofocusSettings, key: string, channel: HeightOffsetChannel, value: number
): void {
  if (key === focusReferenceKey(settings, focus) && channel === focus.channel) return;
  const absolute = Number((value + focusReferenceOffset(settings, focus)).toFixed(10));
  if (isMasterOffset(settings, key, channel)) {
    // Editing the displayed master cell changes the origin of the stored matrix.
    // All other displayed cells stay put, including the selected pair's zero.
    for (const [rowKey, row] of Object.entries(settings.height_offsets_mm)) {
      for (const col of Object.keys(row) as HeightOffsetChannel[]) {
        if (!isUnavailableOffset(settings, rowKey, col) && !isMasterOffset(settings, rowKey, col)) {
          row[col] = Number((row[col] - absolute).toFixed(10));
        }
      }
    }
  } else {
    settings.height_offsets_mm[key][channel] = absolute;
  }
}
