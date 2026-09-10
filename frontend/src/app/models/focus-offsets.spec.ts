import { AutofocusSettings } from './autofocus-settings.models';
import { FilterSettings } from './filter-settings.models';
import { editRelativeFocusOffset, relativeFocusOffset } from './focus-offsets';

describe('Focus offsets', () => {
  let settings: FilterSettings;
  let focus: AutofocusSettings;

  beforeEach(() => {
    settings = {
      filters: [
        { id: 'blue', name: 'Kék', wavelength_range: '450', color: '#0000ff' },
        { id: 'green', name: 'Zöld', wavelength_range: '550', color: '#00ff00' }
      ],
      slots: [null, 'blue', 'green', null, null, null],
      height_offsets_mm: {
        empty: { vis: 1, uv255: 1, uv310: 1, uv365: 1 },
        blue: { vis: 0, uv255: 1, uv310: 2, uv365: 3 },
        green: { vis: 2, uv255: 3, uv310: 4, uv365: 5 }
      }
    };
    focus = { channel: 'vis', brightness: 'full', filter_position: 3 };
  });

  it('shows the selected pair as zero and preserves calibration across repeated reference changes', () => {
    const before = JSON.stringify(settings);
    expect(relativeFocusOffset(settings, focus, 'green', 'vis')).toBe(0);
    expect(relativeFocusOffset(settings, focus, 'blue', 'vis')).toBe(-2);
    expect(relativeFocusOffset(settings, focus, 'green', 'uv365')).toBe(3);
    focus = { ...focus, filter_position: 1, channel: 'uv255' };
    expect(relativeFocusOffset(settings, focus, 'empty', 'uv255')).toBe(0);
    focus = { ...focus, filter_position: 2, channel: 'vis' };
    expect(relativeFocusOffset(settings, focus, 'green', 'vis')).toBe(2);
    expect(JSON.stringify(settings)).toBe(before);
  });

  it('stores an edited relative value in the master calibration', () => {
    editRelativeFocusOffset(settings, focus, 'green', 'uv365', -1.25);
    expect(settings.height_offsets_mm['green'].uv365).toBe(.75);
    expect(relativeFocusOffset(settings, focus, 'green', 'uv365')).toBe(-1.25);
    editRelativeFocusOffset(settings, focus, 'green', 'vis', 123);
    expect(relativeFocusOffset(settings, focus, 'green', 'vis')).toBe(0);
  });

  it('lets the blue VIS cell be edited relative to another pair without moving other displayed cells', () => {
    editRelativeFocusOffset(settings, focus, 'blue', 'vis', -3);
    expect(settings.height_offsets_mm['blue'].vis).toBe(0);
    expect(relativeFocusOffset(settings, focus, 'blue', 'vis')).toBe(-3);
    expect(relativeFocusOffset(settings, focus, 'green', 'vis')).toBe(0);
    expect(relativeFocusOffset(settings, focus, 'green', 'uv365')).toBe(3);
    expect(relativeFocusOffset(settings, focus, 'empty', 'vis')).toBe(-1);
  });
});
