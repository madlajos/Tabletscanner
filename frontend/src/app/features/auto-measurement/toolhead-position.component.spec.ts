import { AdvancedMotionSettings } from '../../models/motion.models';
import { mapToolheadMarker } from './toolhead-position.component';

describe('mapToolheadMarker', () => {
  const settings: AdvancedMotionSettings = {
    use_virtual_com_port: false,
    lower_z_before_xy_move: true,
    xy_move_z_limit_mm: 35,
    max_height_offset_up_mm: 3,
    max_height_offset_down_mm: 3,
    first_tablet_x_mm: 10,
    first_tablet_y_mm: 20,
    first_tablet_z_mm: 20,
    tablet_spacing_mm: 10
  };

  it('maps a position inside the tray to its grid location', () => {
    expect(mapToolheadMarker({ x: 55, y: 45 }, settings)).toEqual(jasmine.objectContaining({
      left: 50,
      top: 72.22222222222221,
      outsideTray: false
    }));
  });

  it('pins an off-tray position to the nearest edge instead of hiding it', () => {
    const marker = mapToolheadMarker({ x: 5, y: 115 }, settings);

    expect(marker).toEqual(jasmine.objectContaining({ left: 0, top: 0, outsideTray: true }));
    expect(marker?.label).toContain('tálcatartományon kívül');
  });

  it('does not render when tray geometry is invalid', () => {
    expect(mapToolheadMarker({ x: 10, y: 20 }, { ...settings, tablet_spacing_mm: 0 })).toBeNull();
  });
});
