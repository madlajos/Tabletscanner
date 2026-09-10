"""Hardware-free contracts for the operator's Z anchor and its invalidation."""

import copy
import unittest
from unittest.mock import patch

import app as backend_app
import globals
import height_offset_control
import porthandler
import settings_manager
from virtual_octopus import VirtualOctopusSerial
from test_height_offset_control import filter_settings


class HeightReferenceApiTests(unittest.TestCase):
    def setUp(self):
        self.settings = copy.deepcopy(settings_manager.get_settings())
        self.device = VirtualOctopusSerial()
        self.device._position['Z'] = 12.5
        globals.motion_platform = porthandler.motion_platform = self.device
        globals.motion_busy = False
        globals.toolhead_homed = globals.filter_revolver_homed = True
        globals.homed_axes = {'x', 'y', 'z', 'a'}
        globals.filter_revolver_position = 2
        globals.last_toolhead_pos = {'x': 5., 'y': 5., 'z': 10.}
        height_offset_control.invalidate_reference()
        settings_manager.set_settings({'filter_settings': filter_settings()})
        self.light = patch.object(backend_app.light_controller, 'status', return_value={'active_channel': 'uv255'})
        self.light.start()
        self.client = backend_app.app.test_client()

    def tearDown(self):
        self.light.stop()
        settings_manager.set_settings(self.settings)
        self.device.close()
        globals.motion_platform = porthandler.motion_platform = None
        globals.motion_busy = False
        globals.toolhead_homed = globals.filter_revolver_homed = False
        globals.homed_axes = set()
        globals.filter_revolver_position = None
        globals.last_toolhead_pos = {'x': None, 'y': None, 'z': None}
        height_offset_control.invalidate_reference()

    def anchor(self):
        return self.client.post('/api/height-offset/reference', json={'enabled': True})

    def test_anchor_reads_physical_z_without_autofocus_or_motion(self):
        with patch.object(backend_app, '_run_configured_manual_autofocus') as autofocus:
            response = self.anchor()
        self.assertEqual(200, response.status_code)
        self.assertEqual('anchor', response.json['source'])
        self.assertEqual(10., response.json['reference_z'])
        self.assertEqual(2.5, response.json['baseline_offset_mm'])
        self.assertEqual(12.5, globals.last_toolhead_pos['z'])
        self.assertEqual(['M400', 'M114'], self.device.command_history)
        autofocus.assert_not_called()
        before = list(self.device.command_history)
        self.assertTrue(self.client.get('/api/height-offset/reference').json['available'])
        self.assertEqual(before, self.device.command_history)

    def test_filter_changes_apply_offsets_and_return_to_anchor_without_drift(self):
        self.anchor()
        empty = self.client.post('/api/filter-revolver/select', json={'position': 1})
        self.assertEqual(10., empty.json['height_offset']['target_z'])
        selected = self.client.post('/api/filter-revolver/select', json={'position': 2})
        self.assertEqual(12.5, selected.json['height_offset']['target_z'])
        self.assertEqual('anchor', height_offset_control.status()['source'])

    def test_second_press_disables_offsets_without_moving(self):
        self.anchor()
        before = list(self.device.command_history)
        response = self.client.post('/api/height-offset/reference', json={'enabled': False})
        self.assertFalse(response.json['available'])
        self.assertIsNone(response.json['source'])
        self.assertEqual(before, self.device.command_history)
        self.client.post('/api/filter-revolver/select', json={'position': 1})
        self.assertEqual(12.5, globals.last_toolhead_pos['z'])

    def test_xy_moves_and_unchanged_z_preserve_anchor_but_z_move_clears_it(self):
        for route, payload in (
            ('relative', {'axis': 'x', 'value': 1}),
            ('absolute', {'y': 7, 'z': 12.5}),
        ):
            self.anchor()
            response = self.client.post('/api/move_toolhead_' + route, json=payload)
            self.assertEqual(200, response.status_code)
            self.assertTrue(height_offset_control.status()['available'])
        for route, payload in (
            ('relative', {'axis': 'z', 'value': 1}),
            ('absolute', {'z': 14}),
        ):
            self.anchor()
            response = self.client.post('/api/move_toolhead_' + route, json=payload)
            self.assertEqual(200, response.status_code)
            self.assertFalse(height_offset_control.status()['available'])

    def test_homing_and_motor_off_clear_anchor(self):
        for route, payload in (('/api/home_toolhead', {'axes': ['z']}), ('/api/disable_steppers', {})):
            self.anchor()
            response = self.client.post(route, json=payload)
            self.assertEqual(200, response.status_code)
            self.assertFalse(height_offset_control.status()['available'])

    def test_rejects_busy_missing_light_unhomed_and_disconnected(self):
        globals.motion_busy = True
        self.assertEqual(409, self.anchor().status_code)
        globals.motion_busy = False
        with patch.object(backend_app.light_controller, 'status', return_value={'active_channel': None}):
            self.assertEqual(409, self.anchor().status_code)
        globals.filter_revolver_homed = False
        self.assertEqual(409, self.anchor().status_code)
        globals.filter_revolver_homed = True
        self.device.close()
        self.assertEqual(503, self.anchor().status_code)
        self.assertEqual([], self.device.command_history)

    def test_malformed_requests_and_busy_manual_motion_do_not_touch_device(self):
        for payload in ({}, {'enabled': 1}, {'enabled': 'true'}, [], {'enabled': True, 'z': 1}):
            response = self.client.post('/api/height-offset/reference', json=payload)
            self.assertEqual(400, response.status_code)
            self.assertEqual('E1206', response.json['code'])
        globals.motion_busy = True
        for route, payload in (('relative', {'axis': 'z', 'value': 1}), ('absolute', {'z': 5})):
            self.assertEqual(409, self.client.post('/api/move_toolhead_' + route, json=payload).status_code)
        self.assertEqual([], self.device.command_history)

    def test_position_query_failure_does_not_fall_back_to_cached_z(self):
        with patch.object(backend_app.motioncontrols, 'get_toolhead_position', side_effect=RuntimeError('No position')):
            response = self.anchor()
        self.assertEqual(503, response.status_code)
        self.assertFalse(height_offset_control.status()['available'])
        self.assertFalse(globals.motion_busy)

    def test_usb_failure_cleans_both_device_handles_and_reference(self):
        self.anchor()
        with patch.object(porthandler, 'write_and_wait', side_effect=OSError('USB disconnected')):
            response = self.anchor()
        self.assertEqual(503, response.status_code)
        self.assertIsNone(globals.motion_platform)
        self.assertIsNone(porthandler.motion_platform)
        self.assertFalse(height_offset_control.status()['available'])

    def test_status_clears_reference_when_device_is_no_longer_live(self):
        self.anchor()
        self.device.close()
        self.assertFalse(self.client.get('/api/height-offset/reference').json['available'])

    def test_calibration_edit_clears_anchor_but_focus_selection_does_not(self):
        self.anchor()
        with patch.object(backend_app, 'update_autofocus_settings', return_value=True):
            response = self.client.put('/api/settings/autofocus', json={
                'channel': 'vis', 'brightness': 'full', 'filter_position': 1,
            })
        self.assertEqual(200, response.status_code)
        self.assertTrue(height_offset_control.status()['available'])
        with patch.object(backend_app, 'update_filter_settings', return_value=True):
            response = self.client.put('/api/settings/filter', json=filter_settings(offset=3))
        self.assertEqual(200, response.status_code)
        self.assertFalse(height_offset_control.status()['available'])


if __name__ == '__main__':
    unittest.main()
