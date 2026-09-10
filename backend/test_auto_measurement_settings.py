import unittest
from unittest.mock import patch

import app as backend_app


class AutoMeasurementSettingsTests(unittest.TestCase):
    def test_wrapped_camera_parameter_failure_is_a_disconnect(self):
        try:
            try:
                raise RuntimeError('The device has been removed from the system.')
            except RuntimeError as cause:
                raise RuntimeError('Error setting ExposureTime for Camera.') from cause
        except RuntimeError as error:
            self.assertTrue(backend_app._is_camera_disconnect(error))

    def test_unknown_progress_is_pending_instead_of_http_error(self):
        response = backend_app.app.test_client().get(
            '/api/auto_measurement/progress?request_id=not-started-yet'
        )

        self.assertEqual(200, response.status_code)
        self.assertEqual('pending', response.get_json()['status'])

    def test_disabled_autofocus_image_omits_only_reference_row(self):
        plan = [{'filter_position': 4}, {'filter_position': 2}, {'filter_position': 3}]

        rows = backend_app._measurement_capture_rows(
            plan,
            {'auto_measurement_settings': {'save_autofocus_image': False}},
            should_autofocus=True,
        )

        self.assertEqual(list(enumerate(plan))[1:], rows)

    def test_missing_setting_preserves_previous_capture_behavior(self):
        plan = [{'filter_position': 4}, {'filter_position': 2}]

        self.assertEqual(
            list(enumerate(plan)),
            backend_app._measurement_capture_rows(plan, {}, should_autofocus=True),
        )

    def test_reference_row_is_only_captured_when_autofocus_ran(self):
        plan = [{'filter_position': 4}, {'filter_position': 2}]

        self.assertEqual(
            [(1, plan[1])],
            backend_app._measurement_capture_rows(plan, {}, should_autofocus=False),
        )

    def test_tablet_presence_check_can_be_disabled(self):
        self.assertFalse(backend_app._check_tablet_presence_enabled({
            'auto_measurement_settings': {'check_tablet_presence': False}
        }))
        self.assertTrue(backend_app._check_tablet_presence_enabled({}))

    def test_presence_check_restores_autofocus_optics_and_camera_row(self):
        motion_platform = object()
        reference_row = {'filter_position': 4, 'exposure_time': 50000, 'gain': 0}

        with (
            patch.object(backend_app, '_select_configured_autofocus_hardware') as select,
            patch.object(
                backend_app.height_offset_control, 'apply_active_combination'
            ) as apply_offset,
            patch.object(backend_app.time, 'sleep') as settle,
        ):
            select.return_value = (
                {'channel': 'vis', 'filter_position': 4},
                {'slots': [None] * 6},
            )
            backend_app._prepare_tablet_presence_check(
                motion_platform,
                [reference_row, {'filter_position': 2}],
            )

        select.assert_called_once_with(
            motion_platform,
            manage_motion_busy=True,
            capture_plan_row=reference_row,
        )
        apply_offset.assert_called_once_with(
            motion_platform,
            {'slots': [None] * 6},
            'vis',
        )
        settle.assert_called_once_with(0.3)


if __name__ == '__main__':
    unittest.main()
