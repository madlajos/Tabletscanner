import unittest

import measurement_progress


class MeasurementProgressTests(unittest.TestCase):
    def test_pending_snapshot_is_non_error_startup_state(self):
        self.assertEqual({
            'request_id': 'new-request',
            'status': 'pending',
            'images': [],
            'warnings': [],
            'active_plan_row_index': None,
        }, measurement_progress.pending_snapshot('new-request'))

    def test_records_independent_snapshot_and_completion(self):
        request_id = 'measurement-test'
        measurement_progress.start(request_id)
        measurement_progress.set_active_plan_row(request_id, 2)
        image = {'path': 'one.jpg', 'masked': False}
        measurement_progress.record_image(request_id, image)
        warning = {'id': 'warning-one', 'code': 'W1205'}
        measurement_progress.record_warning(request_id, warning)
        warning['code'] = 'changed'
        image['path'] = 'changed.jpg'
        measurement_progress.finish(request_id, 'completed')

        self.assertEqual({
            'request_id': request_id,
            'status': 'completed',
            'images': [{'path': 'one.jpg', 'masked': False}],
            'warnings': [{'id': 'warning-one', 'code': 'W1205'}],
            'active_plan_row_index': None,
        }, measurement_progress.snapshot(request_id))

    def test_rejects_invalid_request_id(self):
        with self.assertRaises(ValueError):
            measurement_progress.start('')


if __name__ == '__main__':
    unittest.main()
