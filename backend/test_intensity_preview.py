"""Hardware-free regression tests for intensity previews in unfinished recipes."""
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from pipeline_engine import execute_pipeline, extract_side_outputs
from pipeline_steps import STEP_DEFINITIONS
from pipeline_types import PipelineDocument, StepInstance


class IntensityPreviewTests(unittest.TestCase):
    def test_preview_ignores_downstream_errors_but_full_run_rejects_them(self):
        with tempfile.TemporaryDirectory() as folder:
            for index, value in enumerate((30, 90)):
                path = Path(folder) / f"{index}.png"
                ok, encoded = cv2.imencode('.png', np.full((16, 16), value, np.uint8))
                self.assertTrue(ok)
                path.write_bytes(encoded.tobytes())
            steps = [
                StepInstance.create(key, {p.name: p.default for p in STEP_DEFINITIONS[key].params})
                for key in ('load_image', 'detect_circles', 'calculate_intensity_stats', 'fit_curve')
            ]
            steps[0].param_values['source'] = folder
            doc = PipelineDocument(steps=steps)
            for selected, expected in ((-1, [30, 90]), (1, [90])):
                result = execute_pipeline(doc, up_to_step=2, single_image_index=selected)
                self.assertTrue(result.success, result.errors)
                stats = extract_side_outputs(result.data)['intensity_stats']
                self.assertEqual([s['mean'] for s in stats], expected)
            full = execute_pipeline(doc)
            self.assertFalse(full.success)
            self.assertTrue(any(e.step_def_id == 'fit_curve' for e in full.errors))
            for mode in ('pooled', 'grouped'):
                steps[2].param_values.update(display_mode=mode, group_labels='["A", "A"]')
                result = execute_pipeline(doc, up_to_step=2, single_image_index=1)
                self.assertTrue(result.success, result.errors)
                group = extract_side_outputs(result.data)['intensity_summary']['groups'][0]
                self.assertEqual(group['sample_count'], 2)
                self.assertEqual(group['channels'][0]['mean'], 60)
                self.assertEqual(group['image_indices'], [0, 1])
            steps[2].param_values['group_labels'] = '["A"]'
            mismatch = execute_pipeline(doc, up_to_step=2)
            self.assertFalse(mismatch.success)
            self.assertEqual(mismatch.errors[0].error_code, 'E2509')
            steps[2].param_values['percentiles'] = ''
            invalid = execute_pipeline(doc, up_to_step=2)
            self.assertFalse(invalid.success)
            self.assertTrue(any(e.step_index == 2 for e in invalid.errors))


if __name__ == '__main__':
    unittest.main()
