import cv2
import numpy as np

from pipeline_steps import STEP_DEFINITIONS
from proc_elements.focus_stack import focus_stack


def _data(images):
    return {"images": images, "count": len(images),
            "paths": [f"focus_{i}.png" for i in range(len(images))],
            "meta": {}, "results": {}, "history": [], "error": None}


def _focused_halves(size=96):
    yy, xx = np.indices((size, size))
    detail = np.uint8(((xx // 3 + yy // 3) % 2) * 220 + 20)
    blurred = cv2.GaussianBlur(detail, (15, 15), 0)
    first = blurred.copy()
    first[:, :size // 2] = detail[:, :size // 2]
    second = blurred.copy()
    second[:, size // 2:] = detail[:, size // 2:]
    return first, second, detail


def test_focus_stack_combines_sharp_regions_and_reports_map():
    first, second, detail = _focused_halves()
    result = focus_stack(_data([first, second]), focus_radius=5,
                         blend_radius=3, align_images=False)
    assert result["error"] is None
    assert result["count"] == 1
    assert result["images"][0].shape == detail.shape
    assert np.mean(np.abs(result["images"][0].astype(float) - detail)) < 12
    assert result["results"]["focus_index_map"].dtype == np.uint8
    assert result["meta"]["focus_stack"]["source_count"] == 2


def test_focus_stack_rejects_invalid_input_sets():
    image = np.zeros((32, 32), np.uint8)
    assert focus_stack(_data([image]))["error"] == "E2170"
    assert focus_stack(_data([image, np.zeros((16, 16), np.uint8)]))["error"] == "E2172"


def test_focus_stack_is_registered_in_catalog():
    definition = STEP_DEFINITIONS["focus_stack"]
    assert definition.output_type.value == "IMAGE"
    assert {param.name for param in definition.params} == {
        "focus_radius", "blend_radius", "align_images", "alignment_iterations"
    }
