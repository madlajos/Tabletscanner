import cv2
import numpy as np

from pipeline_steps import STEP_DEFINITIONS
from proc_elements.region_attr import _percentiles_fast, detect_particles


def make_data(image, active_mask=None):
    data = {
        "images": [image],
        "count": 1,
        "error": None,
        "meta": {},
        "history": [],
        "results": {},
    }
    if active_mask is not None:
        data["meta"]["active_masks"] = [active_mask]
    return data


def test_dark_particles_use_circle_mask_and_otsu():
    image = np.full((120, 120), 15, dtype=np.uint8)
    circle_mask = np.zeros_like(image)
    cv2.circle(circle_mask, (60, 60), 48, 255, -1)
    image[circle_mask > 0] = 180
    cv2.circle(image, (45, 55), 5, 30, -1)
    cv2.circle(image, (75, 68), 7, 45, -1)

    result = detect_particles(
        make_data(image, circle_mask),
        particle_polarity="dark",
        segmentation_threshold=0,
        draw=False,
    )

    assert result["error"] is None
    assert result["meta"]["particles_summary"][0]["particle_count"] == 2
    segmentation = result["meta"]["detect_particles_config"]["segmentation"][0]
    assert segmentation["mode"] == "otsu"
    assert segmentation["circle_mask_applied"] is True


def test_bright_particles():
    image = np.full((80, 80), 30, dtype=np.uint8)
    cv2.circle(image, (20, 25), 4, 220, -1)
    cv2.circle(image, (55, 50), 6, 200, -1)

    result = detect_particles(
        make_data(image),
        particle_polarity="bright",
        segmentation_threshold=100,
        draw=False,
    )

    assert result["error"] is None
    assert result["meta"]["particles_summary"][0]["particle_count"] == 2


def test_binary_mask_keeps_legacy_nonzero_foreground():
    image = np.zeros((50, 50), dtype=np.uint8)
    image[5:10, 5:10] = 255
    image[30:38, 32:40] = 255

    result = detect_particles(
        make_data(image),
        particle_polarity="dark",
        draw=False,
    )

    assert result["error"] is None
    assert result["meta"]["particles_summary"][0]["particle_count"] == 2
    segmentation = result["meta"]["detect_particles_config"]["segmentation"][0]
    assert segmentation["mode"] == "binary_mask"


def test_particle_touching_mask_edge_is_not_filtered_or_drawn():
    image = np.full((80, 80), 180, dtype=np.uint8)
    circle_mask = np.zeros_like(image)
    cv2.circle(circle_mask, (40, 40), 30, 255, -1)
    image[circle_mask == 0] = 0
    cv2.circle(image, (40, 40), 4, 20, -1)
    cv2.circle(image, (69, 40), 4, 20, -1)

    result = detect_particles(
        make_data(image, circle_mask),
        particle_polarity="dark",
        segmentation_threshold=100,
        exclude_mask_edge=True,
        draw=True,
    )

    particles = result["meta"]["particles"][0]
    assert len(particles) == 2
    assert sum(p["touches_mask_edge"] for p in particles) == 1
    assert result["meta"]["particles_summary"][0]["particle_count_filtered"] == 1

    overlay = result["meta"]["particles_overlay"][0]
    assert tuple(overlay[40, 36]) == (0, 255, 0)
    assert tuple(overlay[40, 65]) != (0, 255, 0)


def test_characterize_selected_columns_is_optional():
    definition = STEP_DEFINITIONS["characterize_particles"]
    selected_columns = next(
        param for param in definition.params if param.name == "selected_columns"
    )
    assert selected_columns.default == ""
    assert selected_columns.required is False


def test_fast_small_component_percentiles_match_numpy():
    values = np.array([3, 5, 8, 10, 40, 90, 120], dtype=np.uint8)
    percentiles = (5, 25, 50, 75, 95)
    assert np.allclose(
        _percentiles_fast(values, percentiles),
        np.percentile(values, percentiles),
    )


def test_full_image_component_is_not_reported_as_particle():
    image = np.full((80, 100), 220, dtype=np.uint8)
    cv2.circle(image, (50, 40), 8, 20, -1)

    result = detect_particles(
        make_data(image),
        particle_polarity="bright",
        segmentation_threshold=100,
        draw=True,
    )

    assert result["meta"]["particles"][0] == []
    assert result["meta"]["particles_summary"][0]["particle_count"] == 0
    assert np.array_equal(result["images"][0], image)


if __name__ == "__main__":
    test_dark_particles_use_circle_mask_and_otsu()
    test_bright_particles()
    test_binary_mask_keeps_legacy_nonzero_foreground()
    test_particle_touching_mask_edge_is_not_filtered_or_drawn()
    test_characterize_selected_columns_is_optional()
    test_fast_small_component_percentiles_match_numpy()
    test_full_image_component_is_not_reported_as_particle()
    print("detect_particles tests passed")
