"""Checks for number- and volume-based particle-size distributions."""

import numpy as np

from proc_elements.filter_region import characterize_particles
from proc_elements.resize_img import resize_images


def _particle(image_index, label, diameter):
    return {
        "particle_id": f"img{image_index}_label{label}",
        "image_index": image_index,
        "label": label,
        "area_px": float(np.pi * diameter * diameter / 4.0),
        "equivalent_diameter_px": float(diameter),
        "perimeter_px": float(np.pi * diameter),
        "bbox_w_px": float(diameter),
        "bbox_h_px": float(diameter),
        "excluded": False,
    }


def _data():
    rows = [
        [_particle(0, 1, 10), _particle(0, 2, 20)],
        [_particle(1, 1, 30), _particle(1, 2, 40)],
    ]
    return {
        "error": None,
        "meta": {"particles": rows, "particles_filtered": rows},
        "results": {},
        "history": [],
    }


def test_pooled_distribution_and_dv_values():
    result = characterize_particles(
        _data(), size_metric="equivalent_diameter_px", distribution_mode="pooled", bin_count=10
    )
    distribution = result["results"]["particle_size_distribution"]
    assert len(distribution["groups"]) == 1
    group = distribution["groups"][0]
    assert group["particle_count"] == 4
    assert abs(sum(group["number_percent"]) - 100.0) < 1e-9
    assert abs(sum(group["volume_percent"]) - 100.0) < 1e-9
    assert group["dv10"] <= group["dv50"] <= group["dv90"]
    assert group["dv50"] > group["dn50"]


def test_per_image_distribution_uses_shared_bins():
    result = characterize_particles(
        _data(), size_metric="area_px", distribution_mode="overlay", bin_count=8
    )
    groups = result["results"]["particle_size_distribution"]["groups"]
    assert len(groups) == 2
    assert groups[0]["bin_edges"] == groups[1]["bin_edges"]
    assert groups[0]["particle_count"] == groups[1]["particle_count"] == 2


def test_resize_is_reversed_before_micrometer_conversion():
    data = _data()
    data["meta"]["resize"] = {
        "original_width": 1000,
        "original_height": 800,
        "cumulative_scale_x": 0.5,
        "cumulative_scale_y": 0.5,
    }
    result = characterize_particles(
        data,
        size_metric="equivalent_diameter_px",
        distribution_mode="per_image",
        output_unit="um",
        pixels_per_mm=100.0,
    )
    distribution = result["results"]["particle_size_distribution"]
    first_group = distribution["groups"][0]
    # 10 and 20 resized pixels -> 20 and 40 original pixels -> 200 and 400 µm.
    assert first_group["min"] == 200.0
    assert first_group["max"] == 400.0
    assert distribution["resize_correction"]["applied"] is True


def test_smoothing_preserves_histogram_totals():
    result = characterize_particles(
        _data(), distribution_mode="pooled", bin_count=10, smooth_distribution=True
    )
    group = result["results"]["particle_size_distribution"]["groups"][0]
    assert abs(sum(group["number_percent"]) - 100.0) < 1e-9
    assert abs(sum(group["volume_percent"]) - 100.0) < 1e-9


def test_multiple_resize_steps_track_cumulative_scale():
    data = {
        "images": [np.zeros((80, 100), dtype=np.uint8)],
        "count": 1,
        "error": None,
        "meta": {},
        "history": [],
    }
    resize_images(data, scale=0.5)
    resize_images(data, scale=0.5)
    resize_meta = data["meta"]["resize"]
    assert resize_meta["original_width"] == 100
    assert resize_meta["original_height"] == 80
    assert resize_meta["cumulative_scale_x"] == 0.25
    assert resize_meta["cumulative_scale_y"] == 0.25


def test_separate_image_lists_every_included_particle_without_dv():
    from unittest.mock import patch
    from proc_elements.filter_region import _weighted_quantile

    data = _data()
    data["meta"]["particles"][0][0]["excluded"] = True
    with patch("proc_elements.filter_region._weighted_quantile", wraps=_weighted_quantile) as quantile:
        result = characterize_particles(data, distribution_mode="per_image")
    groups = result["results"]["particle_size_distribution"]["groups"]
    assert [[p["value"] for p in g["particle_values"]] for g in groups] == [[20.0], [30.0, 40.0]]
    assert groups[0]["particle_values"][0]["label"] == 2
    assert all(g[key] is None for g in groups for key in ("dv10", "dv50", "dv90"))
    assert quantile.call_count == 6  # Only the three number quantiles per image.

    single = _data()
    single["meta"]["particles"] = single["meta"]["particles"][:1]
    single["meta"]["particles_filtered"] = single["meta"]["particles"]
    result = characterize_particles(single, distribution_mode="per_image", output_unit="um", pixels_per_mm=100)
    assert [p["value"] for p in result["results"]["particle_size_distribution"]["groups"][0]["particle_values"]] == [100.0, 200.0]

    empty = {"error": None, "meta": {"particles": [[]]}}
    assert characterize_particles(empty, distribution_mode="per_image")["results"]["particle_size_distribution"]["groups"] == []


def test_reference_length_calibration():
    from pipeline_steps import _exec_characterize_particles

    for unit, expected in (("um", 5.0), ("mm", 0.005)):
        result = _exec_characterize_particles(_data(), {
            "output_unit": unit, "distribution_mode": "per_image",
            "calibration_pixels": 200, "calibration_length_um": 100,
        })
        assert result["error"] is None
        assert result["results"]["particle_size_distribution"]["particles"][0]["value"] == expected

    data = _data()
    data["meta"]["resize"] = {"cumulative_scale_x": 0.5, "cumulative_scale_y": 0.5}
    result = characterize_particles(data, output_unit="um", size_metric="area_px",
                                    calibration_pixels=200, calibration_length_um=100)
    assert np.isclose(result["results"]["particle_size_distribution"]["groups"][0]["min"], np.pi * 25)
    for pixels, length in ((0, 100), (100, 0), (-1, 100), (float('nan'), 100), (100, float('inf')), ('bad', 100)):
        result = characterize_particles(_data(), output_unit="um", pixels_per_mm=100,
                                        calibration_pixels=pixels, calibration_length_um=length)
        assert result["error"] == "E3112"


if __name__ == "__main__":
    test_pooled_distribution_and_dv_values()
    test_per_image_distribution_uses_shared_bins()
    test_resize_is_reversed_before_micrometer_conversion()
    test_smoothing_preserves_histogram_totals()
    test_multiple_resize_steps_track_cumulative_scale()
    test_separate_image_lists_every_included_particle_without_dv()
    test_reference_length_calibration()
    print("particle size distribution checks passed")
