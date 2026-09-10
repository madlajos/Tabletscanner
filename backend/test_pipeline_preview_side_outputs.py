"""Focused checks for preview-only side-output serialization."""

import base64

import cv2
import numpy as np

from pipeline_engine import (
    _encode_reference_sequence_preview_from_data,
    extract_side_outputs,
)


def test_circle_preview_encodes_only_requested_overlay():
    overlays = [
        np.full((8, 9, 3), 20, dtype=np.uint8),
        np.full((8, 9, 3), 180, dtype=np.uint8),
    ]

    side_outputs = extract_side_outputs(
        {"results": {"circle_overlay": overlays}},
        preview_image_index=1,
    )

    encoded = side_outputs["circle_overlay_base64"]
    assert len(encoded) == 1
    decoded = cv2.imdecode(
        np.frombuffer(base64.b64decode(encoded[0]), dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    assert decoded is not None
    assert float(decoded.mean()) > 150


def test_full_side_output_export_keeps_all_circle_overlays():
    overlays = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(3)]
    side_outputs = extract_side_outputs({"results": {"circle_overlay": overlays}})
    assert len(side_outputs["circle_overlay_base64"]) == 3


def test_particle_preview_contains_only_selected_compact_overlay_data():
    particles = [
        [{"particle_id": "img0_label1", "polygon": [[1, 2]], "contour": [[1, 2], [2, 3]], "area_px": 10.0, "passed_filters": True, "excluded": False}],
        [{"particle_id": "img1_label2", "polygon": [[4, 5]], "contour": [[4, 5], [5, 6]], "area_px": 20.0, "passed_filters": False, "excluded": True}],
    ]

    side_outputs = extract_side_outputs(
        {"meta": {"particles": particles, "particles_filtered": particles}},
        preview_image_index=1,
    )

    preview_particles = side_outputs["meta"]["particles"]
    assert preview_particles == [[{
        "particle_id": "img1_label2",
        "polygon": [[4, 5]],
        "passed_filters": False,
        "excluded": True,
    }]]
    assert "particles_filtered" not in side_outputs["meta"]


def test_characterization_rows_survive_preview_and_exclusion():
    from proc_elements.region_attr import detect_particles
    from proc_elements.filter_region import characterize_particles

    def run(excluded_ids):
        image = np.zeros((40, 40), dtype=np.uint8)
        cv2.circle(image, (20, 20), 5, 255, -1)
        data = {"images": [image], "count": 1, "error": None, "meta": {}, "history": [], "_single_image_index": 3}
        data = detect_particles(data, excluded_ids=excluded_ids)
        data = characterize_particles(data, distribution_mode="per_image")
        return extract_side_outputs(data, preview_image_index=3)["particle_size_distribution"]

    initial = run([])
    assert len(initial["particles"]) == 1
    particle = initial["particles"][0]
    assert particle["particle_id"] == "img3_label1"
    assert particle["polygon"]
    excluded = run([particle["particle_id"]])
    assert excluded["particles"][0]["excluded"] is True
    assert excluded["groups"] == []
    restored = run([])
    assert restored["particles"][0]["excluded"] is False
    assert restored["groups"][0]["particle_count"] == 1

    rows = [dict(particle, particle_id=f"img3_label{i}") for i in range(178)]
    side = extract_side_outputs({"results": {"particle_size_distribution": {
        "mode": "per_image", "particles": rows, "groups": [],
    }}}, preview_image_index=3)
    assert len(side["particle_size_distribution"]["particles"]) == 178


def test_reference_sequence_preview_accepts_bgra_crops():
    bgra_crop = np.zeros((12, 10, 4), dtype=np.uint8)
    bgra_crop[..., 2] = 220
    bgra_crop[..., 3] = 128
    preview = _encode_reference_sequence_preview_from_data({
        "images": [bgra_crop],
        "results": {"reference_crops": [[bgra_crop]]},
    })

    assert preview is not None
    decoded = cv2.imdecode(
        np.frombuffer(base64.b64decode(preview["image_base64"]), dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    assert decoded is not None
    assert decoded.ndim == 3
    assert decoded.shape[2] == 3


if __name__ == "__main__":
    test_circle_preview_encodes_only_requested_overlay()
    test_full_side_output_export_keeps_all_circle_overlays()
    test_particle_preview_contains_only_selected_compact_overlay_data()
    test_characterization_rows_survive_preview_and_exclusion()
    test_reference_sequence_preview_accepts_bgra_crops()
    print("pipeline preview side-output checks passed")
