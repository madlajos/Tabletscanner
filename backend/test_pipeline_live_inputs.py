"""Inline recipe contracts using temporary captures; no devices or operator data."""

from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from pipeline_engine import (
    execute_pipeline, classify_inline_input_scope, extract_side_outputs,
    _encode_preview_image_from_data,
)
from pipeline_types import PipelineDocument, StepInstance
from pipeline_steps import STEP_DEFINITIONS
from pipeline_validators import validate_pipeline


def _step(name, **params):
    defaults = {param.name: param.default for param in STEP_DEFINITIONS[name].params}
    return StepInstance.create(name, {**defaults, **params})


def _recipe(*steps):
    # A saved, obsolete source and explicit file order must never affect live input.
    return PipelineDocument(steps=[_step("load_image", source="obsolete/source", file_order="2,1,0"), *steps])


def _capture(tmp_path, tablet="A4", wavelength="uv255", channel="red", value=25, name=None):
    path = tmp_path / (name or f"arbitrary-{len(list(tmp_path.iterdir()))}.png")
    assert cv2.imwrite(str(path), np.full((8, 9, 3), value, np.uint8))
    return {"path": str(path), "tablet_id": tablet,
            "metadata": {"wavelength": wavelength, "channel": channel, "Errors": []}}


def _trio(tmp_path, tablet="A4", wavelength="uv255", offset=0):
    # Deliberately blue/red/green, with names that lie about their channel.
    return [_capture(tmp_path, tablet, wavelength, channel, value + offset)
            for channel, value in (("blue", 30), ("red", 10), ("green", 20))]


def _pseudo():
    return _step("pseudo_image", blue_source="3-GRAY", green_source="2-GRAY", red_source="1-GRAY")


def test_inline_overrides_source_and_order_without_mutating_recipe_or_records(tmp_path):
    inputs = _trio(tmp_path)
    doc = _recipe(_pseudo())
    before_doc, before_inputs = deepcopy(doc.to_dict()), deepcopy(inputs)
    result = execute_pipeline(doc, live_inputs=inputs)
    assert result.success, result.errors
    assert tuple(result.data["images"][0][0, 0]) == (30, 20, 10)
    assert result.data["image_metadata"][0]["tablet_id"] == "A4"
    assert result.data["image_metadata"][0]["channel"] == "rgb"
    assert doc.to_dict() == before_doc
    assert inputs == before_inputs


def test_live_input_node_allows_empty_editor_path_only_in_inline_execution(tmp_path):
    doc = PipelineDocument(steps=[_step("load_image", source="")])
    assert validate_pipeline(doc)[0].param_name == "source"
    result = execute_pipeline(doc, live_inputs=[_capture(tmp_path)])
    assert result.success, result.errors
    assert len(result.data["images"]) == 1


def test_empty_live_input_never_falls_back_to_editor_source(tmp_path):
    source = _capture(tmp_path)["path"]
    doc = PipelineDocument(steps=[_step("load_image", source=source)])
    assert execute_pipeline(doc).success
    result = execute_pipeline(doc, live_inputs=[])
    assert not result.success
    assert result.errors[0].error_code == "E2180"


def test_unreadable_live_file_rejects_whole_tablet_instead_of_partial_success(tmp_path):
    inputs = _trio(tmp_path)
    inputs[1]["path"] = str(tmp_path / "missing.png")
    result = execute_pipeline(_recipe(), live_inputs=inputs)
    assert not result.success
    assert result.errors[0].error_code == "E2184"


def test_rgb_groups_are_isolated_by_tablet_and_wavelength_with_uv_extras(tmp_path):
    inputs = (_trio(tmp_path, "A4", "uv255") + _trio(tmp_path, "A4", "vis", 40)
              + _trio(tmp_path, "B4", "uv255", 80))
    inputs.insert(3, _capture(tmp_path, channel=None, value=250))
    result = execute_pipeline(_recipe(_pseudo()), live_inputs=inputs)
    assert result.success, result.errors
    assert [tuple(image[0, 0]) for image in result.data["images"]] == [
        (30, 20, 10), (70, 60, 50), (110, 100, 90),
    ]
    assert [(row["tablet_id"], row["wavelength"]) for row in result.data["image_metadata"]] == [
        ("A4", "uv255"), ("A4", "vis"), ("B4", "uv255"),
    ]


def test_single_image_processing_preserves_all_capture_identity(tmp_path):
    inputs = _trio(tmp_path) + [_capture(tmp_path, channel=None)]
    result = execute_pipeline(_recipe(_step("resize_images", scale=0.5)), live_inputs=inputs)
    assert result.success, result.errors
    assert len(result.data["images"]) == len(result.data["image_metadata"]) == 4
    assert all(image.shape[:2] == (4, 4) for image in result.data["images"])


@pytest.mark.parametrize("missing_scope", ["same_tablet", "other_tablet", "other_wavelength"])
def test_missing_channel_cannot_be_filled_by_another_tablet_or_wavelength(tmp_path, missing_scope):
    inputs = _trio(tmp_path)
    if missing_scope == "same_tablet":
        inputs.pop(0)
    elif missing_scope == "other_tablet":
        inputs[0]["tablet_id"] = "B4"
    else:
        inputs[0]["metadata"]["wavelength"] = "vis"
    result = execute_pipeline(_recipe(_pseudo()), live_inputs=inputs)
    assert not result.success
    assert result.errors[0].error_code == "E2182"


def test_duplicate_channel_reports_tablet_and_wavelength(tmp_path):
    result = execute_pipeline(_recipe(_pseudo()), live_inputs=_trio(tmp_path) + [_capture(tmp_path)])
    assert not result.success
    assert result.errors[0].error_code == "E2183"
    assert "A4, uv255: red" in result.errors[0].message


def test_metadata_not_filenames_binds_legacy_hungarian_filter_names(tmp_path):
    inputs = [_capture(tmp_path, channel=None, value=value, name=f"wrong_{suffix}.png")
              for value, suffix in ((10, "b"), (20, "r"), (30, "g"))]
    for record, filter_name in zip(inputs, ("Piros", "Zöld", "Kék")):
        record["metadata"]["filter_name"] = filter_name
    result = execute_pipeline(_recipe(_pseudo()), live_inputs=inputs)
    assert result.success, result.errors
    assert tuple(result.data["images"][0][0, 0]) == (30, 20, 10)


def test_rgb_preview_keeps_complete_groups_and_serializes_output_identity(tmp_path):
    inputs = _trio(tmp_path) + _trio(tmp_path, "B4", offset=40)
    result = execute_pipeline(_recipe(_pseudo()), up_to_step=1, single_image_index=1, live_inputs=inputs)
    assert result.success, result.errors
    assert len(result.data["images"]) == 2
    side = extract_side_outputs(result.data, preview_image_index=1)
    assert len(side["loaded_paths"]) == 2
    assert side["image_metadata"][1]["tablet_id"] == "B4"
    encoded = _encode_preview_image_from_data(result.data, 1)
    assert encoded["image_count"] == 2
    assert encoded["source_name"] == Path(inputs[4]["path"]).name


def test_automatic_rgb_processes_every_tablet_and_keeps_parameter_mapping(tmp_path):
    doc = _recipe(_step("automatic_rgb_composite", blue_image=3, green_image=2, red_image=1,
                        min_scale=1.0, max_scale=1.0, white_balance_strength=0,
                        apply_gamma=False, final_blur=1))
    with patch("proc_elements.automatic_rgb_composite._best_scale", return_value=(1.0, 1.0)) as scale:
        result = execute_pipeline(doc, live_inputs=_trio(tmp_path) + _trio(tmp_path, "B4", offset=40))
    assert result.success, result.errors
    assert scale.call_count == 4
    assert len(result.data["images"]) == 2
    assert len(result.data["results"]["automatic_rgb_metrics"]) == 2
    assert tuple(result.data["images"][0][0, 0]) == (30, 20, 10)


def test_cumulative_rgb_pca_keeps_images_and_diagram_outputs_for_every_tablet(tmp_path):
    doc = _recipe(_pseudo(), _step("select_channel", space="GRAY", channel="R"),
                  _step("calculate_histograms", bins=16), _step("histogram_pca", max_components=2))
    first = _trio(tmp_path)
    initial = execute_pipeline(doc, live_inputs=first)
    assert not initial.success
    assert initial.errors[0].error_code == "E2405"
    records = first + _trio(tmp_path, "B4", offset=60) + _trio(tmp_path, "C4", offset=120)
    result = execute_pipeline(doc, live_inputs=records)
    assert result.success, result.errors
    side = extract_side_outputs(result.data)
    assert len(result.data["images"]) == 3
    assert len(side["histogram_pca_scores"]) == 3
    assert len(side["histogram_pca_explained_ratio"]) == 2
    assert [row["tablet_id"] for row in side["image_metadata"]] == ["A4", "B4", "C4"]


def test_live_binding_preserves_validation_of_processing_parameters(tmp_path):
    result = execute_pipeline(_recipe(_step("pseudo_image", blue_source="4-GRAY")),
                              live_inputs=_trio(tmp_path))
    assert not result.success
    assert result.errors[0].error_code == "E3003"
    assert result.errors[0].param_name == "blue_source"


def test_inline_focus_stack_keeps_tablets_wavelengths_and_filters_separate(tmp_path):
    inputs = []
    for tablet, wavelength, offset in (("A4", "uv255", 0), ("A4", "vis", 40), ("B4", "uv255", 80)):
        inputs += _trio(tmp_path, tablet, wavelength, offset) + _trio(tmp_path, tablet, wavelength, offset)
    doc = _recipe(_step("focus_stack", align_images=False), _pseudo())
    result = execute_pipeline(doc, live_inputs=inputs)
    assert result.success, result.errors
    assert [tuple(image[0, 0]) for image in result.data["images"]] == [
        (30, 20, 10), (70, 60, 50), (110, 100, 90),
    ]
    assert len(result.data["image_metadata"]) == 3


def test_focus_stack_does_not_combine_unrelated_single_captures(tmp_path):
    records = [_capture(tmp_path), _capture(tmp_path, tablet="B4")]
    result = execute_pipeline(_recipe(_step("focus_stack", align_images=False)), live_inputs=records)
    assert not result.success
    assert result.errors[0].error_code == "E2170"


def test_recursive_merge_branch_uses_live_binding_and_preserves_document(tmp_path):
    first, merge, second, gray = _step("load_image", source=""), _step("branch_merge"), _step("load_image", source=""), _step("select_channel", space="GRAY", channel="R")
    doc = PipelineDocument(steps=[first, merge, second, gray], connections=[{
        "from_instance_id": gray.instance_id, "to_instance_id": merge.instance_id, "kind": "merge",
    }])
    before = deepcopy(doc.to_dict())
    result = execute_pipeline(doc, up_to_step=1, live_inputs=[_capture(tmp_path)])
    assert result.success, result.errors
    panels = result.data["results"]["branch_merge_preview"]["panels"]
    assert len(panels) == 2
    assert panels[0]["is_grayscale"] is True
    assert panels[1]["is_grayscale"] is False
    assert doc.to_dict() == before


def test_failed_recursive_inline_branch_is_an_analysis_error(tmp_path):
    first, merge, second, pseudo = _step("load_image", source=""), _step("branch_merge"), _step("load_image", source=""), _pseudo()
    doc = PipelineDocument(steps=[first, merge, second, pseudo], connections=[{
        "from_instance_id": pseudo.instance_id, "to_instance_id": merge.instance_id, "kind": "merge",
    }])
    result = execute_pipeline(doc, up_to_step=1, live_inputs=[_capture(tmp_path)])
    assert not result.success
    assert result.errors[0].error_code == "E2182"


@pytest.mark.parametrize("step_id,params,scope", [
    ("pseudo_image", {}, "tablet"), ("automatic_rgb_composite", {}, "tablet"),
    ("fit_curve", {}, "cumulative"), ("histogram_pca", {}, "cumulative"),
    ("reference_sequence", {}, "cumulative"),
    ("calculate_histograms", {"display_mode": "pooled"}, "cumulative"),
    ("calculate_intensity_stats", {"display_mode": "grouped"}, "cumulative"),
    ("calculate_intensity_stats", {"display_mode": "per_image"}, "tablet"),
    ("characterize_particles", {}, "cumulative"),
])
def test_scope_defaults_classify_batch_operations(step_id, params, scope):
    assert classify_inline_input_scope(_recipe(_step(step_id, **params))) == scope


def test_disabled_batch_operation_does_not_force_cumulative_scope():
    doc = _recipe(_step("fit_curve"))
    doc.steps[-1].enabled = False
    assert classify_inline_input_scope(doc) == "tablet"
