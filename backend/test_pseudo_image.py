import numpy as np
import cv2
import pytest

from pipeline_engine import execute_pipeline
from pipeline_steps import STEP_DEFINITIONS
from pipeline_types import PipelineDocument, StepInstance
from pipeline_validators import validate_pipeline

from proc_elements.pseudo_image import create_pseudo_image


def _data(images):
    return {"images": images, "count": len(images), "paths": ["a", "b"],
            "meta": {}, "results": {}, "history": [], "error": None}


def _pipeline(source, **params):
    defaults = {p.name: p.default for p in STEP_DEFINITIONS["pseudo_image"].params}
    return PipelineDocument(steps=[
        StepInstance.create("load_image", {"source": str(source)}),
        StepInstance.create("pseudo_image", {**defaults, **params}),
    ])


@pytest.mark.parametrize("param", ["blue_source", "green_source", "red_source"])
@pytest.mark.parametrize("selector", ["1-B", "2-GRAY", "3-G"])
def test_validation_accepts_dynamic_image_sources(param, selector):
    assert validate_pipeline(_pipeline("unused", **{param: selector})) == []


@pytest.mark.parametrize("selector", ["0-G", "-1-G", "4-B", "12-R", "2-X", "2", "2-GRAY-extra", 2, ""])
def test_validation_rejects_invalid_image_sources(selector):
    errors = validate_pipeline(_pipeline("unused", green_source=selector))
    assert len(errors) == 1
    assert errors[0].param_name == "green_source"
    assert errors[0].error_code == "E3003"


@pytest.mark.parametrize("preview", [False, True])
def test_pipeline_composes_second_image_gray_as_green(tmp_path, preview):
    for index, value in enumerate([20, 80, 140]):
        assert cv2.imwrite(str(tmp_path / f"{index}.png"), np.full((3, 4, 3), value, np.uint8))
    doc = _pipeline(tmp_path, green_source="2-GRAY", green_multiplier=1.5,
                    red_source="3-R")
    result = execute_pipeline(doc, up_to_step=1 if preview else -1,
                              single_image_index=0 if preview else -1)
    assert result.success, result.errors
    assert tuple(result.data["images"][0][0, 0]) == (20, 120, 140)


def test_pipeline_reports_incomplete_group(tmp_path):
    assert cv2.imwrite(str(tmp_path / "one.png"), np.zeros((3, 4, 3), np.uint8))
    result = execute_pipeline(_pipeline(tmp_path, green_source="2-GRAY"))
    assert not result.success
    assert result.errors[0].error_code == "E2154"


def test_repeats_first_trio_channel_recipe_for_every_three_images():
    first = np.dstack([np.full((2, 3), value, np.uint8) for value in (10, 20, 30)])
    second = np.dstack([np.full((2, 3), value, np.uint8) for value in (40, 50, 60)])

    third = np.zeros_like(first)
    first_group = [first, second, third]
    result = create_pseudo_image(
        _data(first_group + [image + 70 for image in first_group]),
        blue_source="2-R", green_source="1-B", red_source="2-G"
    )

    assert result["error"] is None
    assert result["count"] == 2
    assert np.all(result["images"][0][:, :, 0] == 60)
    assert np.all(result["images"][0][:, :, 1] == 10)
    assert np.all(result["images"][0][:, :, 2] == 50)
    assert tuple(result["images"][1][0, 0]) == (130, 80, 120)
    assert result["meta"]["pseudo_image"]["group_count"] == 2


def test_requires_images_and_matching_selected_source_sizes():
    assert create_pseudo_image(_data([]))["error"] == "E2150"
    result = create_pseudo_image(_data([
        np.zeros((2, 2), np.uint8), np.zeros((3, 2), np.uint8), np.zeros((2, 2), np.uint8)
    ]), red_source="2-R")
    assert result["error"] == "E2152"


def test_rejects_incomplete_three_image_group():
    images = [np.zeros((2, 2, 3), np.uint8) for _ in range(4)]
    assert create_pseudo_image(_data(images))["error"] == "E2154"


def test_shifts_checked_layers_together_without_moving_unchecked_layer():
    first = np.zeros((3, 4, 3), np.uint8)
    second = np.zeros_like(first)
    first[1, 1, 0] = 90
    first[1, 1, 1] = 120
    second[1, 2, 2] = 180

    result = create_pseudo_image(
        _data([first, second, np.zeros_like(first)]),
        blue_source="1-B",
        green_source="1-G",
        red_source="2-R",
        move_blue=True,
        move_green=True,
        move_red=False,
        offset_x=1,
        offset_y=-1,
    )

    output = result["images"][0]
    assert output[0, 2, 0] == 90
    assert output[0, 2, 1] == 120
    assert output[1, 2, 2] == 180
    assert output[1, 1, 0] == 0


def test_scales_each_output_layer_independently_around_image_centre():
    image = np.zeros((5, 5, 3), np.uint8)
    image[2, 3, 0] = 200
    image[2, 3, 1] = 150
    image[2, 3, 2] = 100

    result = create_pseudo_image(
        _data([image, image, image]),
        blue_source="1-B",
        green_source="1-G",
        red_source="1-R",
        blue_scale_percent=200,
        green_scale_percent=100,
        red_scale_percent=50,
    )

    output = result["images"][0]
    assert output[2, 4, 0] == 200
    assert output[2, 3, 1] == 150
    assert output[2, 3, 2] == 0
    assert result["meta"]["pseudo_image"]["scale_percent"] == {
        "blue": 200.0, "green": 100.0, "red": 50.0
    }


def test_multiplies_each_layer_independently_and_clips_to_uint8():
    image = np.dstack([
        np.full((2, 2), 100, np.uint8),
        np.full((2, 2), 80, np.uint8),
        np.full((2, 2), 200, np.uint8),
    ])
    result = create_pseudo_image(
        _data([image, image, image]),
        blue_multiplier=2.0,
        green_multiplier=0.5,
        red_multiplier=2.0,
    )

    assert tuple(result["images"][0][0, 0]) == (200, 40, 255)
    assert result["meta"]["pseudo_image"]["multiplier"] == {
        "blue": 2.0, "green": 0.5, "red": 2.0
    }


@pytest.mark.parametrize("preview", [False, True])
@pytest.mark.parametrize("space,thresholds", [
    ("HSV", {"H_min": 50, "H_max": 70, "S_min": 200}),
    ("BGR", {"G_min": 200, "R_max": 10}),
    ("AUTO", {"H_min": 50, "H_max": 70, "S_min": 200}),
])
def test_threshold_directly_after_pseudo_image(tmp_path, preview, space, thresholds):
    image = np.array([[[0, 255, 0], [0, 0, 255]]], dtype=np.uint8)
    for index in range(3):
        assert cv2.imwrite(str(tmp_path / f"colors-{index}.png"), image)
    doc = _pipeline(tmp_path)
    defaults = {p.name: p.default for p in STEP_DEFINITIONS["color_thresh"].params}
    doc.steps.append(StepInstance.create("color_thresh", {**defaults, "space": space, **thresholds}))
    assert validate_pipeline(doc) == []
    result = execute_pipeline(doc, up_to_step=2 if preview else -1)
    assert result.success, result.errors
    np.testing.assert_array_equal(result.data["images"][0], [[255, 0]])


def test_legacy_threshold_inherits_conversion_space(tmp_path):
    image = np.array([[[0, 255, 0], [0, 0, 255]]], dtype=np.uint8)
    for index in range(3):
        assert cv2.imwrite(str(tmp_path / f"colors-{index}.png"), image)
    doc = _pipeline(tmp_path)
    doc.steps.extend([
        StepInstance.create("select_channel", {"space": "BGR", "channel": "ALL"}),
        StepInstance.create("color_thresh", {
            **{p.name: p.default for p in STEP_DEFINITIONS["color_thresh"].params if p.name != "space"},
            "G_min": 200,
        }),
    ])
    assert validate_pipeline(doc) == []
    result = execute_pipeline(doc)
    assert result.success, result.errors
    np.testing.assert_array_equal(result.data["images"][0], [[255, 0]])
