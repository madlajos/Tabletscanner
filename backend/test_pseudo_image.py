import numpy as np

from proc_elements.pseudo_image import create_pseudo_image


def _data(images):
    return {"images": images, "count": len(images), "paths": ["a", "b"],
            "meta": {}, "results": {}, "history": [], "error": None}


def test_composes_selected_channels_from_two_images():
    first = np.dstack([np.full((2, 3), value, np.uint8) for value in (10, 20, 30)])
    second = np.dstack([np.full((2, 3), value, np.uint8) for value in (40, 50, 60)])

    result = create_pseudo_image(
        _data([first, second]), blue_source="2-R", green_source="1-B", red_source="2-G"
    )

    assert result["error"] is None
    assert result["count"] == 1
    assert np.all(result["images"][0][:, :, 0] == 60)
    assert np.all(result["images"][0][:, :, 1] == 10)
    assert np.all(result["images"][0][:, :, 2] == 50)


def test_requires_images_and_matching_selected_source_sizes():
    assert create_pseudo_image(_data([]))["error"] == "E2150"
    result = create_pseudo_image(_data([
        np.zeros((2, 2), np.uint8), np.zeros((3, 2), np.uint8)
    ]), red_source="2-R")
    assert result["error"] == "E2152"


def test_can_select_channels_from_more_than_two_images():
    images = [np.full((2, 2, 3), index, np.uint8) for index in range(1, 5)]
    result = create_pseudo_image(
        _data(images), blue_source="4-B", green_source="3-G", red_source="2-R"
    )
    assert result["error"] is None
    assert tuple(result["images"][0][0, 0]) == (4, 3, 2)


def test_shifts_checked_layers_together_without_moving_unchecked_layer():
    first = np.zeros((3, 4, 3), np.uint8)
    second = np.zeros_like(first)
    first[1, 1, 0] = 90
    first[1, 1, 1] = 120
    second[1, 2, 2] = 180

    result = create_pseudo_image(
        _data([first, second]),
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
