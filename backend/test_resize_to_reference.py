import numpy as np

from proc_elements.resize_img import resize_to_reference
from pipeline_engine import extract_side_outputs


def make_data(images):
    return {
        "images": images,
        "paths": [],
        "count": len(images),
        "meta": {},
        "results": {},
        "history": [],
        "error": None,
    }


def test_resizes_each_image_to_corresponding_reference():
    data = make_data([
        np.full((10, 20, 3), 50, np.uint8),
        np.full((12, 18, 3), 100, np.uint8),
    ])
    references = [
        np.zeros((30, 40, 3), np.uint8),
        np.zeros((25, 35, 3), np.uint8),
    ]

    result = resize_to_reference(data, references, interpolation="linear")

    assert result["error"] is None
    assert [image.shape for image in result["images"]] == [(30, 40, 3), (25, 35, 3)]
    assert result["meta"]["reference_resize"]["sizes"][0]["scale_x"] == 2.0
    assert result["history"][-1] == "resize_to_reference"


def test_reuses_last_reference_and_preserves_grayscale():
    data = make_data([
        np.zeros((4, 5), np.uint8),
        np.zeros((6, 7), np.uint8),
    ])

    result = resize_to_reference(data, [np.zeros((8, 9), np.uint8)])

    assert [image.shape for image in result["images"]] == [(8, 9), (8, 9)]


def test_percentage_zoom_keeps_reference_canvas_size():
    image = np.zeros((5, 5), np.uint8)
    image[2, 2] = 255
    data = make_data([image])

    result = resize_to_reference(data, [np.zeros((10, 12), np.uint8)], scale_percent=125)

    assert result["images"][0].shape == (10, 12)
    assert result["meta"]["reference_resize"]["scale_percent"] == 125.0


def test_overlay_visibility_and_opacity_only_affect_preview():
    moving = np.full((4, 4, 3), 40, np.uint8)
    reference = np.full((4, 4, 3), 200, np.uint8)

    blended = resize_to_reference(
        make_data([moving]), [reference], reference_opacity=0.25,
    )
    reference_only = resize_to_reference(
        make_data([moving]), [reference], show_image=False, show_reference=True,
    )
    image_only = resize_to_reference(
        make_data([moving]), [reference], show_image=True, show_reference=False,
    )

    assert int(blended["results"]["reference_resize_previews"][0][0, 0, 0]) == 80
    assert np.array_equal(reference_only["results"]["reference_resize_previews"][0], reference)
    assert np.array_equal(image_only["results"]["reference_resize_previews"][0], moving)
    assert np.array_equal(blended["images"][0], moving)
    layers = extract_side_outputs(blended, preview_image_index=0)["reference_resize_layers_base64"]
    assert layers["image"] and layers["reference"]


def test_requires_reference_images():
    data = make_data([np.zeros((4, 5, 3), np.uint8)])

    result = resize_to_reference(data, [])

    assert result["error"] == "E3912"


def test_rejects_non_positive_percentage():
    data = make_data([np.zeros((4, 5, 3), np.uint8)])

    result = resize_to_reference(data, [np.zeros((8, 9, 3), np.uint8)], scale_percent=0)

    assert result["error"] == "E3915"


if __name__ == "__main__":
    test_resizes_each_image_to_corresponding_reference()
    test_reuses_last_reference_and_preserves_grayscale()
    test_percentage_zoom_keeps_reference_canvas_size()
    test_overlay_visibility_and_opacity_only_affect_preview()
    test_requires_reference_images()
    test_rejects_non_positive_percentage()
    print("resize_to_reference: 6 tests passed")
