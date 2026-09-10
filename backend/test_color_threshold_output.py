import numpy as np

from proc_elements import color_threshold, create_data


def _data_with(image):
    data = create_data()
    data["images"] = [image]
    data["count"] = 1
    return data


def _thresholds():
    return {"B": (0, 100), "G": (0, 100), "R": (0, 100)}


def test_mask_output_remains_the_default():
    image = np.array([[[10, 20, 30], [200, 20, 30]]], dtype=np.uint8)

    result = color_threshold(_data_with(image), space="BGR", thresholds=_thresholds())

    np.testing.assert_array_equal(result["images"][0], [[255, 0]])


def test_applied_output_preserves_selected_pixels_and_uses_rgb_background():
    image = np.array([[[10, 20, 30], [200, 20, 30]]], dtype=np.uint8)

    result = color_threshold(
        _data_with(image), space="BGR", thresholds=_thresholds(),
        output_mode="applied", background_color="#3366cc",
    )

    # The public color is RGB; OpenCV image storage is BGR.
    expected = np.array([[[10, 20, 30], [204, 102, 51]]], dtype=np.uint8)
    np.testing.assert_array_equal(result["images"][0], expected)


def test_inverted_applied_output_uses_the_inverted_mask():
    image = np.array([[[10, 20, 30], [200, 20, 30]]], dtype=np.uint8)

    result = color_threshold(
        _data_with(image), space="BGR", thresholds=_thresholds(), invert=True,
        output_mode="applied", background_color="#000000",
    )

    expected = np.array([[[0, 0, 0], [200, 20, 30]]], dtype=np.uint8)
    np.testing.assert_array_equal(result["images"][0], expected)
