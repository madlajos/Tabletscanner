import numpy as np

from proc_elements import apply_threshold, create_data


def _data_with(image):
    data = create_data()
    data["images"] = [image]
    data["count"] = 1
    return data


def test_grayscale_behavior_remains_compatible():
    image = np.array([[10, 200]], dtype=np.uint8)

    result = apply_threshold(_data_with(image), thresh=100, maxval=255, mode="binary")

    np.testing.assert_array_equal(result["images"][0], [[0, 255]])
    assert result["error"] is None


def test_selected_rgb_channel_drives_binary_result_for_full_image():
    # OpenCV stores color images as BGR. Only the red values cross the threshold.
    image = np.array([[[240, 230, 10], [10, 20, 200]]], dtype=np.uint8)

    result = apply_threshold(
        _data_with(image), thresh=100, maxval=255, mode="binary", channel="R"
    )

    expected = np.array([[[0, 0, 0], [255, 255, 255]]], dtype=np.uint8)
    np.testing.assert_array_equal(result["images"][0], expected)
    assert result["meta"]["threshold"]["channel"] == "R"


def test_selected_rgb_channel_masks_all_channels_in_tozero_mode():
    image = np.array([[[5, 150, 250], [80, 20, 10]]], dtype=np.uint8)

    result = apply_threshold(
        _data_with(image), thresh=100, mode="tozero", channel="G"
    )

    expected = np.array([[[5, 150, 250], [0, 0, 0]]], dtype=np.uint8)
    np.testing.assert_array_equal(result["images"][0], expected)


def test_inverse_tozero_keeps_full_pixel_even_when_selected_channel_is_zero():
    image = np.array([[[40, 80, 0], [255, 255, 200]]], dtype=np.uint8)

    result = apply_threshold(
        _data_with(image), thresh=100, mode="tozero_inv", channel="R"
    )

    expected = np.array([[[40, 80, 0], [0, 0, 0]]], dtype=np.uint8)
    np.testing.assert_array_equal(result["images"][0], expected)
