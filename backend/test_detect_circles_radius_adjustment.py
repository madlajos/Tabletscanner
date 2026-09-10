"""Focused checks for detected-circle radius adjustment."""

from unittest.mock import patch

import numpy as np

from proc_elements.detect_circ import detect_circles


def _data():
    return {
        "images": [np.zeros((100, 100), dtype=np.uint8)],
        "count": 1,
        "error": None,
        "results": {},
        "meta": {},
        "history": [],
    }


def _detected_radius(adjustment_percent):
    hough_result = np.array([[[50.0, 50.0, 20.0]]], dtype=np.float32)
    with patch("cv2.HoughCircles", return_value=hough_result):
        result = detect_circles(
            _data(),
            min_diameter=20,
            max_diameter=60,
            radius_adjustment_percent=adjustment_percent,
        )
    return result["results"]["circles"][0][0]["radius"]


def test_radius_can_be_increased_by_percent():
    assert _detected_radius(125) == 25


def test_radius_can_be_decreased_by_percent():
    assert _detected_radius(75) == 15


def test_percentage_scales_detected_radius_and_mask():
    hough_result = np.array([[[50.0, 50.0, 40.0]]], dtype=np.float32)
    with patch("cv2.HoughCircles", return_value=hough_result):
        result = detect_circles(_data(), radius_adjustment_percent=98, apply_mask=True)
    circle = result["results"]["circles"][0][0]
    assert circle["raw_radius"] == 40
    assert circle["radius"] == 39
    mask = result["results"]["masks"][0]
    assert mask[50, 89] == 255
    assert mask[50, 90] == 0


def test_default_preserves_detected_radius():
    hough_result = np.array([[[50.0, 50.0, 20.0]]], dtype=np.float32)
    with patch("cv2.HoughCircles", return_value=hough_result):
        result = detect_circles(_data())
    assert result["results"]["circles"][0][0]["radius"] == 20
    assert _detected_radius(100) == 20


if __name__ == "__main__":
    test_radius_can_be_increased_by_percent()
    test_radius_can_be_decreased_by_percent()
    test_percentage_scales_detected_radius_and_mask()
    test_default_preserves_detected_radius()
    print("circle radius adjustment checks passed")
