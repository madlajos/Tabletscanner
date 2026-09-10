import numpy as np

from proc_elements.automatic_rgb_composite import automatic_rgb_composite


def _data(images):
    return {"images": images, "count": len(images),
            "paths": [f"{i}.png" for i in range(len(images))],
            "meta": {}, "results": {}, "history": [], "error": None}


def _pattern(size=64):
    image = np.zeros((size, size), np.uint8)
    image[12:52, 18:46] = 80
    image[22:42, 24:40] = 180
    return image


def test_builds_one_bgr_image_and_reports_metrics():
    base = _pattern()
    inputs = [np.dstack([base, base, base]) for _ in range(3)]
    result = automatic_rgb_composite(_data(inputs), max_scale=1.05,
                                     search_size=100, final_blur=1)
    assert result["error"] is None
    assert result["count"] == 1
    assert result["images"][0].shape == (64, 64, 3)
    metrics = result["results"]["automatic_rgb_metrics"]
    assert abs(metrics["scales"]["green"] - 1.0) < 0.01
    assert abs(metrics["scales"]["red"] - 1.0) < 0.01
    assert result["results"]["automatic_rgb_signal_mask"].dtype == np.uint8


def test_uses_configured_source_order():
    base = _pattern()
    images = [np.dstack([base, base, base]), np.dstack([base // 2] * 3),
              np.dstack([base // 4] * 3)]
    result = automatic_rgb_composite(
        _data(images), blue_image=3, green_image=2, red_image=1,
        min_scale=1.0, max_scale=1.0, white_balance_strength=0.0,
        target_luminance=50.0, apply_gamma=False, final_blur=1,
    )
    pixel = result["images"][0][30, 30]
    assert pixel[0] < pixel[1] < pixel[2]


def test_rejects_missing_or_differently_sized_sources():
    assert automatic_rgb_composite(_data([_pattern(), _pattern()]))["error"] == "E2160"
    result = automatic_rgb_composite(
        _data([_pattern(), _pattern(), np.zeros((32, 32), np.uint8)]))
    assert result["error"] == "E2163"
