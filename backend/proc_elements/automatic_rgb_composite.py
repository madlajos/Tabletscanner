"""Create a naturally balanced BGR image from three filter captures."""

import cv2
import numpy as np


def _gray(image):
    array = np.asarray(image)
    if array.ndim == 2:
        return array.astype(np.uint8, copy=False)
    if array.ndim == 3 and array.shape[2] in (3, 4):
        return cv2.cvtColor(array[:, :, :3], cv2.COLOR_BGR2GRAY)
    return None


def _scale_center(image, scale):
    height, width = image.shape[:2]
    center = ((width - 1) / 2.0, (height - 1) / 2.0)
    matrix = cv2.getRotationMatrix2D(center, 0.0, float(scale))
    return cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _edge_image(image, max_size):
    gray = _gray(image)
    height, width = gray.shape
    longest = max(height, width)
    if longest > max_size:
        factor = float(max_size) / float(longest)
        gray = cv2.resize(gray, (max(1, round(width * factor)), max(1, round(height * factor))),
                          interpolation=cv2.INTER_AREA)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0).astype(np.float32) / 255.0
    gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    maximum = float(np.percentile(magnitude, 99.5))
    return np.clip(magnitude / maximum, 0.0, 1.0) if maximum > 0 else magnitude


def _similarity(reference, moving):
    height, width = reference.shape
    mx, my = int(width * 0.07), int(height * 0.07)
    ref = reference[my:height - my, mx:width - mx]
    mov = moving[my:height - my, mx:width - mx]
    if ref.size == 0:
        return -1.0
    threshold = max(float(np.percentile(ref, 60)), float(np.percentile(mov, 60)), 0.015)
    valid = (ref > threshold) | (mov > threshold)
    if np.count_nonzero(valid) < min(500, max(10, valid.size // 100)):
        return -1.0
    a, b = ref[valid].astype(np.float32), mov[valid].astype(np.float32)
    a -= np.mean(a)
    b -= np.mean(b)
    denominator = np.sqrt(np.sum(a * a)) * np.sqrt(np.sum(b * b))
    return -1.0 if denominator < 1e-12 else float(np.sum(a * b) / denominator)


def _best_scale(reference_image, moving_image, min_scale, max_scale, search_size):
    reference = _edge_image(reference_image, search_size)
    moving = _edge_image(moving_image, search_size)
    if moving.shape != reference.shape:
        moving = cv2.resize(moving, (reference.shape[1], reference.shape[0]), interpolation=cv2.INTER_AREA)
    best_scale, best_score = float(min_scale), -999.0
    for pass_index in range(3):
        if pass_index == 0:
            scales = np.linspace(min_scale, max_scale, 61)
        elif pass_index == 1:
            scales = np.linspace(max(min_scale, best_scale - 0.012), min(max_scale, best_scale + 0.012), 121)
        else:
            scales = np.linspace(max(min_scale, best_scale - 0.0015), min(max_scale, best_scale + 0.0015), 61)
        for scale in scales:
            score = _similarity(reference, _scale_center(moving, scale))
            if score > best_score:
                best_scale, best_score = float(scale), score
    return best_scale, best_score


def _signal_mask(blue, green, red):
    maximum = np.maximum(np.maximum(blue, green), red).astype(np.float32)
    non_black = maximum[maximum > 2]
    threshold = max(float(np.percentile(non_black, 25)), 4.0) if non_black.size >= 100 else 2.0
    mask = np.uint8(maximum > threshold) * 255
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) > 0


def _robust_level(channel, mask):
    values = channel[mask].astype(np.float32)
    if values.size < 100:
        return 1.0
    low, high = np.percentile(values, (10, 90))
    values = values[(values >= low) & (values <= high)]
    return 1.0 if values.size < 100 else float(np.median(values))


def automatic_rgb_composite(data, blue_image=1, green_image=2, red_image=3,
                            min_scale=1.0, max_scale=1.3, search_size=700,
                            white_balance_strength=0.65, target_luminance=50.0,
                            apply_gamma=True, final_blur=3):
    if not isinstance(data, dict) or data.get("error"):
        return data
    images = data.get("images") or []
    try:
        indices = [int(blue_image) - 1, int(green_image) - 1, int(red_image) - 1]
    except (TypeError, ValueError):
        data["error"] = "E2161"
        return data
    if any(index < 0 or index >= len(images) for index in indices):
        data["error"] = "E2160"
        return data
    source_images = [images[index] for index in indices]
    channels = [_gray(image) for image in source_images]
    if any(channel is None for channel in channels):
        data["error"] = "E2162"
        return data
    if len({channel.shape for channel in channels}) != 1:
        data["error"] = "E2163"
        return data
    if not 0 < min_scale <= max_scale:
        data["error"] = "E2164"
        return data

    scale_g, score_g = _best_scale(source_images[0], source_images[1], min_scale, max_scale, search_size)
    scale_r, score_r = _best_scale(source_images[0], source_images[2], min_scale, max_scale, search_size)
    blue, green, red = channels[0].copy(), _scale_center(channels[1], scale_g), _scale_center(channels[2], scale_r)
    mask = _signal_mask(blue, green, red)

    levels = np.array([_robust_level(channel, mask) for channel in (blue, green, red)], np.float64)
    target = float(np.prod(np.maximum(levels, 1e-6)) ** (1.0 / 3.0))
    gains = np.clip(1.0 + (target / np.maximum(levels, 1e-6) - 1.0) * white_balance_strength, 0.5, 2.0)
    corrected = [np.clip(channel.astype(np.float32) * gain, 0, 255)
                 for channel, gain in zip((blue, green, red), gains)]

    luminance = 0.114 * corrected[0] + 0.587 * corrected[1] + 0.299 * corrected[2]
    values = luminance[mask]
    current = float(np.percentile(values, 65)) if values.size >= 100 else 0.0
    brightness_gain = float(np.clip(target_luminance / current, 0.65, 2.2)) if current > 1.0 else 1.0
    corrected = [np.clip(channel * brightness_gain, 0, 255) for channel in corrected]

    gamma = 1.0
    if apply_gamma:
        luminance = 0.114 * corrected[0] + 0.587 * corrected[1] + 0.299 * corrected[2]
        values = luminance[mask]
        if values.size >= 100:
            median = float(np.clip(np.median(values) / 255.0, 0.01, 0.99))
            exponent = np.log(0.55) / np.log(median)
            gamma = float(np.clip(1.0 / max(exponent, 1e-6), 0.75, 1.8))
            corrected = [np.power(channel / 255.0, 1.0 / gamma) * 255.0 for channel in corrected]

    output = cv2.merge([np.uint8(np.clip(channel, 0, 255)) for channel in corrected])
    if final_blur > 1:
        kernel_size = int(final_blur) | 1
        output = cv2.GaussianBlur(output, (kernel_size, kernel_size), 0)
    metrics = {
        "scales": {"blue": 1.0, "green": scale_g, "red": scale_r},
        "scores": {"green": score_g, "red": score_r},
        "white_balance_gains": {"blue": float(gains[0]), "green": float(gains[1]), "red": float(gains[2])},
        "brightness_gain": brightness_gain, "gamma": gamma,
    }
    results = data.setdefault("results", {})
    results["automatic_rgb_signal_mask"] = np.uint8(mask) * 255
    results["automatic_rgb_metrics"] = metrics
    data["images"], data["count"] = [output], 1
    data["paths"] = (data.get("paths") or [])[:1]
    data.setdefault("meta", {})["automatic_rgb_composite"] = metrics
    data.setdefault("history", []).append("automatic_rgb_composite")
    return data
