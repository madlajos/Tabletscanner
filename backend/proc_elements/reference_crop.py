import cv2
import numpy as np


def _robust_lab_stats(img):
    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        if arr.dtype.kind == "f" and arr.size and float(np.nanmax(arr)) <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = cv2.cvtColor(arr[..., 0], cv2.COLOR_GRAY2BGR)
    elif arr.ndim != 3 or arr.shape[2] < 3:
        return None
    pixels = cv2.cvtColor(arr[..., :3], cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    median = np.median(pixels, axis=0)
    q25, q75 = np.percentile(pixels, (25, 75), axis=0)
    return {
        "median": median.astype(float).tolist(),
        "scale": np.maximum((q75 - q25) / 1.349, 1.0).astype(float).tolist(),
    }


def _normalize_squares(raw_squares, crop_size, image_w, image_h):
    squares = []
    if not isinstance(raw_squares, list):
        return squares

    size = max(1, int(crop_size))
    for item in raw_squares:
        if not isinstance(item, dict):
            continue
        try:
            x = int(round(float(item.get("x", 0))))
            y = int(round(float(item.get("y", 0))))
        except (TypeError, ValueError):
            continue

        item_size = size
        if image_w <= 0 or image_h <= 0:
            continue

        x = max(0, min(image_w - 1, x))
        y = max(0, min(image_h - 1, y))
        w = min(item_size, image_w - x)
        h = min(item_size, image_h - y)
        if w <= 0 or h <= 0:
            continue

        square = {"x": x, "y": y, "size": item_size, "width": w, "height": h}
        name = str(item.get("name", "")).strip()
        if name:
            square["name"] = name
        squares.append(square)

    return squares


def _draw_squares_overlay(img, squares, color=(0, 180, 255), thickness=2):
    out = img.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    for idx, sq in enumerate(squares):
        x = int(sq["x"])
        y = int(sq["y"])
        w = int(sq.get("width", sq.get("size", 1)))
        h = int(sq.get("height", sq.get("size", 1)))
        cv2.rectangle(out, (x, y), (x + w, y + h), color, max(1, int(thickness)))
        cv2.putText(
            out,
            str(idx + 1),
            (x + 4, y + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return out


def reference_crop(data, squares=None, square_overrides=None, crop_size=64, show_overlay=True, source_id="", source_label="", debug=False):
    if data["error"] is not None:
        return data

    if data["images"] is None or data["count"] == 0:
        data["error"] = "E3521"
        return data

    crop_size = max(1, int(crop_size or 1))
    raw_squares = squares if isinstance(squares, list) else []
    overrides = square_overrides if isinstance(square_overrides, dict) else {}
    single_img_idx = data.get("_single_image_index", -1)

    all_crops = []
    all_square_meta = []
    overlays = []
    per_image_crops = []
    per_image_square_meta = []

    for image_index, img in enumerate(data["images"]):
        if img is None:
            data["error"] = "E3523"
            return data

        h, w = img.shape[:2]
        orig_idx = single_img_idx if (single_img_idx >= 0 and data.get("count", 0) == 1) else image_index
        image_raw_squares = overrides.get(str(orig_idx), [] if overrides else raw_squares)
        if not isinstance(image_raw_squares, list):
            image_raw_squares = raw_squares
        image_squares = _normalize_squares(image_raw_squares, crop_size, w, h)
        crops = []

        for sq in image_squares:
            x = sq["x"]
            y = sq["y"]
            cw = sq["width"]
            ch = sq["height"]
            crops.append(img[y:y + ch, x:x + cw].copy())

        for sq in image_squares:
            sq["source_image_index"] = int(orig_idx)

        per_image_crops.append(crops)
        per_image_square_meta.append(image_squares)
        overlays.append(_draw_squares_overlay(img, image_squares) if show_overlay else img.copy())

    if overrides:
        indexed_rows = sorted(
            (
                (idx, crops, meta)
                for idx, crops, meta in zip(
                    [
                        single_img_idx if (single_img_idx >= 0 and data.get("count", 0) == 1) else image_index
                        for image_index in range(len(data["images"]))
                    ],
                    per_image_crops,
                    per_image_square_meta,
                )
            ),
            key=lambda item: item[0],
        )
        merged_crops = [crop for _, crops, _ in indexed_rows for crop in crops]
        merged_square_meta = [sq for _, _, meta in indexed_rows for sq in meta]
        all_crops = [merged_crops for _ in data["images"]]
        all_square_meta = [merged_square_meta for _ in data["images"]]
    else:
        all_crops = per_image_crops
        all_square_meta = per_image_square_meta

    data["results"]["reference_crops"] = all_crops
    data["results"]["reference_crop_overlays"] = overlays
    data["results"]["reference_crop_squares"] = all_square_meta
    source_stats = {}
    for image_index, img in enumerate(data["images"]):
        orig_idx = single_img_idx if (single_img_idx >= 0 and data.get("count", 0) == 1) else image_index
        stats = _robust_lab_stats(img)
        if stats is not None:
            source_stats[str(orig_idx)] = stats
    data["results"]["reference_crop_source_stats"] = source_stats
    if source_id:
        reference_sources = data["results"].setdefault("reference_sources", {})
        reference_sources[source_id] = {
            "id": source_id,
            "label": source_label or "Reference crop",
            "type": "reference_crop",
            "reference_crops": all_crops,
            "reference_crop_squares": all_square_meta,
            "reference_crop_source_stats": source_stats,
        }
    data["meta"]["reference_crop"] = {
        "crop_size": crop_size,
        "square_count": len(all_square_meta[0]) if all_square_meta else 0,
        "squares": raw_squares,
        "square_overrides": overrides,
    }
    data["history"].append("reference_crop")

    return data
