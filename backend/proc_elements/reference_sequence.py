import cv2
import numpy as np


_SORT_COMPONENTS = {
    "GRAY": ("GRAY", 0),
    "B": ("BGR", 0),
    "G": ("BGR", 1),
    "R": ("BGR", 2),
    "H": ("HSV", 0),
    "S": ("HSV", 1),
    "V": ("HSV", 2),
    "L": ("LAB", 0),
    "A": ("LAB", 1),
    "LAB_B": ("LAB", 2),
}

_COMPONENT_GROUPS = {
    "RGB_ALL": ("R", "G", "B"),
    "HSV_ALL": ("H", "S", "V"),
    "LAB_ALL": ("L", "A", "LAB_B"),
}


def _selected_components(component):
    component = str(component or "GRAY").upper()
    return _COMPONENT_GROUPS.get(component, (component,))


def _component_mean(img, component):
    if img is None or not hasattr(img, "shape"):
        return float("inf")

    arr = img
    if arr.size == 0:
        return float("inf")

    component = str(component or "GRAY").upper()
    space, channel = _SORT_COMPONENTS.get(component, _SORT_COMPONENTS["GRAY"])

    if arr.ndim == 2:
        if space == "GRAY":
            comp = arr
        else:
            arr_bgr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
            comp = _extract_component(arr_bgr, space, channel)
    else:
        comp = _extract_component(arr, space, channel)

    return float(np.mean(comp))


def _extract_component(arr, space, channel):
    if arr.ndim == 3 and arr.shape[2] > 3:
        arr = arr[:, :, :3]
    if space == "GRAY":
        return cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    if space == "HSV":
        return cv2.cvtColor(arr, cv2.COLOR_BGR2HSV)[:, :, channel]
    if space == "LAB":
        return cv2.cvtColor(arr, cv2.COLOR_BGR2LAB)[:, :, channel]
    return arr[:, :, channel]


def _item_label(item, fallback_index):
    meta = item.get("meta")
    if isinstance(meta, dict):
        name = str(meta.get("name", "")).strip()
        if name:
            return name
    return str(fallback_index + 1)


def _reference_items(data, results):
    """Return every reference as one logical row, independent of source image."""
    reference_crops = results.get("reference_crops")
    square_rows = results.get("reference_crop_squares")

    if isinstance(reference_crops, list) and reference_crops:
        crop_rows = [row for row in reference_crops if isinstance(row, list)]
        # reference_crop replicates an override-based, already merged row for
        # every input image. Do not duplicate those references a second time.
        if len(crop_rows) > 1 and all(
            len(row) == len(crop_rows[0])
            and all(item is crop_rows[0][idx] for idx, item in enumerate(row))
            for row in crop_rows[1:]
        ):
            crop_rows = crop_rows[:1]

        items = []
        global_index = 0
        for row_index, row in enumerate(crop_rows):
            square_row = square_rows[row_index] if isinstance(square_rows, list) and row_index < len(square_rows) else []
            for crop_index, crop in enumerate(row):
                meta = square_row[crop_index] if isinstance(square_row, list) and crop_index < len(square_row) else {}
                items.append((crop, meta if isinstance(meta, dict) else {}, global_index))
                global_index += 1
        if items:
            return items

    # A crop node is optional: each complete input image becomes one reference.
    images = data.get("images")
    if not isinstance(images, list):
        return []
    return [
        (image, {"source_image_index": index}, index)
        for index, image in enumerate(images)
        if image is not None and hasattr(image, "shape")
    ]


def reference_sequence(data, component="GRAY", direction="ascending", source_id="", source_label="", debug=False):
    if data["error"] is not None:
        return data

    results = data.get("results")
    if not isinstance(results, dict):
        data["error"] = "E3532"
        return data

    reverse = str(direction or "ascending").lower() == "descending"
    components = _selected_components(component)
    references = _reference_items(data, results)
    if not references:
        data["error"] = "E3531"
        return data

    items = []
    for crop, meta, original_index in references:
            scores = {comp: _component_mean(crop, comp) for comp in components}
            score = float(np.mean(list(scores.values()))) if len(scores) > 1 else next(iter(scores.values()), float("inf"))
            items.append({
                "crop": crop,
                "meta": meta,
                "original_index": original_index,
                "score": score,
                "scores": scores,
            })

    if reverse:
        items.sort(key=lambda item: (-item["score"], item["original_index"]))
    else:
        items.sort(key=lambda item: (item["score"], item["original_index"]))

    previous_scores = None
    for item in items:
        scores = item["scores"]
        item["diffs"] = {
            comp: None if previous_scores is None else scores[comp] - previous_scores.get(comp, 0.0)
            for comp in components
        }
        previous_scores = scores

    sorted_rows = [[item["crop"] for item in items]]
    sequence_rows = [{
        "items": [
            {
                "original_index": item["original_index"],
                "label": _item_label(item, item["original_index"]),
                "score": item["score"],
                "scores": item["scores"],
                "diffs": item["diffs"],
            }
            for item in items
        ]
    }]

    results["reference_crops"] = sorted_rows
    results["reference_sequence"] = sequence_rows
    if source_id:
        reference_sources = results.setdefault("reference_sources", {})
        reference_sources[source_id] = {
            "id": source_id,
            "label": source_label or "Reference sequence",
            "type": "reference_sequence",
            "reference_crops": sorted_rows,
            "reference_sequence": sequence_rows,
        }
    data.setdefault("meta", {})["reference_sequence"] = {
        "component": str(component or "GRAY").upper(),
        "components": list(components),
        "direction": "descending" if reverse else "ascending",
        "rows": sequence_rows,
    }
    data.setdefault("history", []).append("reference_sequence")

    if debug:
        print("Reference crops sorted")
        print(data["meta"]["reference_sequence"])

    return data
