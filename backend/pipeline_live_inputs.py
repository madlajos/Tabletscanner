"""Hardware-free binding of complete captured tablets to recipe input nodes.

Bindings belong to an execution, never to the persisted recipe. RGB source
numbers have a stable meaning: 1 = red, 2 = green, 3 = blue, within one tablet
and illumination wavelength. Filenames and filesystem order are irrelevant.
"""

from copy import deepcopy
from pathlib import Path
import unicodedata

from proc_elements.load_img import create_data, _decode_single_image


def classify_inline_input_scope(doc):
    """Choose a useful default; a session may explicitly override this choice."""
    for step in doc.steps:
        if not step.enabled:
            continue
        if step.step_def_id in ("fit_curve", "histogram_pca", "reference_sequence"):
            return "cumulative"
        if (step.step_def_id in ("calculate_intensity_stats", "calculate_histograms")
                and step.param_values.get("display_mode", "per_image") in ("pooled", "grouped")):
            return "cumulative"
        if (step.step_def_id == "characterize_particles"
                and step.param_values.get("distribution_mode", "pooled") in ("pooled", "overlay")):
            return "cumulative"
    return "tablet"


def _channel(metadata):
    raw = metadata.get("channel") or metadata.get("filter_name") or ""
    normalized = "".join(char for char in unicodedata.normalize("NFKD", str(raw))
                         if not unicodedata.combining(char)).strip().lower()
    return {"red": "red", "piros": "red", "green": "green", "zold": "green",
            "blue": "blue", "kek": "blue"}.get(normalized)


def load_live_images(records, thumbnail_max_dim=0):
    """Decode every explicit capture, rejecting partial/missing input sets."""
    data = create_data()
    data["_inline_capture_input"] = True
    data["image_metadata"] = []
    if not isinstance(records, list) or not records:
        data["error"] = "E2180"
        return data
    images, paths, metadata = [], [], []
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            data["error"] = "E2184"
            return data
        path = record["path"]
        try:
            is_file = Path(path).is_file()
        except (OSError, ValueError):
            is_file = False
        if not is_file:
            data["error"] = "E2184"
            return data
        image = _decode_single_image(path, max_dim=thumbnail_max_dim)
        if image is None:
            data["error"] = "E2184"
            return data
        raw_metadata = record.get("metadata")
        row = deepcopy(raw_metadata) if isinstance(raw_metadata, dict) else {}
        if record.get("tablet_id") is not None:
            row["tablet_id"] = str(record["tablet_id"])
        elif isinstance(row.get("tray_position"), (str, int)):
            row["tablet_id"] = str(row["tray_position"])
        row["channel"] = _channel(row)
        row["source_path"] = path
        images.append(image)
        paths.append(path)
        metadata.append(row)
    data.update(images=images, paths=paths, count=len(images), image_metadata=metadata)
    data["meta"]["load"] = {"count": len(images), "source": "inline"}
    data["history"].append("load_image")
    return data


def live_rgb_groups(data):
    """Return (RGB source indices, result metadata) groups or set an error.

    Non-RGB filters remain available to other recipe nodes. They do not form
    their own RGB groups, but cannot fill missing RGB captures either.
    """
    images = data.get("images") or []
    metadata = data.get("image_metadata") or []
    if not images or len(metadata) != len(images):
        data["error"] = "E2181"
        return []
    groups = {}
    for index, row in enumerate(metadata):
        channel = _channel(row)
        if channel is None:
            continue
        tablet_id = row.get("tablet_id")
        wavelength = row.get("wavelength")
        if not isinstance(tablet_id, str) or not tablet_id or not isinstance(wavelength, str) or not wavelength:
            data["error"] = "E2181"
            return []
        key = (tablet_id, wavelength)
        group = groups.setdefault(key, {})
        if channel in group:
            data["error"] = "E2183"
            data["error_detail"] = f"{tablet_id}, {wavelength}: {channel}"
            return []
        group[channel] = index
    if not groups:
        data["error"] = "E2182"
        return []
    output = []
    paths = data.get("paths") or []
    for (tablet_id, wavelength), group in groups.items():
        missing = [channel for channel in ("red", "green", "blue") if channel not in group]
        if missing:
            data["error"] = "E2182"
            data["error_detail"] = f"{tablet_id}, {wavelength}: {', '.join(missing)}"
            return []
        indices = [group[channel] for channel in ("red", "green", "blue")]
        row = deepcopy(metadata[indices[0]])
        row["channel"] = "rgb"
        row["source_paths"] = [paths[index] for index in indices] if len(paths) == len(images) else []
        row["source_channels"] = ["red", "green", "blue"]
        row.pop("filter_name", None)
        row.pop("filter_position", None)
        row.pop("filter_wavelength", None)
        output.append((indices, row))
    return output


def live_focus_groups(data):
    """Keep focus brackets within one tablet, illumination and optical filter."""
    images = data.get("images") or []
    metadata = data.get("image_metadata") or []
    if not images or len(metadata) != len(images):
        data["error"] = "E2181"
        return []
    grouped = {}
    for index, row in enumerate(metadata):
        tablet_id, wavelength = row.get("tablet_id"), row.get("wavelength")
        if not isinstance(tablet_id, str) or not tablet_id or not isinstance(wavelength, str) or not wavelength:
            data["error"] = "E2181"
            return []
        channel = _channel(row) or row.get("channel")
        filter_key = (str(channel or ""), str(row.get("filter_position", "")),
                      str(row.get("filter_name", "")), str(row.get("filter_wavelength", "")))
        grouped.setdefault((tablet_id, wavelength, filter_key), []).append(index)
    output = []
    for indices in grouped.values():
        row = deepcopy(metadata[indices[0]])
        row["source_paths"] = [data["paths"][index] for index in indices]
        row["source_z"] = [metadata[index].get("z") for index in indices]
        output.append((indices, row))
    return output
