import numpy as np


_SIZE_METRICS = {
    "area_px": "Terület (px²)",
    "equivalent_diameter_px": "Egyenértékű átmérő (px)",
    "perimeter_px": "Kerület (px)",
    "bbox_w_px": "Befoglaló téglalap szélessége (px)",
    "bbox_h_px": "Befoglaló téglalap magassága (px)",
}


def _weighted_quantile(values, weights, quantile):
    if values.size == 0:
        return None
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    total = float(sorted_weights.sum())
    if total <= 0:
        return None
    cumulative = np.cumsum(sorted_weights) / total
    return float(np.interp(float(quantile), cumulative, sorted_values))


def _smooth_histogram(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size < 3:
        return values
    padded = np.pad(values, (1, 1), mode="edge")
    smoothed = np.convolve(padded, np.asarray([0.25, 0.5, 0.25]), mode="valid")
    total = float(smoothed.sum())
    return smoothed * (float(values.sum()) / total) if total > 0 else smoothed


def _metric_scale(size_metric, resize_info):
    scale_x = float(resize_info.get("cumulative_scale_x", 1.0) or 1.0)
    scale_y = float(resize_info.get("cumulative_scale_y", 1.0) or 1.0)
    if size_metric == "area_px":
        return max(scale_x * scale_y, 1e-12)
    if size_metric == "bbox_w_px":
        return max(scale_x, 1e-12)
    if size_metric == "bbox_h_px":
        return max(scale_y, 1e-12)
    return max(np.sqrt(scale_x * scale_y), 1e-12)


def _unit_factor(size_metric, output_unit, pixels_per_mm):
    if output_unit == "px":
        return 1.0, "px²" if size_metric == "area_px" else "px"
    linear = (1.0 if output_unit == "mm" else 1000.0) / pixels_per_mm
    factor = linear * linear if size_metric == "area_px" else linear
    unit = output_unit + ("²" if size_metric == "area_px" else "")
    return factor, unit


def _distribution_group(
    particles, size_metric, bin_count, resize_info, output_unit, pixels_per_mm,
    smoothing=False, shared_edges=None, label="", calculate_dv=True,
):
    valid = [
        particle for particle in particles
        if isinstance(particle.get(size_metric), (int, float))
        and np.isfinite(float(particle[size_metric]))
        and float(particle[size_metric]) >= 0
    ]
    if not valid:
        return None

    resize_scale = _metric_scale(size_metric, resize_info)
    unit_factor, unit = _unit_factor(size_metric, output_unit, pixels_per_mm)
    sizes = np.asarray(
        [float(particle[size_metric]) / resize_scale * unit_factor for particle in valid],
        dtype=np.float64,
    )
    diameter_scale = _metric_scale("equivalent_diameter_px", resize_info)
    linear_unit_factor, _ = _unit_factor("equivalent_diameter_px", output_unit, pixels_per_mm)
    diameters = np.asarray([
        max(0.0, float(particle.get("equivalent_diameter_px") or 0.0)) / diameter_scale * linear_unit_factor
        for particle in valid
    ], dtype=np.float64)
    volume_weights = np.power(diameters, 3.0)
    if not np.any(volume_weights > 0):
        volume_weights = np.ones_like(sizes)

    if shared_edges is None:
        lo, hi = float(sizes.min()), float(sizes.max())
        if hi <= lo:
            padding = max(abs(lo) * 0.01, 0.5)
            lo, hi = lo - padding, hi + padding
        edges = np.linspace(lo, hi, int(bin_count) + 1)
    else:
        edges = np.asarray(shared_edges, dtype=np.float64)

    number_counts, _ = np.histogram(sizes, bins=edges)
    volume_counts, _ = np.histogram(sizes, bins=edges, weights=volume_weights)
    number_total = max(float(number_counts.sum()), 1.0)
    volume_total = max(float(volume_counts.sum()), 1.0)
    number_percent = number_counts.astype(np.float64) * 100.0 / number_total
    volume_percent = volume_counts.astype(np.float64) * 100.0 / volume_total
    if smoothing:
        number_percent = _smooth_histogram(number_percent)
        volume_percent = _smooth_histogram(volume_percent)

    return {
        "label": label,
        "particle_count": int(sizes.size),
        "size_metric": size_metric,
        "size_label": _SIZE_METRICS[size_metric],
        "unit": unit,
        "particle_values": [
            {"particle_id": particle.get("particle_id"), "label": particle.get("label"),
             "value": float(size)}
            for particle, size in zip(valid, sizes)
        ] if not calculate_dv else [],
        "bin_edges": edges.tolist(),
        "bin_centers": ((edges[:-1] + edges[1:]) * 0.5).tolist(),
        "number_percent": number_percent.tolist(),
        "volume_percent": volume_percent.tolist(),
        "min": float(sizes.min()),
        "max": float(sizes.max()),
        "mean": float(sizes.mean()),
        "median": float(np.median(sizes)),
        "std": float(sizes.std()),
        "dn10": _weighted_quantile(sizes, np.ones_like(sizes), 0.10),
        "dn50": _weighted_quantile(sizes, np.ones_like(sizes), 0.50),
        "dn90": _weighted_quantile(sizes, np.ones_like(sizes), 0.90),
        "dv10": _weighted_quantile(sizes, volume_weights, 0.10) if calculate_dv else None,
        "dv50": _weighted_quantile(sizes, volume_weights, 0.50) if calculate_dv else None,
        "dv90": _weighted_quantile(sizes, volume_weights, 0.90) if calculate_dv else None,
    }


def characterize_particles(
    data,
    use_filtered=True,
    selected_columns=None,
    include_excluded=False,
    size_metric="equivalent_diameter_px",
    distribution_mode="pooled",
    bin_count=30,
    output_unit="px",
    pixels_per_mm=0.0,
    smooth_distribution=False,
    debug=False,
    calibration_pixels=0.0,
    calibration_length_um=0.0,
):
    """
    A detect_particles eredményéből táblázat épít.

    use_filtered:
        True  -> csak a szűrt szemcséket írja ki
        False -> az összeset

    selected_columns:
        pl. ["particle_id", "image_index", "label", "area_px", "circularity", "intensity_mean"]
        ha None, akkor az összes mezőt visszaadja
    """

    if data["error"] is not None:
        return data

    if "meta" not in data or "particles" not in data["meta"]:
        data["error"] = "E3100"
        return data

    if "results" not in data or data["results"] is None:
        data["results"] = {}

    if "history" not in data or data["history"] is None:
        data["history"] = []

    if selected_columns is not None and not isinstance(selected_columns, (list, tuple)):
        data["error"] = "E3108"
        return data

    if size_metric not in _SIZE_METRICS:
        data["error"] = "E3109"
        return data
    if distribution_mode not in ("pooled", "overlay", "per_image"):
        data["error"] = "E3110"
        return data
    if output_unit not in ("px", "mm", "um"):
        data["error"] = "E3111"
        return data
    try:
        pixels_per_mm = float(pixels_per_mm)
        calibration_pixels = float(calibration_pixels)
        calibration_length_um = float(calibration_length_um)
        if output_unit != "px" and (calibration_pixels != 0 or calibration_length_um != 0):
            if not all(np.isfinite(v) and v > 0 for v in (calibration_pixels, calibration_length_um)):
                raise ValueError("Invalid reference length")
            pixels_per_mm = calibration_pixels / calibration_length_um * 1000.0
    except (TypeError, ValueError):
        pixels_per_mm = 0.0
    if output_unit != "px" and (not np.isfinite(pixels_per_mm) or pixels_per_mm <= 0):
        data["error"] = "E3112"
        return data
    bin_count = max(5, min(100, int(bin_count)))
    resize_info = data.get("meta", {}).get("resize", {})

    particles_source = data["meta"]["particles_filtered"] if use_filtered and "particles_filtered" in data["meta"] else data["meta"]["particles"]

    table = []

    for image_particles in particles_source:
        for particle in image_particles:
            excluded = bool(particle.get("excluded", False))
            if excluded and not include_excluded:
                continue

            if selected_columns is None:
                row = dict(particle)
            else:
                row = {col: particle.get(col, None) for col in selected_columns}

            table.append(row)

    data["results"]["particle_table"] = table
    included_by_image = []
    for image_particles in particles_source:
        included_by_image.append([
            particle for particle in image_particles
            if include_excluded or not bool(particle.get("excluded", False))
        ])

    flattened = [particle for row in included_by_image for particle in row]
    included_ids = {particle.get("particle_id") for particle in flattened}
    if distribution_mode == "pooled":
        distribution_groups = [
            group for group in [
                _distribution_group(
                    flattened, size_metric, bin_count, resize_info, output_unit, pixels_per_mm,
                    smoothing=smooth_distribution, label="Összes kép",
                )
            ] if group is not None
        ]
    else:
        valid_sizes = [
            float(particle[size_metric]) / _metric_scale(size_metric, resize_info)
            * _unit_factor(size_metric, output_unit, pixels_per_mm)[0]
            for particle in flattened
            if isinstance(particle.get(size_metric), (int, float))
            and np.isfinite(float(particle[size_metric]))
            and float(particle[size_metric]) >= 0
        ]
        shared_edges = None
        if valid_sizes:
            lo, hi = min(valid_sizes), max(valid_sizes)
            if hi <= lo:
                padding = max(abs(lo) * 0.01, 0.5)
                lo, hi = lo - padding, hi + padding
            shared_edges = np.linspace(lo, hi, bin_count + 1)
        distribution_groups = []
        for image_index, image_particles in enumerate(included_by_image):
            group = _distribution_group(
                image_particles,
                size_metric,
                bin_count,
                resize_info,
                output_unit,
                pixels_per_mm,
                smoothing=smooth_distribution,
                shared_edges=shared_edges,
                calculate_dv=distribution_mode != "per_image",
                label=f"{image_index + 1}. kép",
            )
            if group is not None:
                group["image_index"] = image_index
                distribution_groups.append(group)

    data["results"]["particle_size_distribution"] = {
        "particles": [
            {
                "particle_id": particle["particle_id"],
                "label": particle.get("label"),
                "image_index": particle.get("image_index", image_index),
                "polygon": particle.get("polygon", []),
                "excluded": bool(particle.get("excluded", False)),
                "value": float(particle[size_metric]) / _metric_scale(size_metric, resize_info)
                * _unit_factor(size_metric, output_unit, pixels_per_mm)[0],
            }
            for image_index, row in enumerate(data["meta"]["particles"])
            for particle in row
            if (not use_filtered or particle.get("passed_filters", particle.get("particle_id") in included_ids)
                or particle.get("excluded", False))
            and isinstance(particle.get(size_metric), (int, float))
            and np.isfinite(float(particle[size_metric]))
            and float(particle[size_metric]) >= 0
        ] if distribution_mode == "per_image" else [],
        "unit": _unit_factor(size_metric, output_unit, pixels_per_mm)[1],
        "mode": distribution_mode,
        "size_metric": size_metric,
        "size_label": _SIZE_METRICS[size_metric],
        "output_unit": output_unit,
        "smoothing": bool(smooth_distribution),
        "resize_correction": {
            "applied": bool(resize_info),
            "scale_x": float(resize_info.get("cumulative_scale_x", 1.0) or 1.0),
            "scale_y": float(resize_info.get("cumulative_scale_y", 1.0) or 1.0),
            "original_width": resize_info.get("original_width"),
            "original_height": resize_info.get("original_height"),
        },
        "groups": distribution_groups,
    }
    data["meta"]["particle_table_config"] = {
        "use_filtered": use_filtered,
        "selected_columns": list(selected_columns) if selected_columns is not None else None,
        "include_excluded": include_excluded,
        "size_metric": size_metric,
        "distribution_mode": distribution_mode,
        "bin_count": bin_count,
        "output_unit": output_unit,
        "pixels_per_mm": pixels_per_mm,
        "calibration_pixels": calibration_pixels,
        "calibration_length_um": calibration_length_um,
        "smooth_distribution": bool(smooth_distribution),
    }

    data["history"].append("characterize_particles")

    if debug:
        print("Particle table complete")
        print(f"Rows: {len(table)}")
        if len(table) > 0:
            print(table[0])

    return data
