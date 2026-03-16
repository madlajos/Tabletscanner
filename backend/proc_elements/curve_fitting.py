import numpy as np
import re
import os


def _safe_float(v):
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def _parse_sample_key_from_path(path):
    name = os.path.splitext(os.path.basename(str(path)))[0]
    m = re.match(r"^(.*?)(?:_([abAB]))$", name)
    if m:
        return m.group(1)
    return name


def _lookup_group_color(colors_map, x_raw, x_float):
    candidates = [str(x_raw), str(x_float)]
    try:
        xf = float(x_float)
        if xf.is_integer():
            candidates.append(str(int(xf)))
        candidates.append(str(round(xf, 6)).rstrip('0').rstrip('.'))
    except Exception:
        pass
    for c in candidates:
        if c in colors_map:
            return colors_map[c]
    return None


def _is_numeric(value):
    return isinstance(value, (int, float, np.integer, np.floating))


def _extract_y_records(results, y_name, expected_len):
    """
    Resolve Y values from result collections.

    Preferred order:
      1. intensity_stats (backward compatible)
      2. any list[dict] result with a numeric y_name field and matching length
    """
    stats = results.get("intensity_stats")
    if isinstance(stats, list) and len(stats) == expected_len:
        if all((s is None or (isinstance(s, dict) and y_name in s)) for s in stats):
            out = []
            for s in stats:
                if s is None:
                    out.append(None)
                    continue
                v = s.get(y_name)
                if v is None:
                    out.append(None)
                    continue
                try:
                    out.append(float(v))
                except Exception:
                    return None, None
            return out, "intensity_stats"

    for source_key, source_val in results.items():
        if source_key == "intensity_stats":
            continue
        if not isinstance(source_val, list) or len(source_val) != expected_len:
            continue
        first = next((item for item in source_val if item is not None), None)
        if not isinstance(first, dict) or y_name not in first:
            continue
        if not _is_numeric(first.get(y_name)):
            continue

        out = []
        valid = True
        for item in source_val:
            if item is None:
                out.append(None)
                continue
            if not isinstance(item, dict) or y_name not in item:
                valid = False
                break
            val = item.get(y_name)
            if not _is_numeric(val):
                valid = False
                break
            out.append(float(val))
        if valid:
            return out, source_key

    return None, None


def fit_curve(data, x_name, y_name, model="linear", degree=2, aggregate=False, agg_method="mean", merge_ab_pairs=False, y_display_name=None, debug=False):

    if data["error"] is not None:
        return data

    if "results" not in data:
        data["error"] = "E2701"
        return data

    if x_name not in data["results"]:
        data["error"] = "E2702"
        return data

    if agg_method not in ("mean", "median"):
        data["error"] = "E2712"
        return data

    x_values = data["results"][x_name]
    y_values_raw, y_source_key = _extract_y_records(data["results"], y_name, len(x_values))
    if y_values_raw is None:
        data["error"] = "E2703"
        return data

    if len(x_values) != len(y_values_raw):
        data["error"] = "E2704"
        return data

    omitted = data.get("_omitted_indices", set())
    paths = data.get("_original_paths", data.get("paths", []))
    colors_map = data["results"].get(f"{x_name}_group_colors", {})
    if not isinstance(colors_map, dict):
        colors_map = {}

    records = []
    all_x = []
    all_y = []
    all_colors = []

    for idx, (x_val, y_val) in enumerate(zip(x_values, y_values_raw)):
        if y_val is None:
            continue

        xf = _safe_float(x_val)
        if xf is None:
            data["error"] = "E2713"
            return data
        yf = float(y_val)
        group_color = _lookup_group_color(colors_map, x_val, xf)
        sample_path = paths[idx] if idx < len(paths) else str(idx)
        sample_key = _parse_sample_key_from_path(sample_path)

        all_x.append(xf)
        all_y.append(yf)
        all_colors.append(group_color)

        if idx not in omitted:
            records.append({
                "index": idx,
                "x": xf,
                "y": yf,
                "path": sample_path,
                "sample_key": sample_key,
                "color": group_color,
            })

    if len(records) < 2:
        data["error"] = "E2706"
        return data

    aggregation_info = None
    merged_records = records

    if merge_ab_pairs:
        by_sample = {}
        for rec in records:
            by_sample.setdefault(rec["sample_key"], []).append(rec)
        merged_records = []
        for sample_key, vals in by_sample.items():
            # Use first record's x (both sides should have the same assigned x)
            x_out = vals[0]["x"]
            y_out = float(np.mean([v["y"] for v in vals]))
            color_out = next((v.get("color") for v in vals if v.get("color")), None)
            merged_records.append({
                "x": x_out,
                "y": y_out,
                "sample_key": sample_key,
                "members": vals,
                "order": min(v.get("index", 0) for v in vals),
                "color": color_out,
            })
        merged_records.sort(key=lambda r: r.get("order", 0))

    result_records = merged_records
    samples_per_value = data.get("meta", {}).get("sequence_values", {}).get(x_name, {}).get("samples_per_value")
    try:
        samples_per_value = int(samples_per_value) if samples_per_value is not None else 1
    except (ValueError, TypeError):
        samples_per_value = 1
    if samples_per_value < 1:
        samples_per_value = 1

    if aggregate:
        chunk_size = samples_per_value
        chunked = []
        for i in range(0, len(merged_records), chunk_size):
            chunk = merged_records[i:i + chunk_size]
            if not chunk:
                continue
            x_out = float(np.mean([r["x"] for r in chunk]))
            y_vals = [r["y"] for r in chunk]
            if agg_method == "mean":
                y_out = float(np.mean(y_vals))
            else:
                y_out = float(np.median(y_vals))
            color_out = next((r.get("color") for r in chunk if r.get("color")), None)
            chunked.append({
                "x": x_out,
                "y": y_out,
                "members": chunk,
                "color": color_out,
                "member_count": len(chunk),
            })
        result_records = chunked

    if len(result_records) < 2:
        data["error"] = "E2706"
        return data

    x = np.array([r["x"] for r in result_records], dtype=np.float64)
    y = np.array([r["y"] for r in result_records], dtype=np.float64)
    point_colors = [r.get("color") for r in result_records]

    try:
        x_all = np.array([float(v) for v in all_x], dtype=np.float64)
    except (ValueError, TypeError):
        x_all = x
    y_all = np.array(all_y, dtype=np.float64)

    unique_x = np.unique(x)

    if model == "linear":
        if len(unique_x) < 2:
            data["error"] = "E2710"
            return data

        coeffs = np.polyfit(x, y, 1)

    elif model == "poly":
        if not isinstance(degree, int) or degree < 1:
            data["error"] = "E2707"
            return data

        if len(x) <= degree:
            data["error"] = "E2708"
            return data

        if len(unique_x) <= degree:
            data["error"] = "E2711"
            return data

        coeffs = np.polyfit(x, y, degree)

    else:
        data["error"] = "E2709"
        return data

    # R² computed on filtered/aggregated data only
    fitted_y_filtered = np.polyval(coeffs, x)
    ss_res = np.sum((y - fitted_y_filtered) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    r2 = 1.0 if ss_tot == 0 else 1.0 - (ss_res / ss_tot)

    result_x = x.tolist()
    result_y = y.tolist()
    result_fitted = fitted_y_filtered.tolist()

    aggregation_info = {
        "enabled": bool(aggregate or merge_ab_pairs),
        "aggregate": bool(aggregate),
        "merge_ab_pairs": bool(merge_ab_pairs),
        "method": agg_method,
        "samples_per_value": int(samples_per_value),
        "result_count": int(len(result_records)),
        "input_count": int(len(records)),
    }

    y_axis_name = str(y_display_name).strip() if isinstance(y_display_name, str) else ""
    if not y_axis_name:
        y_axis_name = str(y_name)

    fit_result = {
        "x_name": x_name,
        "y_name": y_axis_name,
        "y_key": y_name,
        "y_source": y_source_key,
        "model": model,
        "degree": degree if model == "poly" else 1,
        "coefficients": coeffs.tolist(),
        "x_values": result_x,
        "y_values": result_y,
        "fitted_y": result_fitted,
        "x_min": float(np.min(x)),
        "x_max": float(np.max(x)),
        "y_min": float(np.min(y)),
        "y_max": float(np.max(y)),
        "sample_count": int(len(x)),
        "r2": float(r2),
        "aggregation": aggregation_info,
        "point_colors": point_colors,
    }

    if "curve_fits" not in data["results"]:
        data["results"]["curve_fits"] = []

    data["results"]["curve_fits"].append(fit_result)

    data["meta"]["last_curve_fit"] = {
        "x_name": x_name,
        "y_name": y_name,
        "model": model,
        "degree": degree if model == "poly" else 1
    }

    data["history"].append("fit_curve")

    if debug:
        print("Curve fit result:")
        print(fit_result)

    return data
