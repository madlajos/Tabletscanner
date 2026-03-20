import numpy as np
import re


def _lookup_group_color(colors_map, x_raw):
    candidates = [str(x_raw)]
    try:
        xf = float(x_raw)
        candidates.append(str(xf))
        if xf.is_integer():
            candidates.append(str(int(xf)))
        candidates.append(str(round(xf, 6)).rstrip("0").rstrip("."))
    except Exception:
        pass
    for c in candidates:
        if c in colors_map:
            return colors_map[c]
    return None


def fit_curve(
    data,
    x_name,
    y_name,
    model="linear",              # "linear", "poly", "log", "exp"
    degree=2,
    aggregate=False,
    agg_method="mean",
    merge_ab_pairs=False,

    split_enabled=False,
    validation_ratio=0.3,
    split_method="random",       # "random" vagy "ordered"
    random_seed=42,

    manual_split=False,
    calibration_indices=None,
    validation_indices=None,

    debug=False
):
    """
    Görbeillesztés kalibráció / validáció támogatással.

    Modellek:
        - linear: y = a*x + b
        - poly:   y = p(x)
        - log:    y = a*ln(x) + b
        - exp:    y = a*exp(b*x)

    Split módok:
        1) split_enabled=False, manual_split=False
           -> minden adat kalibráció

        2) split_enabled=True, manual_split=False
           -> automatikus split (random / ordered)

        3) manual_split=True
           -> calibration_indices / validation_indices alapján

    MATLAB-szerű metrikák:
        Kalibráció:
            - SSE
            - DFE
            - RMSE = sqrt(SSE / DFE)
            - R2
            - AdjR2

        Validáció:
            - SSE
            - RMSE = sqrt(SSE / n_valid)
    """

    if data["error"] is not None:
        return data

    if "results" not in data:
        data["error"] = "E2701"
        return data

    if x_name not in data["results"]:
        data["error"] = "E2702"
        return data

    if "intensity_stats" not in data["results"]:
        data["error"] = "E2703"
        return data

    if agg_method not in ["mean", "median"]:
        data["error"] = "E2712"
        return data

    if model not in ["linear", "poly", "log", "exp"]:
        data["error"] = "E2709"
        return data

    if split_method not in ["random", "ordered"]:
        data["error"] = "E2714"
        return data

    if isinstance(validation_ratio, (int, float)) and validation_ratio >= 1:
        # Backward/forward compatibility: allow both ratio (0..1) and percent (0..100).
        validation_ratio = float(validation_ratio) / 100.0

    if not isinstance(validation_ratio, (int, float)) or validation_ratio < 0 or validation_ratio >= 1:
        data["error"] = "E2715"
        return data

    if manual_split:
        if calibration_indices is None or validation_indices is None:
            data["error"] = "E2720"
            return data

        if not isinstance(calibration_indices, (list, tuple, np.ndarray)):
            data["error"] = "E2721"
            return data

        if not isinstance(validation_indices, (list, tuple, np.ndarray)):
            data["error"] = "E2722"
            return data

    x_values = data["results"][x_name]
    stats = data["results"]["intensity_stats"]
    colors_map = data["results"].get(f"{x_name}_group_colors", {})
    if not isinstance(colors_map, dict):
        colors_map = {}

    if len(x_values) != len(stats):
        data["error"] = "E2704"
        return data

    filtered_x = []
    filtered_y = []
    filtered_colors = []
    original_indices = []

    for idx, (x_val, stat) in enumerate(zip(x_values, stats)):
        if stat is None:
            continue

        if y_name not in stat:
            data["error"] = "E2705"
            return data

        current_x = x_val

        if merge_ab_pairs:
            x_str = str(current_x)
            m = re.match(r"^(.*?)(?:_([abAB]))$", x_str)
            if m:
                current_x = m.group(1)

        filtered_x.append(current_x)
        filtered_y.append(float(stat[y_name]))
        filtered_colors.append(_lookup_group_color(colors_map, current_x))
        original_indices.append(idx)

    if len(filtered_x) < 2:
        data["error"] = "E2706"
        return data

    aggregation_info = None

    if aggregate or merge_ab_pairs:
        grouped = {}

        for i, (x_val, y_val) in enumerate(zip(filtered_x, filtered_y)):
            key = str(x_val)
            if key not in grouped:
                grouped[key] = {
                    "y": [],
                    "source_indices": [],
                    "colors": []
                }
            grouped[key]["y"].append(y_val)
            grouped[key]["source_indices"].append(original_indices[i])
            grouped[key]["colors"].append(filtered_colors[i])

        agg_x = []
        agg_y = []
        agg_std = []
        agg_count = []
        agg_source_indices = []
        agg_colors = []

        def _to_number_if_possible(v):
            try:
                return float(v)
            except Exception:
                return v

        for key, item in grouped.items():
            vals = item["y"]

            if agg_method == "mean":
                y_out = float(np.mean(vals))
            else:
                y_out = float(np.median(vals))

            agg_x.append(_to_number_if_possible(key))
            agg_y.append(y_out)
            agg_std.append(float(np.std(vals)))
            agg_count.append(int(len(vals)))
            agg_source_indices.append(item["source_indices"])
            agg_colors.append(next((c for c in item["colors"] if c), None))

        filtered_x = agg_x
        filtered_y = agg_y
        filtered_colors = agg_colors
        original_indices = list(range(len(filtered_x)))

        aggregation_info = {
            "enabled": True,
            "method": agg_method,
            "merge_ab_pairs": bool(merge_ab_pairs),
            "x_unique": agg_x,
            "y_aggregated": agg_y,
            "y_std": agg_std,
            "sample_count_per_x": agg_count,
            "source_indices_per_x": agg_source_indices
        }

    try:
        x = np.array([float(v) for v in filtered_x], dtype=np.float64)
    except Exception:
        data["error"] = "E2713"
        return data

    y = np.array(filtered_y, dtype=np.float64)
    point_colors = filtered_colors

    if len(x) < 2:
        data["error"] = "E2706"
        return data

    def _get_param_count(model_name, poly_degree):
        if model_name == "linear":
            return 2
        elif model_name == "poly":
            return poly_degree + 1
        elif model_name == "log":
            return 2
        elif model_name == "exp":
            return 2
        return 2

    def _compute_calibration_metrics_matlab(y_true, y_pred, p):
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)

        n = len(y_true)
        residuals = y_true - y_pred
        sse = float(np.sum(residuals ** 2))

        dfe = int(n - p) if n >= p else 0

        if dfe > 0:
            rmse = float(np.sqrt(sse / dfe))
        else:
            rmse = None

        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        if ss_tot == 0:
            r2 = 1.0
        else:
            r2 = float(1.0 - (sse / ss_tot))

        if dfe > 0:
            adj_r2 = float(1.0 - (1.0 - r2) * (n - 1) / dfe)
        else:
            adj_r2 = None

        return {
            "sse": sse,
            "dfe": dfe,
            "rmse": rmse,
            "r2": r2,
            "adj_r2": adj_r2,
            "sample_count": int(n),
            "param_count": int(p)
        }

    def _compute_validation_metrics_matlab(y_true, y_pred):
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)

        n = len(y_true)
        residuals = y_true - y_pred
        sse = float(np.sum(residuals ** 2))

        if n > 0:
            rmse = float(np.sqrt(sse / n))
        else:
            rmse = None

        return {
            "sse": sse,
            "rmse": rmse,
            "sample_count": int(n)
        }

    def _fit_model(x_fit, y_fit):
        unique_x = np.unique(x_fit)

        if model == "linear":
            if len(unique_x) < 2:
                raise ValueError("E2710")
            coeffs = np.polyfit(x_fit, y_fit, 1)
            return {
                "model": "linear",
                "coefficients": coeffs.tolist()
            }

        elif model == "poly":
            if not isinstance(degree, int) or degree < 1:
                raise ValueError("E2707")
            if len(x_fit) <= degree:
                raise ValueError("E2708")
            if len(unique_x) <= degree:
                raise ValueError("E2711")

            coeffs = np.polyfit(x_fit, y_fit, degree)
            return {
                "model": "poly",
                "degree": degree,
                "coefficients": coeffs.tolist()
            }

        elif model == "log":
            if np.any(x_fit <= 0):
                raise ValueError("E2716")

            logx = np.log(x_fit)
            coeffs = np.polyfit(logx, y_fit, 1)
            return {
                "model": "log",
                "coefficients": coeffs.tolist()
            }

        elif model == "exp":
            if np.any(y_fit <= 0):
                raise ValueError("E2717")

            logy = np.log(y_fit)
            coeffs_lin = np.polyfit(x_fit, logy, 1)
            b = float(coeffs_lin[0])
            ln_a = float(coeffs_lin[1])
            a = float(np.exp(ln_a))

            return {
                "model": "exp",
                "coefficients": [a, b]
            }

        raise ValueError("E2709")

    def _predict_model(model_info, x_pred):
        m = model_info["model"]

        if m == "linear":
            coeffs = np.array(model_info["coefficients"], dtype=np.float64)
            return np.polyval(coeffs, x_pred)

        elif m == "poly":
            coeffs = np.array(model_info["coefficients"], dtype=np.float64)
            return np.polyval(coeffs, x_pred)

        elif m == "log":
            if np.any(x_pred <= 0):
                raise ValueError("E2718")
            coeffs = np.array(model_info["coefficients"], dtype=np.float64)
            return coeffs[0] * np.log(x_pred) + coeffs[1]

        elif m == "exp":
            a, b = model_info["coefficients"]
            return a * np.exp(b * x_pred)

        raise ValueError("E2709")

    n = len(x)
    indices = np.arange(n)

    if manual_split:
        try:
            cal_idx = np.array(calibration_indices, dtype=int)
            val_idx = np.array(validation_indices, dtype=int)
        except Exception:
            data["error"] = "E2723"
            return data

        if len(cal_idx) < 2:
            data["error"] = "E2724"
            return data

        if len(val_idx) < 1:
            data["error"] = "E2725"
            return data

        if np.any(cal_idx < 0) or np.any(cal_idx >= n):
            data["error"] = "E2726"
            return data

        if np.any(val_idx < 0) or np.any(val_idx >= n):
            data["error"] = "E2727"
            return data

        if len(np.unique(cal_idx)) != len(cal_idx):
            data["error"] = "E2728"
            return data

        if len(np.unique(val_idx)) != len(val_idx):
            data["error"] = "E2729"
            return data

        overlap = np.intersect1d(cal_idx, val_idx)
        if len(overlap) > 0:
            data["error"] = "E2730"
            return data

    elif split_enabled:
        val_count = int(np.ceil(n * validation_ratio))
        cal_count = n - val_count

        if cal_count < 2 or val_count < 1:
            data["error"] = "E2719"
            return data

        if split_method == "random":
            rng = np.random.default_rng(random_seed)
            shuffled = indices.copy()
            rng.shuffle(shuffled)
            cal_idx = shuffled[:cal_count]
            val_idx = shuffled[cal_count:]
        else:
            cal_idx = indices[:cal_count]
            val_idx = indices[cal_count:]

    else:
        cal_idx = indices
        val_idx = np.array([], dtype=int)

    x_cal = x[cal_idx]
    y_cal = y[cal_idx]

    x_val = x[val_idx] if len(val_idx) > 0 else np.array([], dtype=np.float64)
    y_val = y[val_idx] if len(val_idx) > 0 else np.array([], dtype=np.float64)

    try:
        model_info = _fit_model(x_cal, y_cal)
        y_cal_pred = _predict_model(model_info, x_cal)
        y_all_pred = _predict_model(model_info, x)

        if len(x_val) > 0:
            y_val_pred = _predict_model(model_info, x_val)
        else:
            y_val_pred = np.array([], dtype=np.float64)

    except ValueError as e:
        data["error"] = str(e)
        return data

    p = _get_param_count(model_info["model"], degree)

    calibration_metrics = _compute_calibration_metrics_matlab(
        y_cal,
        y_cal_pred,
        p
    )

    if len(x_val) > 0:
        validation_metrics = _compute_validation_metrics_matlab(
            y_val,
            y_val_pred
        )
    else:
        validation_metrics = None

    fit_result = {
        "x_name": x_name,
        "y_name": y_name,
        "model": model_info["model"],
        "degree": degree if model == "poly" else None,
        "coefficients": model_info["coefficients"],

        "x_values": x.tolist(),
        "y_values": y.tolist(),
        "fitted_y": y_all_pred.tolist(),
        "r2": calibration_metrics["r2"],
        "point_colors": point_colors,
        "calibration_metrics": calibration_metrics,
        "validation_metrics": validation_metrics,

        "calibration": {
            "indices": cal_idx.tolist(),
            "x_values": x_cal.tolist(),
            "y_values": y_cal.tolist(),
            "fitted_y": y_cal_pred.tolist(),
            "metrics": calibration_metrics
        },

        "validation": None if len(x_val) == 0 else {
            "indices": val_idx.tolist(),
            "x_values": x_val.tolist(),
            "y_values": y_val.tolist(),
            "predicted_y": y_val_pred.tolist(),
            "metrics": validation_metrics
        },

        "x_min": float(np.min(x)),
        "x_max": float(np.max(x)),
        "y_min": float(np.min(y)),
        "y_max": float(np.max(y)),
        "sample_count": int(len(x)),

        "aggregation": aggregation_info,
        "split": {
            "enabled": bool(split_enabled),
            "manual_split": bool(manual_split),
            "validation_ratio": float(validation_ratio),
            "split_method": split_method if not manual_split else "manual",
            "random_seed": random_seed if (split_method == "random" and not manual_split) else None
        }
    }

    if "curve_fits" not in data["results"]:
        data["results"]["curve_fits"] = []

    data["results"]["curve_fits"].append(fit_result)

    if "meta" not in data or data["meta"] is None:
        data["meta"] = {}

    if "history" not in data or data["history"] is None:
        data["history"] = []

    data["meta"]["last_curve_fit"] = {
        "x_name": x_name,
        "y_name": y_name,
        "model": model_info["model"],
        "degree": degree if model == "poly" else None,
        "aggregate": bool(aggregate),
        "agg_method": agg_method,
        "merge_ab_pairs": bool(merge_ab_pairs),
        "split_enabled": bool(split_enabled),
        "manual_split": bool(manual_split),
        "validation_ratio": float(validation_ratio),
        "split_method": split_method if not manual_split else "manual"
    }

    data["history"].append("fit_curve")

    if debug:
        print("Curve fit result:")
        print(fit_result)

    return data
