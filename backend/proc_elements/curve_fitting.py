import numpy as np


def fit_curve(data, x_name, y_name, model="linear", degree=2, debug=False):

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

    x_values = data["results"][x_name]
    stats = data["results"]["intensity_stats"]

    if len(x_values) != len(stats):
        data["error"] = "E2704"
        return data

    filtered_x = []
    filtered_y = []

    for x_val, stat in zip(x_values, stats):
        if stat is None:
            continue

        if y_name not in stat:
            data["error"] = "E2705"
            return data

        filtered_x.append(float(x_val))
        filtered_y.append(float(stat[y_name]))

    if len(filtered_x) < 2:
        data["error"] = "E2706"
        return data

    x = np.array(filtered_x, dtype=np.float64)
    y = np.array(filtered_y, dtype=np.float64)

    unique_x = np.unique(x)

    if model == "linear":
        if len(unique_x) < 2:
            data["error"] = "E2710"
            return data

        coeffs = np.polyfit(x, y, 1)
        fitted_y = np.polyval(coeffs, x)

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
        fitted_y = np.polyval(coeffs, x)

    else:
        data["error"] = "E2709"
        return data

    ss_res = np.sum((y - fitted_y) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    if ss_tot == 0:
        r2 = 1.0
    else:
        r2 = 1.0 - (ss_res / ss_tot)

    fit_result = {
        "x_name": x_name,
        "y_name": y_name,
        "model": model,
        "degree": degree if model == "poly" else 1,
        "coefficients": coeffs.tolist(),
        "x_values": x.tolist(),
        "y_values": y.tolist(),
        "fitted_y": fitted_y.tolist(),
        "x_min": float(np.min(x)),
        "x_max": float(np.max(x)),
        "y_min": float(np.min(y)),
        "y_max": float(np.max(y)),
        "sample_count": int(len(x)),
        "r2": float(r2)
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
