import copy
import numpy as np
import re


def _compile_equation(expr: str):
    """Compile an equation string into f(x). Supports 'y = ...' and '^' power syntax."""
    text = str(expr or "").strip()
    if not text:
        return None

    if "=" in text:
        left, right = text.split("=", 1)
        if left.strip().lower() == "y":
            text = right.strip()

    text = text.replace("^", "**")
    # Insert multiplication between number and x/parenthesis for common user format.
    text = re.sub(r"(\d)(\s*)(x|\()", r"\1*\3", text, flags=re.IGNORECASE)

    allowed = {
        "np": np,
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "exp": np.exp,
        "log": np.log,
        "sqrt": np.sqrt,
        "abs": np.abs,
        "pi": np.pi,
        "e": np.e,
    }

    def _f(x):
        return eval(text, {"__builtins__": {}}, {**allowed, "x": x})

    return _f


def _invert_by_scan(f, target_y: float, x_min: float, x_max: float) -> float | None:
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
        return None
    xs = np.linspace(x_min, x_max, 4096)
    try:
        ys = np.asarray(f(xs), dtype=np.float64)
    except Exception:
        return None

    if ys.shape != xs.shape:
        try:
            ys = np.broadcast_to(ys, xs.shape).astype(np.float64)
        except Exception:
            return None

    mask = np.isfinite(ys)
    if not np.any(mask):
        return None
    xs = xs[mask]
    ys = ys[mask]
    idx = int(np.argmin(np.abs(ys - target_y)))
    return float(xs[idx])


def predict_node(model_data, input_data, fit_index=0, equation="", y_name="mean", x_min=None, x_max=None, debug=False):

    result_data = copy.deepcopy(input_data)

    if model_data["error"] is not None:
        result_data["error"] = "E2801"
        return result_data

    if input_data["error"] is not None:
        return result_data

    if "results" not in result_data or "intensity_stats" not in result_data["results"]:
        result_data["error"] = "E2804"
        return result_data

    fit = None
    eq_callable = None
    eq_text = str(equation or "").strip()

    if eq_text:
        eq_callable = _compile_equation(eq_text)
        if eq_callable is None:
            result_data["error"] = "E2806"
            return result_data
    else:
        if "results" not in model_data or "curve_fits" not in model_data["results"]:
            result_data["error"] = "E2803"
            return result_data

        curve_fits = model_data["results"]["curve_fits"]

        if fit_index < 0 or fit_index >= len(curve_fits):
            result_data["error"] = "E2805"
            return result_data

        fit = curve_fits[fit_index]
        y_name = fit.get("y_key") or fit.get("y_name") or y_name

    if len(result_data["results"]["intensity_stats"]) == 0:
        result_data["error"] = "E2806"
        return result_data

    predictions = []

    if eq_callable is not None:
        try:
            x_min_f = float(x_min) if x_min is not None else 0.0
            x_max_f = float(x_max) if x_max is not None else 255.0
        except (ValueError, TypeError):
            result_data["error"] = "E2806"
            return result_data
    else:
        x_train = np.array(fit["x_values"], dtype=np.float64)
        y_fit = np.array(fit["fitted_y"], dtype=np.float64)

    for stat in result_data["results"]["intensity_stats"]:

        if stat is None:
            predictions.append(None)
            continue

        if y_name not in stat:
            result_data["error"] = "E2808"
            return result_data

        target_y = float(stat[y_name])

        if eq_callable is not None:
            predicted_x = _invert_by_scan(eq_callable, target_y, x_min_f, x_max_f)
            if predicted_x is None:
                result_data["error"] = "E2806"
                return result_data
            model_name = "equation"
            degree = None
            r2 = None
            x_name = "x"
        else:
            idx = np.argmin(np.abs(y_fit - target_y))
            predicted_x = float(x_train[idx])
            model_name = fit["model"]
            degree = fit["degree"]
            r2 = fit["r2"]
            x_name = fit["x_name"]

        predictions.append({
            "x_name": x_name,
            "y_name": y_name,
            "input_y": target_y,
            "predicted_x": predicted_x,
            "model": model_name,
            "degree": degree,
            "r2": r2,
            "equation": eq_text if eq_callable is not None else None,
        })

    if "results" not in result_data:
        result_data["results"] = {}

    result_data["results"]["predictions"] = predictions
    result_data["meta"]["prediction"] = {
        "fit_index": fit_index,
        "x_name": "x" if eq_callable is not None else fit["x_name"],
        "y_name": y_name,
        "equation": eq_text if eq_callable is not None else None,
    }
    result_data["history"].append("predict_node")

    if debug:
        print(predictions[0])

    return result_data
