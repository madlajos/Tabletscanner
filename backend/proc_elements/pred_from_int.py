import copy
import numpy as np


def predict_node(model_data, input_data, fit_index=0, debug=False):

    result_data = copy.deepcopy(input_data)

    if model_data["error"] is not None:
        result_data["error"] = "E2801"
        return result_data

    if input_data["error"] is not None:
        return result_data

    if "results" not in model_data or "curve_fits" not in model_data["results"]:
        result_data["error"] = "E2803"
        return result_data

    if "results" not in result_data or "intensity_stats" not in result_data["results"]:
        result_data["error"] = "E2804"
        return result_data

    curve_fits = model_data["results"]["curve_fits"]

    if fit_index < 0 or fit_index >= len(curve_fits):
        result_data["error"] = "E2805"
        return result_data

    fit = curve_fits[fit_index]
    y_name = fit["y_name"]

    if len(result_data["results"]["intensity_stats"]) == 0:
        result_data["error"] = "E2806"
        return result_data

    predictions = []

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

        idx = np.argmin(np.abs(y_fit - target_y))
        predicted_x = float(x_train[idx])

        predictions.append({
            "x_name": fit["x_name"],
            "y_name": y_name,
            "input_y": target_y,
            "predicted_x": predicted_x,
            "model": fit["model"],
            "degree": fit["degree"],
            "r2": fit["r2"]
        })

    if "results" not in result_data:
        result_data["results"] = {}

    result_data["results"]["predictions"] = predictions
    result_data["meta"]["prediction"] = {
        "fit_index": fit_index,
        "x_name": fit["x_name"],
        "y_name": y_name
    }
    result_data["history"].append("predict_node")

    if debug:
        print(predictions[0])

    return result_data
