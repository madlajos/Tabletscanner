import numpy as np


def add_sequence_values(
    data,
    name="sequence_value",
    values=None,
    start=None,
    step=None,
    stop=None,
    debug=False
):

    if data["error"] is not None:
        return data

    if data["images"] is None or data["count"] == 0:
        data["error"] = "E2631"
        return data

    if not isinstance(name, str) or len(name.strip()) == 0:
        data["error"] = "E2632"
        return data

    generated_values = None
    mode = None

    if values is not None:
        if isinstance(values, (int, float)):
            generated_values = [float(values)] * data["count"]
        elif isinstance(values, (list, tuple, np.ndarray)):
            if len(values) != data["count"]:
                data["error"] = "E2633"
                return data
            try:
                generated_values = [float(v) for v in values]
            except Exception:
                data["error"] = "E2634"
                return data
        else:
            data["error"] = "E2635"
            return data

        mode = "explicit"

    elif start is not None and step is not None and stop is None:
        if not isinstance(start, (int, float)) or not isinstance(step, (int, float)):
            data["error"] = "E2636"
            return data

        generated_values = [float(start + i * step) for i in range(data["count"])]
        mode = "start_step"

    elif start is not None and stop is not None and step is None:
        if not isinstance(start, (int, float)) or not isinstance(stop, (int, float)):
            data["error"] = "E2637"
            return data

        if data["count"] == 1:
            generated_values = [float(start)]
        else:
            generated_values = np.linspace(float(start), float(stop), data["count"]).tolist()

        mode = "start_stop"

    else:
        data["error"] = "E2638"
        return data

    if "results" not in data:
        data["results"] = {}

    data["results"][name] = generated_values

    if "sequence_values" not in data["meta"]:
        data["meta"]["sequence_values"] = {}

    data["meta"]["sequence_values"][name] = {
        "mode": mode,
        "count": len(generated_values)
    }

    if mode == "explicit":
        data["meta"]["sequence_values"][name]["source"] = "values"

    elif mode == "start_step":
        data["meta"]["sequence_values"][name]["start"] = float(start)
        data["meta"]["sequence_values"][name]["step"] = float(step)

    elif mode == "start_stop":
        data["meta"]["sequence_values"][name]["start"] = float(start)
        data["meta"]["sequence_values"][name]["stop"] = float(stop)

    data["history"].append("add_sequence_values")

    if debug:
        print(f"{name}: {generated_values}")
        print(data["meta"]["sequence_values"][name])

    return data
