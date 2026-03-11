import numpy as np


def robust_stretch_gamma(
    data,
    percentile_low=2,
    percentile_high=98,
    gamma=1.0,
    debug=False
):

    if data["error"] is not None:
        return data

    if data["images"] is None or data["count"] == 0:
        data["error"] = "E3221"
        return data

    if percentile_low < 0 or percentile_high > 100 or percentile_low >= percentile_high:
        data["error"] = "E3222"
        return data

    if not isinstance(gamma, (int, float)) or gamma <= 0:
        data["error"] = "E3223"
        return data

    output_images = []

    for img in data["images"]:
        if img is None:
            data["error"] = "E3224"
            return data

        if len(img.shape) != 2:
            data["error"] = "E3225"
            return data

        I = img.astype(np.float32) / 255.0

        lo = np.percentile(I, percentile_low)
        hi = np.percentile(I, percentile_high)

        if hi <= lo:
            data["error"] = "E3226"
            return data

        I2 = np.clip((I - lo) / (hi - lo), 0.0, 1.0)
        I3 = np.clip(I2 ** float(gamma), 0.0, 1.0)

        out = (I3 * 255.0).astype(np.uint8)
        output_images.append(out)

    data["images"] = output_images
    data["count"] = len(output_images)
    data["meta"]["robust_stretch_gamma"] = {
        "percentile_low": float(percentile_low),
        "percentile_high": float(percentile_high),
        "gamma": float(gamma)
    }
    data["history"].append("robust_stretch_gamma")

    if debug:
        print(data["meta"]["robust_stretch_gamma"])

    return data
