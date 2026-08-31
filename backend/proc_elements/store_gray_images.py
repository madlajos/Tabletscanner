"""store_gray_images.py

Saves the current data["images"] list as data["gray_images"] so that a
subsequent load_image step can replace data["images"] with RGB images while
rgb_gray_map_node still has access to the grayscale counterparts.

Typical pipeline order:
    load_image  (grayscale)
    store_as_gray_images       ← this node
    load_image  (RGB)
    rgb_gray_map               ← consumes both data["images"] and data["gray_images"]
"""

import numpy as np


def store_as_gray_images(data, convert_to_gray=False, **_):
    """Copy current data['images'] into data['gray_images'].

    Parameters
    ----------
    convert_to_gray : bool
        If True, convert each image to single-channel float [0,1] before
        storing.  Useful if the first load brought in a colour image but
        you only need its luminance for the measurement step.
    """
    if data.get("error") is not None:
        return data

    images = data.get("images")
    if images is None:
        data["error"] = "E3001"
        return data

    if not convert_to_gray:
        data["gray_images"] = list(images)
    else:
        gray = []
        for img in images:
            if img is None:
                gray.append(None)
                continue
            arr = np.asarray(img, dtype=np.float64)
            if arr.ndim == 3:
                arr = 0.2989 * arr[..., 0] + 0.5870 * arr[..., 1] + 0.1140 * arr[..., 2]
            if arr.size > 0 and np.nanmax(arr) > 1.0:
                arr = arr / 255.0
            gray.append(np.clip(arr, 0.0, 1.0).astype(np.float32))
        data["gray_images"] = gray

    data.setdefault("history", []).append("store_as_gray_images")
    return data
