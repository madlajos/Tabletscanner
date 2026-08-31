import cv2
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor


def create_data():
    return {
        "images": None,
        "paths": [],
        "count": 0,
        "meta": {},
        "results": {},
        "history": [],
        "error": None
    }


def _decode_single_image(path, max_dim=0):
    """Read and convert a single image file. Returns the image or None.
    If max_dim > 0, the image is downscaled so its longest side <= max_dim."""
    try:
        buf = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    except Exception:
        return None

    if img is None:
        return None

    # Handle grayscale
    if len(img.shape) == 2:
        pass  # ok

    # Handle color
    elif len(img.shape) == 3:

    # BGR -> RGB
        if img.shape[2] == 3:
            img = img

    # BGRA -> RGB
        elif img.shape[2] == 4:
            img = img

    # Thumbnail resize – shrink immediately to save memory
    if max_dim > 0:
        h, w = img.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return img


def load_image(paths, debug=False, single_image_index=-1, thumbnail_max_dim=0):

    data = create_data()

    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]

    image_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
    expanded_paths = []

    for p in paths:

        p = os.path.abspath(p)

        if not os.path.exists(p):
            data["error"] = "E2002"
            return data

        if os.path.isdir(p):

            try:
                files = sorted(os.listdir(p))
            except Exception:
                data["error"] = "E2004"
                return data

            for file in files:
                if file.lower().endswith(image_extensions):
                    expanded_paths.append(os.path.join(p, file))

        else:

            if not p.lower().endswith(image_extensions):
                data["error"] = "E2003"
                return data

            expanded_paths.append(p)

    if len(expanded_paths) == 0:
        data["error"] = "E2001"
        return data

    # Fast path: only decode the single requested image, keep all paths
    if single_image_index >= 0:
        idx = min(single_image_index, len(expanded_paths) - 1)
        img = _decode_single_image(expanded_paths[idx], max_dim=thumbnail_max_dim)

        if img is None:
            data["error"] = "E2001"
            return data

        data["images"] = [img]
        data["paths"] = expanded_paths
        data["count"] = 1
        data["_original_count"] = len(expanded_paths)
        data["_single_image_loaded"] = True

        data["meta"]["load"] = {
            "count": len(expanded_paths)
        }

        data["history"].append("load_image")
        return data

    images = []
    valid_paths = []

    # Parallel image loading – cv2.imdecode releases the GIL,
    # so threads give a real I/O + decode speed-up.
    def _load(p):
        return _decode_single_image(p, max_dim=thumbnail_max_dim)

    with ThreadPoolExecutor(max_workers=min(8, len(expanded_paths))) as pool:
        results = list(pool.map(_load, expanded_paths))

    for path, img in zip(expanded_paths, results):
        if img is None:
            print(f"skip: {path}")
            continue
        images.append(img)
        valid_paths.append(path)

    if len(images) == 0:
        data["error"] = "E2001"
        return data

    data["images"] = images
    data["paths"] = valid_paths
    data["count"] = len(images)

    data["meta"]["load"] = {
        "count": len(images)
    }

    data["history"].append("load_image")

    if debug:
        print(f"Loaded: {data['count']} images")
        print(f"Shape: {data['images'][0].shape}")

    return data
