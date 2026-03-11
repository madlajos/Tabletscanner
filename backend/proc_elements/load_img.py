import cv2
import numpy as np
import os


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


def load_image(paths, debug=False):

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

    images = []
    valid_paths = []

    for path in expanded_paths:

        # Use np.fromfile + imdecode to support non-ASCII (Unicode) file paths
        try:
            buf = np.fromfile(path, dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        except Exception:
            img = None

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
