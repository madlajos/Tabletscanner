import cv2


def resize_images(data, width=None, height=None, scale=None,
                  keep_aspect=True, interpolation="linear", debug=False):

    if data["error"] is not None:
        return data

    if data["images"] is None or data["count"] == 0:
        data["error"] = "E3901"
        return data

    interp_map = {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "area": cv2.INTER_AREA,
        "lanczos": cv2.INTER_LANCZOS4,
    }

    if interpolation not in interp_map:
        data["error"] = "E3902"
        return data

    interp_flag = interp_map[interpolation]

    use_scale = scale is not None and scale > 0
    use_wh = width is not None or height is not None

    if not use_scale and not use_wh:
        data["error"] = "E3903"
        return data

    if use_scale:
        if not isinstance(scale, (int, float)) or scale <= 0:
            data["error"] = "E3904"
            return data

    if use_wh and not use_scale:
        if width is not None and (not isinstance(width, int) or width < 1):
            data["error"] = "E3905"
            return data
        if height is not None and (not isinstance(height, int) or height < 1):
            data["error"] = "E3906"
            return data

    resized = []
    new_w, new_h = 0, 0

    for img in data["images"]:
        if img is None or img.size == 0:
            data["error"] = "E3907"
            return data

        h_orig, w_orig = img.shape[:2]

        if use_scale:
            new_w = int(round(w_orig * scale))
            new_h = int(round(h_orig * scale))
        else:
            if width is not None and height is not None:
                if keep_aspect:
                    ratio = min(width / w_orig, height / h_orig)
                    new_w = int(round(w_orig * ratio))
                    new_h = int(round(h_orig * ratio))
                else:
                    new_w = width
                    new_h = height
            elif width is not None:
                ratio = width / w_orig
                new_w = width
                new_h = int(round(h_orig * ratio))
            else:
                ratio = height / h_orig
                new_h = height
                new_w = int(round(w_orig * ratio))

        if new_w < 1 or new_h < 1:
            data["error"] = "E3908"
            return data

        out = cv2.resize(img, (new_w, new_h), interpolation=interp_flag)
        resized.append(out)

    data["images"] = resized
    data["count"] = len(resized)

    data["meta"]["resize"] = {
        "width": new_w,
        "height": new_h,
        "interpolation": interpolation,
        "scale": scale if use_scale else None,
        "keep_aspect": keep_aspect,
    }

    data["history"].append("resize_images")

    if debug:
        print(f"Resized {len(resized)} images to {new_w}x{new_h}")

    return data
