import cv2


def resize_to_reference(data, reference_images, scale_percent=100.0, interpolation="linear",
                        show_image=True, show_reference=True, reference_opacity=0.5):
    """Resize each input image to the corresponding reference image canvas."""
    if not isinstance(data, dict) or data.get("error"):
        return data
    images = data.get("images")
    if not isinstance(images, list) or not images:
        data["error"] = "E3911"
        return data
    if not isinstance(reference_images, list) or not reference_images:
        data["error"] = "E3912"
        return data

    interp_map = {
        "nearest": cv2.INTER_NEAREST, "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC, "area": cv2.INTER_AREA,
        "lanczos": cv2.INTER_LANCZOS4,
    }
    if interpolation not in interp_map:
        data["error"] = "E3913"
        return data
    try:
        scale = float(scale_percent) / 100.0
    except (TypeError, ValueError):
        scale = 0.0
    if scale <= 0:
        data["error"] = "E3915"
        return data

    resized, sizes, previews = [], [], []
    try:
        for index, image in enumerate(images):
            reference = reference_images[min(index, len(reference_images) - 1)]
            if image is None or image.size == 0 or reference is None or reference.size == 0:
                data["error"] = "E3914"
                return data
            source_height, source_width = image.shape[:2]
            target_height, target_width = reference.shape[:2]
            output = cv2.resize(image, (target_width, target_height), interpolation=interp_map[interpolation])
            if scale != 1.0:
                centre = ((target_width - 1) / 2.0, (target_height - 1) / 2.0)
                matrix = cv2.getRotationMatrix2D(centre, 0.0, scale)
                output = cv2.warpAffine(
                    output, matrix, (target_width, target_height),
                    flags=interp_map[interpolation], borderMode=cv2.BORDER_CONSTANT, borderValue=0,
                )
            resized.append(output)
            preview_reference = reference
            if output.ndim == 2 and reference.ndim == 3:
                preview_reference = cv2.cvtColor(reference[:, :, :3], cv2.COLOR_BGR2GRAY)
            elif output.ndim == 3 and reference.ndim == 2:
                preview_reference = cv2.cvtColor(reference, cv2.COLOR_GRAY2BGR)
            elif output.ndim == 3 and reference.ndim == 3 and output.shape[2] != reference.shape[2]:
                preview_reference = reference[:, :, :output.shape[2]]

            if show_image and show_reference:
                opacity = max(0.0, min(1.0, float(reference_opacity)))
                previews.append(cv2.addWeighted(output, 1.0 - opacity, preview_reference, opacity, 0.0))
            elif show_reference:
                previews.append(preview_reference.copy())
            elif show_image:
                previews.append(output.copy())
            else:
                previews.append(output * 0)
            sizes.append({
                "source_width": source_width, "source_height": source_height,
                "target_width": target_width, "target_height": target_height,
                "scale_x": target_width / source_width, "scale_y": target_height / source_height,
            })
    except (TypeError, ValueError, cv2.error):
        data["error"] = "E3914"
        return data

    data["images"] = resized
    data["count"] = len(resized)
    meta = data.setdefault("meta", {})
    meta["reference_resize"] = {
        "interpolation": interpolation, "scale_percent": float(scale_percent), "sizes": sizes,
        "show_image": bool(show_image), "show_reference": bool(show_reference),
        "reference_opacity": float(reference_opacity),
    }
    data.setdefault("results", {})["reference_resize_previews"] = previews
    data["results"]["reference_resize_layers"] = {
        "images": resized,
        "references": [reference_images[min(i, len(reference_images) - 1)] for i in range(len(resized))],
    }
    # Keep the established resize metadata contract for downstream steps that
    # translate pixel measurements back to the original image scale.
    first_size = sizes[0]
    previous_resize = meta.get("resize", {})
    meta["resize"] = {
        "original_width": int(previous_resize.get("original_width", first_size["source_width"])),
        "original_height": int(previous_resize.get("original_height", first_size["source_height"])),
        "width": first_size["target_width"],
        "height": first_size["target_height"],
        "cumulative_scale_x": float(previous_resize.get("cumulative_scale_x", 1.0) or 1.0) * first_size["scale_x"] * scale,
        "cumulative_scale_y": float(previous_resize.get("cumulative_scale_y", 1.0) or 1.0) * first_size["scale_y"] * scale,
        "interpolation": interpolation,
        "reference_branch": True,
        "keep_aspect": False,
    }
    data.setdefault("history", []).append("resize_to_reference")
    return data


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

    previous_resize = data.get("meta", {}).get("resize", {})
    previous_scale_x = float(previous_resize.get("cumulative_scale_x", 1.0) or 1.0)
    previous_scale_y = float(previous_resize.get("cumulative_scale_y", 1.0) or 1.0)
    original_width = int(previous_resize.get("original_width", data["images"][0].shape[1]))
    original_height = int(previous_resize.get("original_height", data["images"][0].shape[0]))

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
        "original_width": original_width,
        "original_height": original_height,
        "width": new_w,
        "height": new_h,
        "cumulative_scale_x": previous_scale_x * (new_w / w_orig),
        "cumulative_scale_y": previous_scale_y * (new_h / h_orig),
        "interpolation": interpolation,
        "scale": scale if use_scale else None,
        "keep_aspect": keep_aspect,
    }

    data["history"].append("resize_images")

    if debug:
        print(f"Resized {len(resized)} images to {new_w}x{new_h}")

    return data
