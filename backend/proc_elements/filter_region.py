import cv2
import numpy as np


def characterize_particles(
    data,
    include_excluded=False,
    pixel_size_um=None,
    percentiles=(5, 25, 50, 75, 95),
    filters=None,
    draw=True,
    draw_only_filtered=True,
    draw_label_key="area_px",
    replace_images=False,
    debug=False
):
    """
    Szemcsekarakterizáló node.

    Feladata:
        - detect_particles által adott szemcsék karakterizálása
        - nagy táblázat készítése
        - geometriai + intenzitás adatok számítása
        - opcionális mikron skálázás
        - szűrés paraméterek alapján
        - overlay kirajzolás

    Output:
        data["results"]["particle_table"]
        data["results"]["particle_table_filtered"]
        data["meta"]["particle_characterization_overlay"]
    """

    if data["error"] is not None:
        return data

    if "meta" not in data or "particles" not in data["meta"]:
        data["error"] = "E3441"
        return data

    if data["images"] is None or data["count"] == 0:
        data["error"] = "E3442"
        return data

    if "results" not in data or data["results"] is None:
        data["results"] = {}

    if "history" not in data or data["history"] is None:
        data["history"] = []

    if pixel_size_um is not None:
        if not isinstance(pixel_size_um, (int, float)) or pixel_size_um <= 0:
            data["error"] = "E3443"
            return data

    if not isinstance(percentiles, (list, tuple)) or len(percentiles) == 0:
        data["error"] = "E3444"
        return data

    for p in percentiles:
        if not isinstance(p, (int, float)) or p < 0 or p > 100:
            data["error"] = "E3445"
            return data

    if filters is None:
        filters = {}

    if not isinstance(filters, dict):
        data["error"] = "E3446"
        return data

    particles_all = data["meta"]["particles"]

    table = []
    filtered_table = []
    overlay_images = []

    for img_index, (img, particles) in enumerate(zip(data["images"], particles_all)):

        if img is None:
            data["error"] = "E3447"
            return data

        if len(img.shape) == 2:
            gray = img
            vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif len(img.shape) == 3 and img.shape[2] == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            vis = img.copy()
        else:
            data["error"] = "E3448"
            return data

        for particle in particles:
            excluded = bool(particle.get("excluded", False))

            if excluded and not include_excluded:
                continue

            polygon = particle.get("polygon")
            if polygon is None or len(polygon) < 3:
                continue

            pts = np.array(polygon, dtype=np.int32)
            cnt = pts.reshape((-1, 1, 2)).astype(np.int32)

            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.fillPoly(mask, [pts], 255)

            pixels = gray[mask > 0]
            if pixels.size == 0:
                continue

            area_px = float(cv2.contourArea(cnt))
            perimeter_px = float(cv2.arcLength(cnt, True))

            x, y, w, h = cv2.boundingRect(cnt)
            bbox_area_px = float(w * h)

            aspect_ratio = float(w / h) if h != 0 else None
            extent = float(area_px / bbox_area_px) if bbox_area_px > 0 else None

            hull = cv2.convexHull(cnt)
            convex_area_px = float(cv2.contourArea(hull))
            solidity = float(area_px / convex_area_px) if convex_area_px > 0 else None

            equivalent_diameter_px = float(np.sqrt(4.0 * area_px / np.pi)) if area_px > 0 else None

            circularity = None
            if perimeter_px > 0:
                circularity = float((4.0 * np.pi * area_px) / (perimeter_px ** 2))

            centroid = particle.get("centroid_px", [None, None])
            centroid_x_px = float(centroid[0]) if centroid[0] is not None else None
            centroid_y_px = float(centroid[1]) if centroid[1] is not None else None

            row = {
                "particle_id": particle.get("particle_id"),
                "image_index": img_index,
                "label": particle.get("label"),
                "excluded": excluded,
                "area_px": area_px,
                "perimeter_px": perimeter_px,
                "bbox_x_px": float(x),
                "bbox_y_px": float(y),
                "bbox_w_px": float(w),
                "bbox_h_px": float(h),
                "centroid_x_px": centroid_x_px,
                "centroid_y_px": centroid_y_px,
                "aspect_ratio": aspect_ratio,
                "extent": extent,
                "solidity": solidity,
                "convex_area_px": convex_area_px,
                "equivalent_diameter_px": equivalent_diameter_px,
                "circularity": circularity,
                "intensity_min": float(np.min(pixels)),
                "intensity_max": float(np.max(pixels)),
                "intensity_mean": float(np.mean(pixels)),
                "intensity_median": float(np.median(pixels)),
                "intensity_std": float(np.std(pixels)),
                "pixel_count": int(pixels.size),
                "polygon": particle.get("polygon"),
                "contour": particle.get("contour"),
            }

            for p in percentiles:
                key = f"intensity_p{int(p)}" if float(p).is_integer() else f"intensity_p{p}"
                row[key] = float(np.percentile(pixels, p))

            if "intensity_p5" in row and "intensity_p95" in row:
                row["intensity_dynamic_range"] = row["intensity_p95"] - row["intensity_p5"]
            else:
                row["intensity_dynamic_range"] = None

            if pixel_size_um is not None:
                row["area_um2"] = area_px * (pixel_size_um ** 2)
                row["perimeter_um"] = perimeter_px * pixel_size_um
                row["bbox_x_um"] = float(x * pixel_size_um)
                row["bbox_y_um"] = float(y * pixel_size_um)
                row["bbox_w_um"] = float(w * pixel_size_um)
                row["bbox_h_um"] = float(h * pixel_size_um)
                row["centroid_x_um"] = centroid_x_px * pixel_size_um if centroid_x_px is not None else None
                row["centroid_y_um"] = centroid_y_px * pixel_size_um if centroid_y_px is not None else None
                row["equivalent_diameter_um"] = (
                    equivalent_diameter_px * pixel_size_um if equivalent_diameter_px is not None else None
                )

            table.append(row)

            keep = True
            for key, rule in filters.items():
                if key not in row:
                    keep = False
                    break

                value = row[key]
                if value is None:
                    keep = False
                    break

                if isinstance(rule, dict):
                    if "min" in rule and value < rule["min"]:
                        keep = False
                    if "max" in rule and value > rule["max"]:
                        keep = False
                    if "eq" in rule and value != rule["eq"]:
                        keep = False
                    if "in" in rule and value not in rule["in"]:
                        keep = False
                else:
                    if value != rule:
                        keep = False

                if not keep:
                    break

            if keep:
                filtered_table.append(row)

            if draw:
                should_draw = keep if draw_only_filtered else True

                if should_draw:
                    poly = np.array(row["polygon"], dtype=np.int32)
                    color = (0, 255, 0) if keep else (0, 0, 255)
                    cv2.polylines(vis, [poly], True, color, 2)

                    if draw_label_key in row:
                        value = row[draw_label_key]
                    else:
                        value = row.get("label", "?")

                    if isinstance(value, float):
                        label_text = f"{draw_label_key}:{value:.2f}"
                    else:
                        label_text = f"{draw_label_key}:{value}"

                    text_x = x
                    text_y = y - 8
                    if text_y < 15:
                        text_y = y + 15

                    (tw, th), baseline = cv2.getTextSize(
                        label_text,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        1,
                    )

                    cv2.rectangle(
                        vis,
                        (text_x, text_y - th - 4),
                        (text_x + tw + 4, text_y + baseline),
                        (0, 0, 0),
                        -1,
                    )

                    cv2.putText(
                        vis,
                        label_text,
                        (text_x + 2, text_y - 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        1,
                        cv2.LINE_AA,
                    )

        overlay_images.append(vis)

    data["results"]["particle_table"] = table
    data["results"]["particle_table_filtered"] = filtered_table
    data["meta"]["particle_characterization_overlay"] = overlay_images
    data["meta"]["particle_characterization_config"] = {
        "include_excluded": include_excluded,
        "pixel_size_um": pixel_size_um,
        "percentiles": tuple(percentiles),
        "filters": filters,
        "draw": draw,
        "draw_only_filtered": draw_only_filtered,
        "draw_label_key": draw_label_key,
        "replace_images": replace_images,
    }

    if replace_images:
        data["images"] = overlay_images
        data["count"] = len(overlay_images)

    data["history"].append("characterize_particles")

    if debug:
        print("Particle characterization complete")
        print(f"Rows: {len(table)}")
        print(f"Filtered rows: {len(filtered_table)}")
        if len(filtered_table) > 0:
            print(filtered_table[0])

    return data
