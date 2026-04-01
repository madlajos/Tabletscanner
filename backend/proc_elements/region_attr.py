import cv2
import numpy as np
from proc_elements.cache_utils import cached_cvtColor


def detect_particles(
    data,
    connectivity=8,
    polygon_epsilon=0.01,
    selected_features=None,
    filters=None,
    percentiles=(5, 25, 50, 75, 95),
    draw=True,
    contour_thickness=2,
    draw_label=True,
    draw_only_filtered=False,
    draw_label_key="label",
    replace_images=False,
    excluded_ids=None,
    debug=False
):
    """
    Szemcsedetektálás + alap feature számítás + szűrés.

    selected_features:
        list[str] | None
        pl. ["area_px", "circularity", "intensity_mean"]

    filters:
        dict
        pl.
        {
            "area_px": {"min": 50, "max": 5000},
            "circularity": {"min": 0.6},
            "intensity_mean": {"min": 80, "max": 200}
        }
    """

    if data["error"] is not None:
        return data

    if data["images"] is None or data["count"] == 0:
        data["error"] = "E3000"
        return data

    if "meta" not in data or data["meta"] is None:
        data["meta"] = {}

    if "history" not in data or data["history"] is None:
        data["history"] = []

    if connectivity not in [4, 8]:
        data["error"] = "E3001"
        return data

    if selected_features is None:
        selected_features = []

    if filters is None:
        filters = {}

    excluded_set = set(excluded_ids) if excluded_ids else set()

    if not isinstance(selected_features, (list, tuple)):
        data["error"] = "E3004"
        return data

    if not isinstance(filters, dict):
        data["error"] = "E3005"
        return data

    if not isinstance(percentiles, (list, tuple)) or len(percentiles) == 0:
        data["error"] = "E3006"
        return data

    for p in percentiles:
        if not isinstance(p, (int, float)) or p < 0 or p > 100:
            data["error"] = "E3007"
            return data

    all_particles = []
    filtered_particles = []
    summary = []
    overlay_images = []

    def passes_filters(row, filters_dict):
        for key, rule in filters_dict.items():
            if key not in row:
                return False

            value = row[key]
            if value is None:
                return False

            if isinstance(rule, dict):
                if "min" in rule and value < rule["min"]:
                    return False
                if "max" in rule and value > rule["max"]:
                    return False
                if "eq" in rule and value != rule["eq"]:
                    return False
                if "in" in rule and value not in rule["in"]:
                    return False
            else:
                if value != rule:
                    return False

        return True

    # Initialize conversion cache in results
    if "results" not in data:
        data["results"] = {}

    for img_index, img in enumerate(data["images"]):

        if img is None:
            data["error"] = "E3002"
            return data

        if len(img.shape) == 2:
            gray = img
            vis = cached_cvtColor(data, img, cv2.COLOR_GRAY2BGR, "cvtColor_GRAY2BGR")
        elif len(img.shape) == 3 and img.shape[2] == 3:
            gray = cached_cvtColor(data, img, cv2.COLOR_BGR2GRAY, "cvtColor_BGR2GRAY")
            vis = img.copy()
        else:
            data["error"] = "E3003"
            return data

        binary = (gray > 0).astype(np.uint8)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary,
            connectivity=connectivity
        )

        image_particles = []
        image_particles_filtered = []

        for label_id in range(1, num_labels):
            x = int(stats[label_id, cv2.CC_STAT_LEFT])
            y = int(stats[label_id, cv2.CC_STAT_TOP])
            w = int(stats[label_id, cv2.CC_STAT_WIDTH])
            h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
            area_cc_px = int(stats[label_id, cv2.CC_STAT_AREA])

            cx, cy = centroids[label_id]
            cx = float(cx)
            cy = float(cy)

            region_mask = np.uint8(labels == label_id) * 255

            contours, _ = cv2.findContours(
                region_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            if len(contours) == 0:
                continue

            cnt = max(contours, key=cv2.contourArea)
            perimeter_px = float(cv2.arcLength(cnt, True))

            epsilon = polygon_epsilon * perimeter_px
            poly = cv2.approxPolyDP(cnt, epsilon, True)

            polygon_points = poly.reshape(-1, 2).tolist()
            contour_points = cnt.reshape(-1, 2).tolist()

            pts = np.array(polygon_points, dtype=np.int32)
            cnt_poly = pts.reshape((-1, 1, 2)).astype(np.int32)

            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.fillPoly(mask, [pts], 255)

            pixels = gray[mask > 0]
            if pixels.size == 0:
                continue

            area_px = float(cv2.contourArea(cnt_poly))
            perimeter_px = float(cv2.arcLength(cnt_poly, True))

            bbox_area_px = float(w * h)
            aspect_ratio = float(w / h) if h != 0 else None
            extent = float(area_px / bbox_area_px) if bbox_area_px > 0 else None

            hull = cv2.convexHull(cnt_poly)
            convex_area_px = float(cv2.contourArea(hull))
            solidity = float(area_px / convex_area_px) if convex_area_px > 0 else None

            equivalent_diameter_px = float(np.sqrt(4.0 * area_px / np.pi)) if area_px > 0 else None

            circularity = None
            if perimeter_px > 0:
                circularity = float((4.0 * np.pi * area_px) / (perimeter_px ** 2))

            particle = {
                "particle_id": f"img{img_index}_label{label_id}",
                "image_index": img_index,
                "label": int(label_id),
                "excluded": False,

                "bbox_px": [x, y, w, h],
                "centroid_px": [cx, cy],

                "area_px": area_px,
                "area_cc_px": int(area_cc_px),
                "perimeter_px": perimeter_px,
                "bbox_x_px": float(x),
                "bbox_y_px": float(y),
                "bbox_w_px": float(w),
                "bbox_h_px": float(h),
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

                "polygon": polygon_points,
                "contour": contour_points
            }

            for p in percentiles:
                key = f"intensity_p{int(p)}" if float(p).is_integer() else f"intensity_p{p}"
                particle[key] = float(np.percentile(pixels, p))

            if "intensity_p5" in particle and "intensity_p95" in particle:
                particle["intensity_dynamic_range"] = particle["intensity_p95"] - particle["intensity_p5"]
            else:
                particle["intensity_dynamic_range"] = None

            particle["selected_values"] = {
                key: particle.get(key, None) for key in selected_features
            }

            is_excluded = particle["particle_id"] in excluded_set
            particle["excluded"] = is_excluded
            keep = passes_filters(particle, filters) and not is_excluded
            particle["passed_filters"] = bool(keep)

            image_particles.append(particle)

            if keep:
                image_particles_filtered.append(particle)

            if draw:
                if is_excluded:
                    # Always draw excluded particles in yellow
                    color = (0, 255, 255)
                    cv2.polylines(vis, [cnt], True, color, max(1, int(contour_thickness)))
                else:
                    should_draw = keep if draw_only_filtered else True
                    if not should_draw:
                        continue
                    color = (0, 255, 0) if keep else (0, 0, 255)
                    cv2.polylines(vis, [cnt], True, color, max(1, int(contour_thickness)))

                    if draw_label:
                        value = particle.get(draw_label_key, particle.get("label", "?"))
                        if isinstance(value, float):
                            text = f"{draw_label_key}:{value:.2f}"
                        else:
                            text = f"{draw_label_key}:{value}"

                        text_x = x
                        text_y = y - 8
                        if text_y < 15:
                            text_y = y + 15

                        (tw, th), baseline = cv2.getTextSize(
                            text,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            1
                        )

                        cv2.rectangle(
                            vis,
                            (text_x, text_y - th - 4),
                            (text_x + tw + 4, text_y + baseline),
                            (0, 0, 0),
                            -1
                        )

                        cv2.putText(
                            vis,
                            text,
                            (text_x + 2, text_y - 2),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            color,
                            1,
                            cv2.LINE_AA
                        )

        all_particles.append(image_particles)
        filtered_particles.append(image_particles_filtered)
        summary.append({
            "image_index": img_index,
            "particle_count": len(image_particles),
            "particle_count_filtered": len(image_particles_filtered)
        })
        overlay_images.append(vis)

    data["meta"]["particles"] = all_particles
    data["meta"]["particles_filtered"] = filtered_particles
    data["meta"]["particles_summary"] = summary
    data["meta"]["particles_overlay"] = overlay_images
    data["meta"]["detect_particles_config"] = {
        "connectivity": connectivity,
        "polygon_epsilon": polygon_epsilon,
        "selected_features": list(selected_features),
        "filters": filters,
        "percentiles": tuple(percentiles),
        "draw": draw,
        "draw_label": draw_label,
        "draw_only_filtered": draw_only_filtered,
        "draw_label_key": draw_label_key,
        "replace_images": replace_images
    }

    if replace_images:
        data["images"] = overlay_images
        data["count"] = len(overlay_images)

    data["history"].append("detect_particles")

    if debug:
        print("Particle detection complete")
        print(summary)
        if len(all_particles) > 0 and len(all_particles[0]) > 0:
            print(all_particles[0][0])

    return data
