import cv2
import numpy as np


def detect_particles(
    data,
    connectivity=8,
    polygon_epsilon=0.01,
    draw=True,
    draw_label=True,
    replace_images=False,
    debug=False
):
    """
    Szemcsedetektáló node.

    Feladata:
        - bináris / maszkolt képből szemcsék detektálása
        - polygon / contour eltárolása
        - preview overlay készítése
        - manuális exclude kezelés előkészítése

    Input:
        - data["images"]: binary / grayscale / BGR képek
        - foreground: nonzero
        - background: 0

    Output:
        - data["meta"]["particles"]
        - data["meta"]["particles_summary"]
        - data["meta"]["particles_overlay"]
    """

    if data["error"] is not None:
        return data

    if data["images"] is None or data["count"] == 0:
        data["error"] = "E3431"
        return data

    if "meta" not in data or data["meta"] is None:
        data["meta"] = {}

    if "history" not in data or data["history"] is None:
        data["history"] = []

    if connectivity not in [4, 8]:
        data["error"] = "E3432"
        return data

    all_particles = []
    summary = []
    overlay_images = []

    for img_index, img in enumerate(data["images"]):

        if img is None:
            data["error"] = "E3433"
            return data

        if len(img.shape) == 2:
            gray = img
            vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif len(img.shape) == 3 and img.shape[2] == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            vis = img.copy()
        else:
            data["error"] = "E3434"
            return data

        binary = (gray > 0).astype(np.uint8)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary,
            connectivity=connectivity
        )

        image_particles = []

        for label_id in range(1, num_labels):
            x = int(stats[label_id, cv2.CC_STAT_LEFT])
            y = int(stats[label_id, cv2.CC_STAT_TOP])
            w = int(stats[label_id, cv2.CC_STAT_WIDTH])
            h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
            area_px = int(stats[label_id, cv2.CC_STAT_AREA])

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

            particle = {
                "particle_id": f"img{img_index}_label{label_id}",
                "image_index": img_index,
                "label": int(label_id),
                "excluded": False,
                "bbox_px": [x, y, w, h],
                "centroid_px": [cx, cy],
                "area_px": area_px,
                "perimeter_px": perimeter_px,
                "polygon": polygon_points,
                "contour": contour_points,
            }

            image_particles.append(particle)

            if draw:
                pts = np.array(polygon_points, dtype=np.int32)
                cv2.polylines(vis, [pts], True, (0, 255, 0), 2)

                if draw_label:
                    text = str(label_id)
                    text_x = x
                    text_y = y - 8
                    if text_y < 15:
                        text_y = y + 15

                    (tw, th), baseline = cv2.getTextSize(
                        text,
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
                        text,
                        (text_x + 2, text_y - 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )

        all_particles.append(image_particles)
        summary.append({
            "image_index": img_index,
            "particle_count": len(image_particles),
        })
        overlay_images.append(vis)

    data["meta"]["particles"] = all_particles
    data["meta"]["particles_summary"] = summary
    data["meta"]["particles_overlay"] = overlay_images
    data["meta"]["detect_particles_config"] = {
        "connectivity": connectivity,
        "polygon_epsilon": polygon_epsilon,
        "draw": draw,
        "draw_label": draw_label,
        "replace_images": replace_images,
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
