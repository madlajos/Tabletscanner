import cv2
import numpy as np
from proc_elements.cache_utils import cached_cvtColor


def detect_circles(
    data,
    dp=1.2,
    min_dist=20,
    min_diameter=40,
    max_diameter=50,
    detect_scale=1.0,
    blur_ksize=5,
    edge_threshold=100,
    accumulator_threshold=20,
    polarity="dark",
    radius_multiplier=1.0,
    apply_mask=False,
    mask_background="black",
    invert_mask=False,
    debug=False
):
    """
    Kördetektálás OpenCV HoughCircles-szel.

    polarity:
        - "dark"   : sötét körök világos háttéren
        - "bright" : világos körök sötét / közepes háttéren
        - "both"   : mindkettő külön futtatva, majd összevonva
    
    radius_multiplier:
        - 1.0 (default): nincs módosítás
        - 0.8: sugár 80%-ra csökkentve
        - 1.5: sugár 150%-ra növelve, stb.

    min_diameter / max_diameter:
        - a teljes felbontású képre értendő átmérő pixelben
        - a belső Hough sugárhatárok a detect_scale alapján kerülnek skálázásra
    
    apply_mask:
        - False (default): körön kívül piros kör és zöld pont jelölés
        - True: maszkként alkalmazza a kört
    
    mask_background:
        - "black" (default): körön kívüli terület fekete
        - "white": körön kívüli terület fehér
    
    invert_mask:
        - False (default): körön belül az eredeti kép
        - True: körön belül fekete/fehér, körön kívül az eredeti kép
    """

    if data["error"] is not None:
        return data

    if data["images"] is None or data["count"] == 0:
        data["error"] = "E3601"
        return data

    if polarity not in ["dark", "bright", "both"]:
        data["error"] = "E3602"
        return data

    if not isinstance(min_diameter, int) or not isinstance(max_diameter, int):
        data["error"] = "E3603"
        return data

    if min_diameter <= 0 or max_diameter <= 0 or min_diameter > max_diameter:
        data["error"] = "E3604"
        return data

    if not isinstance(blur_ksize, int) or blur_ksize <= 0 or blur_ksize % 2 == 0:
        data["error"] = "E3605"
        return data

    if detect_scale <= 0:
        data["error"] = "E3610"
        return data

    try:
        radius_multiplier = float(radius_multiplier)
    except (TypeError, ValueError):
        data["error"] = "E3608"
        return data

    if radius_multiplier <= 0:
        data["error"] = "E3608"
        return data

    if mask_background not in ["black", "white"]:
        data["error"] = "E3609"
        return data

    if "results" not in data or data["results"] is None:
        data["results"] = {}
    if "meta" not in data or data["meta"] is None:
        data["meta"] = {}

    scale = float(detect_scale)
    min_radius = max(1, int(round(min_diameter / 2.0)))
    max_radius = max(1, int(round(max_diameter / 2.0)))

    def _run_hough(gray_img, mode):
        if scale != 1.0:
            interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
            work = cv2.resize(gray_img, None, fx=scale, fy=scale, interpolation=interpolation)
        else:
            work = gray_img

        if mode == "bright":
            work = cv2.bitwise_not(work)

        if blur_ksize > 1:
            work = cv2.medianBlur(work, blur_ksize)

        circles = cv2.HoughCircles(
            work,
            cv2.HOUGH_GRADIENT,
            dp=float(dp),
            minDist=float(min_dist) * scale,
            param1=float(edge_threshold),
            param2=float(accumulator_threshold),
            minRadius=max(1, int(round(min_radius * scale))),
            maxRadius=max(1, int(round(max_radius * scale)))
        )

        results = []
        if circles is not None:
            circles = np.round(circles[0, :]).astype(int)
            for c in circles:
                if scale != 1.0:
                    x = int(round(c[0] / scale))
                    y = int(round(c[1] / scale))
                    r = int(round(c[2] / scale))
                else:
                    x, y, r = int(c[0]), int(c[1]), int(c[2])
                adjusted_r = max(1, int(r * radius_multiplier))
                results.append({
                    "center_x": x,
                    "center_y": y,
                    "raw_radius": max(1, int(r)),
                    "radius": adjusted_r,
                    "polarity": mode
                })

        return results

    def _deduplicate(circle_list, center_tol=6, radius_tol=4):
        unique = []
        for c in circle_list:
            keep = True
            for u in unique:
                dc = np.hypot(c["center_x"] - u["center_x"], c["center_y"] - u["center_y"])
                dr = abs(c["radius"] - u["radius"])
                if dc <= center_tol and dr <= radius_tol:
                    keep = False
                    break
            if keep:
                unique.append(c)
        return unique

    def _refine_radius(gray_img, center_x, center_y, radius_guess, search_margin=None):
        if radius_guess <= 0:
            return radius_guess

        height, width = gray_img.shape[:2]
        if center_x < 0 or center_y < 0 or center_x >= width or center_y >= height:
            return radius_guess

        if search_margin is None:
            search_margin = max(4, int(round(radius_guess * 0.25)))

        r_min = max(1, radius_guess - search_margin)
        r_max = max(r_min, radius_guess + search_margin)

        x0 = max(0, center_x - r_max - 2)
        y0 = max(0, center_y - r_max - 2)
        x1 = min(width, center_x + r_max + 3)
        y1 = min(height, center_y + r_max + 3)
        roi = gray_img[y0:y1, x0:x1]
        if roi.size == 0:
            return radius_guess

        yy, xx = np.ogrid[:roi.shape[0], :roi.shape[1]]
        dx = xx - (center_x - x0)
        dy = yy - (center_y - y0)
        dist = np.sqrt(dx * dx + dy * dy)

        best_radius = radius_guess
        best_score = -1.0

        for radius in range(r_min, r_max + 1):
            inner_ring = (dist >= max(0, radius - 2)) & (dist < radius)
            outer_ring = (dist >= radius) & (dist <= radius + 2)
            inner_vals = roi[inner_ring]
            outer_vals = roi[outer_ring]
            if inner_vals.size == 0 or outer_vals.size == 0:
                continue
            score = abs(float(inner_vals.mean()) - float(outer_vals.mean()))
            if score > best_score:
                best_score = score
                best_radius = radius

        return best_radius

    all_circles = []
    overlay_images = []
    all_masks = []
    all_masked_images = []

    # Initialize conversion cache in results
    if "results" not in data:
        data["results"] = {}

    for img in data["images"]:
        if img is None:
            data["error"] = "E3606"
            return data

        if len(img.shape) == 2:
            gray = img
            vis = cached_cvtColor(data, img, cv2.COLOR_GRAY2BGR, "cvtColor_GRAY2BGR")
        elif len(img.shape) == 3 and img.shape[2] == 3:
            gray = cached_cvtColor(data, img, cv2.COLOR_BGR2GRAY, "cvtColor_BGR2GRAY")
            vis = img.copy()
        else:
            data["error"] = "E3607"
            return data

        circles_img = []

        if polarity in ["dark", "bright"]:
            circles_img.extend(_run_hough(gray, polarity))
        else:
            circles_img.extend(_run_hough(gray, "dark"))
            circles_img.extend(_run_hough(gray, "bright"))
            circles_img = _deduplicate(circles_img)

        # Keep only the circle with highest confidence/viability
        if circles_img:
            best_circle = dict(max(circles_img, key=lambda c: (c.get("raw_radius", c.get("radius", 0)),)))
            raw_radius = max(1, int(best_circle.get("raw_radius", best_circle.get("radius", 0))))
            if scale < 1.0:
                raw_radius = max(
                    1,
                    int(_refine_radius(gray, best_circle["center_x"], best_circle["center_y"], raw_radius))
                )
            best_circle["raw_radius"] = raw_radius
            best_circle["radius"] = max(1, int(raw_radius * radius_multiplier))
            circles_img = [best_circle]

        current_mask = None
        current_masked_image = None

        if apply_mask:
            if circles_img:
                # Mask mode: apply the best circle as ROI mask.
                c = circles_img[0]
                base_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
                cv2.circle(base_mask, (c["center_x"], c["center_y"]), c["radius"], 255, -1)
            else:
                # Keep batch lengths stable when no circle was found on an image.
                base_mask = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8) * 255

            mask = cv2.bitwise_not(base_mask) if invert_mask else base_mask
            current_mask = mask.copy()

            bg_value = 255 if mask_background == "white" else 0
            bg_img = np.full(img.shape, bg_value, dtype=np.uint8)

            if len(img.shape) == 2:
                masked_img = np.where(mask > 0, img, bg_img)
                vis = cached_cvtColor(data, masked_img, cv2.COLOR_GRAY2BGR, "cvtColor_GRAY2BGR_detect_circ")
            else:
                masked_img = np.where(mask[:, :, np.newaxis] > 0, img, bg_img)
                vis = masked_img.copy()

            current_masked_image = masked_img
        else:
            # Visualization mode: draw circle and center point
            for c in circles_img:
                color = (0, 0, 255) if c["polarity"] == "dark" else (255, 0, 0)
                cv2.circle(vis, (c["center_x"], c["center_y"]), c["radius"], color, 2)
                cv2.circle(vis, (c["center_x"], c["center_y"]), 2, (0, 255, 0), 2)

        all_circles.append(circles_img)
        overlay_images.append(vis)
        if apply_mask:
            all_masks.append(current_mask)
            all_masked_images.append(current_masked_image)

    data["results"]["circles"] = all_circles
    data["results"]["circle_overlay"] = overlay_images
    
    if apply_mask:
        data["images"] = all_masked_images
        data["count"] = len(all_masked_images)
        data["results"]["masks"] = all_masks
        data["results"]["masked_images"] = all_masked_images
        # Store masks in meta so downstream steps know about them
        data["meta"]["active_masks"] = all_masks
    
    
    data["meta"]["detect_circles"] = {
        "dp": float(dp),
        "min_dist": float(min_dist),
        "detect_scale": float(scale),
        "min_radius": int(min_radius),
        "max_radius": int(max_radius),
        "blur_ksize": int(blur_ksize),
        "edge_threshold": float(edge_threshold),
        "accumulator_threshold": float(accumulator_threshold),
        "polarity": polarity,
        "radius_multiplier": float(radius_multiplier),
        "apply_mask": bool(apply_mask),
        "mask_background": str(mask_background),
        "invert_mask": bool(invert_mask)
    }
    data["history"].append("detect_circles")

    if debug:
        print(data["meta"]["detect_circles"])
        print(f"First image circles: {len(all_circles[0])}")
        cv2.imshow("circle_overlay", overlay_images[0])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return data

