import cv2
import numpy as np
from proc_elements.cache_utils import cached_cvtColor


def detect_circles(
    data,
    dp=1.2,
    min_dist=20,
    min_radius=20,
    max_radius=25,
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

    if not isinstance(min_radius, int) or not isinstance(max_radius, int):
        data["error"] = "E3603"
        return data

    if min_radius <= 0 or max_radius <= 0 or min_radius > max_radius:
        data["error"] = "E3604"
        return data

    if not isinstance(blur_ksize, int) or blur_ksize <= 0 or blur_ksize % 2 == 0:
        data["error"] = "E3605"
        return data

    if radius_multiplier <= 0:
        data["error"] = "E3608"
        return data

    if mask_background not in ["black", "white"]:
        data["error"] = "E3609"
        return data

    if "results" not in data or data["results"] is None:
        data["results"] = {}

    all_circles = []
    overlay_images = []

    def _run_hough(gray_img, mode):
        work = gray_img.copy()

        if mode == "bright":
            work = cv2.bitwise_not(work)

        if blur_ksize > 1:
            work = cv2.medianBlur(work, blur_ksize)

        circles = cv2.HoughCircles(
            work,
            cv2.HOUGH_GRADIENT,
            dp=float(dp),
            minDist=float(min_dist),
            param1=float(edge_threshold),
            param2=float(accumulator_threshold),
            minRadius=int(min_radius),
            maxRadius=int(max_radius)
        )

        results = []
        if circles is not None:
            circles = np.round(circles[0, :]).astype(int)
            for c in circles:
                x, y, r = int(c[0]), int(c[1]), int(c[2])
                adjusted_r = max(1, int(r * radius_multiplier))
                results.append({
                    "center_x": x,
                    "center_y": y,
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
            best_circle = max(circles_img, key=lambda c: (c.get("radius", 0),))
            circles_img = [best_circle]

        current_mask = None
        current_masked_image = None

        if apply_mask and circles_img:
            # Mask mode: apply circle as mask
            c = circles_img[0]
            mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
            cv2.circle(mask, (c["center_x"], c["center_y"]), c["radius"], 255, -1)
            
            if invert_mask:
                # Invert: körön belül fekete/fehér, körön kívül eredeti
                mask = cv2.bitwise_not(mask)
            
            current_mask = mask.copy()
            
            # Determine background value (0=black, 255=white)
            bg_value = 255 if mask_background == "white" else 0
            
            # Create background image with the selected color
            if len(img.shape) == 2:
                # Grayscale
                bg_img = np.full(img.shape, bg_value, dtype=np.uint8)
                masked_img = np.where(mask[:, :, np.newaxis] > 0, img, bg_img) if len(mask.shape) == 2 else np.where(mask > 0, img, bg_img)
            else:
                # Color (BGR)
                bg_img = np.full(img.shape, bg_value, dtype=np.uint8)
                masked_img = np.where(mask[:, :, np.newaxis] > 0, img, bg_img)
            
            # Convert to BGR for consistency
            if len(masked_img.shape) == 2:
                vis = cv2.cvtColor(masked_img, cv2.COLOR_GRAY2BGR)
            else:
                vis = masked_img
            
            current_masked_image = vis.copy()
        else:
            # Visualization mode: draw circle and center point
            for c in circles_img:
                color = (0, 0, 255) if c["polarity"] == "dark" else (255, 0, 0)
                cv2.circle(vis, (c["center_x"], c["center_y"]), c["radius"], color, 2)
                cv2.circle(vis, (c["center_x"], c["center_y"]), 2, (0, 255, 0), 2)

        all_circles.append(circles_img)
        overlay_images.append(vis)
        if current_mask is not None:
            all_masks.append(current_mask)
        if current_masked_image is not None:
            all_masked_images.append(current_masked_image)

    data["results"]["circles"] = all_circles
    data["results"]["circle_overlay"] = overlay_images
    
    if apply_mask:
        data["images"] = all_masked_images if all_masked_images else overlay_images
        data["results"]["masks"] = all_masks
        data["results"]["masked_images"] = all_masked_images
        # Store masks in meta so downstream steps know about them
        data["meta"]["active_masks"] = all_masks
    
    
    data["meta"]["detect_circles"] = {
        "dp": float(dp),
        "min_dist": float(min_dist),
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

