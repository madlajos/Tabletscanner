import cv2
import numpy as np


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
    debug=False
):
    """
    Kördetektálás OpenCV HoughCircles-szel.

    polarity:
        - "dark"   : sötét körök világos háttéren
        - "bright" : világos körök sötét / közepes háttéren
        - "both"   : mindkettő külön futtatva, majd összevonva
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
                results.append({
                    "center_x": x,
                    "center_y": y,
                    "radius": r,
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

    for img in data["images"]:
        if img is None:
            data["error"] = "E3606"
            return data

        if len(img.shape) == 2:
            gray = img
            vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif len(img.shape) == 3 and img.shape[2] == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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

        for c in circles_img:
            color = (0, 0, 255) if c["polarity"] == "dark" else (255, 0, 0)
            cv2.circle(vis, (c["center_x"], c["center_y"]), c["radius"], color, 2)
            cv2.circle(vis, (c["center_x"], c["center_y"]), 2, (0, 255, 0), 2)

        all_circles.append(circles_img)
        overlay_images.append(vis)

    data["results"]["circles"] = all_circles
    data["results"]["circle_overlay"] = overlay_images
    data["meta"]["detect_circles"] = {
        "dp": float(dp),
        "min_dist": float(min_dist),
        "min_radius": int(min_radius),
        "max_radius": int(max_radius),
        "blur_ksize": int(blur_ksize),
        "edge_threshold": float(edge_threshold),
        "accumulator_threshold": float(accumulator_threshold),
        "polarity": polarity
    }
    data["history"].append("detect_circles")

    if debug:
        print(data["meta"]["detect_circles"])
        print(f"First image circles: {len(all_circles[0])}")
        cv2.imshow("circle_overlay", overlay_images[0])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return data
