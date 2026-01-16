import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
def edge_definition_score(frame_bgr, ring_width=5):
    """
    Objektum-háttér perem definíció:
    - Otsu + legnagyobb kontúr => objektum maszk
    - ring = dilate(mask) - erode(mask)
    - score = átlagos Scharr gradiens a ring pixeleken

    Nagyobb = élesebb kontúr.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return None

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    # Otsu
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # invert ha kell (foreground fehér legyen)
    fg = np.mean(gray[bw == 255]) if np.any(bw == 255) else 0
    bg = np.mean(gray[bw == 0]) if np.any(bw == 0) else 0
    if fg < bg:
        bw = 255 - bw

    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    c = max(contours, key=cv2.contourArea)

    mask = np.zeros_like(bw)
    cv2.drawContours(mask, [c], -1, 255, thickness=cv2.FILLED)

    k = max(1, int(ring_width))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    dil = cv2.dilate(mask, kernel, iterations=1)
    ero = cv2.erode(mask, kernel, iterations=1)
    ring = cv2.subtract(dil, ero)

    gx = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
    gy = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
    mag = np.sqrt(gx * gx + gy * gy)

    vals = mag[ring > 0]
    if vals.size == 0:
        return None

    return round(float(np.mean(vals)), 4)

# ---- 1. Focus score számoló függvény ----
def process_frame(frame, roi=None):
    # ROI kivágása, ha van
    if roi is not None:
        x, y, w, h = roi
        frame = frame[y:y+h, x:x+w]

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    gradient_x = cv2.Sobel(frame_gray, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(frame_gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)

    focus_score = round(np.mean(gradient_magnitude), 4)
    return focus_score

def detect_largest_object_square_roi(img, square_scale=0.8, debug_scale=0.3, show_debug=False):
    """
    Legnagyobb objektum maszkját megkeresi, majd olyan négyzet ROI-t ad,
    ami TELJESEN az objektumon belül van (distance transform alapú).

    square_scale: a bounding box min(w,h)-jának hányad része legyen a négyzet oldala (kezdeti cél).
    """

    # --- grayscale ---
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    img_h, img_w = gray.shape[:2]

    # --- Otsu threshold ---
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # automatikus invert, ha a foreground sötétebb
    foreground_mean = np.mean(gray[bw == 255]) if np.any(bw == 255) else 0
    background_mean = np.mean(gray[bw == 0]) if np.any(bw == 0) else 0
    if foreground_mean < background_mean:
        bw = 255 - bw

    # --- kontúrok, legnagyobb objektum ---
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("Nincs objektum a bináris képen.")
        return None

    largest_contour = max(contours, key=cv2.contourArea)

    # maszk a legnagyobb objektumra (hogy a kisebb zaj-komponensek ne zavarjanak)
    obj_mask = np.zeros_like(bw)
    cv2.drawContours(obj_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    x, y, w, h = cv2.boundingRect(largest_contour)

    width_pct  = (w / img_w) * 100
    height_pct = (h / img_h) * 100
    print(f"Bounding box szelessege: {w} px ({width_pct:.2f} %)")
    print(f"Bounding box magassaga: {h} px ({height_pct:.2f} %)")

    # --- cél oldalméret ---
    side_target = int(min(w, h) * square_scale)
    side_target = max(side_target, 2)

    # --- distance transform: mekkora négyzet "fér el" középen ---
    # distanceTransform bemenet: 0 háttér, >0 foreground (uint8 OK)
    dist = cv2.distanceTransform((obj_mask > 0).astype(np.uint8), cv2.DIST_L2, 5)

    # csak bbox-on belül keressünk középpontot (gyorsabb + releváns)
    dist_roi = dist[y:y+h, x:x+w]
    if dist_roi.size == 0:
        return None

    # legjobb középpont ott, ahol a dist maximális
    max_loc = np.unravel_index(np.argmax(dist_roi), dist_roi.shape)
    cy = y + int(max_loc[0])
    cx = x + int(max_loc[1])

    # maximum beférhető négyzet oldal (konzervatív): 2*dist(center) - 1
    # (a -1 csak biztonsági, elhagyható)
    max_side = int(2 * dist[cy, cx] - 1)
    if max_side < 2:
        print("Az objektum túl keskeny, nincs értelmes belső négyzet.")
        return None

    # tényleges side: cél és max korlát közül a kisebb
    side = min(side_target, max_side)

    # segítség: garantáljuk, hogy a négyzet teljesen maszkban van
    # Ha mégsem (diszkretizálás miatt), csökkentjük.
    def square_fits(cx, cy, side):
        half = side // 2
        x0 = cx - half
        y0 = cy - half
        x1 = x0 + side
        y1 = y0 + side
        if x0 < 0 or y0 < 0 or x1 > img_w or y1 > img_h:
            return False
        patch = obj_mask[y0:y1, x0:x1]
        return np.all(patch == 255)

    # csökkentés, amíg befér
    while side >= 2 and not square_fits(cx, cy, side):
        side -= 2

    if side < 2:
        print("Nem találtunk olyan négyzetet, ami teljesen az objektumban van.")
        return None

    half = side // 2
    sq_x = cx - half
    sq_y = cy - half
    square_roi = (int(sq_x), int(sq_y), int(side), int(side))
    print(f"Negyzet ROI (objektumon belul): x={square_roi[0]}, y={square_roi[1]}, w={side}, h={side}")

    if show_debug:
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(vis, [largest_contour], -1, (0, 0, 255), 2)          # piros kontúr
        cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)                # zöld bbox
        cv2.rectangle(vis, (square_roi[0], square_roi[1]),
                      (square_roi[0]+side, square_roi[1]+side), (255, 0, 0), 2)  # kék négyzet

        text = f"BB: {width_pct:.1f}%, {height_pct:.1f}%, side={side}"
        cv2.putText(vis, text, (x, max(0, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        vis_small = cv2.resize(vis, None, fx=debug_scale, fy=debug_scale)
        mask_small = cv2.resize(obj_mask, None, fx=debug_scale, fy=debug_scale)

        cv2.imshow("Objektum + bbox + belso negyzet ROI", vis_small)
        cv2.imshow("Legnagyobb objektum maszk", mask_small)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return square_roi
