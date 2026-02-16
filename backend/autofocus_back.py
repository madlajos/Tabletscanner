# autofocus_back.py
import cv2
import numpy as np


def grayscale_difference_score(img_bgr, blur_ksize=5):
    if img_bgr is None or img_bgr.size == 0:
        return None, None, None, None
    if img_bgr.ndim == 2:
        gray = img_bgr
    elif img_bgr.shape[2] == 3:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    elif img_bgr.shape[2] == 4:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2GRAY)
    else:
        return None, None, None, None

    if blur_ksize and blur_ksize > 0:
        gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    gray = gray.astype(np.float32)
    mean_gray = np.mean(gray)
    std_gray = float(np.std(gray))
    mean_abs_diff = float(np.mean(np.abs(gray - mean_gray)))
    min_gray = float(np.min(gray))
    max_gray = float(np.max(gray))
    print(std_gray, mean_abs_diff, min_gray, max_gray)
    return std_gray, mean_abs_diff, min_gray, max_gray


# ---------------- SCRIPT-LOGIC HELPERS ----------------

def sobel_topk_score(gray_u8, top_k=500):
    g = cv2.GaussianBlur(gray_u8, (5, 5), 0)
    sx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(sx, sy)

    vals = mag.reshape(-1)
    if vals.size == 0:
        return 0.0

    kk = int(min(max(1, int(top_k)), vals.size))
    return float(np.mean(np.partition(vals, -kk)[-kk:]))


def largest_inscribed_square_from_mask(mask_u8):
    if mask_u8 is None or mask_u8.size == 0:
        return None

    m = (mask_u8 > 0).astype(np.uint8)
    if int(m.sum()) == 0:
        return None

    dist = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    _, maxVal, _, maxLoc = cv2.minMaxLoc(dist)
    r = float(maxVal)
    if r <= 1e-6:
        return None

    side = int(np.floor(np.sqrt(2.0) * r))
    if side < 2:
        return None

    cx, cy = maxLoc
    x1 = int(cx - side // 2)
    y1 = int(cy - side // 2)
    x2 = x1 + side
    y2 = y1 + side

    H, W = m.shape[:2]

    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > W:
        x1 -= (x2 - W)
        x2 = W
    if y2 > H:
        y1 -= (y2 - H)
        y2 = H

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(W, x2)
    y2 = min(H, y2)

    if x2 - x1 < 2 or y2 - y1 < 2:
        return None

    if np.any(m[y1:y2, x1:x2] == 0):
        for shrink in range(1, 15):
            nx1, ny1 = x1 + shrink, y1 + shrink
            nx2, ny2 = x2 - shrink, y2 - shrink
            if nx2 - nx1 < 2 or ny2 - ny1 < 2:
                return None
            if np.all(m[ny1:ny2, nx1:nx2] > 0):
                return (nx1, ny1, nx2, ny2)
        return None

    return (x1, y1, x2, y2)


def laplacian_mean_abs(gray_u8):
    g = cv2.GaussianBlur(gray_u8, (5, 5), 0)
    lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
    return float(np.mean(np.abs(lap))) if lap.size else 0.0


def lap_sq_from_bbox_gray(gray_bbox_u8):
    """
    Script szerinti:
      - bbox gray -> Otsu -> legnagyobb kontúr -> mask
      - largest inscribed square a maskban
      - laplacian_mean_abs a square-en
    """
    g = cv2.GaussianBlur(gray_bbox_u8, (5, 5), 0)
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    fg = np.mean(gray_bbox_u8[bw == 255]) if np.any(bw == 255) else 0.0
    bg = np.mean(gray_bbox_u8[bw == 0]) if np.any(bw == 0) else 0.0
    if fg < bg:
        bw = 255 - bw

    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0, None

    c = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(bw)
    cv2.drawContours(mask, [c], -1, 255, thickness=cv2.FILLED)

    sq = largest_inscribed_square_from_mask(mask)
    if sq is None:
        return 0.0, None

    x1, y1, x2, y2 = sq
    sq_gray = gray_bbox_u8[y1:y2, x1:x2]
    return laplacian_mean_abs(sq_gray), sq


# (Opcionális) kombi-score, ha máshol kell
def _combine_focus_scores(sobel_topk, lap_sq, eps=1e-12):
    a = float(max(0.0, sobel_topk))
    b = float(max(0.0, lap_sq))
    if a <= eps and b <= eps:
        return 0.0
    if a <= eps:
        return b
    if b <= eps:
        return a
    return float(np.sqrt(a * b))


def edge_definition_score(frame_bgr, top_k=500):
    """
    Kombinált score: sqrt( sobel_topk * lap_sq )
    Meghagyva kompatibilitás miatt (AF most nem ezt fogja használni).
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return None

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    sob = sobel_topk_score(gray, top_k=int(top_k))
    lap, _ = lap_sq_from_bbox_gray(gray)
    return float(_combine_focus_scores(sob, lap))


# ---------------------------------------------------------------------
# detect_largest_object_square_roi (marad)
# ---------------------------------------------------------------------
def detect_largest_object_square_roi(img, square_scale=0.8, debug_scale=0.3, show_debug=False):
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    img_h, img_w = gray.shape[:2]

    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    foreground_mean = np.mean(gray[bw == 255]) if np.any(bw == 255) else 0
    background_mean = np.mean(gray[bw == 0]) if np.any(bw == 0) else 0
    if foreground_mean < background_mean:
        bw = 255 - bw

    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("Nincs objektum a bináris képen.")
        return None

    largest_contour = max(contours, key=cv2.contourArea)

    obj_mask = np.zeros_like(bw)
    cv2.drawContours(obj_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    x, y, w, h = cv2.boundingRect(largest_contour)

    width_pct = (w / img_w) * 100
    height_pct = (h / img_h) * 100
    print(f"Bounding box szelessege: {w} px ({width_pct:.2f} %)")
    print(f"Bounding box magassaga: {h} px ({height_pct:.2f} %)")

    side_target = int(min(w, h) * square_scale)
    side_target = max(side_target, 2)

    dist = cv2.distanceTransform((obj_mask > 0).astype(np.uint8), cv2.DIST_L2, 5)

    dist_roi = dist[y:y+h, x:x+w]
    if dist_roi.size == 0:
        return None

    max_loc = np.unravel_index(np.argmax(dist_roi), dist_roi.shape)
    cy = y + int(max_loc[0])
    cx = x + int(max_loc[1])

    max_side = int(2 * dist[cy, cx] - 1)
    if max_side < 2:
        print("Az objektum túl keskeny, nincs értelmes belső négyzet.")
        return None

    side = min(side_target, max_side)

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
        cv2.drawContours(vis, [largest_contour], -1, (0, 0, 255), 2)
        cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.rectangle(vis, (square_roi[0], square_roi[1]),
                      (square_roi[0] + side, square_roi[1] + side), (255, 0, 0), 2)

        text = f"BB: {width_pct:.1f}%, {height_pct:.1f}%, side={side}"
        cv2.putText(vis, text, (x, max(0, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        vis_small = cv2.resize(vis, None, fx=debug_scale, fy=debug_scale)
        mask_small = cv2.resize(obj_mask, None, fx=debug_scale, fy=debug_scale)

        cv2.imshow("Objektum + bbox + belso negyzet ROI", vis_small)
        cv2.imshow("Legnagyobb objektum maszk", mask_small)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return square_roi
