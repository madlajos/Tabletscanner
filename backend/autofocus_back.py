import cv2
import numpy as np


# ---------------------------------------------------------------------
# Basic image stats check (marad)
# ---------------------------------------------------------------------
def grayscale_difference_score(img_bgr, blur_ksize=5):
    if img_bgr is None or img_bgr.size == 0:
        return None, None, None, None

    if img_bgr.ndim == 2:
        gray = img_bgr
    elif img_bgr.ndim == 3 and img_bgr.shape[2] == 3:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    elif img_bgr.ndim == 3 and img_bgr.shape[2] == 4:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2GRAY)
    else:
        return None, None, None, None

    if blur_ksize and blur_ksize > 0:
        gray = cv2.GaussianBlur(gray, (int(blur_ksize), int(blur_ksize)), 0)

    gray = gray.astype(np.float32)
    mean_gray = np.mean(gray)
    std_gray = float(np.std(gray))
    mean_abs_diff = float(np.mean(np.abs(gray - mean_gray)))
    min_gray = float(np.min(gray))
    max_gray = float(np.max(gray))
    print(std_gray, mean_abs_diff, min_gray, max_gray)
    return std_gray, mean_abs_diff, min_gray, max_gray


# ---------------------------------------------------------------------
# Focus metrics (marad)
# ---------------------------------------------------------------------
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

    # ha belelóg 0, akkor kicsit zsugorítsunk
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


# ---------------------------------------------------------------------
# Rounded check (átmozgatva)
# ---------------------------------------------------------------------
def is_near_border_xy(x, y, W, H, margin_px=3):
    return (x <= margin_px) or (y <= margin_px) or \
           (x >= W - 1 - margin_px) or (y >= H - 1 - margin_px)


def rounded_by_curvature_ignore_border(
    contour,
    W, H,
    margin_px=15,
    step=12,
    angle_threshold_deg=55.0,
    max_sharp=20,
    min_used=20,
    downsample=4,
):
    if contour is None:
        return None, {"decision": "NO_CONTOUR", "used": 0}

    pts = contour.reshape(-1, 2).astype(np.int32)

    if downsample and downsample > 1:
        pts = pts[::int(downsample)]

    n = len(pts)
    if n < (2 * step + 3):
        return None, {"decision": "TOO_FEW_POINTS", "used": 0}

    sharp = 0
    used = 0
    min_angle = 180.0

    for i in range(n):
        x, y = int(pts[i][0]), int(pts[i][1])
        if is_near_border_xy(x, y, W, H, int(margin_px)):
            continue

        p_prev = pts[(i - int(step)) % n]
        p_next = pts[(i + int(step)) % n]

        if is_near_border_xy(int(p_prev[0]), int(p_prev[1]), W, H, int(margin_px)):
            continue
        if is_near_border_xy(int(p_next[0]), int(p_next[1]), W, H, int(margin_px)):
            continue

        v1 = (p_prev - pts[i]).astype(np.float32)
        v2 = (p_next - pts[i]).astype(np.float32)

        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))
        if n1 < 1e-6 or n2 < 1e-6:
            continue

        cosang = float(np.dot(v1, v2) / (n1 * n2))
        cosang = max(-1.0, min(1.0, cosang))
        ang = float(np.degrees(np.arccos(cosang)))

        used += 1
        if ang < min_angle:
            min_angle = ang

        if ang < float(angle_threshold_deg):
            sharp += 1

    if used < int(min_used):
        return None, {
            "decision": "NOT_ENOUGH_INTERIOR",
            "used": int(used),
            "sharp": None,
            "min_angle": None,
        }

    ok = (sharp <= int(max_sharp))
    return bool(ok), {
        "decision": "OK",
        "used": int(used),
        "sharp": int(sharp),
        "min_angle": float(min_angle),
        "margin_px": int(margin_px),
        "step": int(step),
        "angle_thr": float(angle_threshold_deg),
        "downsample": int(downsample),
    }


# ---------------------------------------------------------------------
# Edge ring strength (átmozgatva)
# ---------------------------------------------------------------------
def edge_ring_strength_from_roi_gray(roi_gray: np.ndarray, ring_w: int = 5) -> float:
    if roi_gray is None or roi_gray.size == 0:
        return 0.0

    H, W = roi_gray.shape[:2]
    rw = int(max(1, ring_w))
    if H < 2 * rw + 2 or W < 2 * rw + 2:
        return 0.0

    gx = cv2.Sobel(roi_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(roi_gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)

    mask = np.zeros((H, W), dtype=np.uint8)
    mask[:, :] = 255
    mask[rw:H - rw, rw:W - rw] = 0

    vals = mag[mask == 255]
    if vals.size == 0:
        return 0.0

    return float(np.median(vals))


# ---------------------------------------------------------------------
# Full-frame OTSU + largest contour (átmozgatva)
# ---------------------------------------------------------------------
def largest_contour_from_gray_otsu(gray_u8):
    if gray_u8 is None or gray_u8.size == 0:
        return None

    g = cv2.GaussianBlur(gray_u8, (5, 5), 0)
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    fg = np.mean(gray_u8[bw == 255]) if np.any(bw == 255) else 0.0
    bg = np.mean(gray_u8[bw == 0]) if np.any(bw == 0) else 0.0
    if fg < bg:
        bw = 255 - bw

    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)