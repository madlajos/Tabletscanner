import cv2
import numpy as np

def _err(code: str, **extra):
    d = {"status": "ERROR", "code": str(code)}
    if extra:
        d.update(extra)
    return d


def _ok(**extra):
    d = {"status": "OK"}
    if extra:
        d.update(extra)
    return d

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
    mask[rw:H - rw, rw:W - rw] = 0  # keep only border ring

    vals = mag[mask == 255]
    if vals.size == 0:
        return 0.0

    return float(np.median(vals))

def crop_bbox(frame_bgr, bbox):
    x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
    roi = frame_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    return roi


def bbox_roi_gray(frame_bgr, bbox_state):
    roi = crop_bbox(frame_bgr, bbox_state)
    if roi is None:
        return None
    if roi.ndim == 2:
        roi = roi
    else:
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return roi

def grayscale_difference_score(img_bgr, blur_ksize=5):
    if img_bgr is None or img_bgr.size == 0:
        return _err("E2005")

    # --- grayscale konverzió ---
    if img_bgr.ndim == 2:
        gray = img_bgr
    elif img_bgr.ndim == 3 and img_bgr.shape[2] == 3:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    elif img_bgr.ndim == 3 and img_bgr.shape[2] == 4:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2GRAY)
    else:
        return _err("E2005")

    # --- blur ---
    if blur_ksize and blur_ksize > 0:
        gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    gray = gray.astype(np.float32)

    mean_gray = float(np.mean(gray))
    std_gray = float(np.std(gray))
    mean_abs_diff = float(np.mean(np.abs(gray - mean_gray)))
    min_gray = float(np.min(gray))
    max_gray = float(np.max(gray))

    print(std_gray, mean_abs_diff, min_gray, max_gray)

    # -----------------------------
    # Gate logika
    # -----------------------------
    if std_gray < 10 and mean_abs_diff <= 5:
        return _err("E2000",
                    std_gray=std_gray,
                    mean_abs_diff=mean_abs_diff)

    if 5 < mean_abs_diff < 10:
        return _err("E2002",
                    std_gray=std_gray,
                    mean_abs_diff=mean_abs_diff)

    if mean_abs_diff > 100:
        return _err("E2003",
                    std_gray=std_gray,
                    mean_abs_diff=mean_abs_diff)

    # --- OK ---
    return _ok(
        std_gray=std_gray,
        mean_abs_diff=mean_abs_diff,
        min_gray=min_gray,
        max_gray=max_gray
    )


def final_out_of_frame_check(
    frame,
    debug=False,
    debug_buffer=None,
    margin_px=2,
    min_area_ratio=0.001,
    bbox_state=None,
    edge_ring_width=5,
    min_edge_strength=None,
    return_contour=False,   # ha kell: OK/ERR mellé kontúr
):
    # -----------------------------
    # Edge strength check (opcionális)
    # -----------------------------
    if (min_edge_strength is not None) and (bbox_state is not None):
        roi_g = bbox_roi_gray(frame, bbox_state)
        edge_strength = edge_ring_strength_from_roi_gray(
            roi_g, ring_w=int(edge_ring_width)
        )
        print(f"[FINAL_EDGE] edge_strength={edge_strength:.4f} "
              f"(min={float(min_edge_strength):.4f})")

        if edge_strength < float(min_edge_strength):
            # csak ERROR dict
            return _err("E2112", edge_strength=float(edge_strength))

    # -----------------------------
    # Teljes képes OTSU szegmentálás
    # -----------------------------
    if frame is None or frame.size == 0:
        return _err("E2110")  # opcionális: "no frame"

    if frame.ndim == 2:
        gray = frame
    elif frame.ndim == 3 and frame.shape[2] == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif frame.ndim == 3 and frame.shape[2] == 4:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
    else:
        return _err("E2113")

    H, W = gray.shape[:2]

    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    fg = np.mean(gray[bw == 255]) if np.any(bw == 255) else 0
    bg = np.mean(gray[bw == 0]) if np.any(bw == 0) else 0
    if fg < bg:
        bw = 255 - bw

    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return _err("E2114")

    c = max(contours, key=cv2.contourArea)

    area = float(cv2.contourArea(c))
    if area < (float(min_area_ratio) * H * W):
        # kicsi objektum -> OK
        if return_contour:
            return _ok(area=area, note="small_area_ok", contour=c)
        return _ok(area=area, note="small_area_ok")

    x, y, w, h = cv2.boundingRect(c)

    left   = (x <= margin_px)
    top    = (y <= margin_px)
    right  = ((x + w) >= (W - 1 - margin_px))
    bottom = ((y + h) >= (H - 1 - margin_px))

    touch_count = int(left) + int(top) + int(right) + int(bottom)

    print(f"[FRAME_TOUCH] L={left} T={top} R={right} B={bottom} count={touch_count}")

    # ---- 0 oldal -> OK ----
    if touch_count == 0:
        if return_contour:
            return _ok(touch_count=touch_count, bbox=(x, y, w, h), contour=c)
        return _ok(touch_count=touch_count, bbox=(x, y, w, h))

    # ---- 1 oldal -> ERROR ----
    if touch_count == 1:
        return _err("E2004", touch_count=touch_count, bbox=(x, y, w, h))

    # ---- 2 oldal ----
    if touch_count == 2:
        opposite_ok = (left and right) or (top and bottom)
        if opposite_ok:
            if return_contour:
                return _ok(touch_count=touch_count, bbox=(x, y, w, h), contour=c)
            return _ok(touch_count=touch_count, bbox=(x, y, w, h))
        else:
            return _err("E2004", touch_count=touch_count, bbox=(x, y, w, h))

    # ---- 3 oldal -> ERROR ----
    if touch_count == 3:
        return _err("E2004", touch_count=touch_count, bbox=(x, y, w, h))

    # ---- 4 oldal -> OK ----
    if touch_count == 4:
        if return_contour:
            return _ok(touch_count=touch_count, bbox=(x, y, w, h), contour=c)
        return _ok(touch_count=touch_count, bbox=(x, y, w, h))

    # fallback
    if return_contour:
        return _ok(touch_count=touch_count, bbox=(x, y, w, h), contour=c)
    return _ok(touch_count=touch_count, bbox=(x, y, w, h))