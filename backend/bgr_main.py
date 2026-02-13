
import cv2
import numpy as np
import os
from PIL import Image

# ==========================================================
# CONFIG
# ==========================================================
# Az autofocus_main_new-ben a kontúr a frame_scale-es képen keletkezik.
# Nálad default: frame_scale=0.1
AF_FRAME_SCALE = 0.1

# ============================
# keep these (you said must stay)
# ============================
def apply_mask_zero_background(image_bgr, mask_255):
    """mask_255: uint8 0/255"""
    if mask_255 is None:
        return None
    return cv2.bitwise_and(image_bgr, image_bgr, mask=mask_255)

def save_bgr_image_keep_exif(
    image_bgr,
    src_image_path,
    dst_image_path,
    quality=95,
    subsampling=0
):
    """
    OpenCV BGR képet ment el JPEG-be úgy,
    hogy az eredeti EXIF meta megmaradjon.
    """

    # célmappa biztosítása
    out_dir = os.path.dirname(dst_image_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # BGR → RGB → PIL
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    out_img = Image.fromarray(rgb)

    # EXIF betöltése az eredeti képből
    exif_bytes = None
    try:
        with Image.open(src_image_path) as src:
            exif_bytes = src.info.get("exif", None)
    except Exception:
        pass

    save_kwargs = dict(format="JPEG", quality=quality, subsampling=subsampling)

    if exif_bytes is not None:
        save_kwargs["exif"] = exif_bytes

    out_img.save(dst_image_path, **save_kwargs)

# ============================
# (A) Autofocus contour -> mask (SIMPLE, FAST)
# ============================
def _mask_from_autofocus_contour(image_bgr, contour_points, make_convex=True):
    """
    contour_points: (N,2) vagy OpenCV contour (N,1,2) vagy list of points.
    FONTOS: a kontúr az AF-ben frame_scale-elt képről jön, ezért itt visszaskálázzuk full-res-re.
    """
    if contour_points is None:
        return None, None

    H, W = image_bgr.shape[:2]
    pts = np.asarray(contour_points)

    # Accept (N,2) or (N,1,2)
    if pts.ndim == 2 and pts.shape[1] == 2:
        cnt = pts.reshape((-1, 1, 2)).astype(np.float32)
    elif pts.ndim == 3 and pts.shape[1] == 1 and pts.shape[2] == 2:
        cnt = pts.astype(np.float32)
    else:
        return None, None

    if len(cnt) < 3:
        return None, None

    # ---- scale back to full-res ----
    s = float(AF_FRAME_SCALE)
    if s <= 0:
        return None, None

    if s != 1.0:
        cnt[:, 0, 0] = cnt[:, 0, 0] / s
        cnt[:, 0, 1] = cnt[:, 0, 1] / s

    # clamp into image bounds
    cnt[:, 0, 0] = np.clip(cnt[:, 0, 0], 0, W - 1)
    cnt[:, 0, 1] = np.clip(cnt[:, 0, 1], 0, H - 1)

    cnt_i32 = np.round(cnt).astype(np.int32)

    if make_convex:
        cnt_i32 = cv2.convexHull(cnt_i32)

    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [cnt_i32], 255)

    if np.count_nonzero(mask) == 0:
        return None, None

    return mask, cnt_i32


# ============================
# PUBLIC API: called by app.py
# ============================
def make_object_mask_from_bgr_rel(image_bgr, autofocus_contour=None):
    """
    app.py ezt hívja.
    Vissza: (mask_255, kind, metrics)

    Ebben az egyszerűsített verzióban:
      - ha van autofocus_contour -> abból maszk (visszaskálázva full-res-re)
      - ha nincs -> error (mask=None, kind="none")
    """
    if autofocus_contour is None:
        return None, "none", {"error": "No autofocus contour provided"}

    mask, used_cnt = _mask_from_autofocus_contour(image_bgr, autofocus_contour, make_convex=True)
    if mask is None or not np.any(mask):
        return None, "none", {"error": "Contour->mask failed"}

    return mask, "autofocus", {"points": int(len(used_cnt)), "af_frame_scale": float(AF_FRAME_SCALE)}