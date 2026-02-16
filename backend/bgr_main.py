import cv2
import numpy as np
import os
from PIL import Image

# ==========================================================
# CONFIG
# ==========================================================
AF_FRAME_SCALE = 0.1


# ============================
# keep these (must stay)
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

    out_dir = os.path.dirname(dst_image_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    out_img = Image.fromarray(rgb)

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
# unified error format (same as AF)
# ============================
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


# ============================
# (A) Autofocus contour -> mask (SIMPLE, FAST)
# ============================
def _mask_from_autofocus_contour(image_bgr, contour_points, make_convex=True):
    """
    LOGIKA NEM VÁLTOZIK.
    Return: (mask or None, used_cnt or None, metrics)
    """
    if contour_points is None:
        return None, None, _err("E2020")  # no contour

    # (eredetiben ez nem volt védve; itt csak hibakódot adunk, nem változtatunk a logikán)
    if image_bgr is None or getattr(image_bgr, "size", 0) == 0:
        return None, None, _err("E2025")  # empty image

    try:
        H, W = image_bgr.shape[:2]
    except Exception:
        return None, None, _err("E2025")  # bad image shape

    try:
        pts = np.asarray(contour_points)
    except Exception as e:
        return None, None, _err("E2021", error="Contour to array failed", exc=str(e))

    # Accept (N,2) or (N,1,2)  (LOGIKA MARAD)
    if pts.ndim == 2 and pts.shape[1] == 2:
        cnt = pts.reshape((-1, 1, 2)).astype(np.float32)
    elif pts.ndim == 3 and pts.shape[1] == 1 and pts.shape[2] == 2:
        cnt = pts.astype(np.float32)
    else:
        return None, None, _err("E2021", error="Bad contour shape", shape=str(getattr(pts, "shape", None)))

    if len(cnt) < 3:
        return None, None, _err("E2022", n=int(len(cnt)))

    # ---- scale back to full-res ---- (LOGIKA MARAD)
    try:
        s = float(AF_FRAME_SCALE)
    except Exception:
        return None, None, _err("E2023", af_frame_scale=str(AF_FRAME_SCALE))

    if s <= 0:
        return None, None, _err("E2023", af_frame_scale=float(s))

    if s != 1.0:
        cnt[:, 0, 0] = cnt[:, 0, 0] / s
        cnt[:, 0, 1] = cnt[:, 0, 1] / s

    # clamp into image bounds (LOGIKA MARAD)
    cnt[:, 0, 0] = np.clip(cnt[:, 0, 0], 0, W - 1)
    cnt[:, 0, 1] = np.clip(cnt[:, 0, 1], 0, H - 1)

    cnt_i32 = np.round(cnt).astype(np.int32)

    if make_convex:
        try:
            cnt_i32 = cv2.convexHull(cnt_i32)
        except Exception as e:
            return None, None, _err("E2021", error="convexHull failed", exc=str(e))

    mask = np.zeros((H, W), dtype=np.uint8)
    try:
        cv2.fillPoly(mask, [cnt_i32], 255)
    except Exception as e:
        return None, None, _err("E2021", error="fillPoly failed", exc=str(e))

    if np.count_nonzero(mask) == 0:
        return None, None, _err("E2024")  # empty mask

    return mask, cnt_i32, _ok(points=int(len(cnt_i32)), af_frame_scale=float(AF_FRAME_SCALE))


# ============================
# PUBLIC API: called by app.py
# ============================
def make_object_mask_from_bgr_rel(image_bgr, autofocus_contour=None):
    """
    LOGIKA MARAD:
      - ha van autofocus_contour -> abból maszk
      - ha nincs -> error (mask=None, kind="none")
    Return: (mask_255, kind, metrics)
    """
    if autofocus_contour is None:
        return None, "none", _err("E2020")

    mask, used_cnt, metrics = _mask_from_autofocus_contour(
        image_bgr, autofocus_contour, make_convex=True
    )

    if mask is None or not np.any(mask):
        # LOGIKA MARAD: fail
        # ha belül már volt konkrét ok, azt visszaadjuk
        if not isinstance(metrics, dict) or metrics.get("status") != "ERROR":
            metrics = _err("E2024")
        return None, "none", metrics

    # OK ág (ugyanaz, csak egységes metrics)
    if not isinstance(metrics, dict) or metrics.get("status") != "OK":
        metrics = _ok(points=int(len(used_cnt)), af_frame_scale=float(AF_FRAME_SCALE))

    return mask, "autofocus", metrics
