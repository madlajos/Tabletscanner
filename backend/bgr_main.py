import cv2
import numpy as np
import os
from PIL import Image

def _largest_hull_from_binary(bin_img):
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, 0.0
    c = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    hull = cv2.convexHull(c)
    return hull, area

def largest_object_hull_otsu_mask(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th_inv = cv2.bitwise_not(th)

    # válasszuk azt a binárist, ahol a legnagyobb objektum nem a teljes kép
    img_area = image_bgr.shape[0] * image_bgr.shape[1]

    hull1, area1 = _largest_hull_from_binary(th)
    hull2, area2 = _largest_hull_from_binary(th_inv)

    # ha az egyik "túl nagy" (pl. háttér), preferáljuk a másikat
    # (küszöb: 90% képterület)
    def score(area):  # kisebb jobb, de 0 (nincs kontúr) rossz
        if area <= 0:
            return 1e18
        return area

    # preferáld azt, amelyik < 0.9*img_area, ha van ilyen
    candidates = []
    if hull1 is not None:
        candidates.append(("th", hull1, area1))
    if hull2 is not None:
        candidates.append(("th_inv", hull2, area2))

    if not candidates:
        return None, None, None

    under = [c for c in candidates if c[2] < 0.9 * img_area]
    chosen = min(under, key=lambda x: x[2]) if under else min(candidates, key=lambda x: score(x[2]))

    _, hull, _ = chosen

    # hull maszk + kinullázás hullon kívül
    mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)

    masked = cv2.bitwise_and(image_bgr, image_bgr, mask=mask)
    return masked, hull, mask
def save_bgr_image_keep_exif(
    image_bgr,
    src_image_path,
    dst_image_path,
    quality=95,
    subsampling=0
):
    """
    OpenCV BGR képet ment el JPEG-be úgy, hogy az eredeti EXIF meta megmaradjon.

    Paraméterek:
    - image_bgr: numpy array (OpenCV, BGR)
    - src_image_path: eredeti kép elérési útja (innen vesszük az EXIF-et)
    - dst_image_path: célfájl teljes elérési útja (.jpg)
    """

    os.makedirs(os.path.dirname(dst_image_path), exist_ok=True)

    # BGR → RGB → PIL
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    out_img = Image.fromarray(rgb)

    # EXIF betöltése (ha van)
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

import os
import cv2

def process_folder(input_dir):
    """
    Egy mappában lévő képeket feldolgoz:
    - Otsu + legnagyobb objektum konvex hull
    - hullon kívül minden pixel 0
    - mentés input_dir/masked almappába
    - EXIF meta megőrzése

    Visszatér: feldolgozott fájlok száma
    """

    output_dir = os.path.join(input_dir, "masked")
    os.makedirs(output_dir, exist_ok=True)

    valid_exts = {".jpg", ".jpeg", ".JPG", ".JPEG"}
    count = 0

    for fname in os.listdir(input_dir):
        ext = os.path.splitext(fname)[1]
        if ext not in valid_exts:
            continue

        input_path = os.path.join(input_dir, fname)
        if not os.path.isfile(input_path):
            continue

        img = cv2.imread(input_path)
        if img is None:
            continue

        masked, hull, _ = largest_object_hull_otsu_mask(img)
        if masked is None:
            continue

        base = os.path.splitext(fname)[0]
        output_path = os.path.join(output_dir, f"{base}_masked.jpg")

        save_bgr_image_keep_exif(
            image_bgr=masked,
            src_image_path=input_path,
            dst_image_path=output_path
        )

        count += 1

    return count
