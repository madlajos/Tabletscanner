"""
Pipeline step catalog: definitions and execution functions for all V1 steps.

Each step is registered via STEP_REGISTRY (id -> StepDefinition) and
STEP_EXECUTORS (id -> callable).

Executor signature:
    def execute(input_image: np.ndarray | None, params: dict) -> StepResult

All executors must:
 - Accept BGR uint8 or grayscale uint8 (auto-converted by engine)
 - Return StepResult with primary_output as numpy array
 - Never raise — return StepResult(success=False) with warnings instead
"""
import cv2
import numpy as np
import os
from pipeline_types import (
    DataType, ParamSchema, StepDefinition, StepResult,
)

# ---------------------------------------------------------------------------
# Step definitions (catalog)
# ---------------------------------------------------------------------------

STEP_DEFINITIONS: dict[str, StepDefinition] = {}
STEP_EXECUTORS: dict[str, callable] = {}


def _register(defn: StepDefinition, executor):
    STEP_DEFINITIONS[defn.id] = defn
    STEP_EXECUTORS[defn.id] = executor


# ---------------------------------------------------------------------------
# 1. Load Image
# ---------------------------------------------------------------------------
_load_image_def = StepDefinition(
    id="load_image",
    name="Kép betöltése",
    category="io",
    description="Kép betöltése fájlból. Ez az első lépés minden feldolgozási láncban.",
    icon="image",
    input_type=DataType.IMAGE,
    output_type=DataType.IMAGE,
    params=[
        ParamSchema(name="source", label="Forrásfájl", type="file_path",
                    default="", required=True,
                    description="A betöltendő képfájl elérési útja"),
    ],
    side_output_types={"width": "SCALAR", "height": "SCALAR", "channels": "SCALAR"},
)

def _exec_load_image(_input, params: dict) -> StepResult:
    path = params.get("source", "")
    if not path or not os.path.isfile(path):
        return StepResult(success=False, warnings=[f"A fájl nem található: {path}"])
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return StepResult(success=False, warnings=[f"A kép nem olvasható: {path}"])
    h, w = img.shape[:2]
    ch = img.shape[2] if img.ndim == 3 else 1
    return StepResult(
        success=True, primary_output=img, output_type=DataType.IMAGE,
        side_outputs={"width": w, "height": h, "channels": ch},
    )

_register(_load_image_def, _exec_load_image)


# ---------------------------------------------------------------------------
# 2. Brightness / Gamma
# ---------------------------------------------------------------------------
_brightness_gamma_def = StepDefinition(
    id="brightness_gamma",
    name="Fényerő / Gamma",
    category="adjustment",
    description="Fényerő és gamma korrekció. A fényerő hozzáadódik, a gamma hatványozással módosít.",
    icon="brightness_6",
    input_type=DataType.IMAGE,
    output_type=DataType.IMAGE,
    params=[
        ParamSchema(name="brightness", label="Fényerő", type="int",
                    default=0, min=-100, max=100, step=1),
        ParamSchema(name="contrast", label="Kontraszt", type="float",
                    default=1.0, min=0.1, max=3.0, step=0.1),
        ParamSchema(name="gamma", label="Gamma", type="float",
                    default=1.0, min=0.1, max=5.0, step=0.1),
    ],
)

def _exec_brightness_gamma(img: np.ndarray, params: dict) -> StepResult:
    brightness = int(params.get("brightness", 0))
    contrast = float(params.get("contrast", 1.0))
    gamma = float(params.get("gamma", 1.0))

    # Contrast and brightness via convertScaleAbs
    out = cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)

    # Gamma correction via LUT
    if abs(gamma - 1.0) > 1e-3:
        inv_gamma = 1.0 / max(gamma, 0.01)
        table = np.array(
            [(i / 255.0) ** inv_gamma * 255 for i in range(256)]
        ).astype("uint8")
        out = cv2.LUT(out, table)

    return StepResult(success=True, primary_output=out, output_type=DataType.IMAGE)

_register(_brightness_gamma_def, _exec_brightness_gamma)


# ---------------------------------------------------------------------------
# 3. Threshold
# ---------------------------------------------------------------------------
_threshold_def = StepDefinition(
    id="threshold",
    name="Küszöbölés",
    category="analysis",
    description="Bináris vagy adaptív küszöbölés. Az eredmény egy maszk (0/255).",
    icon="tonality",
    input_type=DataType.GRAYSCALE,
    output_type=DataType.MASK,
    params=[
        ParamSchema(name="method", label="Módszer", type="enum",
                    default="binary", options=["binary", "binary_inv", "otsu", "adaptive_mean", "adaptive_gaussian"]),
        ParamSchema(name="thresh_value", label="Küszöbérték", type="int",
                    default=128, min=0, max=255, step=1,
                    description="Használatos binary/binary_inv esetén"),
        ParamSchema(name="block_size", label="Blokk méret", type="int",
                    default=11, min=3, max=255, step=2, odd_only=True,
                    description="Adaptív módszerekhez (páratlan szám)"),
        ParamSchema(name="c_value", label="C konstans", type="int",
                    default=2, min=-50, max=50, step=1,
                    description="Adaptív módszerekhez levonandó konstans"),
    ],
    side_output_types={"threshold_used": "SCALAR"},
)

def _exec_threshold(img: np.ndarray, params: dict) -> StepResult:
    method = params.get("method", "binary")
    thresh_val = int(params.get("thresh_value", 128))
    block_size = int(params.get("block_size", 11))
    c_val = int(params.get("c_value", 2))

    # Ensure grayscale
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Ensure block_size is odd and >= 3
    if block_size < 3:
        block_size = 3
    if block_size % 2 == 0:
        block_size += 1

    used_thresh = thresh_val

    if method == "binary":
        _, out = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
    elif method == "binary_inv":
        _, out = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
    elif method == "otsu":
        used_thresh, out = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        used_thresh = float(used_thresh)
    elif method == "adaptive_mean":
        out = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY, block_size, c_val)
    elif method == "adaptive_gaussian":
        out = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, block_size, c_val)
    else:
        return StepResult(success=False, warnings=[f"Ismeretlen küszöbölési módszer: {method}"])

    return StepResult(
        success=True, primary_output=out, output_type=DataType.MASK,
        side_outputs={"threshold_used": used_thresh},
    )

_register(_threshold_def, _exec_threshold)


# ---------------------------------------------------------------------------
# 4. Histogram
# ---------------------------------------------------------------------------
_histogram_def = StepDefinition(
    id="histogram",
    name="Hisztogram",
    category="analysis",
    description="Hisztogram számítás csatornánként. A kép változatlan marad, az eredmény mellékadatként jelenik meg.",
    icon="bar_chart",
    input_type=DataType.IMAGE,
    output_type=DataType.IMAGE,
    params=[
        ParamSchema(name="bins", label="Osztások száma", type="int",
                    default=256, min=2, max=256, step=1),
    ],
    side_output_types={"histogram": "HISTOGRAM"},
)

def _exec_histogram(img: np.ndarray, params: dict) -> StepResult:
    bins = int(params.get("bins", 256))
    bins = max(2, min(256, bins))

    histograms = {}
    if img.ndim == 3 and img.shape[2] == 3:
        for i, color in enumerate(["blue", "green", "red"]):
            hist = cv2.calcHist([img], [i], None, [bins], [0, 256])
            histograms[color] = hist.flatten().tolist()
    else:
        hist = cv2.calcHist([img], [0], None, [bins], [0, 256])
        histograms["gray"] = hist.flatten().tolist()

    return StepResult(
        success=True, primary_output=img.copy(), output_type=DataType.IMAGE,
        side_outputs={"histogram": histograms},
    )

_register(_histogram_def, _exec_histogram)


# ---------------------------------------------------------------------------
# 5. Color / Brightness Statistics
# ---------------------------------------------------------------------------
_color_stats_def = StepDefinition(
    id="color_stats",
    name="Színstatisztikák",
    category="analysis",
    description="Átlagos fényerő, szórás, min/max csatornánként. Numerikus eredmény, a kép nem változik.",
    icon="analytics",
    input_type=DataType.IMAGE,
    output_type=DataType.IMAGE,
    params=[],
    side_output_types={
        "mean_brightness": "SCALAR",
        "std_brightness": "SCALAR",
        "per_channel_mean": "HISTOGRAM",
        "per_channel_std": "HISTOGRAM",
    },
)

def _exec_color_stats(img: np.ndarray, params: dict) -> StepResult:
    if img.ndim == 3 and img.shape[2] >= 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    mean_br = float(np.mean(gray))
    std_br = float(np.std(gray))

    per_ch_mean = {}
    per_ch_std = {}
    if img.ndim == 3 and img.shape[2] == 3:
        for i, ch_name in enumerate(["blue", "green", "red"]):
            per_ch_mean[ch_name] = round(float(np.mean(img[:, :, i])), 2)
            per_ch_std[ch_name] = round(float(np.std(img[:, :, i])), 2)
    else:
        per_ch_mean["gray"] = round(mean_br, 2)
        per_ch_std["gray"] = round(std_br, 2)

    return StepResult(
        success=True, primary_output=img.copy(), output_type=DataType.IMAGE,
        side_outputs={
            "mean_brightness": round(mean_br, 2),
            "std_brightness": round(std_br, 2),
            "per_channel_mean": per_ch_mean,
            "per_channel_std": per_ch_std,
        },
    )

_register(_color_stats_def, _exec_color_stats)


# ---------------------------------------------------------------------------
# 6. Blur
# ---------------------------------------------------------------------------
_blur_def = StepDefinition(
    id="blur",
    name="Elmosás",
    category="filter",
    description="Gauss, medián vagy bilaterális elmosás.",
    icon="blur_on",
    input_type=DataType.IMAGE,
    output_type=DataType.IMAGE,
    params=[
        ParamSchema(name="method", label="Módszer", type="enum",
                    default="gaussian", options=["gaussian", "median", "bilateral"]),
        ParamSchema(name="kernel_size", label="Kernel méret", type="int",
                    default=5, min=1, max=99, step=2, odd_only=True),
        ParamSchema(name="sigma", label="Sigma", type="float",
                    default=0.0, min=0.0, max=50.0, step=0.5,
                    description="Gauss sigma (0 = automatikus)"),
        ParamSchema(name="sigma_color", label="Szín sigma", type="float",
                    default=75.0, min=1.0, max=300.0, step=1.0,
                    description="Bilaterális szűrő szín sigma"),
        ParamSchema(name="sigma_space", label="Tér sigma", type="float",
                    default=75.0, min=1.0, max=300.0, step=1.0,
                    description="Bilaterális szűrő térbeli sigma"),
    ],
)

def _exec_blur(img: np.ndarray, params: dict) -> StepResult:
    method = params.get("method", "gaussian")
    ksize = int(params.get("kernel_size", 5))
    sigma = float(params.get("sigma", 0.0))
    sigma_color = float(params.get("sigma_color", 75.0))
    sigma_space = float(params.get("sigma_space", 75.0))

    # Ensure odd kernel
    if ksize < 1:
        ksize = 1
    if ksize % 2 == 0:
        ksize += 1

    if method == "gaussian":
        out = cv2.GaussianBlur(img, (ksize, ksize), sigma)
    elif method == "median":
        out = cv2.medianBlur(img, ksize)
    elif method == "bilateral":
        out = cv2.bilateralFilter(img, ksize, sigma_color, sigma_space)
    else:
        return StepResult(success=False, warnings=[f"Ismeretlen elmosási módszer: {method}"])

    return StepResult(success=True, primary_output=out, output_type=DataType.IMAGE)

_register(_blur_def, _exec_blur)


# ---------------------------------------------------------------------------
# 7. Edge Detection
# ---------------------------------------------------------------------------
_edge_detection_def = StepDefinition(
    id="edge_detection",
    name="Éldetektálás",
    category="detection",
    description="Canny, Sobel vagy Laplacian éldetektálás.",
    icon="gradient",
    input_type=DataType.GRAYSCALE,
    output_type=DataType.GRAYSCALE,
    params=[
        ParamSchema(name="method", label="Módszer", type="enum",
                    default="canny", options=["canny", "sobel", "laplacian"]),
        ParamSchema(name="threshold1", label="Alsó küszöb", type="int",
                    default=50, min=0, max=500, step=1,
                    description="Canny alsó küszöb"),
        ParamSchema(name="threshold2", label="Felső küszöb", type="int",
                    default=150, min=0, max=500, step=1,
                    description="Canny felső küszöb"),
        ParamSchema(name="ksize", label="Kernel méret", type="int",
                    default=3, min=1, max=31, step=2, odd_only=True,
                    description="Sobel/Laplacian kernel méret"),
    ],
    side_output_types={"edge_pixel_count": "SCALAR", "edge_ratio": "SCALAR"},
)

def _exec_edge_detection(img: np.ndarray, params: dict) -> StepResult:
    method = params.get("method", "canny")
    t1 = int(params.get("threshold1", 50))
    t2 = int(params.get("threshold2", 150))
    ksize = int(params.get("ksize", 3))

    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if ksize < 1:
        ksize = 1
    if ksize % 2 == 0:
        ksize += 1
    ksize = min(ksize, 31)

    if method == "canny":
        out = cv2.Canny(gray, t1, t2)
    elif method == "sobel":
        sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
        sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
        mag = cv2.magnitude(sx, sy)
        out = np.clip(mag, 0, 255).astype(np.uint8)
    elif method == "laplacian":
        lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize)
        out = np.clip(np.abs(lap), 0, 255).astype(np.uint8)
    else:
        return StepResult(success=False, warnings=[f"Ismeretlen éldetektálási módszer: {method}"])

    edge_count = int(np.count_nonzero(out))
    total = out.shape[0] * out.shape[1]
    edge_ratio = round(edge_count / max(total, 1), 4)

    return StepResult(
        success=True, primary_output=out, output_type=DataType.GRAYSCALE,
        side_outputs={"edge_pixel_count": edge_count, "edge_ratio": edge_ratio},
    )

_register(_edge_detection_def, _exec_edge_detection)


# ---------------------------------------------------------------------------
# 8. Channel Operations
# ---------------------------------------------------------------------------
_channel_ops_def = StepDefinition(
    id="channel_ops",
    name="Csatorna műveletek",
    category="adjustment",
    description="Szín csatorna kiválasztása (B/G/R/H/S/V) vagy szürkeárnyalatossá alakítás.",
    icon="palette",
    input_type=DataType.IMAGE,
    output_type=DataType.GRAYSCALE,
    params=[
        ParamSchema(name="channel", label="Csatorna", type="enum",
                    default="gray",
                    options=["gray", "blue", "green", "red", "hue", "saturation", "value"]),
    ],
)

def _exec_channel_ops(img: np.ndarray, params: dict) -> StepResult:
    channel = params.get("channel", "gray")

    if img.ndim == 2:
        # Already grayscale — return as-is for most channels
        return StepResult(success=True, primary_output=img.copy(), output_type=DataType.GRAYSCALE)

    if channel == "gray":
        out = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif channel == "blue":
        out = img[:, :, 0]
    elif channel == "green":
        out = img[:, :, 1]
    elif channel == "red":
        out = img[:, :, 2]
    elif channel in ("hue", "saturation", "value"):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        idx = {"hue": 0, "saturation": 1, "value": 2}[channel]
        out = hsv[:, :, idx]
    else:
        return StepResult(success=False, warnings=[f"Ismeretlen csatorna: {channel}"])

    return StepResult(success=True, primary_output=out.copy(), output_type=DataType.GRAYSCALE)

_register(_channel_ops_def, _exec_channel_ops)


# ---------------------------------------------------------------------------
# 9. Crop / ROI
# ---------------------------------------------------------------------------
_crop_roi_def = StepDefinition(
    id="crop_roi",
    name="Kivágás (ROI)",
    category="adjustment",
    description="Téglalap alakú kivágás a képből. A koordináták automatikusan a kép méretéhez igazodnak.",
    icon="crop",
    input_type=DataType.IMAGE,
    output_type=DataType.IMAGE,
    params=[
        ParamSchema(name="x", label="X kezdőpont", type="int", default=0, min=0, max=100000, step=1),
        ParamSchema(name="y", label="Y kezdőpont", type="int", default=0, min=0, max=100000, step=1),
        ParamSchema(name="width", label="Szélesség", type="int", default=100, min=1, max=100000, step=1),
        ParamSchema(name="height", label="Magasság", type="int", default=100, min=1, max=100000, step=1),
    ],
    side_output_types={"actual_width": "SCALAR", "actual_height": "SCALAR"},
)

def _exec_crop_roi(img: np.ndarray, params: dict) -> StepResult:
    h, w = img.shape[:2]
    x = int(params.get("x", 0))
    y = int(params.get("y", 0))
    cw = int(params.get("width", 100))
    ch = int(params.get("height", 100))

    # Clamp to image bounds
    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))
    x2 = max(x + 1, min(x + cw, w))
    y2 = max(y + 1, min(y + ch, h))

    out = img[y:y2, x:x2].copy()
    return StepResult(
        success=True, primary_output=out, output_type=DataType.IMAGE,
        side_outputs={"actual_width": x2 - x, "actual_height": y2 - y},
    )

_register(_crop_roi_def, _exec_crop_roi)


# ---------------------------------------------------------------------------
# 10. Contour Detection
# ---------------------------------------------------------------------------
_contour_detection_def = StepDefinition(
    id="contour_detection",
    name="Kontúr detektálás",
    category="detection",
    description="Kontúrok keresése és megjelenítése. Az eredmény a kontúrokat rárajzolja a képre.",
    icon="polyline",
    input_type=DataType.IMAGE,
    output_type=DataType.IMAGE,
    params=[
        ParamSchema(name="mode", label="Keresési mód", type="enum",
                    default="external", options=["external", "list", "tree"]),
        ParamSchema(name="min_area", label="Min. terület", type="int",
                    default=100, min=0, max=1000000, step=10,
                    description="Kontúrok szűrése minimális terület alapján"),
        ParamSchema(name="draw_color_b", label="Szín B", type="int", default=0, min=0, max=255, step=1),
        ParamSchema(name="draw_color_g", label="Szín G", type="int", default=255, min=0, max=255, step=1),
        ParamSchema(name="draw_color_r", label="Szín R", type="int", default=0, min=0, max=255, step=1),
        ParamSchema(name="thickness", label="Vastagság", type="int", default=2, min=1, max=20, step=1),
    ],
    side_output_types={
        "contour_count": "SCALAR",
        "total_area": "SCALAR",
        "areas": "HISTOGRAM",
    },
)

def _exec_contour_detection(img: np.ndarray, params: dict) -> StepResult:
    mode_str = params.get("mode", "external")
    min_area = int(params.get("min_area", 100))
    color_b = int(params.get("draw_color_b", 0))
    color_g = int(params.get("draw_color_g", 255))
    color_r = int(params.get("draw_color_r", 0))
    thickness = int(params.get("thickness", 2))

    mode_map = {
        "external": cv2.RETR_EXTERNAL,
        "list": cv2.RETR_LIST,
        "tree": cv2.RETR_TREE,
    }
    mode = mode_map.get(mode_str, cv2.RETR_EXTERNAL)

    # Need a binary image for findContours
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, mode, cv2.CHAIN_APPROX_SIMPLE)

    # Filter by area
    filtered = [c for c in contours if cv2.contourArea(c) >= min_area]

    # Draw on a copy (ensure BGR for colored drawing)
    if img.ndim == 2:
        canvas = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        canvas = img.copy()

    cv2.drawContours(canvas, filtered, -1, (color_b, color_g, color_r), thickness)

    areas = [round(cv2.contourArea(c), 1) for c in filtered]
    total_area = round(sum(areas), 1)

    return StepResult(
        success=True, primary_output=canvas, output_type=DataType.IMAGE,
        side_outputs={
            "contour_count": len(filtered),
            "total_area": total_area,
            "areas": areas,
        },
    )

_register(_contour_detection_def, _exec_contour_detection)


# ---------------------------------------------------------------------------
# 11. Save / Export
# ---------------------------------------------------------------------------
_save_export_def = StepDefinition(
    id="save_export",
    name="Mentés",
    category="io",
    description="Eredmény kép mentése fájlba. A kép változatlanul továbbhalad.",
    icon="save",
    input_type=DataType.IMAGE,
    output_type=DataType.IMAGE,
    params=[
        ParamSchema(name="output_path", label="Kimeneti útvonal", type="file_path",
                    default="", required=True,
                    description="A mentendő fájl elérési útja"),
        ParamSchema(name="quality", label="JPEG minőség", type="int",
                    default=95, min=1, max=100, step=1),
    ],
    side_output_types={"saved_path": "SCALAR"},
)

def _exec_save_export(img: np.ndarray, params: dict) -> StepResult:
    path = params.get("output_path", "")
    quality = int(params.get("quality", 95))

    if not path:
        return StepResult(success=False, warnings=["Nincs megadva kimeneti útvonal"])

    # Ensure directory exists
    directory = os.path.dirname(path)
    if directory and not os.path.isdir(directory):
        try:
            os.makedirs(directory, exist_ok=True)
        except OSError as e:
            return StepResult(success=False, warnings=[f"Könyvtár létrehozása sikertelen: {e}"])

    ext = os.path.splitext(path)[1].lower()
    encode_params = []
    if ext in (".jpg", ".jpeg"):
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    elif ext == ".png":
        encode_params = [cv2.IMWRITE_PNG_COMPRESSION, max(0, min(9, (100 - quality) // 11))]

    success = cv2.imwrite(path, img, encode_params)
    if not success:
        return StepResult(success=False, warnings=[f"Kép mentése sikertelen: {path}"])

    return StepResult(
        success=True, primary_output=img.copy(), output_type=DataType.IMAGE,
        side_outputs={"saved_path": path},
    )

_register(_save_export_def, _exec_save_export)
