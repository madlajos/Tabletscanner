"""
Pipeline step catalog: definitions and execution functions for all steps.

Each step is registered via STEP_DEFINITIONS (id -> StepDefinition) and
STEP_EXECUTORS (id -> callable).

Executor signature (data-dict pattern):
    def execute(data: dict, params: dict) -> dict

The data dict flows through the entire pipeline, accumulating images,
results, metadata, and history.  Processing elements live in
proc_elements/ and are wrapped here with parameter mapping.
"""
from pipeline_types import (
    DataType, ParamSchema, StepDefinition,
)
from proc_elements import (
    load_image as _pe_load_image,
    select_channel as _pe_select_channel,
    apply_threshold as _pe_apply_threshold,
    calculate_histograms as _pe_calculate_histograms,
    apply_range_mask as _pe_apply_range_mask,
    calculate_intensity_stats as _pe_calculate_intensity_stats,
    add_sequence_values as _pe_add_sequence_values,
    fit_curve as _pe_fit_curve,
    predict_node as _pe_predict_node,
    apply_blur as _pe_apply_blur,
    histogram_equalization as _pe_histogram_equalization,
    apply_clahe as _pe_apply_clahe,
    normalize_images as _pe_normalize_images,
    adjust_brightness_contrast as _pe_adjust_brightness_contrast,
    gamma_correction as _pe_gamma_correction,
    flat_field_correction as _pe_flat_field_correction,
    robust_stretch_gamma as _pe_robust_stretch_gamma,
    advanced_illumin_corr as _pe_advanced_illumin_corr,
    mask_rect_roi as _pe_mask_rect_roi,
)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

STEP_DEFINITIONS: dict[str, StepDefinition] = {}
STEP_EXECUTORS: dict[str, callable] = {}


def _register(defn: StepDefinition, executor):
    STEP_DEFINITIONS[defn.id] = defn
    STEP_EXECUTORS[defn.id] = executor


# ---------------------------------------------------------------------------
# 1. Load Image  (load_img.py)
# ---------------------------------------------------------------------------
_load_image_def = StepDefinition(
    id="load_image",
    name="Kép betöltése",
    category="io",
    description="Kép(ek) betöltése fájlból vagy mappából. Ez az első lépés minden feldolgozási láncban.",
    icon="image",
    input_type=DataType.IMAGE,
    output_type=DataType.IMAGE,
    params=[
        ParamSchema(name="source", label="Forrás útvonal", type="file_path",
                    default="", required=True,
                    description="Képfájl vagy mappa elérési útja"),
        ParamSchema(name="file_order", label="Sorrend", type="string",
                    default="", required=False,
                    description="Képek egyéni sorrendje (vesszővel elválasztott indexek)"),
    ],
    side_output_types={"count": "SCALAR", "loaded_paths": "SCALAR"},
)


def _exec_load_image(data: dict, params: dict) -> dict:
    path = params.get("source", "")
    if not path:
        from proc_elements import create_data
        data = create_data()
        data["error"] = "E2002"
        return data
    result = _pe_load_image(path)
    # Apply custom image order if specified
    order_str = params.get("file_order", "")
    if order_str and result.get("images") and not result.get("error"):
        try:
            indices = [int(x.strip()) for x in order_str.split(",") if x.strip()]
            n = len(result["images"])
            # Validate indices
            if indices and all(0 <= i < n for i in indices) and len(indices) == n:
                result["images"] = [result["images"][i] for i in indices]
                result["paths"] = [result["paths"][i] for i in indices]
        except (ValueError, IndexError):
            pass  # Ignore invalid order, keep original
    return result


_register(_load_image_def, _exec_load_image)


# ---------------------------------------------------------------------------
# 2. Select Channel  (select_channel.py)
# ---------------------------------------------------------------------------
_select_channel_def = StepDefinition(
    id="select_channel",
    name="Színtér konverzió",
    category="adjustment",
    description="Színtér konverzió és csatorna kiválasztás (BGR, HSV, LAB, szürkeárnyalat).",
    icon="palette",
    input_type=DataType.IMAGE,
    output_type=DataType.GRAYSCALE,
    params=[
        ParamSchema(name="space", label="Színtér", type="enum",
                    default="GRAY",
                    options=["BGR", "HSV", "LAB", "GRAY"]),
        ParamSchema(name="channel", label="Csatorna", type="enum",
                    default="GRAY",
                    options=["R", "G", "B", "H", "S", "V", "L", "A", "GRAY"],
                    description="A kiválasztott csatorna a megadott színtérből"),
    ],
)


def _exec_select_channel(data: dict, params: dict) -> dict:
    space = params.get("space", "GRAY")
    channel = params.get("channel", "GRAY")
    return _pe_select_channel(data, space=space, channel=channel)


_register(_select_channel_def, _exec_select_channel)


# ---------------------------------------------------------------------------
# 3. Apply Threshold  (apply_thresh.py)
# ---------------------------------------------------------------------------
_apply_threshold_def = StepDefinition(
    id="apply_threshold",
    name="Küszöbölés",
    category="filter",
    description="Bináris küszöbölés. A bemenet szürkeárnyalatos kép kell legyen.",
    icon="tonality",
    input_type=DataType.GRAYSCALE,
    output_type=DataType.MASK,
    params=[
        ParamSchema(name="thresh", label="Küszöbérték", type="int",
                    default=127, min=0, max=255, step=1),
        ParamSchema(name="maxval", label="Max. érték", type="int",
                    default=255, min=0, max=255, step=1),
        ParamSchema(name="mode", label="Mód", type="enum",
                    default="binary",
                    options=["binary", "binary_inv", "trunc", "tozero", "tozero_inv"]),
    ],
)


def _exec_apply_threshold(data: dict, params: dict) -> dict:
    thresh = int(params.get("thresh", 127))
    maxval = int(params.get("maxval", 255))
    mode = params.get("mode", "binary")
    return _pe_apply_threshold(data, thresh=thresh, maxval=maxval, mode=mode)


_register(_apply_threshold_def, _exec_apply_threshold)


# ---------------------------------------------------------------------------
# 4. Calculate Histograms  (generate_histogram.py)
# ---------------------------------------------------------------------------
_calculate_histograms_def = StepDefinition(
    id="calculate_histograms",
    name="Hisztogram",
    category="analysis",
    description="Hisztogram számítás szürkeárnyalatos képekhez. A kép változatlan marad, az eredmény mellékadatként jelenik meg.",
    icon="bar_chart",
    input_type=DataType.GRAYSCALE,
    output_type=DataType.GRAYSCALE,
    params=[
        ParamSchema(name="bins", label="Osztások száma", type="int",
                    default=256, min=2, max=1024, step=1),
        ParamSchema(name="range_min", label="Tartomány min", type="int",
                    default=0, min=0, max=65535, step=1),
        ParamSchema(name="range_max", label="Tartomány max", type="int",
                    default=256, min=1, max=65536, step=1),
    ],
    side_output_types={"histograms": "HISTOGRAM"},
)


def _exec_calculate_histograms(data: dict, params: dict) -> dict:
    bins = int(params.get("bins", 256))
    range_min = int(params.get("range_min", 0))
    range_max = int(params.get("range_max", 256))
    return _pe_calculate_histograms(data, bins=bins, hist_range=(range_min, range_max))


_register(_calculate_histograms_def, _exec_calculate_histograms)


# ---------------------------------------------------------------------------
# 5. Apply Range Mask  (range_mask.py)
# ---------------------------------------------------------------------------
_apply_range_mask_def = StepDefinition(
    id="apply_range_mask",
    name="Tartomány maszk",
    category="filter",
    description="Intenzitás tartomány alapú maszkolás. A tartományon kívüli pixelek nullázódnak.",
    icon="filter_alt",
    input_type=DataType.GRAYSCALE,
    output_type=DataType.GRAYSCALE,
    params=[
        ParamSchema(name="low", label="Alsó határ", type="int",
                    default=0, min=0, max=255, step=1),
        ParamSchema(name="high", label="Felső határ", type="int",
                    default=255, min=0, max=255, step=1),
        ParamSchema(name="keep_mode", label="Megtartás", type="enum",
                    default="inside",
                    options=["inside", "outside"],
                    description="Belső: tartományon belüli, Külső: tartományon kívüli pixelek"),
    ],
    side_output_types={"range_masks": "MASK"},
)


def _exec_apply_range_mask(data: dict, params: dict) -> dict:
    low = int(params.get("low", 0))
    high = int(params.get("high", 255))
    keep_mode = params.get("keep_mode", "inside")
    return _pe_apply_range_mask(data, low=low, high=high, keep_mode=keep_mode)


_register(_apply_range_mask_def, _exec_apply_range_mask)


# ---------------------------------------------------------------------------
# 6. Calculate Intensity Stats  (calc_intensity.py)
# ---------------------------------------------------------------------------
_calculate_intensity_stats_def = StepDefinition(
    id="calculate_intensity_stats",
    name="Intenzitás statisztikák",
    category="analysis",
    description="Intenzitás statisztikák számítása a maszkolt területen (min, max, átlag, medián, szórás, percentilisek).",
    icon="analytics",
    input_type=DataType.GRAYSCALE,
    output_type=DataType.GRAYSCALE,
    params=[
        ParamSchema(name="percentiles", label="Percentilisek", type="string",
                    default="5,25,50,75,95", required=True,
                    description="Vesszővel elválasztott percentilis értékek (0-100)"),
    ],
    side_output_types={"intensity_stats": "SCALAR"},
)


def _exec_calculate_intensity_stats(data: dict, params: dict) -> dict:
    pct_str = params.get("percentiles", "5,25,50,75,95")
    try:
        percentiles = tuple(float(x.strip()) for x in str(pct_str).split(",") if x.strip())
    except (ValueError, TypeError):
        percentiles = (5, 25, 50, 75, 95)
    return _pe_calculate_intensity_stats(data, percentiles=percentiles)


_register(_calculate_intensity_stats_def, _exec_calculate_intensity_stats)


# ---------------------------------------------------------------------------
# 7. Add Sequence Values  (add_measured.py)
# ---------------------------------------------------------------------------
_add_sequence_values_def = StepDefinition(
    id="add_sequence_values",
    name="Szekvencia értékek",
    category="io",
    description="Mért vagy generált értéksorozat hozzárendelése a képekhez (pl. expozíciós idő, hőmérséklet).",
    icon="pin",
    input_type=DataType.IMAGE,
    output_type=DataType.IMAGE,
    params=[
        ParamSchema(name="name", label="Változó neve", type="string",
                    default="sequence_value", required=True,
                    description="Az értéksorozat azonosítója"),
        ParamSchema(name="mode", label="Generálás módja", type="enum",
                    default="start_step",
                    options=["start_step", "start_stop", "explicit"],
                    description="start_step: kezdőérték+lépésköz, start_stop: egyenletes elosztás, explicit: kézi értékek"),
        ParamSchema(name="start", label="Kezdőérték", type="float",
                    default=0.0, min=-1e9, max=1e9, step=0.1),
        ParamSchema(name="step_val", label="Lépésköz", type="float",
                    default=1.0, min=-1e9, max=1e9, step=0.1,
                    description="Használatos start_step módban"),
        ParamSchema(name="stop", label="Végérték", type="float",
                    default=100.0, min=-1e9, max=1e9, step=0.1,
                    description="Használatos start_stop módban"),
        ParamSchema(name="values", label="Explicit értékek", type="string",
                    default="", required=False,
                    description="Vesszővel elválasztott értékek (explicit módban)"),
    ],
)


def _exec_add_sequence_values(data: dict, params: dict) -> dict:
    name = params.get("name", "sequence_value")
    mode = params.get("mode", "start_step")
    start = float(params.get("start", 0.0))
    step_val = float(params.get("step_val", 1.0))
    stop = float(params.get("stop", 100.0))
    values_str = params.get("values", "")

    if mode == "explicit":
        try:
            values = [float(x.strip()) for x in str(values_str).split(",") if x.strip()]
        except (ValueError, TypeError):
            data["error"] = "E2634"
            return data
        return _pe_add_sequence_values(data, name=name, values=values)
    elif mode == "start_step":
        return _pe_add_sequence_values(data, name=name, start=start, step=step_val)
    elif mode == "start_stop":
        return _pe_add_sequence_values(data, name=name, start=start, stop=stop)
    else:
        data["error"] = "E2638"
        return data


_register(_add_sequence_values_def, _exec_add_sequence_values)


# ---------------------------------------------------------------------------
# 8. Fit Curve  (curve_fitting.py)
# ---------------------------------------------------------------------------
_fit_curve_def = StepDefinition(
    id="fit_curve",
    name="Görbe illesztés",
    category="analysis",
    description="Lineáris vagy polinomiális görbe illesztése a szekvencia értékek és az intenzitás statisztikák alapján.",
    icon="show_chart",
    input_type=DataType.IMAGE,
    output_type=DataType.IMAGE,
    params=[
        ParamSchema(name="x_name", label="X tengely (értéknév)", type="string",
                    default="sequence_value", required=True,
                    description="A results-ben tárolt szekvencia változó neve"),
        ParamSchema(name="y_name", label="Y tengely (statisztika)", type="enum",
                    default="mean",
                    options=["mean", "median", "min", "max", "std", "p5", "p25", "p50", "p75", "p95", "dynamic_range"],
                    description="Intenzitás statisztika mező neve"),
        ParamSchema(name="model", label="Illesztési modell", type="enum",
                    default="linear",
                    options=["linear", "poly"]),
        ParamSchema(name="degree", label="Polinom fok", type="int",
                    default=2, min=1, max=10, step=1,
                    description="Polinom illesztés fokszáma (poly módban)"),
    ],
    side_output_types={"r2": "SCALAR", "coefficients": "SCALAR"},
)


def _exec_fit_curve(data: dict, params: dict) -> dict:
    x_name = params.get("x_name", "sequence_value")
    y_name = params.get("y_name", "mean")
    model = params.get("model", "linear")
    degree = int(params.get("degree", 2))
    return _pe_fit_curve(data, x_name=x_name, y_name=y_name, model=model, degree=degree)


_register(_fit_curve_def, _exec_fit_curve)


# ---------------------------------------------------------------------------
# 9. Predict from Intensity  (pred_from_int.py)
# ---------------------------------------------------------------------------
_predict_node_def = StepDefinition(
    id="predict_node",
    name="Predikció",
    category="analysis",
    description="Előrejelzés a korábban illesztett görbe alapján. A pipeline saját curve_fits eredményét használja modellként.",
    icon="trending_up",
    input_type=DataType.IMAGE,
    output_type=DataType.IMAGE,
    params=[
        ParamSchema(name="fit_index", label="Görbe illesztés index", type="int",
                    default=0, min=0, max=100, step=1,
                    description="Melyik korábban illesztett görbét használja (0-tól)"),
    ],
    side_output_types={"predictions": "SCALAR"},
)


def _exec_predict_node(data: dict, params: dict) -> dict:
    fit_index = int(params.get("fit_index", 0))
    # In a linear pipeline, model_data = data (same pipeline's curve_fits)
    return _pe_predict_node(model_data=data, input_data=data, fit_index=fit_index)


_register(_predict_node_def, _exec_predict_node)


# ---------------------------------------------------------------------------
# 10. Apply Blur  (apply_blur.py)
# ---------------------------------------------------------------------------
_apply_blur_def = StepDefinition(
    id="apply_blur",
    name="Elmosás",
    category="filter",
    description="Gauss, medián, átlag vagy bilaterális elmosás.",
    icon="blur_on",
    input_type=DataType.IMAGE,
    output_type=DataType.IMAGE,
    params=[
        ParamSchema(name="method", label="Módszer", type="enum",
                    default="gaussian",
                    options=["gaussian", "median", "bilateral", "average"]),
        ParamSchema(name="ksize", label="Kernel méret", type="int",
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


def _exec_apply_blur(data: dict, params: dict) -> dict:
    method = params.get("method", "gaussian")
    ksize = int(params.get("ksize", 5))
    sigma = float(params.get("sigma", 0.0))
    sigma_color = float(params.get("sigma_color", 75.0))
    sigma_space = float(params.get("sigma_space", 75.0))
    return _pe_apply_blur(data, method=method, ksize=ksize, sigma=sigma,
                          sigma_color=sigma_color, sigma_space=sigma_space)


_register(_apply_blur_def, _exec_apply_blur)


# ---------------------------------------------------------------------------
# 11. Histogram Equalization  (histogram_eq.py)
# ---------------------------------------------------------------------------
_histogram_eq_def = StepDefinition(
    id="histogram_equalization",
    name="Hisztogram kiegyenlítés",
    category="adjustment",
    description="Hisztogram kiegyenlítés szürkeárnyalatos képekhez. Javítja a kontrasztot.",
    icon="equalizer",
    input_type=DataType.GRAYSCALE,
    output_type=DataType.GRAYSCALE,
    params=[],
)


def _exec_histogram_eq(data: dict, params: dict) -> dict:
    return _pe_histogram_equalization(data)


_register(_histogram_eq_def, _exec_histogram_eq)


# ---------------------------------------------------------------------------
# 12. CLAHE  (clahe.py)
# ---------------------------------------------------------------------------
_clahe_def = StepDefinition(
    id="apply_clahe",
    name="CLAHE",
    category="adjustment",
    description="Kontrasztkorlátos adaptív hisztogram kiegyenlítés (CLAHE). Szürkeárnyalatos képekhez.",
    icon="auto_fix_high",
    input_type=DataType.GRAYSCALE,
    output_type=DataType.GRAYSCALE,
    params=[
        ParamSchema(name="clip_limit", label="Levágási határ", type="float",
                    default=2.0, min=0.1, max=40.0, step=0.1),
        ParamSchema(name="tile_x", label="Rács X", type="int",
                    default=8, min=1, max=64, step=1,
                    description="Csempe rács vízszintes mérete"),
        ParamSchema(name="tile_y", label="Rács Y", type="int",
                    default=8, min=1, max=64, step=1,
                    description="Csempe rács függőleges mérete"),
    ],
)


def _exec_clahe(data: dict, params: dict) -> dict:
    clip_limit = float(params.get("clip_limit", 2.0))
    tile_x = int(params.get("tile_x", 8))
    tile_y = int(params.get("tile_y", 8))
    return _pe_apply_clahe(data, clip_limit=clip_limit, tile_grid_size=(tile_x, tile_y))


_register(_clahe_def, _exec_clahe)


# ---------------------------------------------------------------------------
# 13. Normalize  (normalization.py)
# ---------------------------------------------------------------------------
_normalize_def = StepDefinition(
    id="normalize_images",
    name="Normalizálás",
    category="adjustment",
    description="Kép normalizálás (MinMax, L1, L2). Az intenzitás tartományt az alpha-beta közé skálázza.",
    icon="tune",
    input_type=DataType.IMAGE,
    output_type=DataType.IMAGE,
    params=[
        ParamSchema(name="alpha", label="Alpha (min)", type="float",
                    default=0.0, min=0.0, max=255.0, step=1.0),
        ParamSchema(name="beta", label="Beta (max)", type="float",
                    default=255.0, min=0.0, max=255.0, step=1.0),
        ParamSchema(name="norm_type", label="Típus", type="enum",
                    default="minmax",
                    options=["minmax", "l1", "l2"]),
    ],
)


def _exec_normalize(data: dict, params: dict) -> dict:
    alpha = float(params.get("alpha", 0.0))
    beta = float(params.get("beta", 255.0))
    norm_type = params.get("norm_type", "minmax")
    return _pe_normalize_images(data, alpha=alpha, beta=beta, norm_type=norm_type)


_register(_normalize_def, _exec_normalize)


# ---------------------------------------------------------------------------
# 14. Brightness / Contrast  (bright_contr.py)
# ---------------------------------------------------------------------------
_brightness_contrast_def = StepDefinition(
    id="brightness_contrast",
    name="Fényerő / Kontraszt",
    category="adjustment",
    description="Fényerő és kontraszt beállítás. A kontraszt szorzóként, a fényerő hozzáadásként hat.",
    icon="brightness_6",
    input_type=DataType.IMAGE,
    output_type=DataType.IMAGE,
    params=[
        ParamSchema(name="brightness", label="Fényerő", type="int",
                    default=0, min=-255, max=255, step=1),
        ParamSchema(name="contrast", label="Kontraszt", type="float",
                    default=1.0, min=0.1, max=5.0, step=0.1),
    ],
)


def _exec_brightness_contrast(data: dict, params: dict) -> dict:
    brightness = int(params.get("brightness", 0))
    contrast = float(params.get("contrast", 1.0))
    return _pe_adjust_brightness_contrast(data, brightness=brightness, contrast=contrast)


_register(_brightness_contrast_def, _exec_brightness_contrast)


# ---------------------------------------------------------------------------
# 15. Gamma Correction  (gamma_corr.py)
# ---------------------------------------------------------------------------
_gamma_corr_def = StepDefinition(
    id="gamma_correction",
    name="Gamma korrekció",
    category="adjustment",
    description="Gamma korrekció LUT táblával. Gamma < 1 világosít, > 1 sötétít.",
    icon="contrast",
    input_type=DataType.IMAGE,
    output_type=DataType.IMAGE,
    params=[
        ParamSchema(name="gamma", label="Gamma", type="float",
                    default=1.0, min=0.1, max=5.0, step=0.05),
    ],
)


def _exec_gamma_corr(data: dict, params: dict) -> dict:
    gamma = float(params.get("gamma", 1.0))
    return _pe_gamma_correction(data, gamma=gamma)


_register(_gamma_corr_def, _exec_gamma_corr)


# ---------------------------------------------------------------------------
# 16. Flat-Field Correction  (flat_field_corr.py)
# ---------------------------------------------------------------------------
_flat_field_def = StepDefinition(
    id="flat_field_correction",
    name="Flat-field korrekció",
    category="filter",
    description="Megvilágítás egyenetlenség korrekció szürkeárnyalatos képekhez. Háttérbecslés + flat-field osztás.",
    icon="wb_sunny",
    input_type=DataType.GRAYSCALE,
    output_type=DataType.GRAYSCALE,
    params=[
        ParamSchema(name="method", label="Háttér módszer", type="enum",
                    default="downsampled",
                    options=["gaussian", "downsampled"]),
        ParamSchema(name="alpha", label="Korrekció erősség", type="float",
                    default=1.0, min=0.1, max=5.0, step=0.1),
        ParamSchema(name="use_nlm", label="NLM zajszűrés", type="bool",
                    default=False,
                    description="fastNlMeansDenoising alkalmazása a korrekció előtt"),
        ParamSchema(name="nlm_h", label="NLM erősség (h)", type="float",
                    default=3.0, min=0.0, max=30.0, step=0.5),
        ParamSchema(name="sigma_bg", label="Háttér sigma", type="float",
                    default=80.0, min=1.0, max=500.0, step=1.0,
                    description="Gaussian módszer sigma értéke"),
        ParamSchema(name="down", label="Lekicsinyítési arány", type="float",
                    default=0.1, min=0.01, max=1.0, step=0.01,
                    description="Downsampled módszer kicsinyítési aránya"),
        ParamSchema(name="sigma_small", label="Kis kép sigma", type="float",
                    default=8.0, min=0.5, max=100.0, step=0.5),
        ParamSchema(name="final_blur", label="Végső simítás", type="float",
                    default=3.0, min=0.0, max=50.0, step=0.5),
        ParamSchema(name="percentile_low", label="Percentilis alsó", type="float",
                    default=1.0, min=0.0, max=49.0, step=0.5),
        ParamSchema(name="percentile_high", label="Percentilis felső", type="float",
                    default=99.0, min=51.0, max=100.0, step=0.5),
    ],
    side_output_types={"background_images": "IMAGE"},
)


def _exec_flat_field(data: dict, params: dict) -> dict:
    return _pe_flat_field_correction(
        data,
        method=params.get("method", "downsampled"),
        alpha=float(params.get("alpha", 1.0)),
        use_nlm=bool(params.get("use_nlm", False)),
        nlm_h=float(params.get("nlm_h", 3.0)),
        sigma_bg=float(params.get("sigma_bg", 80.0)),
        down=float(params.get("down", 0.1)),
        sigma_small=float(params.get("sigma_small", 8.0)),
        final_blur=float(params.get("final_blur", 3.0)),
        percentile_low=float(params.get("percentile_low", 1.0)),
        percentile_high=float(params.get("percentile_high", 99.0)),
    )


_register(_flat_field_def, _exec_flat_field)


# ---------------------------------------------------------------------------
# 17. Robust Stretch + Gamma  (robust_stretch.py)
# ---------------------------------------------------------------------------
_robust_stretch_def = StepDefinition(
    id="robust_stretch_gamma",
    name="Robusztus nyújtás + Gamma",
    category="adjustment",
    description="Percentilis alapú kontraszt nyújtás opcionális gamma korrekcióval. Szürkeárnyalatos képekhez.",
    icon="expand",
    input_type=DataType.GRAYSCALE,
    output_type=DataType.GRAYSCALE,
    params=[
        ParamSchema(name="percentile_low", label="Alsó percentilis", type="float",
                    default=2.0, min=0.0, max=49.0, step=0.5),
        ParamSchema(name="percentile_high", label="Felső percentilis", type="float",
                    default=98.0, min=51.0, max=100.0, step=0.5),
        ParamSchema(name="gamma", label="Gamma", type="float",
                    default=1.0, min=0.1, max=5.0, step=0.05),
    ],
)


def _exec_robust_stretch(data: dict, params: dict) -> dict:
    percentile_low = float(params.get("percentile_low", 2.0))
    percentile_high = float(params.get("percentile_high", 98.0))
    gamma = float(params.get("gamma", 1.0))
    return _pe_robust_stretch_gamma(data, percentile_low=percentile_low,
                                    percentile_high=percentile_high, gamma=gamma)


_register(_robust_stretch_def, _exec_robust_stretch)


# ---------------------------------------------------------------------------
# 18. Advanced Illumination Correction  (advanced_ill_corr.py)
# ---------------------------------------------------------------------------
_advanced_illum_def = StepDefinition(
    id="advanced_illumin_corr",
    name="Haladó megvilágítás korrekció",
    category="filter",
    description="Komplex megvilágítás korrekció: NLM zajszűrés + flat-field + normalizálás + percentilis skálázás + gamma.",
    icon="auto_awesome",
    input_type=DataType.GRAYSCALE,
    output_type=DataType.GRAYSCALE,
    params=[
        ParamSchema(name="bg_method", label="Háttér módszer", type="enum",
                    default="downsampled",
                    options=["gaussian", "downsampled"]),
        ParamSchema(name="alpha", label="Korrekció erősség", type="float",
                    default=1.0, min=0.1, max=5.0, step=0.1),
        ParamSchema(name="use_nlm", label="NLM zajszűrés", type="bool",
                    default=False),
        ParamSchema(name="nlm_h", label="NLM erősség (h)", type="float",
                    default=3.0, min=0.0, max=30.0, step=0.5),
        ParamSchema(name="sigma_bg", label="Háttér sigma", type="float",
                    default=80.0, min=1.0, max=500.0, step=1.0),
        ParamSchema(name="down", label="Lekicsinyítési arány", type="float",
                    default=0.1, min=0.01, max=1.0, step=0.01),
        ParamSchema(name="sigma_small", label="Kis kép sigma", type="float",
                    default=8.0, min=0.5, max=100.0, step=0.5),
        ParamSchema(name="final_blur", label="Végső simítás", type="float",
                    default=3.0, min=0.0, max=50.0, step=0.5),
        ParamSchema(name="percentile_low", label="Percentilis alsó", type="float",
                    default=1.0, min=0.0, max=49.0, step=0.5),
        ParamSchema(name="percentile_high", label="Percentilis felső", type="float",
                    default=99.0, min=51.0, max=100.0, step=0.5),
        ParamSchema(name="use_gamma", label="Gamma használata", type="bool",
                    default=False),
        ParamSchema(name="gamma", label="Gamma", type="float",
                    default=1.0, min=0.1, max=5.0, step=0.05),
    ],
    side_output_types={"advanced_illum_backgrounds": "IMAGE", "advanced_illum_denoised": "IMAGE"},
)


def _exec_advanced_illum(data: dict, params: dict) -> dict:
    return _pe_advanced_illumin_corr(
        data,
        bg_method=params.get("bg_method", "downsampled"),
        alpha=float(params.get("alpha", 1.0)),
        use_nlm=bool(params.get("use_nlm", False)),
        nlm_h=float(params.get("nlm_h", 3.0)),
        sigma_bg=float(params.get("sigma_bg", 80.0)),
        down=float(params.get("down", 0.1)),
        sigma_small=float(params.get("sigma_small", 8.0)),
        final_blur=float(params.get("final_blur", 3.0)),
        percentile_low=float(params.get("percentile_low", 1.0)),
        percentile_high=float(params.get("percentile_high", 99.0)),
        use_gamma=bool(params.get("use_gamma", False)),
        gamma=float(params.get("gamma", 1.0)),
    )


_register(_advanced_illum_def, _exec_advanced_illum)


# ---------------------------------------------------------------------------
# 19. Mask Rect ROI  (draw_roi.py)
# ---------------------------------------------------------------------------
_mask_roi_def = StepDefinition(
    id="mask_rect_roi",
    name="Téglalap ROI maszk",
    category="filter",
    description="Téglalap alakú érdeklődési terület (ROI) maszkolás. A területen kívüli pixelek a háttérszínnel töltődnek.",
    icon="crop",
    input_type=DataType.IMAGE,
    output_type=DataType.IMAGE,
    params=[
        ParamSchema(name="roi_x", label="X kezdőpont", type="int",
                    default=0, min=0, max=100000, step=1),
        ParamSchema(name="roi_y", label="Y kezdőpont", type="int",
                    default=0, min=0, max=100000, step=1),
        ParamSchema(name="roi_width", label="Szélesség", type="int",
                    default=100, min=1, max=100000, step=1),
        ParamSchema(name="roi_height", label="Magasság", type="int",
                    default=100, min=1, max=100000, step=1),
        ParamSchema(name="background_color", label="Háttérszín", type="int",
                    default=0, min=0, max=255, step=1),
        ParamSchema(name="keep_outside", label="Külső megtartása", type="bool",
                    default=False,
                    description="True: a ROI-n kívüli rész marad; False: a ROI-n belüli rész marad"),
    ],
    side_output_types={"roi_masks": "MASK"},
)


def _exec_mask_roi(data: dict, params: dict) -> dict:
    roi = {
        "type": "rect",
        "x": int(params.get("roi_x", 0)),
        "y": int(params.get("roi_y", 0)),
        "width": int(params.get("roi_width", 100)),
        "height": int(params.get("roi_height", 100)),
    }
    background_color = int(params.get("background_color", 0))
    keep_outside = bool(params.get("keep_outside", False))
    return _pe_mask_rect_roi(data, roi=roi, background_color=background_color,
                             keep_outside=keep_outside)


_register(_mask_roi_def, _exec_mask_roi)
