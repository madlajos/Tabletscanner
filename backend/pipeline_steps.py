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
import json

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
    mask_roi as _pe_mask_roi,
    resize_images as _pe_resize_images,
    detect_particles as _pe_detect_particles,
    histogram_pca as _pe_histogram_pca,
    detect_circles as _pe_detect_circles,
    characterize_particles as _pe_characterize_particles,
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
    name="Referencia értékek",
    category="io",
    description="Mért vagy generált értéksorozat hozzárendelése a képekhez (pl. expozíciós idő, hőmérséklet).",
    icon="pin",
    input_type=DataType.IMAGE,
    output_type=DataType.IMAGE,
    params=[
        ParamSchema(name="name", label="Változó neve", type="string",
                    default="sequence_value", required=True,
                    description="Az értéksorozat azonosítója"),
        ParamSchema(name="values", label="X értékek", type="string",
                    default="", required=False,
                    description="Vesszővel elválasztott szintértékek (pl. 1,2,3)"),
        ParamSchema(name="num_levels", label="Szintek száma", type="int",
                    default=5, min=1, max=10000, step=1, required=False,
                    description="Hány különböző szintet rendelünk a képekhez"),
        ParamSchema(name="start", label="Kezdőérték", type="float",
                    default=0.0, min=-1e9, max=1e9, step=0.1),
        ParamSchema(name="step_val", label="Lépésköz", type="float",
                    default=1.0, min=-1e9, max=1e9, step=0.1),
        ParamSchema(name="group_colors", label="Csoport színek", type="string",
                    default="", required=False,
                    description="JSON szín térkép az egyedi értékekhez (pl. {\"10\":\"#ff0000\"})"),
    ],
)


def _exec_add_sequence_values(data: dict, params: dict) -> dict:
    name = params.get("name", "sequence_value")
    values_str = params.get("values", "")
    group_colors_str = params.get("group_colors", "")
    num_levels = params.get("num_levels", None)
    if num_levels is not None:
        num_levels = int(num_levels)

    group_colors = None
    if isinstance(group_colors_str, str) and group_colors_str.strip():
        try:
            parsed = json.loads(group_colors_str)
            if isinstance(parsed, dict):
                group_colors = parsed
        except (ValueError, TypeError):
            group_colors = None

    # Always use the explicit values string.
    # The frontend provides short-form (unique level values) which we expand
    # by repeating each value (total_images / num_levels) times.
    try:
        unique_vals = [float(x.strip()) for x in str(values_str).split(",") if x.strip()]
    except (ValueError, TypeError):
        data["error"] = "E2634"
        return data

    if not unique_vals:
        data["error"] = "E2634"
        return data

    total_images = data.get("count", 0)
    n_levels = len(unique_vals)
    if total_images > 0 and n_levels > 0:
        samples_per_value = total_images // n_levels
        if samples_per_value < 1:
            samples_per_value = 1
        expanded = []
        for v in unique_vals:
            expanded.extend([v] * samples_per_value)
        # If there are remaining images, assign them to the last level
        while len(expanded) < total_images:
            expanded.append(unique_vals[-1])
        expanded = expanded[:total_images]
    else:
        expanded = unique_vals
        samples_per_value = 1

    return _pe_add_sequence_values(data, name=name, values=expanded,
                                  group_colors=group_colors,
                                  _samples_per_value=samples_per_value)


_register(_add_sequence_values_def, _exec_add_sequence_values)


# ---------------------------------------------------------------------------
# 8. Fit Curve  (curve_fitting.py)
# ---------------------------------------------------------------------------
_fit_curve_def = StepDefinition(
    id="fit_curve",
    name="Görbe illesztés",
    category="analysis",
    description="Lineáris vagy polinomiális görbe illesztése a referencia értékek és az intenzitás statisztikák alapján.",
    icon="show_chart",
    input_type=DataType.IMAGE,
    output_type=DataType.IMAGE,
    params=[
        ParamSchema(name="y_name", label="Y tengely értékei", type="string",
                    default="mean",
                    description="A görbe illesztéshez használt Y mező"),
        ParamSchema(name="y_label", label="Y tengely neve", type="string",
                    default="mean",
                    description="A grafikonon megjelenő Y tengelyfelirat"),
        ParamSchema(name="model", label="Illesztett görbe", type="enum",
                    default="linear",
                    options=["linear", "poly"]),
        ParamSchema(name="degree", label="Polinom fok", type="int",
                    default=2, min=1, max=10, step=1,
                    description="Polinom illesztés fokszáma (poly módban)"),
        ParamSchema(name="aggregate", label="Szintenkénti összevonás", type="bool",
                    default=False, required=False,
                    description="Azonos X értékű pontok összevonása"),
        ParamSchema(name="agg_method", label="Összevont értékek számítása", type="enum",
                    default="mean", options=["mean", "median"], required=False,
                    description="Aggregálás módja (átlag vagy medián)"),
        ParamSchema(name="merge_ab_pairs", label="Tablettaoldalak összevonása", type="bool",
                    default=False, required=False,
                    description="_a/_b utótagú X értékek összevonása"),
    ],
    side_output_types={"r2": "SCALAR", "coefficients": "SCALAR"},
    secondary_inputs=["add_sequence_values"],
)


def _exec_fit_curve(data: dict, params: dict) -> dict:
    x_name = "sequence_value"
    seq_meta = data.get("meta", {}).get("sequence_values", {})
    if isinstance(seq_meta, dict) and seq_meta:
        # Prefer the most recently added reference variable.
        x_name = next(reversed(seq_meta.keys()))
    if x_name not in data.get("results", {}):
        fallback = params.get("x_name")
        if isinstance(fallback, str) and fallback in data.get("results", {}):
            x_name = fallback

    y_name = params.get("y_name", "mean")
    y_label = params.get("y_label")
    model = params.get("model", "linear")
    degree = int(params.get("degree", 2))
    aggregate = bool(params.get("aggregate", False))
    agg_method = params.get("agg_method", "mean")
    merge_ab_pairs = bool(params.get("merge_ab_pairs", False))
    return _pe_fit_curve(data, x_name=x_name, y_name=y_name, model=model,
                         degree=degree, aggregate=aggregate,
                         agg_method=agg_method, merge_ab_pairs=merge_ab_pairs,
                         y_display_name=y_label)


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
    side_output_types={"histeq_input_histograms": "HISTOGRAM", "histeq_output_histograms": "HISTOGRAM"},
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
# 19. ROI Mask  (draw_roi.py)
# ---------------------------------------------------------------------------
_mask_roi_def = StepDefinition(
    id="mask_rect_roi",
    name="ROI beállítása",
    category="filter",
    description="Érdeklődési terület (ROI) beállítása és maszkolás. Téglalap, ellipszis vagy sokszög alakú ROI alkalmazható.",
    icon="crop",
    input_type=DataType.IMAGE,
    output_type=DataType.IMAGE,
    params=[
        ParamSchema(name="roi_type", label="ROI típusa", type="enum",
                    default="rect", options=["rect", "ellipse", "polygon"]),
        # Rectangle params
        ParamSchema(name="roi_x", label="X kezdőpont", type="int",
                    default=0, min=0, max=100000, step=1),
        ParamSchema(name="roi_y", label="Y kezdőpont", type="int",
                    default=0, min=0, max=100000, step=1),
        ParamSchema(name="roi_width", label="Szélesség", type="int",
                    default=0, min=0, max=100000, step=1),
        ParamSchema(name="roi_height", label="Magasság", type="int",
                    default=0, min=0, max=100000, step=1),
        # Ellipse params
        ParamSchema(name="roi_cx", label="Közép X", type="int",
                    default=0, min=0, max=100000, step=1),
        ParamSchema(name="roi_cy", label="Közép Y", type="int",
                    default=0, min=0, max=100000, step=1),
        ParamSchema(name="roi_rx", label="Sugár X", type="int",
                    default=0, min=0, max=100000, step=1),
        ParamSchema(name="roi_ry", label="Sugár Y", type="int",
                    default=0, min=0, max=100000, step=1),
        # Polygon params (JSON string of [{x,y}, ...])
        ParamSchema(name="roi_points", label="Pontok (JSON)", type="string",
                    default="[]"),
    ],
    side_output_types={"roi_masks": "MASK"},
)


def _exec_mask_roi(data: dict, params: dict) -> dict:
    import json as _json
    roi_type = params.get("roi_type", "rect")

    if roi_type == "rect":
        w = int(params.get("roi_width", 0))
        h = int(params.get("roi_height", 0))
        if w <= 0 or h <= 0:
            return data  # No ROI defined
        roi = {
            "type": "rect",
            "x": int(params.get("roi_x", 0)),
            "y": int(params.get("roi_y", 0)),
            "width": w,
            "height": h,
        }
    elif roi_type == "ellipse":
        rx = int(params.get("roi_rx", 0))
        ry = int(params.get("roi_ry", 0))
        if rx <= 0 or ry <= 0:
            return data  # No ROI defined
        roi = {
            "type": "ellipse",
            "cx": int(params.get("roi_cx", 0)),
            "cy": int(params.get("roi_cy", 0)),
            "rx": rx,
            "ry": ry,
        }
    else:
        pts_raw = params.get("roi_points", "[]")
        try:
            pts = _json.loads(pts_raw) if isinstance(pts_raw, str) else pts_raw
        except Exception:
            pts = []
        if not pts or len(pts) < 3:
            return data  # No ROI defined
        roi = {
            "type": "polygon",
            "points": [{'x': int(p.get('x', 0)), 'y': int(p.get('y', 0))} for p in pts],
        }

    return _pe_mask_roi(data, roi=roi)


_register(_mask_roi_def, _exec_mask_roi)


# ---------------------------------------------------------------------------
# 21. Resize Images  (resize_img.py)
# ---------------------------------------------------------------------------
_resize_images_def = StepDefinition(
    id="resize_images",
    name="Képek átméretezése",
    category="adjustment",
    description="Képek átméretezése skálázási aránnyal.",
    icon="photo_size_select_large",
    input_type=DataType.IMAGE,
    output_type=DataType.IMAGE,
    params=[
        ParamSchema(name="scale", label="Skálázási arány", type="float",
                    default=1.0, min=0.0, max=1.0, step=0.01, required=False,
                    description="Skálázási tényező (0-1 csúszka, kézi bevitellel nagyobb is megadható)"),
    ],
)


def _exec_resize_images(data: dict, params: dict) -> dict:
    scale_val = float(params.get("scale", 1.0))
    scale = scale_val if scale_val > 0 else None
    return _pe_resize_images(data, scale=scale)


_register(_resize_images_def, _exec_resize_images)


# ---------------------------------------------------------------------------
# 22. Detect Particles  (region_attr.py)
# ---------------------------------------------------------------------------
_detect_particles_def = StepDefinition(
    id="detect_particles",
    name="Szemcsedetektálás",
    category="detection",
    description="Szemcsék detektálása bináris/maszkolt képeken, polygon és contour adatok előállításával.",
    icon="center_focus_strong",
    input_type=DataType.GRAYSCALE,
    output_type=DataType.MASK,
    params=[
        ParamSchema(name="connectivity", label="Szomszédság", type="enum",
                    default="8", options=["4", "8"],
                    description="4-es vagy 8-as szomszédság"),
        ParamSchema(name="polygon_epsilon", label="Polygon közelítés", type="float",
                    default=0.01, min=0.001, max=0.5, step=0.001,
                    description="Polygon közelítés pontossága (kisebb = pontosabb)"),
        ParamSchema(name="draw", label="Rajzolás", type="bool",
                    default=True,
                    description="Detektált szemcsék körvonalainak kirajzolása"),
        ParamSchema(name="draw_label", label="Címkék rajzolása", type="bool",
                    default=True,
                    description="Szemcse azonosítók megjelenítése"),
        ParamSchema(name="replace_images", label="Overlay csere", type="bool",
                    default=False,
                    description="A kimeneti képek cseréje az overlay képekre"),
    ],
    side_output_types={"particles": "SCALAR", "particles_summary": "SCALAR"},
)


def _exec_detect_particles(data: dict, params: dict) -> dict:
    connectivity = int(params.get("connectivity", "8"))
    polygon_epsilon = float(params.get("polygon_epsilon", 0.01))
    draw = bool(params.get("draw", True))
    draw_label = bool(params.get("draw_label", True))
    replace_images = bool(params.get("replace_images", False))
    return _pe_detect_particles(
        data,
        connectivity=connectivity,
        polygon_epsilon=polygon_epsilon,
        draw=draw,
        draw_label=draw_label,
        replace_images=replace_images,
    )


_register(_detect_particles_def, _exec_detect_particles)


# ---------------------------------------------------------------------------
# 23. Histogram PCA  (create_pca.py)
# ---------------------------------------------------------------------------
_histogram_pca_def = StepDefinition(
    id="histogram_pca",
    name="Hisztogram PCA",
    category="analysis",
    description="Főkomponens-analízis (PCA) a hisztogramok alapján. Előtte a 'Hisztogram' lépés szükséges.",
    icon="scatter_plot",
    input_type=DataType.GRAYSCALE,
    output_type=DataType.GRAYSCALE,
    params=[
        ParamSchema(name="n_components", label="Komponensek száma", type="int",
                    default=2, min=1, max=50, step=1,
                    description="Az előállítandó főkomponensek száma"),
        ParamSchema(name="center", label="Középre igazítás", type="bool",
                    default=True,
                    description="Átlag kivonása a PCA előtt"),
    ],
    side_output_types={
        "histogram_pca_scores": "SCALAR",
        "histogram_pca_explained_ratio": "SCALAR",
    },
    required_preceding_steps=["calculate_histograms"],
)


def _exec_histogram_pca(data: dict, params: dict) -> dict:
    n_components = int(params.get("n_components", 2))
    center = bool(params.get("center", True))
    return _pe_histogram_pca(data, n_components=n_components, center=center)


_register(_histogram_pca_def, _exec_histogram_pca)


# ---------------------------------------------------------------------------
# 24. Detect Circles  (detect_circ.py)
# ---------------------------------------------------------------------------
_detect_circles_def = StepDefinition(
    id="detect_circles",
    name="Kör detektálás",
    category="detection",
    description="Kör alakú objektumok detektálása Hough-transzformációval.",
    icon="radio_button_unchecked",
    input_type=DataType.IMAGE,
    output_type=DataType.IMAGE,
    params=[
        ParamSchema(name="dp", label="Felbontás arány", type="float",
                    default=1.2, min=0.5, max=5.0, step=0.1,
                    description="Az akkumulátor felbontásának inverz aránya"),
        ParamSchema(name="min_dist", label="Min. távolság", type="int",
                    default=20, min=1, max=1000, step=1,
                    description="Minimális távolság a detektált körök középpontjai között"),
        ParamSchema(name="min_radius", label="Min. sugár", type="int",
                    default=20, min=1, max=5000, step=1,
                    description="Minimális kör sugár pixelben"),
        ParamSchema(name="max_radius", label="Max. sugár", type="int",
                    default=25, min=1, max=5000, step=1,
                    description="Maximális kör sugár pixelben"),
        ParamSchema(name="blur_ksize", label="Elmosás kernel", type="int",
                    default=5, min=1, max=99, step=2, odd_only=True,
                    description="Medián elmosás kernel mérete"),
        ParamSchema(name="edge_threshold", label="Él küszöb", type="int",
                    default=100, min=1, max=500, step=1,
                    description="Canny él-detektálás felső küszöbe"),
        ParamSchema(name="accumulator_threshold", label="Akkumulátor küszöb", type="int",
                    default=20, min=1, max=300, step=1,
                    description="Akkumulátor küszöbérték a kör jelöltek elfogadásához"),
        ParamSchema(name="polarity", label="Polaritás", type="enum",
                    default="dark",
                    options=["dark", "bright", "both"],
                    description="Sötét körök világos háttéren / világos körök sötét háttéren / mindkettő"),
    ],
    side_output_types={"circles": "SCALAR"},
)


def _exec_detect_circles(data: dict, params: dict) -> dict:
    dp = float(params.get("dp", 1.2))
    min_dist = int(params.get("min_dist", 20))
    min_radius = int(params.get("min_radius", 20))
    max_radius = int(params.get("max_radius", 25))
    blur_ksize = int(params.get("blur_ksize", 5))
    edge_threshold = int(params.get("edge_threshold", 100))
    accumulator_threshold = int(params.get("accumulator_threshold", 20))
    polarity = params.get("polarity", "dark")
    return _pe_detect_circles(
        data,
        dp=dp,
        min_dist=min_dist,
        min_radius=min_radius,
        max_radius=max_radius,
        blur_ksize=blur_ksize,
        edge_threshold=edge_threshold,
        accumulator_threshold=accumulator_threshold,
        polarity=polarity,
    )


_register(_detect_circles_def, _exec_detect_circles)


# ---------------------------------------------------------------------------
# 25. Characterize Particles  (filter_region.py)
# ---------------------------------------------------------------------------
_characterize_particles_def = StepDefinition(
    id="characterize_particles",
    name="Szemcsekarakterizálás",
    category="analysis",
    description="Detektált szemcsék geometriai és intenzitás alapú karakterizálása, táblázatos eredménnyel.",
    icon="table_chart",
    input_type=DataType.GRAYSCALE,
    output_type=DataType.GRAYSCALE,
    params=[
        ParamSchema(name="include_excluded", label="Kizárt szemcsék számítása", type="bool",
                    default=False,
                    description="A kizárt szemcsék is bekerüljenek a karakterizálásba"),
        ParamSchema(name="pixel_size_um", label="Pixel méret (µm)", type="float",
                    default=0.0, min=0.0, max=1000.0, step=0.01,
                    description="Pixel méret mikronban (0 = pixel egység)"),
        ParamSchema(name="percentiles", label="Percentilisek", type="string",
                    default="5,25,50,75,95", required=True,
                    description="Vesszővel elválasztott percentilis értékek (0-100)"),
        ParamSchema(name="draw", label="Rajzolás", type="bool",
                    default=True,
                    description="Szemcsék kirajzolása a preview képre"),
        ParamSchema(name="draw_only_filtered", label="Csak szűrtek rajzolása", type="bool",
                    default=True,
                    description="Csak a szűrésen átment szemcsék rajzolása"),
        ParamSchema(name="draw_label_key", label="Felirat típusa", type="enum",
                    default="area_px",
                    options=["label", "area_px", "perimeter_px", "equivalent_diameter_px", "circularity", "intensity_mean"],
                    description="A szemcsék feliratának típusa"),
        ParamSchema(name="filter_min_area", label="Min. terület", type="int",
                    default=0, min=0, max=10000000, step=1,
                    description="Minimális szemcse terület (0 = nincs szűrés)"),
        ParamSchema(name="filter_max_area", label="Max. terület", type="int",
                    default=0, min=0, max=10000000, step=1,
                    description="Maximális szemcse terület (0 = nincs limit)"),
        ParamSchema(name="replace_images", label="Overlay csere", type="bool",
                    default=False,
                    description="A kimeneti képek cseréje az overlay képekre"),
    ],
    side_output_types={"particle_table": "SCALAR", "particle_table_filtered": "SCALAR"},
    required_preceding_steps=["detect_particles"],
)


def _exec_characterize_particles(data: dict, params: dict) -> dict:
    pixel_size_val = float(params.get("pixel_size_um", 0.0))
    pixel_size_um = pixel_size_val if pixel_size_val > 0 else None

    pct_str = params.get("percentiles", "5,25,50,75,95")
    try:
        percentiles = tuple(float(x.strip()) for x in str(pct_str).split(",") if x.strip())
    except (ValueError, TypeError):
        percentiles = (5, 25, 50, 75, 95)

    filters = {}
    min_area = int(params.get("filter_min_area", 0))
    max_area = int(params.get("filter_max_area", 0))
    if min_area > 0 or max_area > 0:
        rule = {}
        if min_area > 0:
            rule["min"] = min_area
        if max_area > 0:
            rule["max"] = max_area
        filters["area_px"] = rule

    return _pe_characterize_particles(
        data,
        include_excluded=bool(params.get("include_excluded", False)),
        pixel_size_um=pixel_size_um,
        percentiles=percentiles,
        filters=filters,
        draw=bool(params.get("draw", True)),
        draw_only_filtered=bool(params.get("draw_only_filtered", True)),
        draw_label_key=params.get("draw_label_key", "area_px"),
        replace_images=bool(params.get("replace_images", False)),
    )


_register(_characterize_particles_def, _exec_characterize_particles)
