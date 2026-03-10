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
    ],
    side_output_types={"count": "SCALAR"},
)


def _exec_load_image(data: dict, params: dict) -> dict:
    path = params.get("source", "")
    if not path:
        data["error"] = "E2002"
        return data
    return _pe_load_image(path)


_register(_load_image_def, _exec_load_image)


# ---------------------------------------------------------------------------
# 2. Select Channel  (select_channel.py)
# ---------------------------------------------------------------------------
_select_channel_def = StepDefinition(
    id="select_channel",
    name="Csatorna kiválasztás",
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
