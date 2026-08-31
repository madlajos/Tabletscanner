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
import os
import re

from pipeline_types import (
    DataType, ParamSchema, StepDefinition,
)
from proc_elements import (
    load_image as _pe_load_image,
    select_channel as _pe_select_channel,
    create_pseudo_image as _pe_create_pseudo_image,
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
    reference_crop as _pe_reference_crop,
    reference_color_align as _pe_reference_color_align,
    reference_sequence as _pe_reference_sequence,
    scale_bar_overlay as _pe_scale_bar_overlay,
    resize_images as _pe_resize_images,
    detect_particles as _pe_detect_particles,
    histogram_pca as _pe_histogram_pca,
    detect_circles as _pe_detect_circles,
    characterize_particles as _pe_characterize_particles,
    color_threshold as _pe_color_threshold,
    kmeans_cluster as _pe_kmeans_cluster,
    cluster_reference_map as _pe_cluster_reference_map,
)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

STEP_DEFINITIONS: dict[str, StepDefinition] = {}
STEP_EXECUTORS: dict[str, callable] = {}


def _register(defn: StepDefinition, executor):
    STEP_DEFINITIONS[defn.id] = defn
    STEP_EXECUTORS[defn.id] = executor


def _parse_float_sequence_param(raw_value, default_values, param_name, allowed_lengths=None):
    if raw_value is None or raw_value == "":
        return tuple(default_values), None

    if isinstance(raw_value, (list, tuple)):
        raw_items = raw_value
    else:
        text = str(raw_value).strip()
        if not text:
            return tuple(default_values), None
        text = text.strip("[]()")
        raw_items = [part.strip() for part in text.split(",") if part.strip()]

    try:
        values = tuple(float(item) for item in raw_items)
    except (TypeError, ValueError):
        return None, "E2605"

    if allowed_lengths is None:
        allowed_lengths = (len(default_values),)

    if len(values) not in allowed_lengths:
        return None, "E2602" if param_name == "initial_centroids" else "E2605"

    return values, None


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
    single_idx = params.get("_single_image_index", -1)
    thumb_dim = params.get("_thumbnail_max_dim", 0)
    # When reordering is active, we need all images to reorder, then trim in engine
    order_str = params.get("file_order", "")
    if order_str and single_idx >= 0:
        # Load all images for reordering, engine will trim afterwards
        result = _pe_load_image(path, thumbnail_max_dim=thumb_dim)
    else:
        result = _pe_load_image(path, single_image_index=single_idx, thumbnail_max_dim=thumb_dim)
    # Apply custom image order if specified
    if order_str and result.get("images") and not result.get("error"):
        try:
            indices = [int(x.strip()) for x in order_str.split(",") if x.strip()]
            n = len(result["paths"])
            # Validate indices
            if indices and all(0 <= i < n for i in indices) and len(indices) == n:
                result["paths"] = [result["paths"][i] for i in indices]
                if not result.get("_single_image_loaded"):
                    result["images"] = [result["images"][i] for i in indices]
        except (ValueError, IndexError):
            pass  # Ignore invalid order, keep original
    return result


_register(_load_image_def, _exec_load_image)


# ---------------------------------------------------------------------------
# 1/b. Save Images  (output sink)
# ---------------------------------------------------------------------------
_save_images_def = StepDefinition(
    id="save_images",
    name="Kép mentése",
    category="io",
    description="A feldolgozott képek mentése mappába az eredeti fájlnév prefix/suffix formátumával.",
    icon="save",
    input_type=DataType.IMAGE,
    output_type=DataType.IMAGE,
    params=[
        ParamSchema(name="output_folder", label="Kimeneti mappa", type="file_path",
                    default="", required=True,
                    description="A mentett képek célmappája"),
        ParamSchema(name="name_prefix", label="Név előtag", type="string",
                    default="", required=False,
                    description="A mentett fájlnév elejére kerül"),
        ParamSchema(name="name_suffix", label="Név utótag", type="string",
                    default="", required=False,
                    description="A mentett fájlnév végére kerül az eredeti név után"),
    ],
    side_output_types={"save_preview": "SCALAR"},
)


def _sanitize_filename_part(value: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', '_', str(value or ''))


def _exec_save_images(data: dict, params: dict) -> dict:
    # This node is an explicit sink action in the UI. Preview/normal pipeline
    # execution should not write files as a side effect.
    if not isinstance(data, dict):
        return data

    paths = data.get("_original_paths", data.get("paths", []))
    if paths:
        first_name = os.path.basename(paths[0])
    else:
        first_name = "image_001.png"

    stem, ext = os.path.splitext(first_name)
    if not ext:
        ext = ".png"

    prefix = _sanitize_filename_part(params.get("name_prefix", ""))
    suffix = _sanitize_filename_part(params.get("name_suffix", ""))
    preview_name = f"{prefix}{_sanitize_filename_part(stem)}{suffix}{ext}"

    results = data.setdefault("results", {})
    results["save_preview"] = {
        "example_original": first_name,
        "example_saved": preview_name,
    }
    return data


_register(_save_images_def, _exec_save_images)


# ---------------------------------------------------------------------------
# 1/c. Save Data Array  (output sink)
# ---------------------------------------------------------------------------
_save_array_def = StepDefinition(
    id="save_array",
    name="Adattömb mentése",
    category="io",
    description="Numerikus eredmények mentése CSV formátumban.",
    icon="table_view",
    input_type=DataType.SCALAR,
    output_type=DataType.SCALAR,
    params=[
        ParamSchema(name="output_folder", label="Mentési hely", type="file_path",
                    default="", required=True,
                    description="A CSV fájl célmappája"),
        ParamSchema(name="filename", label="Fájlnév", type="string",
                    default="adattomb.csv", required=True,
                    description="A kimeneti CSV fájl neve"),
    ],
    side_output_types={"array_save_preview": "SCALAR"},
)


def _is_numeric_scalar(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _table_from_value(value):
    if value is None:
        return None, None

    if isinstance(value, dict):
        if value and all(_is_numeric_scalar(v) for v in value.values()):
            rows = [[str(k), v] for k, v in value.items()]
            return ["key", "value"], rows
        return None, None

    if isinstance(value, list):
        if not value:
            return None, None

        if all(isinstance(r, dict) for r in value):
            keys = []
            seen = set()
            for rec in value:
                for k in rec.keys():
                    if k not in seen:
                        seen.add(k)
                        keys.append(str(k))
            rows = [[rec.get(k, "") for k in keys] for rec in value]
            return keys, rows

        if all(isinstance(r, (list, tuple)) for r in value):
            width = max((len(r) for r in value), default=0)
            rows = [list(r) + [""] * max(0, width - len(r)) for r in value]
            headers = [f"col_{i + 1}" for i in range(width)]
            return headers, rows

        if all(_is_numeric_scalar(v) for v in value):
            return ["value"], [[v] for v in value]

    return None, None


def _exec_save_array(data: dict, params: dict) -> dict:
    # Preview-only sink: do not write to disk during pipeline preview execution.
    if not isinstance(data, dict):
        return data

    results = data.get("results", {}) if isinstance(data.get("results", {}), dict) else {}

    source_key = ""
    headers = []
    rows = []
    for key, value in results.items():
        h, r = _table_from_value(value)
        if h and r:
            source_key = str(key)
            headers = h
            rows = r
            break

    if not headers:
        preview = {
            "source_key": "",
            "headers": [],
            "rows": [],
            "message": "Nem található menthető numerikus adattömb az előző lépésekben.",
        }
    else:
        max_cols = min(10, len(headers))
        preview = {
            "source_key": source_key,
            "headers": headers[:max_cols],
            "rows": [list(r[:max_cols]) for r in rows[:10]],
            "total_rows": len(rows),
            "total_cols": len(headers),
        }

    out_results = data.setdefault("results", {})
    out_results["array_save_preview"] = preview
    return data


_register(_save_array_def, _exec_save_array)


# ---------------------------------------------------------------------------
# 1/d. Branch Merge  (preview/pass-through helper)
# ---------------------------------------------------------------------------
_branch_merge_def = StepDefinition(
    id="branch_merge",
    name="Agak osszevonasa",
    category="io",
    description="Ket feldolgozasi ag osszehasonlitasa es osszevonasa. A fo kepet valtozatlanul tovabbadja, az elonezetben osztott nezetet ad.",
    icon="merge_type",
    input_type=DataType.IMAGE,
    output_type=DataType.IMAGE,
    params=[],
    side_output_types={"branch_merge_preview": "SCALAR"},
)


def _exec_branch_merge(data: dict, params: dict) -> dict:
    if not isinstance(data, dict):
        return data

    results = data.setdefault("results", {})
    meta = data.setdefault("meta", {})
    branch_sources = params.get("_branch_reference_sources")
    if isinstance(branch_sources, list):
        meta["branch_reference_sources"] = branch_sources

    # Keep the current branch's image stream, but import auxiliary outputs from
    # every explicitly connected branch. This is what carries ROI/range masks
    # into downstream processing such as k-means clustering.
    mask_result_keys = {"masks", "roi_masks", "range_masks", "region_masks"}
    for branch_data in params.get("_branch_merge_inputs", []):
        if not isinstance(branch_data, dict):
            continue
        for key, value in (branch_data.get("results") or {}).items():
            if key in mask_result_keys:
                results[key] = value
            else:
                results.setdefault(key, value)
        for key, value in (branch_data.get("meta") or {}).items():
            if key == "active_masks":
                meta[key] = value
            else:
                meta.setdefault(key, value)

    preview = params.get("_branch_merge_preview")
    if isinstance(preview, dict):
        results["branch_merge_preview"] = preview
    data.setdefault("history", []).append("branch_merge")
    return data


_register(_branch_merge_def, _exec_branch_merge)


# ---------------------------------------------------------------------------
# 2. Select Channel  (select_channel.py)
# ---------------------------------------------------------------------------
_select_channel_def = StepDefinition(
    id="select_channel",
    name="Színtér konverzió",
    category="adjustment",
    description="Színtér konverzió és csatorna kiválasztás (BGR, HSV, LAB, szürkeárnyalat). HSV/LAB esetén az 'ALL' opcióval mind a 3 csatorna kimenetre kerül.",
    icon="palette",
    input_type=DataType.IMAGE,
    output_type=DataType.GRAYSCALE,
    params=[
        ParamSchema(name="space", label="Színtér", type="enum",
                    default="GRAY",
                    options=["BGR", "HSV", "LAB", "GRAY"]),
        ParamSchema(name="channel", label="Csatorna", type="enum",
                    default="GRAY",
                    options=["R", "G", "B", "H", "S", "V", "L", "A", "GRAY", "ALL"],
                    description="A kiválasztott csatorna a megadott színtérből, vagy 'ALL' az összes csatornához"),
    ],
)


def _exec_select_channel(data: dict, params: dict) -> dict:
    space = params.get("space", "GRAY")
    channel = params.get("channel", "GRAY")
    return _pe_select_channel(data, space=space, channel=channel)


_register(_select_channel_def, _exec_select_channel)


# ---------------------------------------------------------------------------
# 2/b. Pseudo image from two loaded images
# ---------------------------------------------------------------------------
_pseudo_image_sources = [f"1-{channel}" for channel in ("B", "G", "R", "GRAY")]

_pseudo_image_def = StepDefinition(
    id="pseudo_image",
    name="Pszeudokép képekből",
    category="adjustment",
    description="Tetszőleges számú betöltött kép választott csatornáiból egy új színes képet állít elő.",
    icon="filter_vintage",
    input_type=DataType.IMAGE,
    output_type=DataType.IMAGE,
    params=[
        ParamSchema(name="blue_source", label="Kimeneti kék csatorna", type="enum",
                    default="1-B", options=_pseudo_image_sources),
        ParamSchema(name="green_source", label="Kimeneti zöld csatorna", type="enum",
                    default="1-G", options=_pseudo_image_sources),
        ParamSchema(name="red_source", label="Kimeneti piros csatorna", type="enum",
                    default="1-R", options=_pseudo_image_sources),
        ParamSchema(name="move_blue", label="Kék réteg mozgatása", type="bool", default=False),
        ParamSchema(name="move_green", label="Zöld réteg mozgatása", type="bool", default=False),
        ParamSchema(name="move_red", label="Piros réteg mozgatása", type="bool", default=False),
        ParamSchema(name="offset_x", label="Közös X eltolás (px)", type="int",
                    default=0, min=-5000, max=5000, step=1),
        ParamSchema(name="offset_y", label="Közös Y eltolás (px)", type="int",
                    default=0, min=-5000, max=5000, step=1),
    ],
)


def _exec_pseudo_image(data: dict, params: dict) -> dict:
    return _pe_create_pseudo_image(
        data,
        blue_source=params.get("blue_source", "1-B"),
        green_source=params.get("green_source", "1-G"),
        red_source=params.get("red_source", "1-R"),
        move_blue=params.get("move_blue", False),
        move_green=params.get("move_green", False),
        move_red=params.get("move_red", False),
        offset_x=params.get("offset_x", 0),
        offset_y=params.get("offset_y", 0),
    )


_register(_pseudo_image_def, _exec_pseudo_image)


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
    hist_range = (range_min, range_max) if range_min != 0 or range_max != 256 else None
    return _pe_calculate_histograms(data, bins=bins, hist_range=hist_range)


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
    description="Görbe illesztése a referencia értékek és az intenzitás statisztikák alapján (lineáris, polinom, logaritmikus, exponenciális).",
    icon="show_chart",
    input_type=DataType.IMAGE,
    output_type=DataType.IMAGE,
    params=[
        ParamSchema(name="y_name", label="Y tengely értékei", type="string",
                    default="mean", required=False,
                    description="A görbe illesztéshez használt Y mező"),
        ParamSchema(name="model", label="Illesztett görbe", type="enum",
                    default="linear",
                    options=["linear", "poly", "log", "exp"]),
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
        ParamSchema(name="split_enabled", label="Kalibráció/validáció felosztás", type="bool",
                    default=False, required=False,
                    description="Automatikus kalibráció/validáció split engedélyezése"),
        ParamSchema(name="validation_ratio", label="Validáló halmaz aránya", type="int",
                default=20, min=0, max=99, step=1, required=False,
                description="A validáló halmaz aránya százalékban (0-99%)"),
        ParamSchema(name="split_method", label="Felosztás módja", type="enum",
                    default="random", options=["random", "ordered"], required=False,
                    description="Automatikus split módja (véletlenszerű vagy sorrend)"),
    ],
    side_output_types={"calibration_metrics": "SCALAR", "validation_metrics": "SCALAR", "coefficients": "SCALAR"},
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
    model = params.get("model", "linear")
    degree = int(params.get("degree", 2))
    aggregate = bool(params.get("aggregate", False))
    agg_method = params.get("agg_method", "mean")
    merge_ab_pairs = bool(params.get("merge_ab_pairs", False))
    split_enabled = bool(params.get("split_enabled", False))
    raw_validation_ratio = float(params.get("validation_ratio", 20))
    validation_ratio = raw_validation_ratio / 100.0 if raw_validation_ratio >= 1 else raw_validation_ratio
    split_method = params.get("split_method", "random")
    return _pe_fit_curve(data, x_name=x_name, y_name=y_name, model=model,
                         degree=degree, aggregate=aggregate,
                         agg_method=agg_method, merge_ab_pairs=merge_ab_pairs,
                         split_enabled=split_enabled,
                         validation_ratio=validation_ratio,
                         split_method=split_method)


_register(_fit_curve_def, _exec_fit_curve)


# ---------------------------------------------------------------------------
# 9. Predict from Intensity  (pred_from_int.py)
# ---------------------------------------------------------------------------
_predict_node_def = StepDefinition(
    id="predict_node",
    name="Predikció",
    category="analysis",
    description="Előrejelzés kalibrációs egyenlet alapján (kézi egyenlet vagy mentett kalibráció).",
    icon="trending_up",
    input_type=DataType.SCALAR,
    output_type=DataType.IMAGE,
    params=[
        ParamSchema(name="equation", label="Kalibrációs egyenlet", type="string",
                    default="", required=False,
                    description="y = f(x) alakú egyenlet. Pl: y = 1.23x + 0.4"),
        ParamSchema(name="y_name", label="Bemeneti Y mező", type="string",
                    default="mean", required=False,
                    description="Melyik intenzitás mezőből számoljon predikciót"),
        ParamSchema(name="fit_index", label="Görbe illesztés index", type="int",
                    default=0, min=0, max=100, step=1,
                    description="Legacy: melyik korábbi görbét használja, ha nincs egyenlet"),
    ],
    side_output_types={"predictions": "SCALAR"},
)


def _exec_predict_node(data: dict, params: dict) -> dict:
    fit_index = int(params.get("fit_index", 0))
    equation = str(params.get("equation", "")).strip()
    y_name = str(params.get("y_name", "mean")).strip() or "mean"
    return _pe_predict_node(
        model_data=data,
        input_data=data,
        fit_index=fit_index,
        equation=equation,
        y_name=y_name,
    )


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
    params=[
        ParamSchema(name="output_mode", label="Kimenet", type="enum",
                    default="image",
                    options=["image", "histogram"],
                    description="Kimenet típusa: korrigált kép vagy korrigált hisztogram adatok"),
    ],
    side_output_types={"histeq_input_histograms": "HISTOGRAM", "histeq_output_histograms": "HISTOGRAM"},
)


def _exec_histogram_eq(data: dict, params: dict) -> dict:
    out_mode = str(params.get("output_mode", "image"))
    data = _pe_histogram_equalization(data)
    if data.get("error"):
        return data

    if out_mode == "histogram":
        output_hist = data.get("results", {}).get("histeq_output_histograms", [])
        data["results"]["histograms"] = output_hist
        data.setdefault("meta", {})["histogram_equalization"]["output_mode"] = "histogram"
    else:
        data.setdefault("meta", {})["histogram_equalization"]["output_mode"] = "image"

    return data


_register(_histogram_eq_def, _exec_histogram_eq)


# ---------------------------------------------------------------------------
# 12. CLA Histogram Equalization  (clahe.py)
# ---------------------------------------------------------------------------
_clahe_def = StepDefinition(
    id="apply_clahe",
    name="CLA hisztogram kiegyenlítés",
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
# 19. Scale Bar Overlay  (scale_bar.py)
# ---------------------------------------------------------------------------
_scale_bar_overlay_def = StepDefinition(
    id="scale_bar_overlay",
    name="Skála feliratozás",
    category="adjustment",
    description="Draggable scale bar overlay a képre, testre szabható betűtípussal, betűmérettel és pozícióval.",
    icon="straighten",
    input_type=DataType.IMAGE,
    output_type=DataType.IMAGE,
    params=[
        ParamSchema(name="pixels_per_mm", label="Pixels per mm", type="float",
                    default=0.0, min=0.0, max=100000.0, step=0.01,
                    description="A kalibrált pixels/mm érték. Ha 0, a node nem rajzol skálát."),
        ParamSchema(name="bar_length_mm", label="Skála hossza (mm)", type="float",
                    default=0.0, min=0.0, max=100000.0, step=1,
                    description="0 esetén automatikus, szép érték választódik a képméret és a kalibráció alapján."),
        ParamSchema(name="label_unit", label="Felirat mértékegysége", type="enum",
                default="mm", options=["mm", "cm", "um"],
                description="A skála feliratának megjelenítési egysége."),
        ParamSchema(name="position_x", label="Pozíció X", type="int",
                    default=-1, min=-1, max=100000, step=1,
                    description="A skáladoboz bal felső sarka. -1 esetén automatikusan jobb alsó sarokba kerül."),
        ParamSchema(name="position_y", label="Pozíció Y", type="int",
                    default=-1, min=-1, max=100000, step=1,
                    description="A skáladoboz bal felső sarka. -1 esetén automatikusan jobb alsó sarokba kerül."),
        ParamSchema(name="font_family", label="Betűtípus", type="enum",
                    default="sans", options=["sans", "serif", "mono", "complex", "script"],
                    description="A felirat rajzolásához használt OpenCV betűtípus."),
        ParamSchema(name="font_size", label="Betűméret", type="int",
                    default=24, min=8, max=120, step=1),
        ParamSchema(name="font_thickness", label="Betűvastagság", type="int",
                    default=1, min=1, max=8, step=1),
        ParamSchema(name="bar_thickness", label="Skála vonal vastagság", type="int",
                    default=3, min=1, max=12, step=1),
        ParamSchema(name="box_padding", label="Keret belső margó", type="int",
                    default=14, min=0, max=80, step=1),
        ParamSchema(name="text_gap", label="Felirat távolság", type="int",
                    default=8, min=0, max=50, step=1),
        ParamSchema(name="background_opacity", label="Háttér átlátszóság", type="float",
                    default=0.55, min=0.0, max=1.0, step=0.05),
        ParamSchema(name="bar_color", label="Vonal színe", type="enum",
                    default="white", options=["white", "black", "yellow"]),
        ParamSchema(name="text_color", label="Felirat színe", type="enum",
                    default="white", options=["white", "black", "yellow"]),
        ParamSchema(name="background_color", label="Háttér színe", type="enum",
                    default="black", options=["black", "white"]),
        ParamSchema(name="show_background", label="Háttér megjelenítése", type="bool",
                    default=True),
    ],
)


def _exec_scale_bar_overlay(data: dict, params: dict) -> dict:
    return _pe_scale_bar_overlay(data, **params)


_register(_scale_bar_overlay_def, _exec_scale_bar_overlay)


# ---------------------------------------------------------------------------
# 20. ROI Mask  (draw_roi.py)
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
                        default="rect", options=["rect", "ellipse", "circle", "polygon"]),
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
        # Rotation angle (degrees, applies to rect and ellipse)
        ParamSchema(name="roi_angle", label="Forgatás (°)", type="float",
                    default=0.0, min=-180.0, max=180.0, step=0.1, required=False),
        # Polygon params (JSON string of [{x,y}, ...])
        ParamSchema(name="roi_points", label="Pontok (JSON)", type="string",
                    default="[]"),
        ParamSchema(name="output_mode", label="Kimenet módja", type="enum",
                default="mask", options=["mask", "crop"],
                description="Mask mód: a ROI-n kívüli területet kezeli. Crop mód: az eredményt a ROI határoló téglalapjára vágja."),
        ParamSchema(name="apply_mask", label="Alkalmaz maszkként", type="bool",
                    default=True,
                    description="Ha engedélyezve: a ROI-n kívüli terület lesz fekete/fehér."),
        ParamSchema(name="shape_only", label="Csak körvonal", type="bool",
            default=False,
            description="Ha engedélyezve: a ROI csak körvonalként jelenik meg, kitöltés nélkül."),
        ParamSchema(name="shape_outline_color", label="Körvonal színe", type="enum",
                default="fehér", options=["fekete", "fehér"],
                description="A körvonal színe csak körvonal mód esetén."),
        ParamSchema(name="shape_outline_thickness", label="Körvonal vastagsága", type="int",
                default=2, min=1, max=100, step=1,
                description="A körvonal vastagsága pixelben csak körvonal mód esetén."),
        ParamSchema(name="background_color", label="Háttér színe", type="enum",
                    default="fekete", options=["fekete", "fehér"],
                    description="A ROI-n kívüli terület színe (csak maszk mód esetén aktív)"),
        ParamSchema(name="invert_mask", label="Maszk invertálása", type="bool",
                    default=False,
                    description="Ha engedélyezve: ROI-n belül háttér, ROI-n kívül eredeti kép (csak maszk mód esetén aktív)"),
        ParamSchema(name="roi_overrides", label="Képenkénti ROI", type="string",
                    default="{}",
                    description="Képenkénti ROI felülírások (JSON, automatikusan kezelve)"),
    ],
    side_output_types={"roi_masks": "MASK"},
)


def _exec_mask_roi(data: dict, params: dict) -> dict:
    import json as _json
    roi_type = params.get("roi_type", "rect")
    roi_angle = float(params.get("roi_angle", 0.0))

    # Parse per-image overrides
    overrides_raw = params.get("roi_overrides", "{}")
    try:
        overrides = _json.loads(overrides_raw) if isinstance(overrides_raw, str) else (overrides_raw or {})
    except Exception:
        overrides = {}

    def _build_roi_from_params(p):
        """Build a single ROI dict from a flat param dict."""
        t = p.get("roi_type", roi_type)
        angle = float(p.get("roi_angle", roi_angle))
        if t == "rect":
            w = int(p.get("roi_width", 0))
            h = int(p.get("roi_height", 0))
            if w <= 0 or h <= 0:
                return None
            return {
                "type": "rect",
                "x": int(p.get("roi_x", 0)),
                "y": int(p.get("roi_y", 0)),
                "width": w,
                "height": h,
                "angle": angle,
            }
        elif t in ("ellipse", "circle"):
            rx = int(p.get("roi_rx", 0))
            ry = int(p.get("roi_ry", 0))
            if t == "circle":
                r = int(p.get("roi_r", p.get("roi_radius", 0)))
                if r > 0:
                    rx = r
                    ry = r
                elif rx > 0 or ry > 0:
                    r = max(rx, ry)
                    rx = r
                    ry = r
            if rx <= 0 or ry <= 0:
                return None
            return {
                "type": "ellipse",
                "cx": int(p.get("roi_cx", 0)),
                "cy": int(p.get("roi_cy", 0)),
                "rx": rx,
                "ry": ry,
                "angle": angle,
            }
        else:
            pts_raw = p.get("roi_points", "[]")
            try:
                pts = _json.loads(pts_raw) if isinstance(pts_raw, str) else pts_raw
            except Exception:
                pts = []
            if not pts or len(pts) < 3:
                return None
            return {
                "type": "polygon",
                "points": [{'x': int(pt.get('x', 0)), 'y': int(pt.get('y', 0))} for pt in pts],
                "angle": angle,
            }

    # Build default ROI from top-level params
    default_roi = _build_roi_from_params(params)

    # Build per-image ROI list
    image_count = data.get("count", 0) if data else 0
    single_img_idx = data.get("_single_image_index", -1) if data else -1
    roi_list = []
    for i in range(image_count):
        # In single-image preview mode, use the original image index for override lookup
        orig_idx = single_img_idx if (single_img_idx >= 0 and image_count == 1) else i
        img_key = str(orig_idx)
        if img_key in overrides and overrides[img_key]:
            override_roi = _build_roi_from_params(overrides[img_key])
            roi_list.append(override_roi if override_roi else default_roi)
        else:
            roi_list.append(default_roi)

    # If no valid ROI at all, pass through
    if not any(roi_list):
        return data

    bg_color_str = params.get("background_color", "fekete")
    background_color = 255 if bg_color_str == "fehér" else 0
    output_mode = params.get("output_mode", "mask")
    apply_mask = params.get("apply_mask", True)
    invert_mask = params.get("invert_mask", False)
    shape_only = params.get("shape_only", False)
    shape_outline_color = params.get("shape_outline_color", "fehér")
    shape_outline_thickness = int(params.get("shape_outline_thickness", 2))

    return _pe_mask_roi(data, roi_list=roi_list, background_color=background_color,
                        apply_mask=apply_mask, invert_mask=invert_mask,
                        shape_only=shape_only,
                        shape_outline_color=shape_outline_color,
                        shape_outline_thickness=shape_outline_thickness,
                        output_mode=output_mode)


_register(_mask_roi_def, _exec_mask_roi)


# ---------------------------------------------------------------------------
# 21. Reference Crop  (reference_crop.py)
# ---------------------------------------------------------------------------
_reference_crop_def = StepDefinition(
    id="reference_crop",
    name="Reference crop",
    category="filter",
    description="Tobb azonos meretu negyzet kijelolese es referencia kivagasok megjelenitese.",
    icon="crop_free",
    input_type=DataType.IMAGE,
    output_type=DataType.IMAGE,
    params=[
        ParamSchema(name="crop_size", label="Negyzet merete", type="int",
                    default=64, min=1, max=100000, step=1,
                    description="A kijelolt referencia negyzetek oldalmérete pixelben."),
        ParamSchema(name="show_references", label="Kivagott referenciak mutatasa", type="bool",
                    default=False, required=False,
                    description="Bekapcsolva az elonezet a kivagott referenciakat sorba rendezve mutatja."),
        ParamSchema(name="reference_squares", label="Referencia negyzetek", type="string",
                    default="[]", required=False,
                    description="Kijelolt negyzetek JSON listaja, automatikusan kezeli a kepnezegeto."),
        ParamSchema(name="reference_square_overrides", label="Kepenkenti referencia negyzetek", type="string",
                    default="{}", required=False,
                    description="Kepenkenti referencia negyzet felulirasok (JSON, automatikusan kezelve)."),
    ],
    side_output_types={
        "reference_crops": "IMAGE",
        "reference_crop_overlays": "IMAGE",
        "reference_crop_squares": "SCALAR",
    },
)


def _exec_reference_crop(data: dict, params: dict) -> dict:
    import json as _json

    raw = params.get("reference_squares", "[]")
    try:
        squares = _json.loads(raw) if isinstance(raw, str) else (raw or [])
    except Exception:
        squares = []
    overrides_raw = params.get("reference_square_overrides", "{}")
    try:
        square_overrides = _json.loads(overrides_raw) if isinstance(overrides_raw, str) else (overrides_raw or {})
    except Exception:
        square_overrides = {}

    crop_size = int(params.get("crop_size", 64) or 64)
    step_index = int(params.get("_step_index", -1))
    return _pe_reference_crop(
        data,
        squares=squares,
        square_overrides=square_overrides,
        crop_size=crop_size,
        show_overlay=True,
        source_id=params.get("_step_instance_id", ""),
        source_label=f"Reference crop #{step_index + 1}" if step_index >= 0 else "Reference crop",
    )


_register(_reference_crop_def, _exec_reference_crop)


# ---------------------------------------------------------------------------
# 22. Reference Color Alignment  (reference_color_align.py)
# ---------------------------------------------------------------------------
_reference_color_align_def = StepDefinition(
    id="reference_color_align",
    name="Referenciaszin-illesztes",
    category="adjustment",
    description="A fokepek robusztus LAB tonuspontjait a referencia cropok medianjaihoz illeszti.",
    icon="colorize",
    input_type=DataType.IMAGE,
    output_type=DataType.IMAGE,
    params=[
        ParamSchema(name="reference_branch", label="Referencia crop ag", type="enum",
                    # Branch ids are recipe-specific and are supplied dynamically
                    # by the inspector, so they cannot be restricted here.
                    default="auto", options=None,
                    description="A kivalasztott ag Reference crop kivagasaihoz illeszti a node sajat aganak kepeit; agak osszevonasa nem szukseges."),
        ParamSchema(name="mode", label="Illesztes modja", type="enum",
                    default="location_scale", options=["location", "location_scale"],
                    description="Csak tonusillesztes, illetve tonusfuggo LAB szinillesztes."),
        ParamSchema(name="strength", label="Illesztes erossege", type="float",
                    default=1.0, min=0.0, max=1.0, step=0.05,
                    description="A szinkorrekcio erossege 0 es 1 kozott."),
        ParamSchema(name="output_dark", label="Sotet referencia celértéke", type="float",
                    default=0.0, min=0.0, max=254.0, step=1.0,
                    description="Kompatibilitasi parameter; a celertket a referencia adja."),
        ParamSchema(name="output_light", label="Vilagos referencia celértéke", type="float",
                    default=255.0, min=1.0, max=255.0, step=1.0,
                    description="Kompatibilitasi parameter; a celertket a referencia adja."),
    ],
    side_output_types={"reference_color_aligned_crops": "IMAGE"},
    # Either a legacy reference_crop or branch data imported by branch_merge
    # can supply the reference.
    required_preceding_steps=[],
)


def _exec_reference_color_align(data: dict, params: dict) -> dict:
    step_index = int(params.get("_step_index", -1))
    return _pe_reference_color_align(
        data,
        reference_branch=params.get("reference_branch", "auto"),
        mode=params.get("mode", "location_scale"),
        strength=float(params.get("strength", 1.0)),
        output_dark=float(params.get("output_dark", 0.0)),
        output_light=float(params.get("output_light", 255.0)),
        source_id=params.get("_step_instance_id", ""),
        source_label=f"Referenciaszin-illesztes #{step_index + 1}" if step_index >= 0 else "Referenciaszin-illesztes",
    )


_register(_reference_color_align_def, _exec_reference_color_align)


# ---------------------------------------------------------------------------
# 23. Reference Sequence  (reference_sequence.py)
# ---------------------------------------------------------------------------
_reference_sequence_def = StepDefinition(
    id="reference_sequence",
    name="Reference sequence",
    category="filter",
    description="Referencia-kivagasok, vagy crop nelkul a teljes kepek sorba rendezese valasztott szinkomponens atlaga alapjan.",
    icon="sort",
    input_type=DataType.IMAGE,
    output_type=DataType.IMAGE,
    params=[
        ParamSchema(name="component", label="Rendezes komponense", type="enum",
                    default="GRAY",
                    options=["GRAY", "R", "G", "B", "RGB_ALL", "H", "S", "V", "HSV_ALL", "L", "A", "LAB_B", "LAB_ALL"],
                    description="Az a szin- vagy fenyessegkomponens, amelynek crop-atlaga alapjan tortenik a rendezes."),
        ParamSchema(name="direction", label="Rendezes iranya", type="enum",
                    default="ascending", options=["ascending", "descending"],
                    description="Novekvo vagy csokkeno sorrend a komponens atlaga szerint."),
    ],
    side_output_types={
        "reference_crops": "IMAGE",
        "reference_sequence": "SCALAR",
    },
)


def _exec_reference_sequence(data: dict, params: dict) -> dict:
    return _pe_reference_sequence(
        data,
        component=params.get("component", "GRAY"),
        direction=params.get("direction", "ascending"),
        source_id=params.get("_step_instance_id", ""),
        source_label=f"Reference sequence #{int(params.get('_step_index', -1)) + 1}" if int(params.get("_step_index", -1)) >= 0 else "Reference sequence",
    )


_register(_reference_sequence_def, _exec_reference_sequence)


# ---------------------------------------------------------------------------
# 23. Resize Images  (resize_img.py)
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
    description="Szemcsék detektálása bináris/maszkolt képeken, feature számítás és szűrés.",
    icon="center_focus_strong",
    input_type=DataType.GRAYSCALE,
    output_type=DataType.MASK,
    params=[
        ParamSchema(name="draw", label="Szemcsekontúrok mutatása", type="bool",
                    default=True,
                    description="Detektált szemcsék körvonalainak kirajzolása"),
        ParamSchema(name="contour_thickness", label="Kontúrvastagság", type="int",
                    default=1, min=1, max=10, step=1,
                    description="A kirajzolt szemcsekontúrok vastagsága"),
        ParamSchema(name="draw_label", label="Címkék rajzolása", type="bool",
                    default=False,
                    description="Szemcse azonosítók megjelenítése"),
        ParamSchema(name="draw_only_filtered", label="Csak szűrtek rajzolása", type="bool",
                    default=False,
                    description="Csak a szűrésen átment szemcsék rajzolása"),
        ParamSchema(name="draw_label_key", label="Felirat típusa", type="enum",
                    default="area_px",
                    options=["label", "area_px", "perimeter_px", "equivalent_diameter_px", "circularity", "intensity_mean"],
                    description="A szemcsék feliratának típusa"),
        # --- Szemcsék szűrése ---
        ParamSchema(name="filter_by_area", label="Terület alapján", type="bool",
                    default=False,
                    description="Szűrés terület alapján"),
        ParamSchema(name="filter_min_area", label="Min. terület", type="int",
                    default=0, min=0, max=10000000, step=1,
                    description="Minimális szemcse terület (0 = nincs szűrés)"),
        ParamSchema(name="filter_max_area", label="Max. terület", type="int",
                    default=10000, min=0, max=10000000, step=1,
                    description="Maximális szemcse terület (0 = nincs limit)"),
        ParamSchema(name="filter_by_circularity", label="Kerekdedség alapján", type="bool",
                    default=False,
                    description="Szűrés kerekdedség alapján"),
        ParamSchema(name="filter_min_circularity", label="Min. kerekdedség", type="float",
                    default=0.0, min=0.0, max=1.0, step=0.01,
                    description="Minimális kerekdedségi érték"),
        ParamSchema(name="filter_max_circularity", label="Max. kerekdedség", type="float",
                    default=1.0, min=0.0, max=1.0, step=0.01,
                    description="Maximális kerekdedségi érték"),
        ParamSchema(name="filter_by_convexity", label="Konvexitás alapján", type="bool",
                    default=False,
                    description="Szűrés konvexitás alapján"),
        ParamSchema(name="filter_convex", label="Konvex", type="bool",
                    default=True,
                    description="Konvex szemcsék megjelenítése"),
        ParamSchema(name="filter_concave", label="Konkáv", type="bool",
                    default=True,
                    description="Konkáv szemcsék megjelenítése"),
    ],
    side_output_types={"particles": "SCALAR", "particles_summary": "SCALAR", "particles_filtered": "SCALAR"},
)


def _exec_detect_particles(data: dict, params: dict) -> dict:
    filters = {}

    if bool(params.get("filter_by_area", False)):
        min_area = int(params.get("filter_min_area", 0))
        max_area = int(params.get("filter_max_area", 50000))
        rule = {}
        if min_area > 0:
            rule["min"] = min_area
        if max_area > 0:
            rule["max"] = max_area
        if rule:
            filters["area_px"] = rule

    if bool(params.get("filter_by_circularity", False)):
        min_circ = float(params.get("filter_min_circularity", 0.0))
        max_circ = float(params.get("filter_max_circularity", 1.0))
        rule = {}
        if min_circ > 0:
            rule["min"] = min_circ
        if max_circ < 1.0:
            rule["max"] = max_circ
        if rule:
            filters["circularity"] = rule

    if bool(params.get("filter_by_convexity", False)):
        want_convex = bool(params.get("filter_convex", True))
        want_concave = bool(params.get("filter_concave", True))
        if want_convex and not want_concave:
            filters["solidity"] = {"min": 0.95}
        elif want_concave and not want_convex:
            filters["solidity"] = {"max": 0.95}
        elif not want_convex and not want_concave:
            filters["solidity"] = {"min": 2.0}  # impossible → filter all

    draw = bool(params.get("draw", True))
    contour_thickness = max(1, int(params.get("contour_thickness", 1)))
    draw_label = bool(params.get("draw_label", False))
    draw_only_filtered = bool(params.get("draw_only_filtered", False))
    draw_label_key = params.get("draw_label_key", "area_px")
    excluded_ids = params.get("excluded_ids", [])
    if not isinstance(excluded_ids, list):
        excluded_ids = []

    return _pe_detect_particles(
        data,
        filters=filters,
        draw=draw,
        contour_thickness=contour_thickness,
        draw_label=draw_label,
        draw_only_filtered=draw_only_filtered,
        draw_label_key=draw_label_key,
        replace_images=draw,
        excluded_ids=excluded_ids,
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
        ParamSchema(name="max_components", label="Max. komponensek", type="int",
                    default=5, min=1, max=50, step=1,
                    description="Az előállítandó főkomponensek maximális száma"),
        ParamSchema(name="preprocessing", label="Előfeldolgozás", type="enum",
                    default="center",
                    options=["none", "center", "standardize", "l1", "l2"],
                    description="Előfeldolgozási módszer a PCA előtt"),
    ],
    side_output_types={
        "histogram_pca_scores": "SCALAR",
        "histogram_pca_explained_ratio": "SCALAR",
        "histogram_pca_cumulative_ratio": "SCALAR",
    },
    required_preceding_steps=["calculate_histograms"],
)


def _exec_histogram_pca(data: dict, params: dict) -> dict:
    max_components = int(params.get("max_components", 5))
    preprocessing = params.get("preprocessing", "center")
    return _pe_histogram_pca(data, max_components=max_components, preprocessing=preprocessing)


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
        ParamSchema(name="detect_scale", label="Detektálási méretarány", type="float",
                    default=1.0, min=0.25, max=1.0, step=0.05,
                    description="A kördetektálásra használt kép méretezési aránya. 1.0 = teljes méret, kisebb érték = gyorsabb futás, de finomhangolást igényelhet."),
        ParamSchema(name="edge_threshold", label="Élküszöb", type="float",
                    default=100.0, min=1.0, max=255.0, step=1.0,
                    description="A Canny élkereső felső küszöbe. Kisebb érték érzékenyebb a halványabb, kevésbé éles élekre, de nagyobb méretarány mellett lassabb lehet."),
        ParamSchema(name="min_diameter", label="Min. átmérő", type="int",
                    default=40, min=1, max=10000, step=1,
                    description="Minimális kör átmérő pixelben, a teljes felbontású képre értendő."),
        ParamSchema(name="max_diameter", label="Max. átmérő", type="int",
                    default=50, min=1, max=10000, step=1,
                    description="Maximális kör átmérő pixelben, a teljes felbontású képre értendő."),
        ParamSchema(name="radius_multiplier", label="Sugár szorzó", type="float",
                    default=1.0, min=0.1, max=5.0, step=0.1,
                    description="A detektált körök sugarának szorzója (1.0 = nincs módosítás, 0.8 = 80%, 1.2 = 120%)"),
        ParamSchema(name="apply_mask", label="Alkalmaz maszkként", type="bool",
                    default=False,
                    description="Ha engedélyezve: a körön kívüli terület lesz fekete/fehér. Nem rajzol kör jelölést."),
        ParamSchema(name="mask_background", label="Háttér szín", type="enum",
                    default="black",
                    options=["black", "white"],
                    description="A maszk háttérszíne (csak apply_mask engedélyezése esetén aktív)"),
        ParamSchema(name="invert_mask", label="Maszk invertálása", type="bool",
                    default=False,
                    description="Ha engedélyezve: körön belül fekete/fehér, körön kívül eredeti (csak apply_mask engedélyezése esetén aktív)"),
        ParamSchema(name="polarity", label="Polaritás", type="enum",
                    default="dark",
                    options=["dark", "bright", "both"],
                    description="Sötét körök világos háttéren / világos körök sötét háttéren / mindkettő"),
    ],
    side_output_types={"circles": "SCALAR"},
)


def _exec_detect_circles(data: dict, params: dict) -> dict:
    dp = float(params.get("dp", 1.2))
    detect_scale = float(params.get("detect_scale", 1.0))
    edge_threshold = float(params.get("edge_threshold", 100.0))
    min_diameter = int(params.get("min_diameter", 40))
    max_diameter = int(params.get("max_diameter", 50))
    radius_multiplier = float(params.get("radius_multiplier", 1.0))
    apply_mask = params.get("apply_mask", False)
    mask_background = params.get("mask_background", "black")
    invert_mask = params.get("invert_mask", False)
    polarity = params.get("polarity", "dark")
    # These parameters keep stable defaults; only the edge threshold is exposed in the UI.
    min_dist = 20
    blur_ksize = 5
    accumulator_threshold = 20
    return _pe_detect_circles(
        data,
        dp=dp,
        min_dist=min_dist,
        min_diameter=min_diameter,
        max_diameter=max_diameter,
        detect_scale=detect_scale,
        blur_ksize=blur_ksize,
        edge_threshold=edge_threshold,
        accumulator_threshold=accumulator_threshold,
        polarity=polarity,
        radius_multiplier=radius_multiplier,
        apply_mask=apply_mask,
        mask_background=mask_background,
        invert_mask=invert_mask,
    )


_register(_detect_circles_def, _exec_detect_circles)


# ---------------------------------------------------------------------------
# 25. Characterize Particles  (filter_region.py)
# ---------------------------------------------------------------------------
_characterize_particles_def = StepDefinition(
    id="characterize_particles",
    name="Szemcsekarakterizálás",
    category="analysis",
    description="Detektált szemcsék táblázatos összegzése.",
    icon="table_chart",
    input_type=DataType.GRAYSCALE,
    output_type=DataType.GRAYSCALE,
    params=[
        ParamSchema(name="use_filtered", label="Szűrt szemcsék használata", type="bool",
                    default=True,
                    description="Csak a szűrt szemcsék megjelenítése a táblázatban"),
        ParamSchema(name="include_excluded", label="Kizárt szemcsék bevétele", type="bool",
                    default=False,
                    description="A kizárt szemcsék is bekerüljenek a táblázatba"),
        ParamSchema(name="selected_columns", label="Kiválasztott oszlopok", type="string",
                    default="",
                    description="Vesszővel elválasztott oszlopnevek (üres = összes)"),
    ],
    side_output_types={"particle_table": "SCALAR"},
    required_preceding_steps=["detect_particles"],
)


def _exec_characterize_particles(data: dict, params: dict) -> dict:
    use_filtered = bool(params.get("use_filtered", True))
    include_excluded = bool(params.get("include_excluded", False))

    cols_str = params.get("selected_columns", "")
    if cols_str and str(cols_str).strip():
        selected_columns = [c.strip() for c in str(cols_str).split(",") if c.strip()]
    else:
        selected_columns = None

    return _pe_characterize_particles(
        data,
        use_filtered=use_filtered,
        selected_columns=selected_columns,
        include_excluded=include_excluded,
    )


_register(_characterize_particles_def, _exec_characterize_particles)


# ---------------------------------------------------------------------------
# 26. Color Threshold (color_thresh.py)
# ---------------------------------------------------------------------------
_color_thresh_def = StepDefinition(
    id="color_thresh",
    name="Szín alapú küszöb",
    category="adjustment",
    description="Szín alapú küszöbölés a kiválasztott színtérben. Szükséges: Színtér konverzió lépés 'összes csatorna' opcióval.",
    icon="palette",
    input_type=DataType.IMAGE,
    output_type=DataType.MASK,
    params=[
        ParamSchema(name="H_min", label="H min", type="int", default=0, min=0, max=179),
        ParamSchema(name="H_max", label="H max", type="int", default=179, min=0, max=179),
        ParamSchema(name="S_min", label="S min", type="int", default=0, min=0, max=255),
        ParamSchema(name="S_max", label="S max", type="int", default=255, min=0, max=255),
        ParamSchema(name="V_min", label="V min", type="int", default=0, min=0, max=255),
        ParamSchema(name="V_max", label="V max", type="int", default=255, min=0, max=255),
        ParamSchema(name="B_min", label="B min", type="int", default=0, min=0, max=255),
        ParamSchema(name="B_max", label="B max", type="int", default=255, min=0, max=255),
        ParamSchema(name="G_min", label="G min", type="int", default=0, min=0, max=255),
        ParamSchema(name="G_max", label="G max", type="int", default=255, min=0, max=255),
        ParamSchema(name="R_min", label="R min", type="int", default=0, min=0, max=255),
        ParamSchema(name="R_max", label="R max", type="int", default=255, min=0, max=255),
        ParamSchema(name="L_min", label="L min", type="int", default=0, min=0, max=255),
        ParamSchema(name="L_max", label="L max", type="int", default=255, min=0, max=255),
        ParamSchema(name="A_min", label="A min", type="int", default=0, min=0, max=255),
        ParamSchema(name="A_max", label="A max", type="int", default=255, min=0, max=255),
        ParamSchema(name="Lab_B_min", label="B min", type="int", default=0, min=0, max=255),
        ParamSchema(name="Lab_B_max", label="B max", type="int", default=255, min=0, max=255),
        ParamSchema(name="GRAY_min", label="GRAY min", type="int", default=0, min=0, max=255),
        ParamSchema(name="GRAY_max", label="GRAY max", type="int", default=255, min=0, max=255),
        ParamSchema(name="invert", label="Invertálás", type="bool",
                    default=False,
                    description="A kimeneti maszk invertálása"),
        ParamSchema(name="white_background", label="Fehér háttér", type="bool",
                    default=False,
                    description="A levágott területek fehérrel töltése (alapértelmezés: fekete)"),
    ],
    side_output_types={"color_thresh_channel_histograms": "SCALAR", "color_thresh_input_images": "SCALAR", "color_thresh_mask_overlays": "SCALAR"},
    required_preceding_steps=["select_channel"],
)


def _exec_color_thresh(data: dict, params: dict) -> dict:
    # Detect color space from the metadata of the previous select_channel step
    space = "HSV"  # default
    
    if "meta" in data and "select_channel" in data["meta"]:
        space = data["meta"]["select_channel"].get("space", "HSV")
    
    invert = bool(params.get("invert", False))
    white_background = bool(params.get("white_background", False))
    
    channel_mapping = {
        "HSV": [("H", "H_min", "H_max"), ("S", "S_min", "S_max"), ("V", "V_min", "V_max")],
        "BGR": [("B", "B_min", "B_max"), ("G", "G_min", "G_max"), ("R", "R_min", "R_max")],
        "LAB": [("L", "L_min", "L_max"), ("A", "A_min", "A_max"), ("B", "Lab_B_min", "Lab_B_max")],
        "GRAY": [("GRAY", "GRAY_min", "GRAY_max")],
    }
    
    thresholds = {}
    if space in channel_mapping:
        for ch_name, min_key, max_key in channel_mapping[space]:
            min_val = int(params.get(min_key, 0))
            max_val = int(params.get(max_key, 255))
            thresholds[ch_name] = (min_val, max_val)
    else:
        data["error"] = "E2201"
        return data
    
    result = _pe_color_threshold(data, space=space, thresholds=thresholds, invert=invert, white_background=white_background)
    
    # Convert input images and mask overlays to base64 for frontend display
    if "results" in result:
        import cv2
        import base64
        
        # Convert input images
        if "color_thresh_input_images" in result["results"]:
            input_images = result["results"]["color_thresh_input_images"]
            if isinstance(input_images, list):
                b64_images = []
                for img in input_images:
                    try:
                        success, jpeg_buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])
                        if success:
                            b64_str = base64.b64encode(jpeg_buf.tobytes()).decode('ascii')
                            b64_images.append(f"data:image/jpeg;base64,{b64_str}")
                        else:
                            b64_images.append(None)
                    except Exception:
                        b64_images.append(None)
                result["results"]["color_thresh_input_images"] = b64_images
        
        # Convert mask overlays
        if "color_thresh_mask_overlays" in result["results"]:
            mask_overlays = result["results"]["color_thresh_mask_overlays"]
            if isinstance(mask_overlays, list):
                b64_overlays = []
                for img in mask_overlays:
                    try:
                        success, jpeg_buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])
                        if success:
                            b64_str = base64.b64encode(jpeg_buf.tobytes()).decode('ascii')
                            b64_overlays.append(f"data:image/jpeg;base64,{b64_str}")
                        else:
                            b64_overlays.append(None)
                    except Exception:
                        b64_overlays.append(None)
                result["results"]["color_thresh_mask_overlays"] = b64_overlays
    
    return result


_register(_color_thresh_def, _exec_color_thresh)


# ---------------------------------------------------------------------------
# 27. K-means Cluster (kmeans_cluster.py)
# ---------------------------------------------------------------------------
_kmeans_cluster_def = StepDefinition(
    id="kmeans_cluster",
    name="K-kozep klaszterezes",
    category="analysis",
    description="Pixelek k-kozep klaszterezese valasztott szinterben. A kimenet szinezett klaszterterkep.",
    icon="hub",
    input_type=DataType.IMAGE,
    output_type=DataType.IMAGE,
    params=[
        ParamSchema(name="k", label="Klaszterek szama", type="int",
                    default=3, min=2, max=32, step=1,
                    description="A keresett klaszterek szama."),
        ParamSchema(name="color_space", label="Szinter", type="enum",
                    default="BGR", options=["BGR", "HSV", "LAB", "GRAY"],
                    description="A pixeljellemzok szintere a k-kozep algoritmushoz."),
        ParamSchema(name="init_mode", label="Referenciak hasznalata", type="enum",
                    default="auto", options=["auto", "reference_fixed", "reference_seeded"],
                    description="Automatikus: referencia nelkuli k-means. Fix referencia: pixelek besorolasa a referencia cropok atlagaihoz. Referenciaval inditott: k-means inditasa a referencia cropok alapjan."),
        ParamSchema(name="reference_source", label="Referencia forras node", type="string",
                    default="auto", required=False,
                    description="Melyik Reference crop vagy Reference sequence node eredmenyet hasznalja. Auto eseten a pipeline aktualis referencia eredmenyet hasznalja."),
        ParamSchema(name="attempts", label="Probalkozasok", type="int",
                    default=3, min=1, max=20, step=1,
                    description="Tobb inditas kozul a legjobb eredmenyt valasztja."),
        ParamSchema(name="max_iter", label="Max. iteracio", type="int",
                    default=30, min=1, max=300, step=1,
                    description="A k-kozep optimalizalas maximalis iteracioszama."),
        ParamSchema(name="epsilon", label="Pontossag", type="float",
                    default=1.0, min=0.001, max=100.0, step=0.1,
                    description="Leallasi kuszob az iteraciohoz."),
        ParamSchema(name="sort_by_brightness", label="Rendezes fenyesseg szerint", type="bool",
                    default=True,
                    description="Bekapcsolva a klaszter indexek sotett-vilagos sorrendben stabilabbak."),
        ParamSchema(name="output_mode", label="Kimeneti szinezes", type="enum",
                    default="palette", options=["palette", "centroid"],
                    description="Paletta: kontrasztos jeloloszinek. Centroid: a klaszter kozeppontjanak szine."),
        ParamSchema(name="background", label="Hatter", type="enum",
                    default="black", options=["black", "white", "original"],
                    description="Maszkolt teruleten kivuli hatter szine."),
    ],
    side_output_types={
        "kmeans_overlay_images": "IMAGE",
        "kmeans_legend": "SCALAR",
        "kmeans_label_maps": "MASK",
        "kmeans_centers": "SCALAR",
        "kmeans_counts": "SCALAR",
        "kmeans_percentages": "SCALAR",
        "kmeans_compactness": "SCALAR",
        "kmeans_reference_info": "SCALAR",
    },
)


def _exec_kmeans_cluster(data: dict, params: dict) -> dict:
    return _pe_kmeans_cluster(
        data,
        k=int(params.get("k", 3)),
        color_space=params.get("color_space", "BGR"),
        init_mode=params.get("init_mode", "auto"),
        reference_source=params.get("reference_source", "auto"),
        attempts=int(params.get("attempts", 3)),
        max_iter=int(params.get("max_iter", 30)),
        epsilon=float(params.get("epsilon", 1.0)),
        sort_by_brightness=bool(params.get("sort_by_brightness", True)),
        output_mode=params.get("output_mode", "palette"),
        background=params.get("background", "black"),
        cluster_colors=params.get("cluster_colors", "{}"),
    )


_register(_kmeans_cluster_def, _exec_kmeans_cluster)


# ---------------------------------------------------------------------------
# 28. Label-based reference map
# ---------------------------------------------------------------------------
_cluster_map_def = StepDefinition(
    id="cluster_reference_map",
    name="Klaszter referencia map",
    category="analysis",
    description="A választott k-means klaszterből referenciát képez, majd a kijelölt klasztereken hasonlósági térképet készít.",
    icon="map",
    input_type=DataType.IMAGE,
    output_type=DataType.IMAGE,
    params=[
        ParamSchema(name="selected_labels", label="Értékelt klaszterek", type="string", default="1",
                    description="Azok a k-means klaszterek, amelyeken a térkép értéket kaphat."),
        ParamSchema(name="reference_label", label="Referenciaklaszter", type="string", default="1",
                    description="Kizárólag ennek a klaszternek a pixeleiből készül a referenciaérték."),
        ParamSchema(name="center_mode", label="Klaszterközép számítása", type="enum",
                    default="cluster_median",
                    options=["min_max_midpoint", "cluster_median", "reference_mean", "reference_mean_half"],
                    description="A kiválasztott klaszterrégió referenciaértékének számítása."),
        ParamSchema(name="map_multiplier", label="Térkép szorzó", type="float", default=1.0,
                    min=0.0, max=1.0, step=0.05,
                    description="A JET színezés előtti hasonlósági érték szorzója. Kisebb érték lejjebb tolja a színeket."),
        ParamSchema(name="accepted_components", label="Elfogadott térképek", type="string", default="[]",
                    description="A pipával elfogadott komponensek mentett beállításai."),
        ParamSchema(name="remainder_as_last", label="Maradék az utolsó komponens", type="bool", default=False,
                    description="Az utolsó komponenst klaszterezés nélkül, a 100%-ból megmaradt értékként számítja."),
        ParamSchema(name="remainder_name", label="Maradék neve", type="string", default="Maradék"),
        ParamSchema(name="remainder_display_multiplier", label="Maradék megjelenítési szorzó", type="float",
                    default=1.0, min=0.0, max=1.0, step=0.05,
                    description="Csak a színezett előnézetet módosítja; a százalékos maradék értékét nem."),
        ParamSchema(name="remainder_invert", label="Maradék színskála megfordítása", type="bool", default=False),
        ParamSchema(name="colormap", label="Szinskala", type="enum", default="jet",
                    options=["turbo", "jet", "viridis"]),
        ParamSchema(name="invert", label="Szinskala megforditasa", type="bool", default=False),
    ],
    side_output_types={
        "kmeans_labeled_images": "IMAGE",
        "kmeans_overlay_images": "IMAGE",
        "kmeans_legend": "SCALAR",
        "cluster_map_images": "IMAGE",
        "cluster_map_overlay_images": "IMAGE",
        "cluster_map_raw": "SCALAR",
        "cluster_map_component_images": "IMAGE",
        "cluster_map_reference": "SCALAR",
        "cluster_map_label_values": "SCALAR",
        "cluster_map_selected_labels": "SCALAR",
        "cluster_map_reference_label": "SCALAR",
    },
    required_preceding_steps=["kmeans_cluster"],
)


def _exec_cluster_map(data: dict, params: dict) -> dict:
    return _pe_cluster_reference_map(
        data,
        selected_labels=params.get("selected_labels", "1"),
        reference_label=params.get("reference_label", "1"),
        center_mode=params.get("center_mode", "cluster_median"),
        map_multiplier=float(params.get("map_multiplier", 1.0)),
        accepted_components=params.get("accepted_components", "[]"),
        remainder_as_last=bool(params.get("remainder_as_last", False)),
        remainder_name=params.get("remainder_name", "Maradék"),
        remainder_display_multiplier=float(params.get("remainder_display_multiplier", 1.0)),
        remainder_invert=bool(params.get("remainder_invert", False)),
        colormap=params.get("colormap", "jet"),
        invert=bool(params.get("invert", False)),
    )


_register(_cluster_map_def, _exec_cluster_map)
