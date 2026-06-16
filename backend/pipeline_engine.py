"""
Pipeline execution engine.

Runs pipeline steps sequentially, passing a shared *data dict* through
each step.  The data dict is created by the first step (``load_image``)
and accumulates images, results, and metadata as it flows through.

Processing-element error codes (E2xxx) are translated to human-readable
messages via PROC_ELEMENT_MESSAGES; these are separate from the scanner
error codes that share the same numbering range.
"""
import cv2
import numpy as np
from typing import Any, Optional
from pipeline_types import (
    StepError, PipelineDocument, PipelineResult,
)
from pipeline_steps import STEP_DEFINITIONS, STEP_EXECUTORS
from pipeline_validators import validate_pipeline

import logging
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Proc-element error code → Hungarian message mapping
# ---------------------------------------------------------------------------

PROC_ELEMENT_MESSAGES: dict[str, str] = {
    # load_image
    "E2001": "A betöltés után nem áll rendelkezésre egyetlen kép sem.",
    "E2002": "Érvénytelen vagy nem létező útvonal.",
    "E2003": "Egyetlen érvényes kép sem tölthető be a megadott útvonalból.",
    "E2004": "Nem támogatott képformátum.",
    # select_channel
    "E2100": "Nincsenek feldolgozandó képek.",
    "E2101": "A kép nem 3 csatornás, a csatorna kiválasztás nem lehetséges.",
    "E2102": "Érvénytelen színtér megadva.",
    "E2103": "Érvénytelen csatorna megadva.",
    "E2104": "Színtér konverzió sikertelen.",
    # apply_threshold
    "E2201": "Nincsenek feldolgozandó képek.",
    "E2202": "A képek nem egycsatornásak, küszöbölés előtt csatorna kiválasztás szükséges.",
    # calculate_histograms
    "E2301": "Nincsenek feldolgozandó képek.",
    "E2302": "A képeknek szürkeárnyalatosnak kell lenniük a hisztogram számításhoz.",
    "E2303": "Érvénytelen osztásszám (bins) megadva.",
    "E2304": "Érvénytelen hisztogram tartomány.",
    "E2305": "Hisztogram számítás sikertelen.",
    "E2306": "Váratlan hiba a hisztogram számítás során.",
    # apply_range_mask
    "E2401": "Hiányzó hisztogramok. Futtassa előbb a 'Hisztogram' lépést.",
    "E2402": "Üres vagy érvénytelen hisztogram adatok.",
    "E2403": "A hisztogram adatok nem konvertálhatók tömbbé.",
    "E2404": "A hisztogram adatoknak 2D tömbnek kell lenniük.",
    # calculate_intensity_stats
    "E2501": "Nincsenek feldolgozandó képek.",
    "E2502": "Hiányzó tartomány maszkok. Futtassa előbb a 'Tartomány maszk' lépést.",
    "E2503": "A maszkok száma nem egyezik a képek számával.",
    "E2504": "Érvénytelen percentilis értékek.",
    "E2505": "Nincsenek érvényes pixelek a maszkban.",
    "E2506": "Intenzitás statisztika számítás sikertelen.",
    "E2507": "Váratlan hiba az intenzitás statisztika számítás során.",
    # add_sequence_values
    "E2631": "Nincsenek feldolgozandó képek.",
    "E2632": "Hiányzó változó név.",
    "E2633": "Nem adott meg értékeket vagy generálási paramétereket.",
    "E2634": "Az explicit értékek érvénytelenek (nem számok).",
    "E2635": "Az értékek száma nem egyezik a képek számával.",
    "E2636": "Érvénytelen kezdőérték/lépésköz.",
    "E2637": "Érvénytelen kezdő/végérték.",
    "E2638": "Váratlan hiba a szekvencia értékek hozzáadásakor.",
    "E2639": "Érvénytelen típusú kezdő/vég/lépésköz érték (szám szükséges).",
    "E2640": "Érvénytelen képek száma szintenként (pozitív egész szám szükséges).",
    "E2641": "A lépésköz nem lehet nulla.",
    "E2642": "A generált értékek száma nem egyezik a képek számával.",
    # gray_map
    "E2601": "Nincsenek feldolgozandó képek vagy a bemeneti adatok hiányoznak.",
    "E2602": "A kezdő centroidok száma nem egyezik a megadott komponensek számával.",
    "E2603": "Hiányzó vagy érvénytelen maszkok. Futtassa előbb a ROI vagy maszkoló lépést.",
    "E2604": "A kép és a maszk mérete nem egyezik.",
    "E2605": "Érvénytelen numerikus lista a gray_map node paramétereiben.",
    # fit_curve
    "E2701": "Nincsenek eredmények (results) az adatokban.",
    "E2702": "Az X tengely változó nem található.",
    "E2703": "Hiányzó intenzitás statisztikák (intensity_stats).",
    "E2704": "Az X és statisztika adatsorok hossza nem egyezik.",
    "E2705": "Az Y tengely mező nem található a statisztikákban.",
    "E2706": "Nincs elegendő adatpont a görbe illesztéshez.",
    "E2707": "Érvénytelen polinom fokszám (pozitív egész szám szükséges).",
    "E2708": "Túl kevés adatpont a megadott polinom fokszámhoz.",
    "E2709": "Érvénytelen illesztési modell (linear, poly, log, exp).",
    "E2710": "Túl kevés egyedi X érték lineáris illesztéshez.",
    "E2711": "Túl kevés egyedi X érték a polinom fokszámához képest.",
    "E2712": "Érvénytelen aggregálási módszer (mean vagy median szükséges).",
    "E2713": "Az X tengely értékei nem konvertálhatók számmá.",
    "E2714": "Érvénytelen split módszer (random vagy ordered szükséges).",
    "E2715": "Érvénytelen validációs arány (0-1 arány vagy 0-100% megadható).",
    "E2716": "Logaritmikus modell pozitív X értékeket igényel.",
    "E2717": "Exponenciális modell pozitív Y értékeket igényel.",
    "E2718": "Logaritmikus predikció pozitív X értékeket igényel.",
    "E2719": "A split eredménye túl kevés kalibráció/validáció adatot adna.",
    "E2720": "Manuális split: hiányzó kalibráció/validáció indexek.",
    "E2721": "Kalibrációs indexek nem lista típusúak.",
    "E2722": "Validációs indexek nem lista típusúak.",
    "E2723": "Indexek nem konvertálhatók egész számra.",
    "E2724": "Túl kevés kalibrációs adatpont.",
    "E2725": "Túl kevés validációs adatpont.",
    "E2726": "Kalibrációs index tartományon kívül.",
    "E2727": "Validációs index tartományon kívül.",
    "E2728": "Duplikált kalibrációs indexek.",
    "E2729": "Duplikált validációs indexek.",
    "E2730": "Átfedés a kalibrációs és validációs indexek között.",
    # predict_node
    "E2801": "Hiányzó modell adatok.",
    "E2803": "Nincsenek illesztett görbék a modell adatokban.",
    "E2804": "A görbe illesztés index tartományon kívül esik.",
    "E2805": "Hiányzó intenzitás statisztikák a bemeneti adatokban.",
    "E2806": "Predikció számítás sikertelen.",
    "E2808": "Váratlan hiba a predikció során.",
    # apply_blur
    "E3041": "Nincsenek feldolgozandó képek.",
    "E3042": "Érvénytelen elmosási módszer.",
    "E3043": "Érvénytelen kernel méret.",
    "E3044": "A kernel méretnek páratlannak kell lennie.",
    "E3045": "Érvénytelen sigma érték.",
    "E3046": "Érvénytelen szín sigma érték.",
    "E3047": "Érvénytelen tér sigma érték.",
    "E3048": "Üres kép az elmosás bemeneteként.",
    # histogram_equalization
    "E3111": "Nincsenek feldolgozandó képek.",
    "E3112": "Üres kép a hisztogram kiegyenlítés bemeneteként.",
    "E3113": "A képeknek szürkeárnyalatosnak kell lenniük.",
    # apply_clahe
    "E3121": "Nincsenek feldolgozandó képek.",
    "E3122": "Érvénytelen levágási határ (clip_limit).",
    "E3123": "Érvénytelen csempe rács méret.",
    "E3124": "A csempe méreteknek pozitív egész számnak kell lenniük.",
    "E3125": "Üres kép a CLAHE bemeneteként.",
    "E3126": "A képeknek szürkeárnyalatosnak kell lenniük.",
    # normalize_images
    "E3131": "Nincsenek feldolgozandó képek.",
    "E3132": "Érvénytelen normalizálási típus.",
    "E3133": "Érvénytelen alpha vagy beta érték.",
    "E3134": "Üres kép a normalizálás bemeneteként.",
    # adjust_brightness_contrast
    "E3141": "Nincsenek feldolgozandó képek.",
    "E3142": "Érvénytelen fényerő érték.",
    "E3143": "Érvénytelen kontraszt érték.",
    "E3144": "Üres kép a fényerő/kontraszt bemeneteként.",
    # gamma_correction
    "E3151": "Nincsenek feldolgozandó képek.",
    "E3152": "Érvénytelen gamma érték.",
    "E3153": "Üres kép a gamma korrekció bemeneteként.",
    # flat_field_correction
    "E3201": "Nincsenek feldolgozandó képek.",
    "E3202": "Érvénytelen háttérbecslési módszer.",
    "E3203": "Érvénytelen korrekció erősség (alpha).",
    "E3204": "Érvénytelen NLM erősség (h).",
    "E3205": "Érvénytelen háttér sigma.",
    "E3206": "Érvénytelen lekicsinyítési arány.",
    "E3207": "Érvénytelen kis kép sigma.",
    "E3208": "Érvénytelen végső simítás érték.",
    "E3209": "Érvénytelen percentilis értékek.",
    "E3210": "Percentilis tartomány hiba (alsó >= felső).",
    "E3211": "Üres kép a flat-field korrekció bemeneteként.",
    "E3212": "A képeknek szürkeárnyalatosnak kell lenniük.",
    "E3213": "Robusztus skálázás sikertelen (hi <= lo).",
    # robust_stretch_gamma
    "E3221": "Nincsenek feldolgozandó képek.",
    "E3222": "Érvénytelen percentilis tartomány.",
    "E3223": "Érvénytelen gamma érték.",
    "E3224": "Üres kép a robusztus nyújtás bemeneteként.",
    "E3225": "A képeknek szürkeárnyalatosnak kell lenniük.",
    "E3226": "Nyújtás sikertelen (hi <= lo).",
    # advanced_illumin_corr
    "E3301": "Nincsenek feldolgozandó képek.",
    "E3302": "Érvénytelen háttérbecslési módszer.",
    "E3303": "Érvénytelen korrekció erősség (alpha).",
    "E3304": "Érvénytelen NLM kapcsoló.",
    "E3305": "Érvénytelen NLM erősség (h).",
    "E3306": "Érvénytelen háttér sigma.",
    "E3307": "Érvénytelen lekicsinyítési arány.",
    "E3308": "Érvénytelen kis kép sigma.",
    "E3309": "Érvénytelen végső simítás érték.",
    "E3310": "Érvénytelen percentilis értékek.",
    "E3311": "Percentilis tartomány hiba (alsó >= felső).",
    "E3312": "Érvénytelen gamma kapcsoló.",
    "E3313": "Érvénytelen gamma érték.",
    "E3314": "Üres kép a haladó megvilágítás korrekció bemeneteként.",
    "E3315": "A képeknek szürkeárnyalatosnak kell lenniük.",
    "E3316": "Üres képadat.",
    "E3317": "Robusztus skálázás sikertelen (hi <= lo).",
    # mask_rect_roi
    "E3521": "Nincsenek feldolgozandó képek.",
    "E3522": "Érvénytelen ROI típus (csak 'rect' támogatott).",
    "E3523": "Üres kép a ROI maszkolás bemeneteként.",
    # resize_images
    "E3901": "Nincsenek feldolgozandó képek.",
    "E3902": "Érvénytelen interpolációs módszer.",
    "E3903": "Nincs megadva szélesség, magasság vagy skálázási arány.",
    "E3904": "Érvénytelen skálázási arány (pozitív szám szükséges).",
    "E3905": "Érvénytelen szélesség (pozitív egész szám szükséges).",
    "E3906": "Érvénytelen magasság (pozitív egész szám szükséges).",
    "E3907": "Üres kép az átméretezés bemeneteként.",
    "E3908": "Az átméretezett méret túl kicsi (min. 1x1 pixel).",
    "E3909": "Váratlan hiba az átméretezés során.",
    # detect_particles
    "E3000": "Nincsenek feldolgozandó képek.",
    "E3001": "Érvénytelen szomszédsági érték (4 vagy 8 szükséges).",
    "E3002": "Üres kép a szemcsedetektálás bemeneteként.",
    "E3003": "Nem támogatott képformátum.",
    "E3004": "Érvénytelen feature lista.",
    "E3005": "Érvénytelen szűrő konfiguráció.",
    "E3006": "Érvénytelen percentilis lista.",
    "E3007": "Érvénytelen percentilis érték (0-100 szükséges).",
    # characterize_particles
    "E3100": "Hiányzó szemcse adatok. Futtassa előbb a 'Szemcsedetektálás' lépést.",
    "E3108": "Érvénytelen oszlop lista (list vagy tuple szükséges).",
    # histogram_pca
    "E2405": "Legalább 2 minta szükséges a PCA-hoz.",
    "E2406": "Érvénytelen komponensszám (pozitív egész szám szükséges).",
    "E2407": "SVD számítás sikertelen.",
    "E2408": "Érvénytelen előfeldolgozási módszer.",
    # detect_circles
    "E3601": "Nincsenek feldolgozandó képek.",
    "E3602": "Érvénytelen polaritás (dark, bright vagy both szükséges).",
    "E3603": "A sugár értékeknek egész számnak kell lenniük.",
    "E3604": "Érvénytelen sugár tartomány (min > 0, max > 0, min <= max).",
    "E3605": "Az elmosás kernel méretnek páratlan pozitív számnak kell lennie.",
    "E3606": "Üres kép a kör detektálás bemeneteként.",
    "E3607": "Nem támogatott képformátum.",
}


def _clamp_params(step_def_id: str, params: dict) -> dict:
    """Clamp parameter values to their schema-defined ranges."""
    defn = STEP_DEFINITIONS.get(step_def_id)
    if not defn:
        return params

    clamped = dict(params)
    for ps in defn.params:
        val = clamped.get(ps.name)
        if val is None:
            if ps.default is not None:
                clamped[ps.name] = ps.default
            continue

        if ps.type in ("int", "float"):
            try:
                num = float(val)
                if ps.min is not None:
                    num = max(num, ps.min)
                if ps.max is not None:
                    num = min(num, ps.max)
                if ps.type == "int":
                    num = int(num)
                    if ps.odd_only and num % 2 == 0:
                        num = max(num + 1, int(ps.min or 1))
                clamped[ps.name] = num
            except (ValueError, TypeError):
                clamped[ps.name] = ps.default

        elif ps.type == "enum":
            if ps.options and str(val) not in ps.options:
                clamped[ps.name] = ps.default

    return clamped


def _serialize_value(val: Any) -> Any:
    """Convert numpy types to JSON-serializable Python types."""
    if isinstance(val, np.ndarray):
        # Allow 2D arrays up to reasonable size (for PCA scores, etc.)
        total_size = val.size
        if val.ndim <= 2 and total_size <= 10000:  # ~10k elements for 2D arrays
            return val.tolist()
        elif val.ndim <= 1 and val.size <= 1024:
            return val.tolist()
        return f"<array shape={val.shape}>"
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, dict):
        return {k: _serialize_value(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_serialize_value(v) for v in val]
    return val


def extract_side_outputs(data: Optional[dict]) -> dict:
    """
    Extract JSON-serializable side outputs from the pipeline data dict.

    Skips large arrays (masks, images) and returns results, meta,
    and scalar summaries.
    """
    if data is None:
        return {}

    side = {}

    # Copy serializable results
    results = data.get("results", {})
    _skip_result_keys = {"range_masks", "region_masks"}
    for key, val in results.items():
        if key in _skip_result_keys:
            side[f"{key}_count"] = len(val) if isinstance(val, list) else 0
            continue
        elif key == "circle_overlay":
            # Convert circle overlay images to base64
            import cv2
            import base64
            circle_overlay_b64 = []
            if isinstance(val, list):
                for img in val:
                    if img is not None and hasattr(img, 'shape'):
                        success, jpeg_buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])
                        if success:
                            b64_str = base64.b64encode(jpeg_buf.tobytes()).decode('ascii')
                            circle_overlay_b64.append(b64_str)
            side["circle_overlay_base64"] = circle_overlay_b64
            continue
        elif key in {"gray_source_images", "rgb_source_images"}:
            # Dual-map original-image previews: encode each as a plain JPEG.
            import cv2
            import base64
            b64_list = []
            if isinstance(val, list):
                for img in val:
                    if img is None or not hasattr(img, 'shape'):
                        b64_list.append(None)
                        continue
                    arr = img
                    if arr.dtype != np.uint8:
                        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
                    if arr.ndim == 2:
                        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
                    success, jpeg_buf = cv2.imencode('.jpg', arr, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    b64_list.append(base64.b64encode(jpeg_buf.tobytes()).decode('ascii') if success else None)
            side[f"{key}_base64"] = b64_list
            continue
        elif key in {
            "hard_composite_rgb", "component_map", "soft_membership",
            "hard_jet", "component_map_jet", "soft_membership_jet",
            "cls_residual",
            # dual_map RGB-side equivalents
            "rgb_hard_composite_rgb", "rgb_component_map", "rgb_soft_membership",
            "rgb_hard_jet", "rgb_component_map_jet", "rgb_soft_membership_jet",
        }:
            import cv2
            import base64

            # Normalise key so inner routing works for both gray and rgb_ variants
            _norm_key = key[4:] if key.startswith("rgb_") else key

            encoded_images = []
            if isinstance(val, list):
                for img in val:
                    if img is None or not hasattr(img, 'shape'):
                        if _norm_key in {"soft_membership_jet", "component_map_jet", "hard_jet"} and isinstance(img, list):
                            component_images = []
                            for comp_img in img:
                                if comp_img is None or not hasattr(comp_img, 'shape'):
                                    component_images.append(None)
                                    continue

                                comp_arr = comp_img
                                if comp_arr.dtype != np.uint8:
                                    comp_arr = np.clip(comp_arr * 255.0, 0, 255).astype(np.uint8)
                                success, jpeg_buf = cv2.imencode('.jpg', comp_arr, [cv2.IMWRITE_JPEG_QUALITY, 90])
                                if success:
                                    component_images.append(base64.b64encode(jpeg_buf.tobytes()).decode('ascii'))
                                else:
                                    component_images.append(None)
                            encoded_images.append(component_images)
                            continue

                        encoded_images.append(None)
                        continue

                    arr = img
                    if _norm_key in {"hard_composite_rgb", "hard_jet", "component_map_jet"}:
                        if arr.dtype != np.uint8:
                            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
                    elif _norm_key == "cls_residual":
                        # CLS reconstruction residual: convert to JET visualization
                        if arr.ndim == 2 and arr.dtype != np.uint8:
                            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
                        if arr.ndim == 2:
                            # Apply JET colormap
                            arr = cv2.applyColorMap(arr, cv2.COLORMAP_JET)
                    elif _norm_key == "component_map":
                        if arr.ndim == 2:
                            palette = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.uint8)
                            labels = np.asarray(arr)
                            palette[labels == 1] = (38, 38, 220)
                            palette[labels == 2] = (74, 163, 22)
                            palette[labels == 3] = (235, 99, 37)
                            arr = palette
                        elif arr.ndim == 3 and arr.shape[2] == 3 and arr.dtype != np.uint8:
                            arr = np.clip(arr * 85.0, 0, 255).astype(np.uint8)
                    elif _norm_key == "soft_membership":
                        # Store one image per membership component so the frontend can render all centroids.
                        component_images = []
                        if arr.ndim == 3 and arr.shape[2] >= 1:
                            component_count = arr.shape[2]
                            for comp_idx in range(component_count):
                                comp_arr = arr[..., comp_idx]
                                if comp_arr.dtype != np.uint8:
                                    comp_arr = np.clip(comp_arr * 255.0, 0, 255).astype(np.uint8)
                                success, jpeg_buf = cv2.imencode('.jpg', comp_arr, [cv2.IMWRITE_JPEG_QUALITY, 90])
                                if success:
                                    component_images.append(base64.b64encode(jpeg_buf.tobytes()).decode('ascii'))
                                else:
                                    component_images.append(None)
                        encoded_images.append(component_images)
                        continue
                    elif _norm_key in {"soft_membership_jet", "component_map_jet", "hard_jet"}:
                        # Keep one encoded image per component so the frontend can render all memberships.
                        component_images = []
                        if arr.ndim == 4 and arr.shape[-1] == 3:
                            component_count = arr.shape[2]
                            for comp_idx in range(component_count):
                                comp_arr = arr[..., comp_idx, :]
                                if comp_arr.dtype != np.uint8:
                                    comp_arr = np.clip(comp_arr * 255.0, 0, 255).astype(np.uint8)
                                success, jpeg_buf = cv2.imencode('.jpg', comp_arr, [cv2.IMWRITE_JPEG_QUALITY, 90])
                                if success:
                                    component_images.append(base64.b64encode(jpeg_buf.tobytes()).decode('ascii'))
                                else:
                                    component_images.append(None)
                        encoded_images.extend(component_images)
                        continue

                    if arr.ndim == 2:
                        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)

                    success, jpeg_buf = cv2.imencode('.jpg', arr, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    if success:
                        encoded_images.append(base64.b64encode(jpeg_buf.tobytes()).decode('ascii'))
                    else:
                        encoded_images.append(None)

            side[f"{key}_base64"] = encoded_images
            continue
        side[key] = _serialize_value(val)

    # Copy meta (skip overlay image arrays)
    meta = data.get("meta", {})
    if meta:
        _skip_meta_keys = {
            "region_props_overlay",
            "filtered_regions_overlay",
            "particles_overlay",
            "particle_characterization_overlay",
        }
        filtered_meta = {k: v for k, v in meta.items() if k not in _skip_meta_keys}
        side["meta"] = _serialize_value(filtered_meta)

    # Image count
    images = data.get("images", [])
    side["image_count"] = len(images)

    # Loaded file paths (basenames for UI) — prefer originals for single-image mode
    paths = data.get("_original_paths", data.get("paths", []))
    if paths:
        import os
        side["loaded_paths"] = [os.path.basename(p) for p in paths]
        side["loaded_full_paths"] = paths

    # History
    history = data.get("history", [])
    if history:
        side["history"] = history

    return side


def execute_pipeline(
    doc: PipelineDocument,
    up_to_step: int = -1,
    single_image_index: int = -1,
    omitted_indices: list = None,
    thumbnail_max_dim: int = 0,
) -> PipelineResult:
    """
    Execute pipeline steps 0..up_to_step (inclusive).

    Each step receives and returns a shared *data dict*.  The first step
    (``load_image``) creates the dict; subsequent steps modify it in
    place.  If ``data["error"]`` is set by a processing element, execution
    stops immediately.

    If *thumbnail_max_dim* > 0, images are down-scaled after the
    ``load_image`` step so that subsequent processing is much faster.
    Intended for montage / thumbnail preview, not analytical accuracy.
    """
    validation_errors = validate_pipeline(doc)
    if validation_errors:
        return PipelineResult(success=False, errors=validation_errors)

    if up_to_step < 0 or up_to_step >= len(doc.steps):
        up_to_step = len(doc.steps) - 1

    data: Optional[dict] = None

    for i in range(up_to_step + 1):
        step_inst = doc.steps[i]
        defn = STEP_DEFINITIONS.get(step_inst.step_def_id)
        if defn is None:
            return PipelineResult(
                success=False,
                errors=[StepError(
                    step_index=i, step_def_id=step_inst.step_def_id,
                    error_code="E3005",
                    message=f"Ismeretlen lépés: {step_inst.step_def_id}",
                )],
                executed_up_to=max(0, i - 1),
            )

        executor = STEP_EXECUTORS.get(step_inst.step_def_id)
        if executor is None:
            return PipelineResult(
                success=False,
                errors=[StepError(
                    step_index=i, step_def_id=step_inst.step_def_id,
                    error_code="E3005",
                    message=f"Nincs végrehajtó a lépéshez: {step_inst.step_def_id}",
                )],
                executed_up_to=max(0, i - 1),
            )

        # For non-load steps, data must already exist
        if step_inst.step_def_id != "load_image" and data is None:
            return PipelineResult(
                success=False,
                errors=[StepError(
                    step_index=i, step_def_id=step_inst.step_def_id,
                    error_code="E3001",
                    message="Nincs bemeneti adat. Az első lépésnek 'Kép betöltése' típusúnak kell lennie.",
                )],
                executed_up_to=max(0, i - 1),
            )

        params = _clamp_params(step_inst.step_def_id, step_inst.param_values)

        # Pass single_image_index hint to load_image executor
        if step_inst.step_def_id == "load_image" and single_image_index >= 0:
            params = dict(params)
            params["_single_image_index"] = single_image_index

        # Pass thumbnail_max_dim hint to load_image executor
        if step_inst.step_def_id == "load_image" and thumbnail_max_dim > 0:
            params = dict(params) if not isinstance(params, dict) else params
            params["_thumbnail_max_dim"] = thumbnail_max_dim

        try:
            data = executor(data, params)
        except cv2.error as e:
            logger.error("OpenCV error in step %d (%s): %s", i, step_inst.step_def_id, e)
            return PipelineResult(
                success=False,
                errors=[StepError(
                    step_index=i, step_def_id=step_inst.step_def_id,
                    error_code="E3005",
                    message=f"OpenCV hiba a(z) {defn.name} lépésben: {str(e)[:200]}",
                )],
                executed_up_to=max(0, i - 1),
                data=data,
            )
        except Exception as e:
            logger.error("Error in step %d (%s): %s", i, step_inst.step_def_id, e)
            return PipelineResult(
                success=False,
                errors=[StepError(
                    step_index=i, step_def_id=step_inst.step_def_id,
                    error_code="E3005",
                    message=f"Hiba a(z) {defn.name} lépésben: {str(e)[:200]}",
                )],
                executed_up_to=max(0, i - 1),
                data=data,
            )

        # Check for processing-element error
        if data is not None and data.get("error"):
            error_code = data["error"]
            message = PROC_ELEMENT_MESSAGES.get(
                error_code, f"Feldolgozási hiba: {error_code}"
            )
            return PipelineResult(
                success=False,
                errors=[StepError(
                    step_index=i, step_def_id=step_inst.step_def_id,
                    error_code=error_code, message=message,
                )],
                executed_up_to=i,
                data=data,
            )

        # After load_image, optionally trim to a single image for faster preview
        if (step_inst.step_def_id == "load_image"
                and single_image_index >= 0
                and data is not None
                and data.get("images")):
            if not data.get("_single_image_loaded"):
                # Fallback: full load happened (e.g. reordering), trim now
                data["_original_count"] = data["count"]
                data["_original_paths"] = list(data.get("paths", []))
                idx = min(single_image_index, len(data["images"]) - 1)
                data["images"] = [data["images"][idx]]
                data["paths"] = [data["paths"][idx]] if data.get("paths") else []
                data["count"] = 1
                data["_single_image_index"] = idx
            else:
                # Already loaded single image in load_image(), just set original paths
                data["_original_paths"] = list(data.get("paths", []))
                data["_single_image_index"] = single_image_index

        # Inject omitted indices into data dict for curve fitting
        if step_inst.step_def_id == "load_image" and data is not None and omitted_indices:
            data["_omitted_indices"] = set(omitted_indices)

    return PipelineResult(
        success=True,
        executed_up_to=up_to_step,
        data=data,
    )
