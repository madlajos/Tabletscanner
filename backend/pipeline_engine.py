"""
Pipeline execution engine: runs pipeline steps with auto-conversion,
parameter clamping, runtime safety wrappers, and intermediate caching.
"""
import hashlib
import json
import cv2
import numpy as np
from typing import Optional
from pipeline_types import (
    DataType, AUTO_CONVERSIONS, StepResult, StepError,
    PipelineDocument, PipelineResult,
)
from pipeline_steps import STEP_DEFINITIONS, STEP_EXECUTORS
from pipeline_validators import validate_pipeline

import logging
logger = logging.getLogger(__name__)

# In-memory cache: step_hash -> (StepResult.primary_output, StepResult.side_outputs, output_type)
_step_cache: dict[str, tuple] = {}
_MAX_CACHE_ENTRIES = 50


def _compute_input_hash(img: Optional[np.ndarray]) -> str:
    """Produce a fast hash of a numpy image for cache keying."""
    if img is None:
        return "none"
    # Use a strided sample for speed on large images
    flat = img.flat
    sample_size = min(10000, len(flat))
    step = max(1, len(flat) // sample_size)
    sample = flat[::step]
    h = hashlib.md5(sample.tobytes(), usedforsecurity=False)
    h.update(f"{img.shape}_{img.dtype}".encode())
    return h.hexdigest()


def _compute_step_hash(step_def_id: str, param_values: dict, input_hash: str) -> str:
    """Cache key for one step execution."""
    param_str = json.dumps(param_values, sort_keys=True, default=str)
    raw = f"{step_def_id}|{param_str}|{input_hash}"
    return hashlib.md5(raw.encode(), usedforsecurity=False).hexdigest()


def _auto_convert(image: np.ndarray, from_type: DataType, to_type: DataType) -> np.ndarray:
    """Apply auto-conversion between compatible data types."""
    if from_type == to_type:
        return image

    key = (from_type, to_type)
    conv_name = AUTO_CONVERSIONS.get(key)
    if conv_name is None:
        return image  # identity (same-type or compatible pair with no-op)

    if conv_name == "bgr_to_gray":
        if image.ndim == 3 and image.shape[2] >= 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
    elif conv_name == "gray_to_bgr":
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return image

    return image


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


def safe_execute_step(
    step_def_id: str,
    params: dict,
    input_image: Optional[np.ndarray],
    step_index: int,
) -> StepResult:
    """
    Execute a single pipeline step with full safety wrapping.

    1. Validate input (non-empty for non-load steps)
    2. Auto-convert input type
    3. Clamp params
    4. Execute in try/except
    5. Validate output
    """
    defn = STEP_DEFINITIONS.get(step_def_id)
    if defn is None:
        return StepResult(
            success=False,
            warnings=[f"Ismeretlen lépés: {step_def_id}"],
        )

    executor = STEP_EXECUTORS.get(step_def_id)
    if executor is None:
        return StepResult(
            success=False,
            warnings=[f"Nincs végrehajtó a lépéshez: {step_def_id}"],
        )

    # For non-load steps, validate input image
    if step_def_id != "load_image":
        if input_image is None or (isinstance(input_image, np.ndarray) and input_image.size == 0):
            return StepResult(
                success=False,
                warnings=["Nincs bemeneti kép ehhez a lépéshez."],
            )

        # Determine previous output type from image shape
        if input_image.ndim == 2:
            actual_type = DataType.GRAYSCALE
        else:
            actual_type = DataType.IMAGE

        # Auto-convert if needed
        input_image = _auto_convert(input_image, actual_type, defn.input_type)

    # Clamp parameters
    params = _clamp_params(step_def_id, params)

    # Execute with safety wrapper
    try:
        result = executor(input_image, params)
    except cv2.error as e:
        logger.error(f"OpenCV error in step {step_index} ({step_def_id}): {e}")
        return StepResult(
            success=False,
            warnings=[f"OpenCV hiba a(z) {defn.name} lépésben: {str(e)[:200]}"],
        )
    except (ValueError, TypeError) as e:
        logger.error(f"Value/Type error in step {step_index} ({step_def_id}): {e}")
        return StepResult(
            success=False,
            warnings=[f"Paraméter hiba a(z) {defn.name} lépésben: {str(e)[:200]}"],
        )
    except Exception as e:
        logger.error(f"Unexpected error in step {step_index} ({step_def_id}): {e}")
        return StepResult(
            success=False,
            warnings=[f"Váratlan hiba a(z) {defn.name} lépésben: {str(e)[:200]}"],
        )

    # Validate output
    if result.success and result.primary_output is not None:
        if not isinstance(result.primary_output, np.ndarray):
            return StepResult(
                success=False,
                warnings=[f"A(z) {defn.name} lépés nem numpy tömböt adott vissza."],
            )
        if result.primary_output.size == 0:
            return StepResult(
                success=False,
                warnings=[f"A(z) {defn.name} lépés üres képet adott vissza."],
            )

    return result


def execute_pipeline(
    doc: PipelineDocument,
    up_to_step: int = -1,
    use_cache: bool = True,
) -> PipelineResult:
    """
    Execute pipeline steps 0..up_to_step (inclusive).
    If up_to_step < 0, execute all steps.
    """
    # Validate first
    validation_errors = validate_pipeline(doc)
    if validation_errors:
        return PipelineResult(
            success=False,
            errors=validation_errors,
        )

    if up_to_step < 0 or up_to_step >= len(doc.steps):
        up_to_step = len(doc.steps) - 1

    current_image = None
    step_results = []

    for i in range(up_to_step + 1):
        step_inst = doc.steps[i]
        defn = STEP_DEFINITIONS.get(step_inst.step_def_id)
        if defn is None:
            error = StepError(
                step_index=i, step_def_id=step_inst.step_def_id,
                error_code="E3005", message=f"Ismeretlen lépés: {step_inst.step_def_id}",
            )
            return PipelineResult(
                success=False, step_results=step_results,
                errors=[error], executed_up_to=i - 1,
            )

        # Check cache
        input_hash = _compute_input_hash(current_image)
        step_hash = _compute_step_hash(step_inst.step_def_id, step_inst.param_values, input_hash)

        cached = _step_cache.get(step_hash) if use_cache else None
        if cached is not None:
            img, side_outputs, out_type = cached
            result = StepResult(
                success=True, primary_output=img,
                side_outputs=side_outputs, output_type=out_type,
            )
        else:
            result = safe_execute_step(
                step_inst.step_def_id,
                step_inst.param_values,
                current_image,
                i,
            )

            # Store in cache if successful
            if result.success and use_cache:
                if len(_step_cache) >= _MAX_CACHE_ENTRIES:
                    # Evict oldest entries (simple strategy)
                    keys = list(_step_cache.keys())
                    for k in keys[:len(keys) // 2]:
                        del _step_cache[k]
                _step_cache[step_hash] = (
                    result.primary_output,
                    result.side_outputs,
                    result.output_type,
                )

        step_results.append(result)

        if not result.success:
            error = StepError(
                step_index=i, step_def_id=step_inst.step_def_id,
                error_code="E3005",
                message=result.warnings[0] if result.warnings else "Ismeretlen hiba",
            )
            return PipelineResult(
                success=False, step_results=step_results,
                errors=[error], executed_up_to=i,
            )

        current_image = result.primary_output

    return PipelineResult(
        success=True, step_results=step_results,
        executed_up_to=up_to_step,
    )


def clear_cache():
    """Clear the pipeline step cache."""
    _step_cache.clear()
