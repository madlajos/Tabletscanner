"""
Pipeline validation: structural and parameter validation for pipeline documents.
"""
from typing import List
from pipeline_types import StepError, StepInstance, PipelineDocument
from pipeline_steps import STEP_DEFINITIONS


def validate_pipeline(doc: PipelineDocument) -> List[StepError]:
    """
    Validate a full pipeline document.
    Returns a list of StepError objects (empty list = valid).
    """
    errors: List[StepError] = []

    if not doc.steps:
        errors.append(StepError(
            step_index=-1, step_def_id="",
            error_code="E3001",
            message="A feldolgozási lánc üres.",
        ))
        return errors

    # Check first step is load_image
    first = doc.steps[0]
    if first.step_def_id != "load_image":
        errors.append(StepError(
            step_index=0, step_def_id=first.step_def_id,
            error_code="E3001",
            message="Az első lépésnek 'Kép betöltése' típusúnak kell lennie.",
        ))

    for i, step_inst in enumerate(doc.steps):
        defn = STEP_DEFINITIONS.get(step_inst.step_def_id)
        if defn is None:
            errors.append(StepError(
                step_index=i, step_def_id=step_inst.step_def_id,
                error_code="E3001",
                message=f"Ismeretlen lépés típus: {step_inst.step_def_id}",
            ))
            continue

        # Validate required preceding steps
        if defn.required_preceding_steps:
            preceding_ids = {s.step_def_id for s in doc.steps[:i]}
            for req_id in defn.required_preceding_steps:
                if req_id not in preceding_ids:
                    req_defn = STEP_DEFINITIONS.get(req_id)
                    req_name = req_defn.name if req_defn else req_id
                    errors.append(StepError(
                        step_index=i,
                        step_def_id=step_inst.step_def_id,
                        error_code="E3004",
                        message=f"A(z) '{defn.name}' lépéshez szükséges egy '{req_name}' lépés előtte.",
                    ))

        # Validate secondary inputs (must also precede)
        if defn.secondary_inputs:
            preceding_ids = preceding_ids if defn.required_preceding_steps else {s.step_def_id for s in doc.steps[:i]}
            for sec_id in defn.secondary_inputs:
                if sec_id not in preceding_ids:
                    sec_defn = STEP_DEFINITIONS.get(sec_id)
                    sec_name = sec_defn.name if sec_defn else sec_id
                    errors.append(StepError(
                        step_index=i,
                        step_def_id=step_inst.step_def_id,
                        error_code="E3004",
                        message=f"A(z) '{defn.name}' lépéshez szükséges egy '{sec_name}' lépés előtte.",
                    ))

        # Validate parameters
        param_errors = _validate_params(i, step_inst, defn)
        errors.extend(param_errors)

    return errors


def _validate_params(step_index: int, inst: StepInstance, defn) -> List[StepError]:
    """Validate parameter values against the step definition's schema."""
    errors = []
    for ps in defn.params:
        value = inst.param_values.get(ps.name)

        # Required check
        if ps.required and (value is None or value == ""):
            errors.append(StepError(
                step_index=step_index,
                step_def_id=inst.step_def_id,
                error_code="E3003",
                message=f"Kötelező paraméter hiányzik: {ps.label}",
                param_name=ps.name,
            ))
            continue

        if value is None:
            continue

        # Type-specific validation
        if ps.type in ("int", "float"):
            try:
                num = float(value)
            except (ValueError, TypeError):
                errors.append(StepError(
                    step_index=step_index,
                    step_def_id=inst.step_def_id,
                    error_code="E3003",
                    message=f"'{ps.label}' értéke nem szám: {value}",
                    param_name=ps.name,
                ))
                continue

            if ps.min is not None and num < ps.min:
                errors.append(StepError(
                    step_index=step_index,
                    step_def_id=inst.step_def_id,
                    error_code="E3003",
                    message=f"'{ps.label}' értéke ({num}) kisebb a minimuménál ({ps.min})",
                    param_name=ps.name,
                ))
            if ps.max is not None and num > ps.max:
                errors.append(StepError(
                    step_index=step_index,
                    step_def_id=inst.step_def_id,
                    error_code="E3003",
                    message=f"'{ps.label}' értéke ({num}) nagyobb a maximuménál ({ps.max})",
                    param_name=ps.name,
                ))
            if ps.odd_only and ps.type == "int":
                ival = int(num)
                if ival % 2 == 0:
                    errors.append(StepError(
                        step_index=step_index,
                        step_def_id=inst.step_def_id,
                        error_code="E3003",
                        message=f"'{ps.label}' értékének páratlannak kell lennie: {ival}",
                        param_name=ps.name,
                    ))

        elif ps.type == "enum":
            if ps.options and str(value) not in ps.options:
                errors.append(StepError(
                    step_index=step_index,
                    step_def_id=inst.step_def_id,
                    error_code="E3003",
                    message=f"'{ps.label}' érvénytelen értéke: {value}. Engedélyezett: {ps.options}",
                    param_name=ps.name,
                ))

        elif ps.type == "bool":
            if not isinstance(value, bool):
                errors.append(StepError(
                    step_index=step_index,
                    step_def_id=inst.step_def_id,
                    error_code="E3003",
                    message=f"'{ps.label}' logikai értéknek kell lennie.",
                    param_name=ps.name,
                ))

    return errors
