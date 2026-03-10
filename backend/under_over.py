
import numpy as np
import cv2

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

def _to_gray_u8(frame_bgr):
    if frame_bgr is None or getattr(frame_bgr, "size", 0) == 0:
        return None
    if frame_bgr.ndim == 2:
        return frame_bgr
    if frame_bgr.ndim == 3 and frame_bgr.shape[2] == 3:
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    if frame_bgr.ndim == 3 and frame_bgr.shape[2] == 4:
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGRA2GRAY)
    return None


def exposure_gate(
    gray_u8,
    white_thr=250,
    frac_white_thr=0.20,
    under_p95_thr=30,
    under_dr_thr=30,
    # extra "nagyon sötét" kapu
    under_p10_thr=10,
    under_p95_hi=55,
    under_p99_thr=90,
):
    """
    Return: _ok(...) or _err("E2002"/"E2003", ...)
    Extra mezők: p95, dr, white, (opcionálisan p10, p99)
    """
    if gray_u8 is None or getattr(gray_u8, "size", 0) == 0:
        return _err("E2302", p95=0.0, dr=0.0, white=0.0)

    g = gray_u8.astype(np.float32)
    p1, p10, p95, p99 = np.percentile(g, [1, 10, 95, 99])
    dr = float(p99 - p1)

    frac_white = float((gray_u8 >= int(white_thr)).mean())
    metrics = {
        "p95": float(p95),
        "dr": dr,
        "white": frac_white,
        "p10": float(p10),
        "p99": float(p99),
    }

    # OVER -> E2003
    if frac_white > float(frac_white_thr) or p95 > 240:
        return _err("E2303", **metrics)

    # UNDER -> E2002 (sötét + lapos)
    if (p95 < float(under_p95_thr)) and (dr < float(under_dr_thr)):
        return _err("E2302", **metrics)

    # EXTRA UNDER -> E2002 (nagyon sötét összkép)
    if (p10 < float(under_p10_thr)) and (p95 < float(under_p95_hi)) and (p99 < float(under_p99_thr)):
        return _err("E2302", **metrics)

    return _ok(**metrics)


def exposure_gate_from_frame(frame_bgr, **kwargs):
    """
    Input: BGR frame
    Output: _ok(...) or _err(...)
    """
    gray = _to_gray_u8(frame_bgr)

    return exposure_gate(gray, **kwargs)