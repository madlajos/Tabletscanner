"""
Calibration curve persistence for fit/predict workflows.
Stored in backend/calibrations/calibrations.json.
"""

import json
import os
import re
import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

_cal_lock = threading.Lock()


def _calibrations_dir() -> str:
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibrations")
    os.makedirs(d, exist_ok=True)
    return d


def _calibrations_file() -> str:
    return os.path.join(_calibrations_dir(), "calibrations.json")


def _sanitize_name(name: str) -> str:
    safe = re.sub(r"[^\w\s\-]", "", str(name), flags=re.UNICODE).strip()
    return safe or "unnamed_calibration"


def _make_id(name: str) -> str:
    slug = re.sub(r"\s+", "-", _sanitize_name(name).lower())
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    return f"{slug}-{stamp}"


def _load_all() -> List[Dict]:
    path = _calibrations_file()
    if not os.path.isfile(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        return []
    except (OSError, json.JSONDecodeError):
        return []


def _save_all(records: List[Dict]) -> Tuple[bool, Optional[str]]:
    path = _calibrations_file()
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        return True, None
    except OSError as e:
        return False, f"Kalibráció mentési hiba: {e}"


def list_calibrations() -> List[Dict]:
    with _cal_lock:
        recs = _load_all()
    recs.sort(key=lambda r: str(r.get("created_at", "")), reverse=True)
    return recs


def save_calibration(payload: Dict) -> Tuple[Optional[Dict], Optional[str]]:
    name = _sanitize_name(payload.get("name", ""))
    equation = str(payload.get("equation", "")).strip()
    if not name:
        return None, "A kalibráció neve nem lehet üres."
    if not equation:
        return None, "A kalibráció egyenlete nem lehet üres."

    now = datetime.now(timezone.utc).isoformat()
    record = {
        "id": _make_id(name),
        "name": name,
        "equation": equation,
        "comment": str(payload.get("comment", "")).strip(),
        "x_name": str(payload.get("x_name", "x")).strip() or "x",
        "y_name": str(payload.get("y_name", "y")).strip() or "y",
        "y_key": str(payload.get("y_key", "")).strip(),
        "model": str(payload.get("model", "")).strip(),
        "degree": int(payload.get("degree", 1)) if payload.get("degree") is not None else 1,
        "coefficients": payload.get("coefficients") if isinstance(payload.get("coefficients"), list) else [],
        "x_min": payload.get("x_min"),
        "x_max": payload.get("x_max"),
        "created_at": now,
    }

    with _cal_lock:
        recs = _load_all()
        recs.append(record)
        ok, err = _save_all(recs)
        if not ok:
            return None, err

    return record, None
