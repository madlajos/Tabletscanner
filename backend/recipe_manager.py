"""
Recipe persistence: save, load, list, delete recipe JSON documents.
Recipes are stored in a 'recipes' subdirectory next to this file.
"""
import json
import os
import re
import threading
from datetime import datetime, timezone
from typing import List, Optional, Tuple

from pipeline_types import PipelineDocument

import logging
logger = logging.getLogger(__name__)

_recipe_lock = threading.Lock()

def _recipes_dir() -> str:
    """Get (and ensure) the recipes directory path."""
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recipes")
    os.makedirs(d, exist_ok=True)
    return d


def _sanitize_name(name: str) -> str:
    """Sanitize recipe name for filesystem safety."""
    # Keep alphanumeric, spaces, hyphens, underscores; strip the rest
    safe = re.sub(r'[^\w\s\-]', '', name, flags=re.UNICODE)
    safe = safe.strip()
    if not safe:
        safe = "unnamed_recipe"
    return safe


def _recipe_path(name: str) -> str:
    return os.path.join(_recipes_dir(), f"{_sanitize_name(name)}.json")


def list_recipes() -> List[dict]:
    """List all saved recipes with summary info."""
    recipes = []
    recipes_path = _recipes_dir()
    with _recipe_lock:
        for fname in os.listdir(recipes_path):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(recipes_path, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                recipes.append({
                    "name": data.get("name", fname[:-5]),
                    "description": data.get("description", ""),
                    "step_count": len(data.get("steps", [])),
                    "modified_at": data.get("modified_at", ""),
                })
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Skipping invalid recipe file {fname}: {e}")
    return recipes


def load_recipe(name: str) -> Tuple[Optional[PipelineDocument], Optional[str]]:
    """
    Load a recipe by name.
    Returns (PipelineDocument, None) on success, (None, error_message) on failure.
    """
    path = _recipe_path(name)
    with _recipe_lock:
        if not os.path.isfile(path):
            return None, f"A recept nem található: {name}"
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            doc = PipelineDocument.from_dict(data)
            return doc, None
        except json.JSONDecodeError as e:
            return None, f"Érvénytelen JSON formátum: {e}"
        except Exception as e:
            return None, f"Recept betöltési hiba: {e}"


def save_recipe(doc: PipelineDocument) -> Tuple[bool, Optional[str]]:
    """
    Save a recipe. Name is taken from doc.name.
    Returns (True, None) on success, (False, error_message) on failure.
    """
    if not doc.name:
        return False, "A recept neve nem lehet üres."

    now = datetime.now(timezone.utc).isoformat()
    if not doc.created_at:
        doc.created_at = now
    doc.modified_at = now

    path = _recipe_path(doc.name)
    data = doc.to_dict()

    with _recipe_lock:
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True, None
        except OSError as e:
            return False, f"Recept mentési hiba: {e}"


def delete_recipe(name: str) -> Tuple[bool, Optional[str]]:
    """Delete a recipe by name."""
    path = _recipe_path(name)
    with _recipe_lock:
        if not os.path.isfile(path):
            return False, f"A recept nem található: {name}"
        try:
            os.remove(path)
            return True, None
        except OSError as e:
            return False, f"Recept törlési hiba: {e}"
