"""
Recipe persistence: save, load, list, delete recipe JSON documents.
Recipes are stored in a 'recipes' subdirectory next to this file.
"""
import json
import os
import re
import sys
import threading
import uuid
from datetime import datetime, timezone
from typing import List, Optional, Tuple

from pipeline_types import PipelineDocument

import logging
logger = logging.getLogger(__name__)

_recipe_lock = threading.Lock()
_FOLDER_METADATA_FILENAME = ".recipe-folders.json"
_FOLDER_METADATA_SCHEMA_VERSION = 1

def _recipes_dir() -> str:
    """Get (and ensure) the persistent recipes directory path.

    In a PyInstaller build ``__file__`` points into a temporary ``_MEI...``
    extraction directory which is removed when the application exits.  Store
    recipes next to the executable instead so they survive restarts.
    """
    if getattr(sys, "frozen", False):
        base_dir = os.path.dirname(os.path.abspath(sys.executable))
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    d = os.path.join(base_dir, "recipes")
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


def _folder_metadata_path() -> str:
    return os.path.join(_recipes_dir(), _FOLDER_METADATA_FILENAME)


def _empty_folder_state() -> dict:
    return {
        "schema_version": _FOLDER_METADATA_SCHEMA_VERSION,
        "folders": [],
        "recipe_folders": {},
    }


def _load_folder_state_locked() -> dict:
    path = _folder_metadata_path()
    if not os.path.isfile(path):
        return _empty_folder_state()
    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read recipe folder metadata: %s", exc)
        return _empty_folder_state()

    folders = data.get("folders", [])
    assignments = data.get("recipe_folders", {})
    if not isinstance(folders, list) or not isinstance(assignments, dict):
        logger.warning("Ignoring invalid recipe folder metadata structure")
        return _empty_folder_state()
    valid_folders = [
        {"id": str(folder["id"]), "name": str(folder["name"])}
        for folder in folders
        if isinstance(folder, dict) and folder.get("id") and folder.get("name")
    ]
    valid_ids = {folder["id"] for folder in valid_folders}
    return {
        "schema_version": _FOLDER_METADATA_SCHEMA_VERSION,
        "folders": valid_folders,
        "recipe_folders": {
            str(name): str(folder_id)
            for name, folder_id in assignments.items()
            if str(folder_id) in valid_ids
        },
    }


def _write_folder_state_locked(state: dict) -> None:
    """Atomically persist organizational metadata without changing recipe documents."""
    path = _folder_metadata_path()
    temp_path = f"{path}.{uuid.uuid4().hex}.tmp"
    try:
        with open(temp_path, "w", encoding="utf-8") as file:
            json.dump(state, file, ensure_ascii=False, indent=2)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def _validate_folder_name(name: str) -> Tuple[Optional[str], Optional[str]]:
    normalized = str(name).strip()
    if not normalized:
        return None, "A mappa neve nem lehet üres."
    if len(normalized) > 80:
        return None, "A mappa neve legfeljebb 80 karakter lehet."
    return normalized, None


def list_recipes() -> List[dict]:
    """List all saved recipes with summary info."""
    recipes = []
    recipes_path = _recipes_dir()
    with _recipe_lock:
        folder_state = _load_folder_state_locked()
        assignments = folder_state["recipe_folders"]
        for fname in os.listdir(recipes_path):
            if not fname.endswith(".json") or fname == _FOLDER_METADATA_FILENAME:
                continue
            fpath = os.path.join(recipes_path, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                recipe_name = data.get("name", fname[:-5])
                recipes.append({
                    "name": recipe_name,
                    "description": data.get("description", ""),
                    "step_count": len(data.get("steps", [])),
                    "modified_at": data.get("modified_at", ""),
                    "folder_id": assignments.get(recipe_name),
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
            state = _load_folder_state_locked()
            if state["recipe_folders"].pop(name, None) is not None:
                _write_folder_state_locked(state)
            return True, None
        except OSError as e:
            return False, f"Recept törlési hiba: {e}"


def update_recipe_description(name: str, description: str) -> Tuple[bool, Optional[str]]:
    """Update only the description field of a recipe."""
    path = _recipe_path(name)
    with _recipe_lock:
        if not os.path.isfile(path):
            return False, f"A recept nem található: {name}"
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["description"] = description
            data["modified_at"] = datetime.now(timezone.utc).isoformat()
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True, None
        except (json.JSONDecodeError, OSError) as e:
            return False, f"Recept módosítási hiba: {e}"


def list_recipe_folders() -> List[dict]:
    with _recipe_lock:
        return list(_load_folder_state_locked()["folders"])


def create_recipe_folder(name: str) -> Tuple[Optional[dict], Optional[str]]:
    normalized, error = _validate_folder_name(name)
    if error:
        return None, error
    with _recipe_lock:
        state = _load_folder_state_locked()
        if any(folder["name"].casefold() == normalized.casefold() for folder in state["folders"]):
            return None, "Már létezik ilyen nevű receptmappa."
        folder = {"id": uuid.uuid4().hex, "name": normalized}
        state["folders"].append(folder)
        try:
            _write_folder_state_locked(state)
        except OSError as exc:
            return None, f"Receptmappa mentési hiba: {exc}"
        return folder, None


def rename_recipe_folder(folder_id: str, name: str) -> Tuple[Optional[dict], Optional[str]]:
    normalized, error = _validate_folder_name(name)
    if error:
        return None, error
    with _recipe_lock:
        state = _load_folder_state_locked()
        folder = next((item for item in state["folders"] if item["id"] == folder_id), None)
        if folder is None:
            return None, "A receptmappa nem található."
        if any(
            item["id"] != folder_id and item["name"].casefold() == normalized.casefold()
            for item in state["folders"]
        ):
            return None, "Már létezik ilyen nevű receptmappa."
        folder["name"] = normalized
        try:
            _write_folder_state_locked(state)
        except OSError as exc:
            return None, f"Receptmappa mentési hiba: {exc}"
        return dict(folder), None


def delete_recipe_folder(folder_id: str) -> Tuple[bool, Optional[str]]:
    """Delete a folder while retaining its recipes as unfiled recipes."""
    with _recipe_lock:
        state = _load_folder_state_locked()
        if not any(folder["id"] == folder_id for folder in state["folders"]):
            return False, "A receptmappa nem található."
        state["folders"] = [folder for folder in state["folders"] if folder["id"] != folder_id]
        state["recipe_folders"] = {
            name: assigned_id
            for name, assigned_id in state["recipe_folders"].items()
            if assigned_id != folder_id
        }
        try:
            _write_folder_state_locked(state)
        except OSError as exc:
            return False, f"Receptmappa mentési hiba: {exc}"
        return True, None


def assign_recipe_folder(name: str, folder_id: Optional[str]) -> Tuple[bool, Optional[str]]:
    with _recipe_lock:
        if not os.path.isfile(_recipe_path(name)):
            return False, f"A recept nem található: {name}"
        state = _load_folder_state_locked()
        if folder_id is not None and not any(
            folder["id"] == folder_id for folder in state["folders"]
        ):
            return False, "A receptmappa nem található."
        if folder_id is None:
            state["recipe_folders"].pop(name, None)
        else:
            state["recipe_folders"][name] = folder_id
        try:
            _write_folder_state_locked(state)
        except OSError as exc:
            return False, f"Receptmappa mentési hiba: {exc}"
        return True, None


def duplicate_recipe(name: str) -> Tuple[Optional[str], Optional[str]]:
    """Duplicate a recipe, returning (new_name, None) on success or (None, error)."""
    path = _recipe_path(name)
    with _recipe_lock:
        if not os.path.isfile(path):
            return None, f"A recept nem található: {name}"
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            return None, f"Recept olvasási hiba: {e}"

    base_new_name = f"{name} (másolat)"
    new_name = base_new_name
    counter = 2
    while os.path.isfile(_recipe_path(new_name)):
        new_name = f"{base_new_name} {counter}"
        counter += 1

    now = datetime.now(timezone.utc).isoformat()
    data["name"] = new_name
    data["created_at"] = now
    data["modified_at"] = now

    new_path = _recipe_path(new_name)
    with _recipe_lock:
        try:
            with open(new_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            state = _load_folder_state_locked()
            source_folder = state["recipe_folders"].get(name)
            if source_folder:
                state["recipe_folders"][new_name] = source_folder
                _write_folder_state_locked(state)
            return new_name, None
        except OSError as e:
            return None, f"Recept másolási hiba: {e}"
