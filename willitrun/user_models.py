"""Models people add themselves.

`willitrun add org/repo` reads the model's config from Hugging Face and saves
it here, so it shows up in every scan after that. The file is plain JSON in
the same format as the built-in models.json, so it can be edited by hand.
"""

from __future__ import annotations

import json
import os
import platform

from .catalog import CATALOG, ModelSpec, spec_from_dict, spec_to_dict


def models_path() -> str:
    home = os.environ.get("WILLITRUN_HOME")
    if not home:
        if platform.system() == "Windows":
            home = os.path.join(os.environ.get("APPDATA") or os.path.expanduser("~"), "willitrun")
        else:
            base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
            home = os.path.join(base, "willitrun")
    return os.path.join(home, "models.json")


def load() -> list[ModelSpec]:
    path = models_path()
    if not os.path.exists(path):
        return []
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return [spec_from_dict(d) for d in data]
    except (OSError, ValueError, TypeError, KeyError) as e:
        raise SystemExit(f"Couldn't read your models file {path}: {e}\n"
                         "Fix or delete it, then run willitrun again.")


def save(models: list[ModelSpec]) -> str:
    path = models_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = [json.dumps(spec_to_dict(m)) for m in models]
    with open(path, "w", encoding="utf-8") as f:
        f.write("[\n" + ",\n".join(f"  {r}" for r in rows) + "\n]\n")
    return path


def add(spec: ModelSpec) -> tuple[str, bool]:
    """Save a model, replacing one with the same id. Returns (path, replaced)."""
    models = load()
    replaced = any(m.id == spec.id for m in models)
    models = [m for m in models if m.id != spec.id] + [spec]
    return save(models), replaced


def remove(model_id: str) -> bool:
    models = load()
    keep = [m for m in models if m.id != model_id]
    if len(keep) == len(models):
        return False
    save(keep)
    return True


def full_catalog() -> tuple[list[ModelSpec], set[str]]:
    """Built-in plus user models. A user model with a built-in id replaces it."""
    mine = load()
    ids = {m.id for m in mine}
    return [m for m in CATALOG if m.id not in ids] + mine, ids
