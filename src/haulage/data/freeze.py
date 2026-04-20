from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from haulage.data.cache import cache_dir


def freeze(tag: str) -> Path:
    """Pin the current cache contents under a named vintage.

    A vintage is a hard-copy of the `manifests/` directory plus a top-level lockfile
    recording the vintage tag. Downstream code calls `load_vintage(tag)` to read back
    the exact manifests used in a paper.
    """
    src = cache_dir() / "manifests"
    dst = cache_dir() / "vintages" / tag
    if dst.exists():
        raise FileExistsError(f"vintage {tag!r} already exists at {dst}")
    dst.mkdir(parents=True)
    for m in src.glob("*.json"):
        shutil.copy2(m, dst / m.name)
    (dst / "vintage.json").write_text(json.dumps({"tag": tag}, indent=2))
    return dst


def load_vintage(tag: str) -> list[dict[str, Any]]:
    """Load all manifests belonging to a frozen vintage."""
    root = cache_dir() / "vintages" / tag
    if not root.exists():
        raise FileNotFoundError(f"no vintage {tag!r} at {root}")
    return [json.loads(m.read_text()) for m in sorted(root.glob("*.json")) if m.name != "vintage.json"]
