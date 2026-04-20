from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from joblib import Memory

from haulage._logging import get_logger

_log = get_logger("data.cache")


def cache_dir() -> Path:
    """Root cache directory at $HAULAGE_CACHE or ~/.haulage/cache."""
    root = Path(os.environ.get("HAULAGE_CACHE", str(Path.home() / ".haulage" / "cache")))
    root.mkdir(parents=True, exist_ok=True)
    return root


_MEM: Memory | None = None


def get_cache() -> Memory:
    """Return the shared joblib.Memory instance."""
    global _MEM
    if _MEM is None:
        _MEM = Memory(location=str(cache_dir()), verbose=0)
    return _MEM


def _hash_frame(df: pd.DataFrame) -> str:
    blob = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    return hashlib.sha256(blob).hexdigest()


def cache_manifest(
    df: pd.DataFrame,
    *,
    source: str,
    endpoint: str,
    tag: str,
) -> dict[str, Any]:
    """Write a vintage manifest to the cache and return it.

    The manifest records fetch timestamp, source URL, endpoint, row count, and a
    SHA-256 hash of the returned frame. Used by `freeze.freeze` to pin a vintage.
    """
    manifest: dict[str, Any] = {
        "tag": tag,
        "source": source,
        "endpoint": endpoint,
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "columns": list(map(str, df.columns)),
        "sha256": _hash_frame(df),
        "fetched_at": datetime.now(tz=UTC).isoformat(),
    }
    manifests = cache_dir() / "manifests"
    manifests.mkdir(exist_ok=True)
    target = manifests / f"{tag}.json"
    target.write_text(json.dumps(manifest, indent=2))
    _log.info(
        "wrote manifest %s (rows=%d, sha256=%s...)",
        tag,
        manifest["rows"],
        str(manifest["sha256"])[:8],
    )
    return manifest
