from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def as_float_array(x: Any, name: str = "array") -> NDArray[np.float64]:
    """Convert to a contiguous float64 ndarray, raising on NaN or non-finite entries."""
    arr = np.ascontiguousarray(np.asarray(x, dtype=np.float64))
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name!r} contains non-finite values")
    return arr


def check_shapes(**arrays: NDArray[np.float64] | None) -> None:
    """Assert that all non-None inputs share the same leading dimension."""
    shapes = {k: v.shape[0] for k, v in arrays.items() if v is not None}
    if len(set(shapes.values())) > 1:
        raise ValueError(f"inconsistent leading dimensions: {shapes}")


def add_constant(X: NDArray[np.float64], prepend: bool = True) -> NDArray[np.float64]:
    """Prepend (or append) a column of ones to X."""
    ones = np.ones((X.shape[0], 1), dtype=np.float64)
    return np.hstack([ones, X]) if prepend else np.hstack([X, ones])


def ensure_panel(df: pd.DataFrame, unit: str, time: str) -> pd.DataFrame:
    """Sort a long-form DataFrame by (unit, time) and check no duplicates."""
    missing = [c for c in (unit, time) if c not in df.columns]
    if missing:
        raise KeyError(f"missing columns: {missing}")
    out = df.sort_values([unit, time]).reset_index(drop=True)
    if out.duplicated(subset=[unit, time]).any():
        raise ValueError(f"duplicate (unit, time) pairs found on ({unit}, {time})")
    return out
