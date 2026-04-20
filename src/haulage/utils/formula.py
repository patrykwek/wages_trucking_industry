from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def design_from_formula(
    df: pd.DataFrame,
    lhs: str,
    rhs: list[str],
    add_const: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.float64], list[str]]:
    """Build (y, X, column_names) from a DataFrame and a simple list of column names.

    Categorical columns are one-hot encoded with the first level dropped.
    """
    y = df[lhs].to_numpy(dtype=np.float64)
    pieces: list[NDArray[np.float64]] = []
    names: list[str] = []
    if add_const:
        pieces.append(np.ones((df.shape[0], 1)))
        names.append("const")
    for col in rhs:
        if col not in df.columns:
            raise KeyError(f"column {col!r} not in frame")
        s = df[col]
        if isinstance(s.dtype, pd.CategoricalDtype) or s.dtype == object:
            dummies = pd.get_dummies(s, prefix=col, drop_first=True)
            pieces.append(dummies.to_numpy(dtype=np.float64))
            names.extend(dummies.columns.tolist())
        else:
            pieces.append(s.to_numpy(dtype=np.float64).reshape(-1, 1))
            names.append(col)
    X = np.hstack(pieces)
    return y, X, names
