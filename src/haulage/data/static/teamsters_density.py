from __future__ import annotations

import numpy as np
import pandas as pd

_ANCHORS: dict[int, float] = {
    1970: 0.75,
    1975: 0.72,
    1980: 0.60,
    1985: 0.40,
    1990: 0.28,
    1995: 0.22,
    2000: 0.18,
    2005: 0.15,
    2010: 0.13,
    2015: 0.11,
    2020: 0.10,
    2025: 0.09,
}


def load_teamsters_density() -> pd.DataFrame:
    """Return an annual IBT-density series 1970-2025 from archival UIC/BLS anchors.

    Linearly interpolated between 5-year anchor points. Columns: year, density.
    """
    years = np.arange(1970, 2026)
    anchor_years = np.array(sorted(_ANCHORS))
    anchor_vals = np.array([_ANCHORS[y] for y in anchor_years])
    vals = np.interp(years, anchor_years, anchor_vals)
    return pd.DataFrame({"year": years, "density": vals})
