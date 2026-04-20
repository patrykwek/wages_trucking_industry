from __future__ import annotations

import pandas as pd


def load_freight_analysis_framework(year: int) -> pd.DataFrame:
    """State-level ton-mile data by commodity from BTS Freight Analysis Framework.

    Public CSV distributions, downloaded on demand. Schema documented at
    https://ops.fhwa.dot.gov/freight/freight_analysis/faf/.
    """
    raise NotImplementedError(
        "FAF tables ship as multi-sheet Excel workbooks; implement the year-specific "
        "parser in a follow-up PR."
    )
