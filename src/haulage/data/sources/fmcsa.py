from __future__ import annotations

import pandas as pd
import requests

from haulage.data.cache import cache_manifest, get_cache

_BASE = "https://mobile.fmcsa.dot.gov/qc/services"


def load_fmcsa_carriers(usdot_numbers: list[int]) -> pd.DataFrame:
    """Carrier-level registration and safety-rating data from FMCSA QCMobile.

    Public; no auth required for basic queries, though rate-limited.
    """
    mem = get_cache()
    df = mem.cache(_pull)(tuple(sorted(usdot_numbers)))
    cache_manifest(df, source=_BASE, endpoint="carriers", tag=f"fmcsa_{len(usdot_numbers)}")
    return df


def _pull(usdots: tuple[int, ...]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dot in usdots:
        r = requests.get(f"{_BASE}/carriers/{dot}?webKey=PUBLIC", timeout=30)
        if r.status_code == 200:
            rows.append({"usdot": dot, **r.json().get("content", {})})
    return pd.DataFrame(rows)
