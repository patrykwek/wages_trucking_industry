from __future__ import annotations

import os
from collections.abc import Iterable

import pandas as pd
import requests

from haulage.data.cache import cache_manifest, get_cache

_BLS_TIMESERIES = "https://api.bls.gov/publicAPI/v2/timeseries/data/"


def load_ces_trucking(years: Iterable[int]) -> pd.DataFrame:
    """NAICS 484 state-level employment and payrolls from BLS CES/QCEW.

    Requires BLS_API_KEY; returns columns state_abbr, year, employment, payroll.
    """
    mem = get_cache()
    years_t = tuple(sorted(years))
    df = mem.cache(_pull)(years_t)
    cache_manifest(
        df,
        source="https://www.bls.gov/ces/",
        endpoint="timeseries",
        tag=f"bls_ces_{min(years_t)}_{max(years_t)}",
    )
    return df


def _pull(years: tuple[int, ...]) -> pd.DataFrame:
    key = os.environ.get("BLS_API_KEY")
    payload: dict[str, object] = {
        "seriesid": ["CES4348400001"],
        "startyear": str(min(years)),
        "endyear": str(max(years)),
    }
    if key:
        payload["registrationkey"] = key
    r = requests.post(_BLS_TIMESERIES, json=payload, timeout=60)
    r.raise_for_status()
    rows = [
        {"series_id": s["seriesID"], "year": int(p["year"]), "value": float(p["value"])}
        for s in r.json().get("Results", {}).get("series", [])
        for p in s.get("data", [])
    ]
    return pd.DataFrame(rows)
