from __future__ import annotations

from collections.abc import Iterable

import pandas as pd
import requests

from haulage.data.cache import cache_manifest, get_cache

_BASE = "https://api.census.gov/data"


def load_cbp_trucking(years: Iterable[int]) -> pd.DataFrame:
    """County Business Patterns, NAICS 484 establishment counts by size class.

    Used for GVW/firm-size RDD specifications. No auth required for basic extracts
    (the Census API key is optional and raises the daily quota).
    """
    mem = get_cache()
    years_t = tuple(sorted(years))
    df = mem.cache(_pull)(years_t)
    cache_manifest(df, source=_BASE, endpoint="cbp", tag=f"cbp_trucking_{min(years_t)}_{max(years_t)}")
    return df


def _pull(years: tuple[int, ...]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for y in years:
        url = f"{_BASE}/{y}/cbp?get=NAME,NAICS2017_LABEL,ESTAB,EMPSZES_LABEL&NAICS2017=484&for=state:*"
        r = requests.get(url, timeout=60)
        if not r.ok:
            continue
        rows = r.json()
        header, body = rows[0], rows[1:]
        frames.append(pd.DataFrame(body, columns=header).assign(year=y))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
