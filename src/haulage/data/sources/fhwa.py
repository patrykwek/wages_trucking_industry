from __future__ import annotations

from collections.abc import Iterable

import pandas as pd
import requests

from haulage.data.cache import cache_manifest, get_cache

_VMT_URL = "https://www.fhwa.dot.gov/policyinformation/travel_monitoring/{yr}tvt.cfm"


def load_vmt(years: Iterable[int]) -> pd.DataFrame:
    """Monthly VMT by state from FHWA Travel Monitoring Analysis."""
    mem = get_cache()
    years_t = tuple(sorted(years))
    df = mem.cache(_pull)(years_t)
    cache_manifest(
        df,
        source="https://www.fhwa.dot.gov",
        endpoint="travel_monitoring",
        tag=f"fhwa_vmt_{min(years_t)}_{max(years_t)}",
    )
    return df


def _pull(years: tuple[int, ...]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for y in years:
        r = requests.get(_VMT_URL.format(yr=y), timeout=60)
        if r.ok:
            try:
                frames.append(pd.read_html(r.text)[0])
            except Exception:
                pass
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
