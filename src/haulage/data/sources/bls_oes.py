from __future__ import annotations

import os
from collections.abc import Iterable

import pandas as pd
import requests

from haulage._logging import get_logger
from haulage.data.cache import cache_manifest, get_cache

_log = get_logger("data.bls_oes")

_BLS_TIMESERIES = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
_SOC_TRUCKING_DRIVERS = "53-3032"


def _key() -> str | None:
    return os.environ.get("BLS_API_KEY")


def load_oes_trucking(years: Iterable[int], soc: str = _SOC_TRUCKING_DRIVERS) -> pd.DataFrame:
    """State x year mean-hourly-wage and employment for heavy truckers from BLS OES.

    Requires BLS_API_KEY for v2 access (higher daily quota). Returns columns:
    state_abbr, year, wage_mean, employment, series_id.
    """
    mem = get_cache()
    years_t = tuple(sorted(years))
    df = mem.cache(_pull_oes)(years_t, soc)
    cache_manifest(
        df,
        source="https://www.bls.gov/oes/",
        endpoint="timeseries",
        tag=f"bls_oes_{min(years_t)}_{max(years_t)}_{soc}",
    )
    return df


def _pull_oes(years: tuple[int, ...], soc: str) -> pd.DataFrame:
    from haulage.data.static.state_dereg_dates import load_state_dereg_table

    states = load_state_dereg_table()["state_abbr"].tolist()
    series = [_series_id_oes_state(s, soc) for s in states]
    key = _key()
    payload: dict[str, object] = {
        "seriesid": series,
        "startyear": str(min(years)),
        "endyear": str(max(years)),
    }
    if key:
        payload["registrationkey"] = key
    _log.info("BLS OES pull for %d states x %d years", len(states), len(years))
    r = requests.post(_BLS_TIMESERIES, json=payload, timeout=60)
    r.raise_for_status()
    body = r.json()
    rows: list[dict[str, object]] = []
    for s in body.get("Results", {}).get("series", []):
        for p in s.get("data", []):
            rows.append({"series_id": s["seriesID"], "year": int(p["year"]), "value": float(p["value"])})
    return pd.DataFrame(rows)


def _series_id_oes_state(state: str, soc: str) -> str:
    """Build a state-level OES series ID; schema documented at BLS OES handbook."""
    return f"OEUS{state}00000000{soc.replace('-', '')}03"
