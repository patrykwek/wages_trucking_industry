from __future__ import annotations

import os
from collections.abc import Iterable

import pandas as pd

from haulage._logging import get_logger
from haulage.data.cache import cache_manifest, get_cache

_log = get_logger("data.fred")


def _client() -> object:
    try:
        from fredapi import Fred
    except ImportError as e:
        raise ImportError("fredapi not installed; `pip install fredapi`") from e
    key = os.environ.get("FRED_API_KEY")
    if not key:
        raise OSError(
            "FRED_API_KEY not set. Obtain a key at https://fred.stlouisfed.org/docs/api/ "
            "and export FRED_API_KEY before calling FRED loaders."
        )
    return Fred(api_key=key)


_CANONICAL: dict[str, str] = {
    "ppi_trucking": "PCU484484",
    "diesel_price": "GASDESM",
    "rail_freight_index": "RAILFRTINTERMODAL",
    "cpi_u": "CPIAUCSL",
    "unemployment_rate": "UNRATE",
    "real_gdp": "GDPC1",
}


def load_fred_series(names: Iterable[str]) -> pd.DataFrame:
    """Pull a collection of FRED series by canonical short name.

    Canonical names: ppi_trucking, diesel_price, rail_freight_index, cpi_u,
    unemployment_rate, real_gdp. Pass any raw FRED ID to fetch it directly.
    """
    mem = get_cache()
    names_t = tuple(sorted(set(names)))
    df = mem.cache(_pull)(names_t)
    cache_manifest(
        df, source="https://fred.stlouisfed.org", endpoint="series", tag=f"fred_{'_'.join(names_t)[:40]}"
    )
    return df


def _pull(names: tuple[str, ...]) -> pd.DataFrame:
    fred = _client()
    frames: list[pd.DataFrame] = []
    for name in names:
        series_id = _CANONICAL.get(name, name)
        _log.info("FRED pull %s -> %s", name, series_id)
        s = fred.get_series(series_id)  # type: ignore[attr-defined]
        frames.append(s.rename(name).to_frame())
    return pd.concat(frames, axis=1).sort_index()
