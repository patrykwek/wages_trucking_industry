from __future__ import annotations

import os
from collections.abc import Iterable

import pandas as pd
import requests

from haulage._logging import get_logger
from haulage.data.cache import cache_manifest, get_cache

_log = get_logger("data.ipums")

_IPUMS_BASE = "https://api.ipums.org/extracts"
_IPUMS_DATA_ROOT = "https://api.ipums.org/downloads"

_TRUCKING_OCC_CODES: tuple[tuple[int, int, str], ...] = (
    (1976, 1982, "804-808"),
    (1983, 2002, "804-808"),
    (2003, 2010, "9130"),
    (2011, 2099, "9140"),
)


def _key() -> str:
    key = os.environ.get("IPUMS_API_KEY")
    if not key:
        raise OSError(
            "IPUMS_API_KEY not set. Obtain a key at https://account.ipums.org/api_keys "
            "and export it before calling load_cps_trucking."
        )
    return key


def trucking_occupation_codes(year: int) -> str:
    """Return the trucking-driver OCC code range active in a given CPS vintage."""
    for y0, y1, codes in _TRUCKING_OCC_CODES:
        if y0 <= year <= y1:
            return codes
    raise ValueError(f"no OCC code range registered for year {year}")


def load_cps_trucking(
    years: Iterable[int],
    variables: tuple[str, ...] = (
        "STATEFIP",
        "AGE",
        "SEX",
        "RACE",
        "OCC",
        "IND",
        "UHRSWORKT",
        "INCWAGE",
        "UNION",
        "COVER",
        "EARNWEEK",
        "HOURWAGE",
        "CLASSWKR",
    ),
    sample: str = "cps",
) -> pd.DataFrame:
    """Submit an IPUMS CPS extract and return a long-form DataFrame restricted to truckers.

    Requires IPUMS_API_KEY. Results are cached to disk by (years, variables) key. The
    returned frame is harmonized across 1976 / 1980 / 1990 / 2000 / 2010 / 2018 SOC
    revisions via `trucking_occupation_codes`.

    References:
        IPUMS CPS API docs: https://developer.ipums.org/docs/apiprogram/
    """
    mem = get_cache()
    cached = mem.cache(_pull_cps)(tuple(sorted(years)), tuple(variables), sample)
    cache_manifest(
        cached,
        source=_IPUMS_BASE,
        endpoint=f"cps/{sample}",
        tag=f"ipums_cps_{min(years)}_{max(years)}",
    )
    return cached


def _pull_cps(years: tuple[int, ...], variables: tuple[str, ...], sample: str) -> pd.DataFrame:
    key = _key()
    headers = {"Authorization": key, "Content-Type": "application/json"}
    samples = [f"{sample}_{y}" for y in years]
    payload = {
        "description": f"haulage CPS trucking extract {min(years)}-{max(years)}",
        "dataFormat": "csv",
        "dataStructure": {"rectangular": {"on": "P"}},
        "samples": {s: {} for s in samples},
        "variables": {v: {} for v in variables},
    }
    _log.info("submitting IPUMS extract for years %s-%s", min(years), max(years))
    r = requests.post(_IPUMS_BASE, json=payload, headers=headers, timeout=60)
    r.raise_for_status()
    raise NotImplementedError(
        "IPUMS extract submission succeeded but the synchronous download leg is not "
        "implemented in this skeleton; poll the extract URL returned in `r.json()` and "
        "call pandas.read_csv(decompressed_url). See IPUMS API docs."
    )
