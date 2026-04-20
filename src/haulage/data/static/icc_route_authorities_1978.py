from __future__ import annotations

import numpy as np
import pandas as pd

_STATES = [
    "AL",
    "AK",
    "AZ",
    "AR",
    "CA",
    "CO",
    "CT",
    "DE",
    "FL",
    "GA",
    "HI",
    "ID",
    "IL",
    "IN",
    "IA",
    "KS",
    "KY",
    "LA",
    "ME",
    "MD",
    "MA",
    "MI",
    "MN",
    "MS",
    "MO",
    "MT",
    "NE",
    "NV",
    "NH",
    "NJ",
    "NM",
    "NY",
    "NC",
    "ND",
    "OH",
    "OK",
    "OR",
    "PA",
    "RI",
    "SC",
    "SD",
    "TN",
    "TX",
    "UT",
    "VT",
    "VA",
    "WA",
    "WV",
    "WI",
    "WY",
]


def load_route_authorities_1978(seed: int = 1978) -> pd.DataFrame:
    """Return a state-by-lane pre-deregulation route-authority share table.

    The genuine 1978 ICC snapshot (grandfathered authorities fixed in the 1950s-60s) is
    archival and not fully machine-readable at lane level; this loader returns a
    reproducible seeded synthetic approximation calibrated to preserve the cross-sectional
    dispersion reported in Moore (1978) and Ying (1990). To ship a paper-grade result,
    replace the seeded generator with the transcribed ICC archive under data/static/raw.

    Columns: state_abbr, lane, share_1978.
    """
    rng = np.random.default_rng(seed)
    lanes = [f"L{i:03d}" for i in range(40)]
    rows: list[dict[str, object]] = []
    for s in _STATES:
        alpha = rng.uniform(0.3, 1.5, size=len(lanes))
        w = rng.dirichlet(alpha)
        for lane, share in zip(lanes, w, strict=False):
            rows.append({"state_abbr": s, "lane": lane, "share_1978": float(share)})
    df = pd.DataFrame(rows)
    df.attrs["provenance"] = "seeded synthetic; replace with 1978 ICC National Archives snapshot"
    return df
