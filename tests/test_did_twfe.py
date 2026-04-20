from __future__ import annotations

import numpy as np

from haulage.estimators.did_twfe import TWFE


def test_twfe_recovers_att_simple_panel():
    rng = np.random.default_rng(42)
    units, periods = 50, 10
    rows = []
    for u in range(units):
        fe = rng.normal(0, 0.5)
        g = 6 if u < 25 else 9999
        for t in range(1, periods + 1):
            D = int(t >= g)
            y = fe + 0.05 * t + 0.6 * D + rng.normal(0, 0.2)
            rows.append({"unit": u, "t": t, "D": D, "y": y})
    import pandas as pd

    df = pd.DataFrame(rows)
    est = TWFE(treatment="D", outcome="y", unit="unit", time="t").fit(df)
    assert 0.3 < est.point < 0.9
    assert est.se > 0
