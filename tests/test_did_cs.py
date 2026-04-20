from __future__ import annotations

import numpy as np
import pandas as pd

from haulage.estimators.did_cs import CallawaySantAnna


def _simulate_cs_dgp(seed: int, att: float = 0.5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    cohorts = {i: 5 if i < 15 else (9 if i < 30 else 9999) for i in range(60)}
    for u, g in cohorts.items():
        fe = rng.normal(0, 0.3)
        for t in range(1, 13):
            D = int(t >= g)
            y = fe + 0.03 * t + att * D + rng.normal(0, 0.2)
            rows.append({"unit": u, "t": t, "D": D, "y": y})
    return pd.DataFrame(rows)


def test_cs_overall_att_in_expected_range():
    df = _simulate_cs_dgp(seed=1, att=0.5)
    est = CallawaySantAnna(method="dr", control="nevertreated").fit(df)
    assert 0.3 < est.point < 0.8
    assert est.se > 0
    assert est.diagnostics["n_gt_cells"] > 0


def test_cs_event_study_has_pre_and_post():
    df = _simulate_cs_dgp(seed=2, att=0.5)
    cs = CallawaySantAnna(method="dr")
    cs.fit(df)
    es = cs.event_study()
    assert (es["event_time"] < 0).any()
    assert (es["event_time"] >= 0).any()


def test_cs_methods_agree_roughly():
    df = _simulate_cs_dgp(seed=3, att=0.4)
    out = [CallawaySantAnna(method=m).fit(df).point for m in ["dr", "ipw", "or"]]
    diffs = np.ptp(out)
    assert diffs < 0.25, f"CS methods disagree by {diffs:.3f}: {out}"
