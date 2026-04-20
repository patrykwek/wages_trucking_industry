from __future__ import annotations

import numpy as np

from haulage.estimators.did_bacon import GoodmanBacon


def test_bacon_decomposition_weights_sum_to_one(staggered_panel):
    decomp = GoodmanBacon(treatment="D", outcome="y", unit="unit", time="t").fit(staggered_panel)
    total_w = sum(c.weight for c in decomp.components)
    assert abs(total_w - 1.0) < 1e-8 or total_w == 0.0
    assert np.isfinite(decomp.total)


def test_bacon_has_all_three_comparison_kinds(staggered_panel):
    decomp = GoodmanBacon(treatment="D", outcome="y", unit="unit", time="t").fit(staggered_panel)
    kinds = {c.kind for c in decomp.components}
    assert "treated_vs_never" in kinds
    assert "early_vs_later_control" in kinds
    assert "later_vs_already_treated" in kinds
