from __future__ import annotations

import numpy as np

from haulage.influence import InfluenceFunction


def test_influence_variance_matches_mean_square():
    rng = np.random.default_rng(0)
    psi = rng.standard_normal(1000)
    infl = InfluenceFunction(values=psi, tau_hat=1.5, name="unit")
    assert abs(infl.variance - np.mean(psi**2)) < 1e-12
    assert abs(infl.se - np.sqrt(infl.variance / 1000)) < 1e-12


def test_influence_ci_is_symmetric():
    psi = np.ones(500) * 0.5
    infl = InfluenceFunction(values=psi, tau_hat=2.0, name="u")
    lo, hi = infl.ci(alpha=0.05)
    assert abs((lo + hi) / 2 - 2.0) < 1e-10


def test_influence_cluster_se_matches_analytic_iid_limit():
    rng = np.random.default_rng(2)
    n = 800
    psi = rng.standard_normal(n)
    infl = InfluenceFunction(values=psi, tau_hat=0.0, name="u")
    clusters = np.arange(n, dtype=np.int64)
    cl_se = infl.cluster_se(clusters)
    assert abs(cl_se - infl.se) < 1e-8
