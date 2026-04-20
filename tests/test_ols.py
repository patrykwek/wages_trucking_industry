from __future__ import annotations

import numpy as np
import pytest
import statsmodels.api as sm

from haulage.estimators.ols import OLS


def test_ols_matches_numpy_lstsq(simple_ols_data):
    y, X, beta_true = simple_ols_data
    res = OLS(se="hc0").fit(y, X)
    beta_np, *_ = np.linalg.lstsq(X, y, rcond=None)
    np.testing.assert_allclose(res.coef, beta_np, atol=1e-9)


@pytest.mark.parametrize("kind", ["nonrobust", "hc0", "hc1", "hc2", "hc3"])
def test_ols_matches_statsmodels(simple_ols_data, kind):
    y, X, _ = simple_ols_data
    sm_kind = {"nonrobust": "nonrobust", "hc0": "HC0", "hc1": "HC1", "hc2": "HC2", "hc3": "HC3"}[kind]
    sm_res = sm.OLS(y, X).fit(cov_type=sm_kind)
    our = OLS(se=kind, coef_of_interest=1).fit(y, X)
    np.testing.assert_allclose(our.coef, sm_res.params, atol=1e-8)
    np.testing.assert_allclose(our.cov, sm_res.cov_params(), rtol=1e-7, atol=1e-10)


def test_ols_cluster_se_reasonable(cluster_panel):
    y = cluster_panel["y"].to_numpy(dtype=np.float64)
    X = np.column_stack([np.ones(len(cluster_panel)), cluster_panel["x"].to_numpy()])
    clusters = cluster_panel["g"].to_numpy(dtype=np.int64)
    res_cl = OLS(se="cluster").fit(y, X, clusters=clusters)
    res_hc = OLS(se="hc1").fit(y, X)
    # intra-cluster correlation should inflate SE relative to heteroskedastic
    assert res_cl.coef_se(1) > res_hc.coef_se(1) * 0.95


def test_ols_hac_positive_semidefinite(simple_ols_data):
    y, X, _ = simple_ols_data
    res = OLS(se="hac", kernel="bartlett", hac_lags=5).fit(y, X)
    eigs = np.linalg.eigvalsh((res.cov + res.cov.T) / 2)
    assert eigs.min() > -1e-8


def test_ols_driscoll_kraay_shape(cluster_panel):
    y = cluster_panel["y"].to_numpy(dtype=np.float64)
    X = np.column_stack([np.ones(len(cluster_panel)), cluster_panel["x"].to_numpy()])
    time = cluster_panel["t"].to_numpy(dtype=np.int64)
    res = OLS(se="driscoll_kraay", hac_lags=3).fit(y, X, time=time)
    assert res.cov.shape == (2, 2)
    assert res.coef_se(1) > 0


def test_ols_wild_bootstrap():
    rng = np.random.default_rng(0)
    n = 400
    g = rng.integers(0, 25, size=n)
    x = rng.standard_normal(n)
    alpha = rng.normal(0, 1.0, size=25)[g]
    y = 0.5 + 0.0 * x + alpha + rng.normal(0, 0.5, size=n)
    X = np.column_stack([np.ones(n), x])
    from haulage.inference.wild_bootstrap import wild_cluster_bootstrap

    out = wild_cluster_bootstrap(
        X,
        y,
        clusters=g,
        R=np.array([[0.0, 1.0]]),
        r=0.0,
        B=199,
        seed=1,
    )
    assert 0.0 <= out["p_value_two_sided"] <= 1.0


def test_ols_mc_coverage():
    """Monte Carlo: 95% CIs cover the truth at ~95% across simulated draws."""
    rng = np.random.default_rng(3)
    reps, n = 300, 200
    cover = 0
    for _ in range(reps):
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        y = 1.0 + 2.0 * X[:, 1] + rng.standard_normal(n) * (0.5 + 0.5 * np.abs(X[:, 1]))
        res = OLS(se="hc1", coef_of_interest=1).fit(y, X)
        lo, hi = res.ci()
        cover += int(lo <= 2.0 <= hi)
    cov = cover / reps
    assert 0.90 <= cov <= 0.99
