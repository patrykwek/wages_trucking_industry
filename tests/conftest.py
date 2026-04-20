from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture
def simple_ols_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    n, k = 500, 3
    X = np.hstack([np.ones((n, 1)), rng.standard_normal((n, k))])
    beta = np.array([1.0, 2.0, -1.0, 0.5])
    u = rng.standard_normal(n) * 0.8
    y = X @ beta + u
    return y, X, beta


@pytest.fixture
def cluster_panel() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n_groups, n_per = 40, 20
    rows: list[dict[str, float | int]] = []
    for g in range(n_groups):
        alpha = rng.normal(0, 1.0)
        for t in range(n_per):
            x = rng.standard_normal()
            y = 1.0 + 0.5 * x + alpha + rng.standard_normal() * 0.3
            rows.append({"g": g, "t": t, "x": x, "y": y})
    return pd.DataFrame(rows)


@pytest.fixture
def staggered_panel() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    rows: list[dict[str, float | int]] = []
    cohorts = {i: (i // 3) * 2 + 5 if i % 4 != 0 else 9999 for i in range(40)}
    for unit, g in cohorts.items():
        fe = rng.normal(0, 0.5)
        for t in range(1, 16):
            D = int(t >= g)
            y = fe + 0.02 * t + 0.8 * D + rng.normal(0, 0.3)
            rows.append({"unit": unit, "t": t, "D": D, "y": y, "g": g})
    return pd.DataFrame(rows)
