from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from haulage.utils.linalg import solve_ols


def _rademacher(rng: np.random.Generator, G: int) -> NDArray[np.float64]:
    return rng.choice(np.array([-1.0, 1.0]), size=G).astype(np.float64)


def _mammen(rng: np.random.Generator, G: int) -> NDArray[np.float64]:
    p = (np.sqrt(5) + 1) / (2 * np.sqrt(5))
    a = -(np.sqrt(5) - 1) / 2
    b = (np.sqrt(5) + 1) / 2
    draws = rng.random(G) < p
    out: NDArray[np.float64] = np.where(draws, a, b).astype(np.float64)
    return out


def _webb(rng: np.random.Generator, G: int) -> NDArray[np.float64]:
    vals = np.array(
        [-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)],
        dtype=np.float64,
    )
    return rng.choice(vals, size=G)


def wild_cluster_bootstrap(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    clusters: NDArray[np.int64],
    R: NDArray[np.float64],
    r: float,
    B: int = 999,
    weights: str = "rademacher",
    seed: int | None = None,
) -> dict[str, float]:
    r"""Cameron-Gelbach-Miller (2008) wild cluster bootstrap-t for $H_0: R\beta = r$.

    Imposes the null when computing bootstrap residuals, then draws cluster-level multipliers.

    Args:
        X: (n, k).
        y: (n,).
        clusters: (n,) cluster labels.
        R: (1, k) linear restriction row.
        r: scalar restriction value.
        B: bootstrap reps.
        weights: "rademacher" | "mammen" | "webb".
        seed: RNG seed.

    Returns:
        dict with "t", "p_value_two_sided", and the bootstrap t-distribution mean/SD.

    References:
        Cameron, Gelbach & Miller (2008); Roodman et al. (2019).
    """
    rng = np.random.default_rng(seed)
    draw_fn: Callable[[np.random.Generator, int], NDArray[np.float64]]
    draw_fn = {"rademacher": _rademacher, "mammen": _mammen, "webb": _webb}[weights]

    beta, _, _ = solve_ols(X, y)
    u = y - X @ beta
    t_obs = float(((R @ beta - r) / _se_linear(X, u, clusters, R)).item())

    beta_r, u_r = _restricted_fit(X, y, R, r)

    labels, inv = np.unique(clusters, return_inverse=True)
    G = labels.shape[0]

    t_star = np.empty(B, dtype=np.float64)
    for b in range(B):
        w = draw_fn(rng, G)[inv]
        y_star = X @ beta_r + w * u_r
        beta_b, _, _ = solve_ols(X, y_star)
        u_b = y_star - X @ beta_b
        t_star[b] = float(((R @ beta_b - r) / _se_linear(X, u_b, clusters, R)).item())

    p_two = float(np.mean(np.abs(t_star) >= abs(t_obs)))
    return {
        "t": t_obs,
        "p_value_two_sided": p_two,
        "bootstrap_mean": float(np.mean(t_star)),
        "bootstrap_sd": float(np.std(t_star, ddof=1)),
        "B": float(B),
        "G": float(G),
    }


def _se_linear(
    X: NDArray[np.float64],
    u: NDArray[np.float64],
    clusters: NDArray[np.int64],
    R: NDArray[np.float64],
) -> float:
    from haulage.inference.cluster import cluster_one_way

    V = cluster_one_way(X, u, clusters, dof_correction=True)
    var = float((R @ V @ R.T).item())
    return max(float(np.sqrt(var)), 1e-12)


def _restricted_fit(
    X: NDArray[np.float64], y: NDArray[np.float64], R: NDArray[np.float64], r: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""OLS subject to the linear constraint $R\beta = r$ (via Lagrangian projection)."""
    from haulage.utils.linalg import xtx_inv

    A = xtx_inv(X)
    beta_u = A @ X.T @ y
    adj = A @ R.T @ np.linalg.solve(R @ A @ R.T, R @ beta_u - np.asarray([r], dtype=np.float64))
    beta_r = beta_u - adj
    u_r = y - X @ beta_r
    return beta_r, u_r
