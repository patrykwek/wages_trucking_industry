from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import qr, solve_triangular


def solve_ols(
    X: NDArray[np.float64], y: NDArray[np.float64], rcond: float = 1e-10
) -> tuple[NDArray[np.float64], NDArray[np.float64], int]:
    r"""Solve $\min_\beta \|y - X\beta\|_2^2$ via a rank-revealing QR.

    Args:
        X: Design matrix (n, k).
        y: Outcome (n,) or (n, m).
        rcond: Rank threshold relative to the largest diagonal of R.

    Returns:
        beta: Least-squares solution (k,) or (k, m).
        R: Upper-triangular factor so that X^T X = R^T R on the retained columns.
        rank: Estimated numerical rank.
    """
    Q, R, piv = qr(X, mode="economic", pivoting=True)
    R_abs = np.abs(np.diag(R))
    rank = int(np.sum(R_abs > rcond * R_abs[0])) if R_abs.size else 0
    Qty = Q.T @ y
    beta_piv = np.zeros((X.shape[1], *y.shape[1:]), dtype=np.float64)
    if rank > 0:
        beta_piv[:rank] = solve_triangular(R[:rank, :rank], Qty[:rank], lower=False)
    unpiv = np.empty_like(piv)
    unpiv[piv] = np.arange(piv.size)
    return beta_piv[unpiv], R, rank


def xtx_inv(X: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Return $(X^T X)^{-1}$ via QR, without forming the normal equations directly."""
    _Q, R = qr(X, mode="economic")
    R_inv = solve_triangular(R, np.eye(R.shape[0]), lower=False)
    out: NDArray[np.float64] = R_inv @ R_inv.T
    return out


def hat_diagonal(X: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Diagonal of the hat matrix $H = X(X'X)^{-1}X'$, computed from the Q factor of a QR.

    This avoids forming the full hat matrix.
    """
    Q, _ = qr(X, mode="economic")
    out: NDArray[np.float64] = np.einsum("ij,ij->i", Q, Q).astype(np.float64)
    return out


def demean_within(y: NDArray[np.float64], groups: NDArray[np.int64]) -> NDArray[np.float64]:
    r"""Within-group demean: $\tilde y_{ig} = y_{ig} - \bar y_g$.

    O(n) using np.add.at; works for 1-D or 2-D y.
    """
    labels, inv = np.unique(groups, return_inverse=True)
    G = labels.shape[0]
    if y.ndim == 1:
        sums = np.zeros(G, dtype=np.float64)
        np.add.at(sums, inv, y)
        counts = np.bincount(inv, minlength=G).astype(np.float64)
        means = sums / np.where(counts > 0, counts, 1.0)
        out: NDArray[np.float64] = y - means[inv]
        return out
    sums2 = np.zeros((G, y.shape[1]), dtype=np.float64)
    np.add.at(sums2, inv, y)
    counts = np.bincount(inv, minlength=G).astype(np.float64)[:, None]
    means2 = sums2 / np.where(counts > 0, counts, 1.0)
    out2: NDArray[np.float64] = y - means2[inv]
    return out2


def two_way_demean(
    y: NDArray[np.float64],
    g1: NDArray[np.int64],
    g2: NDArray[np.int64],
    tol: float = 1e-10,
    max_iter: int = 200,
) -> NDArray[np.float64]:
    r"""Iterative within transform for two-way FE (Guimaraes-Portugal / method of alternating projections).

    Converges linearly on balanced-enough panels; stops when max-abs change drops below tol.
    """
    z = y.astype(np.float64, copy=True)
    for _ in range(max_iter):
        z0 = z.copy()
        z = demean_within(z, g1)
        z = demean_within(z, g2)
        if np.max(np.abs(z - z0)) < tol:
            break
    return z
