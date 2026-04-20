from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray

from haulage.utils.linalg import xtx_inv


def _cluster_meat(
    X: NDArray[np.float64],
    u: NDArray[np.float64],
    clusters: NDArray[np.int64],
) -> NDArray[np.float64]:
    labels, inv = np.unique(clusters, return_inverse=True)
    G = labels.shape[0]
    k = X.shape[1]
    S = np.zeros((G, k), dtype=np.float64)
    np.add.at(S, inv, X * u[:, None])
    return S.T @ S


def cluster_one_way(
    X: NDArray[np.float64],
    u: NDArray[np.float64],
    clusters: NDArray[np.int64],
    dof_correction: bool = True,
) -> NDArray[np.float64]:
    r"""Liang-Zeger (1986) one-way cluster-robust covariance.

    $\hat V = \frac{G}{G-1}\,\frac{n-1}{n-k}\,(X'X)^{-1}\left(\sum_g S_g S_g'\right)(X'X)^{-1}$
    with $S_g = \sum_{i \in g} x_i \hat u_i$.
    """
    A = xtx_inv(X)
    B = _cluster_meat(X, u, clusters)
    V: NDArray[np.float64] = A @ B @ A
    if dof_correction:
        n, k = X.shape
        G = int(np.unique(clusters).shape[0])
        V = V * ((G / (G - 1)) * ((n - 1) / (n - k)))
    return V


def cluster_two_way(
    X: NDArray[np.float64],
    u: NDArray[np.float64],
    cluster_a: NDArray[np.int64],
    cluster_b: NDArray[np.int64],
    dof_correction: bool = True,
) -> NDArray[np.float64]:
    r"""Cameron-Gelbach-Miller (2011) two-way cluster-robust covariance.

    $V_{AB} = V_A + V_B - V_{A \cap B}$ where $V_{A \cap B}$ clusters on the intersection.
    """
    V_a = cluster_one_way(X, u, cluster_a, dof_correction=dof_correction)
    V_b = cluster_one_way(X, u, cluster_b, dof_correction=dof_correction)
    intersect = cluster_a.astype(np.int64) * (int(cluster_b.max()) + 1) + cluster_b.astype(np.int64)
    V_ab = cluster_one_way(X, u, intersect, dof_correction=dof_correction)
    V = V_a + V_b - V_ab
    eigvals = np.linalg.eigvalsh((V + V.T) / 2)
    if eigvals.min() < -1e-8 * max(abs(eigvals.max()), 1.0):
        V = _psd_project(V)
    return V


def _psd_project(V: NDArray[np.float64]) -> NDArray[np.float64]:
    """Cameron-Gelbach-Miller PSD projection: clip negative eigenvalues to zero."""
    V_sym = (V + V.T) / 2
    w, U = np.linalg.eigh(V_sym)
    w_clipped = np.clip(w, 0.0, None)
    return cast("NDArray[np.float64]", U @ np.diag(w_clipped) @ U.T)
