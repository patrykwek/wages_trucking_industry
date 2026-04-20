from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from haulage.utils.linalg import xtx_inv


def _bartlett(L: int) -> NDArray[np.float64]:
    return np.array([1 - l / (L + 1) for l in range(L + 1)], dtype=np.float64)


def _parzen(L: int) -> NDArray[np.float64]:
    out = np.zeros(L + 1, dtype=np.float64)
    for l in range(L + 1):
        x = l / (L + 1)
        if x <= 0.5:
            out[l] = 1 - 6 * x * x + 6 * x * x * x
        else:
            out[l] = 2 * (1 - x) ** 3
    return out


def _qs(L: int) -> NDArray[np.float64]:
    out = np.zeros(L + 1, dtype=np.float64)
    out[0] = 1.0
    for l in range(1, L + 1):
        x = 6 * np.pi * (l / L) / 5
        out[l] = (25 / (12 * np.pi**2 * (l / L) ** 2)) * (np.sin(x) / x - np.cos(x))
    return out


def _kernel_weights(kind: str, L: int) -> NDArray[np.float64]:
    kind = kind.lower()
    if kind == "bartlett":
        return _bartlett(L)
    if kind == "parzen":
        return _parzen(L)
    if kind in ("qs", "quadratic-spectral"):
        return _qs(L)
    raise ValueError(f"unknown kernel {kind!r}")


def _andrews_bandwidth(scores: NDArray[np.float64]) -> int:
    r"""Andrews (1991) optimal bandwidth for a Bartlett kernel, pooled AR(1) approx."""
    n = scores.shape[0]
    rho_num, rho_den, sig4 = 0.0, 0.0, 0.0
    for j in range(scores.shape[1]):
        s = scores[:, j]
        num = float(np.sum(s[1:] * s[:-1]))
        den = float(np.sum(s[:-1] ** 2))
        rho = num / den if den > 0 else 0.0
        sig2 = float(np.var(s)) + 1e-12
        rho_num = rho_num + 4 * rho**2 * sig2**2 / ((1 - rho) ** 6 * (1 + rho) ** 2)
        rho_den = rho_den + sig2**2 / (1 - rho) ** 4
        sig4 = sig4 + sig2**2
    alpha = rho_num / max(rho_den, 1e-12)
    L = int(np.ceil(1.1447 * (alpha * n) ** (1 / 3)))
    return max(1, L)


def newey_west(
    X: NDArray[np.float64],
    u: NDArray[np.float64],
    lags: int | None = None,
    kernel: str = "bartlett",
) -> NDArray[np.float64]:
    r"""Newey-West HAC covariance with Bartlett / Parzen / Quadratic-Spectral kernel.

    $\hat V = A \left(\Gamma_0 + \sum_{l=1}^L k(l/L)(\Gamma_l + \Gamma_l')\right) A$
    with $\Gamma_l = \sum_{t=l+1}^T g_t g_{t-l}'$, $g_t = x_t u_t$, $A = (X'X)^{-1}$.

    Args:
        X: Design matrix (n, k).
        u: Residuals (n,), assumed time-sorted.
        lags: Kernel truncation; if None, use Andrews (1991) data-driven bandwidth.
        kernel: "bartlett" | "parzen" | "qs".

    References:
        Newey & West (1987), Andrews (1991).
    """
    g = X * u[:, None]
    L = _andrews_bandwidth(g) if lags is None else lags
    weights = _kernel_weights(kernel, L)
    n = g.shape[0]
    S = g.T @ g
    for l in range(1, L + 1):
        Gamma_l = g[l:].T @ g[: n - l]
        S += weights[l] * (Gamma_l + Gamma_l.T)
    A = xtx_inv(X)
    return A @ S @ A
