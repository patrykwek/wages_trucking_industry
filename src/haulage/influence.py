from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class InfluenceFunction:
    r"""Efficient influence function values $\hat\psi_i$ for a Z-estimator.

    For a regular estimator with $\sqrt n(\hat\tau - \tau) \Rightarrow \mathcal N(0, V)$,
    we have $V = \mathbb E[\psi^2]$ under Neyman orthogonality.

    Attributes:
        values: Shape (n,) array of influence-function evaluations.
        tau_hat: Plug-in point estimate.
        name: Tag, e.g. "AIPW", "CS(g=1982,t=1985)".

    References:
        Hines et al. 2022, `Demystifying statistical learning based on efficient influence
        functions`, The American Statistician.
    """

    values: NDArray[np.float64]
    tau_hat: float
    name: str

    def __post_init__(self) -> None:
        if self.values.ndim != 1:
            raise ValueError(f"values must be 1-D, got shape {self.values.shape}")

    @property
    def n(self) -> int:
        return int(self.values.shape[0])

    @property
    def variance(self) -> float:
        r"""Asymptotic variance $V = \mathbb E[\psi^2]$, sample version."""
        return float(np.mean(self.values**2))

    @property
    def se(self) -> float:
        r"""Plug-in standard error $\sqrt{V/n}$."""
        return float(np.sqrt(self.variance / self.n))

    def ci(self, alpha: float = 0.05) -> tuple[float, float]:
        """Wald two-sided confidence interval at level 1 - alpha."""
        from scipy.stats import norm

        z = float(norm.ppf(1 - alpha / 2))
        return (self.tau_hat - z * self.se, self.tau_hat + z * self.se)

    def cluster_se(self, clusters: NDArray[np.int64]) -> float:
        r"""Cluster-robust standard error using the influence-function aggregation.

        Uses $\hat V = \frac{1}{n^2} \sum_c \left(\sum_{i \in c} \psi_i\right)^2$.

        Args:
            clusters: Shape (n,) integer cluster labels.

        Returns:
            Cluster-robust SE.
        """
        if clusters.shape[0] != self.n:
            raise ValueError("clusters length must match influence-function length")
        n = self.n
        labels, inv = np.unique(clusters, return_inverse=True)
        sums = np.zeros(labels.shape[0], dtype=np.float64)
        np.add.at(sums, inv, self.values)
        return float(np.sqrt(np.sum(sums**2) / (n * n)))

    def subgroup(self, mask: NDArray[np.bool_]) -> InfluenceFunction:
        """Restrict to a subgroup, re-centering tau_hat to the subgroup mean of psi.

        Warning: this is an approximation; correct subgroup inference generally requires
        the original-sample estimator re-run on the subgroup.
        """
        sub = self.values[mask]
        return InfluenceFunction(
            values=sub,
            tau_hat=float(np.mean(sub)) + self.tau_hat,
            name=f"{self.name}[sub]",
        )
