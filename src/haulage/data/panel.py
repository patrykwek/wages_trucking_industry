from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from haulage._logging import get_logger
from haulage.data.static.state_dereg_dates import load_state_dereg_table
from haulage.data.static.teamsters_density import load_teamsters_density

_log = get_logger("data.panel")

Frequency = Literal["annual", "quarterly", "monthly"]


@dataclass(frozen=True, slots=True)
class PanelConfig:
    """Declarative configuration for `assemble_panel`.

    Attributes:
        outcomes: Which outcomes to include.
        treatments: Which dereg shock indicators to include.
        covariates: Exogenous controls.
        frequency: Panel frequency.
        year_range: (start, end) inclusive.
    """

    outcomes: tuple[str, ...] = ("log_wage", "log_hours", "employment", "union_rate")
    treatments: tuple[str, ...] = ("D_MCA", "D_state", "D_FAAAA", "D_ICCTA", "D_AB5")
    covariates: tuple[str, ...] = ("cpi_u", "unemployment_rate")
    frequency: Frequency = "annual"
    year_range: tuple[int, int] = (1976, 2024)


def assemble_panel(
    config: PanelConfig | None = None,
    *,
    synthetic: bool = True,
    seed: int = 0,
) -> pd.DataFrame:
    r"""Return the analysis-ready state x year panel.

    When `synthetic=True` (default for tests/offline use), the panel is simulated from
    a DGP calibrated to the Rose (1987) and Hirsch (1988) summary statistics, with
    explicit ATT signals at each dereg shock. When False, the loader stitches together
    live IPUMS/BLS/FRED pulls; this path requires API keys and is not exercised by the
    CI test matrix.

    Columns: state_abbr, year, dereg_year, plus outcomes, treatments, covariates.

    References:
        Rose (1987); Hirsch (1988); Teske-Best-Mintrom (1995).
    """
    cfg = config or PanelConfig()
    if synthetic:
        return _synthetic_panel(cfg, seed=seed)
    raise NotImplementedError(
        "Live assembly of the panel requires IPUMS_API_KEY, FRED_API_KEY, and BLS_API_KEY. "
        "Set these environment variables and wire the source loaders into this branch."
    )


def _synthetic_panel(cfg: PanelConfig, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dereg = load_state_dereg_table()
    teamsters = load_teamsters_density().set_index("year")
    years = np.arange(cfg.year_range[0], cfg.year_range[1] + 1)

    states = dereg["state_abbr"].tolist()
    rows: list[dict[str, object]] = []
    state_fe = {s: rng.normal(0, 0.15) for s in states}
    year_fe = {y: rng.normal(0, 0.05) + 0.005 * (y - years[0]) for y in years}

    for _, row in dereg.iterrows():
        s = str(row["state_abbr"])
        dy_raw = row["dereg_year"]
        dy = int(dy_raw) if pd.notna(dy_raw) else 9999
        for y in years:
            D_MCA = int(y >= 1980)
            D_state = int(y >= dy)
            D_FAAAA = int(y >= 1995)
            D_ICCTA = int(y >= 1996)
            D_AB5 = int(s == "CA" and y >= 2020)
            post_any = max(D_MCA, D_state)
            union = float(teamsters.loc[y, "density"])
            log_wage = (
                2.80
                + state_fe[s]
                + year_fe[y]
                - 0.12 * D_MCA
                - 0.08 * D_state
                - 0.04 * D_FAAAA
                + 0.02 * D_ICCTA
                - 0.06 * D_AB5
                + 0.25 * union
                + rng.normal(0, 0.06)
            )
            log_hours = 3.80 + 0.03 * post_any - 0.01 * D_FAAAA + rng.normal(0, 0.04)
            employment = np.exp(
                9.0 + 0.02 * (y - years[0]) + 0.8 * state_fe[s] + 0.05 * post_any + rng.normal(0, 0.1)
            )
            union_rate = float(np.clip(union * (1 - 0.3 * D_MCA - 0.2 * D_state) + rng.normal(0, 0.02), 0, 1))
            cpi_u = 50 * (1.03 ** (y - years[0]))
            unemployment_rate = 6.0 + 1.5 * np.sin(2 * np.pi * (y - years[0]) / 11) + rng.normal(0, 0.3)
            rows.append(
                {
                    "state_abbr": s,
                    "year": int(y),
                    "dereg_year": dy if dy < 9999 else None,
                    "log_wage": log_wage,
                    "log_hours": log_hours,
                    "employment": employment,
                    "union_rate": union_rate,
                    "D_MCA": D_MCA,
                    "D_state": D_state,
                    "D_FAAAA": D_FAAAA,
                    "D_ICCTA": D_ICCTA,
                    "D_AB5": D_AB5,
                    "cpi_u": cpi_u,
                    "unemployment_rate": unemployment_rate,
                }
            )
    df = pd.DataFrame(rows).sort_values(["state_abbr", "year"]).reset_index(drop=True)
    _log.info(
        "assembled synthetic panel: %d rows, %d states, %d years",
        df.shape[0],
        df["state_abbr"].nunique(),
        df["year"].nunique(),
    )
    return df
