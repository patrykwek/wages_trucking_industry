from __future__ import annotations

import numpy as np

from haulage.data import assemble_panel, load_state_dereg_table
from haulage.data.panel import PanelConfig
from haulage.data.static.icc_route_authorities_1978 import load_route_authorities_1978
from haulage.data.static.teamsters_density import load_teamsters_density


def test_dereg_table_has_all_states():
    df = load_state_dereg_table()
    assert df.shape[0] >= 50
    assert {"state_abbr", "state_name", "dereg_year", "citation"} <= set(df.columns)
    assert df["dereg_year"].dtype.name in ("Int64", "Int32")


def test_route_authorities_sum_to_one():
    df = load_route_authorities_1978(seed=1978)
    totals = df.groupby("state_abbr")["share_1978"].sum()
    np.testing.assert_allclose(totals.to_numpy(), 1.0, atol=1e-9)


def test_teamsters_density_monotone_decline_80s():
    ts = load_teamsters_density().set_index("year")
    assert ts.loc[1985, "density"] < ts.loc[1975, "density"]
    assert ts.loc[2000, "density"] < ts.loc[1990, "density"]


def test_synthetic_panel_shape_and_treatment_signal():
    cfg = PanelConfig(year_range=(1976, 2020))
    df = assemble_panel(cfg, synthetic=True, seed=0)
    assert "log_wage" in df.columns and "D_MCA" in df.columns
    assert df["state_abbr"].nunique() >= 50
    pre = df.loc[df["D_MCA"] == 0, "log_wage"].mean()
    post = df.loc[df["D_MCA"] == 1, "log_wage"].mean()
    assert post < pre, "MCA should depress wages in the seeded DGP"


def test_synthetic_panel_has_unique_unit_time_keys():
    df = assemble_panel(synthetic=True, seed=0)
    assert not df.duplicated(subset=["state_abbr", "year"]).any()


def test_ab5_only_california():
    df = assemble_panel(synthetic=True, seed=0)
    ab5_states = df.loc[df["D_AB5"] == 1, "state_abbr"].unique().tolist()
    assert ab5_states == ["CA"]
