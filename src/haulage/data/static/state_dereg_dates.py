from __future__ import annotations

import pandas as pd

_DEREG_TABLE: list[tuple[str, str, int | None, str]] = [
    ("AK", "Alaska", None, "Never meaningfully regulated intrastate trucking."),
    ("AL", "Alabama", 1994, "FAAAA-driven dereg."),
    ("AR", "Arkansas", 1994, "FAAAA-driven dereg."),
    ("AZ", "Arizona", 1982, "Rose (1985); Teske et al. (1995)."),
    ("CA", "California", 1994, "FAAAA preemption; AB5 recoded 2020 separately."),
    ("CO", "Colorado", 1994, "FAAAA-driven dereg."),
    ("CT", "Connecticut", 1994, "FAAAA-driven dereg."),
    ("DE", "Delaware", 1994, "FAAAA-driven dereg."),
    ("FL", "Florida", 1980, "Early dereg; Rose (1985)."),
    ("GA", "Georgia", 1994, "FAAAA-driven dereg."),
    ("HI", "Hawaii", 1994, "FAAAA-driven dereg."),
    ("IA", "Iowa", 1994, "FAAAA-driven dereg."),
    ("ID", "Idaho", 1994, "FAAAA-driven dereg."),
    ("IL", "Illinois", 1994, "FAAAA-driven dereg."),
    ("IN", "Indiana", 1994, "FAAAA-driven dereg."),
    ("KS", "Kansas", 1994, "FAAAA-driven dereg."),
    ("KY", "Kentucky", 1994, "FAAAA-driven dereg."),
    ("LA", "Louisiana", 1994, "FAAAA-driven dereg."),
    ("MA", "Massachusetts", 1994, "FAAAA-driven dereg."),
    ("MD", "Maryland", 1994, "FAAAA-driven dereg."),
    ("ME", "Maine", 1994, "FAAAA-driven dereg."),
    ("MI", "Michigan", 1982, "Rose (1985)."),
    ("MN", "Minnesota", 1994, "FAAAA-driven dereg."),
    ("MO", "Missouri", 1994, "FAAAA-driven dereg."),
    ("MS", "Mississippi", 1994, "FAAAA-driven dereg."),
    ("MT", "Montana", 1994, "FAAAA-driven dereg."),
    ("NC", "North Carolina", 1994, "FAAAA-driven dereg."),
    ("ND", "North Dakota", 1994, "FAAAA-driven dereg."),
    ("NE", "Nebraska", 1994, "FAAAA-driven dereg."),
    ("NH", "New Hampshire", 1994, "FAAAA-driven dereg."),
    ("NJ", "New Jersey", 1988, "State dereg in advance of FAAAA."),
    ("NM", "New Mexico", 1994, "FAAAA-driven dereg."),
    ("NV", "Nevada", 1994, "FAAAA-driven dereg."),
    ("NY", "New York", 1994, "FAAAA-driven dereg."),
    ("OH", "Ohio", 1994, "FAAAA-driven dereg."),
    ("OK", "Oklahoma", 1994, "FAAAA-driven dereg."),
    ("OR", "Oregon", 1994, "FAAAA-driven dereg."),
    ("PA", "Pennsylvania", 1994, "FAAAA-driven dereg."),
    ("RI", "Rhode Island", 1994, "FAAAA-driven dereg."),
    ("SC", "South Carolina", 1994, "FAAAA-driven dereg."),
    ("SD", "South Dakota", 1994, "FAAAA-driven dereg."),
    ("TN", "Tennessee", 1994, "FAAAA-driven dereg."),
    ("TX", "Texas", 1994, "FAAAA-driven dereg."),
    ("UT", "Utah", 1994, "FAAAA-driven dereg."),
    ("VA", "Virginia", 1994, "FAAAA-driven dereg."),
    ("VT", "Vermont", 1994, "FAAAA-driven dereg."),
    ("WA", "Washington", 1994, "FAAAA-driven dereg."),
    ("WI", "Wisconsin", 1982, "Rose (1985); Teske et al. (1995)."),
    ("WV", "West Virginia", 1994, "FAAAA-driven dereg."),
    ("WY", "Wyoming", 1994, "FAAAA-driven dereg."),
    ("DC", "District of Columbia", 1994, "FAAAA-driven dereg."),
]


def load_state_dereg_table() -> pd.DataFrame:
    """Return a typed DataFrame of state intrastate deregulation years.

    Columns: state_fips (optional), state_abbr, state_name, dereg_year, citation.
    Sources: Rose (1985); Teske, Best & Mintrom (1995); FAAAA 1994 preempted remaining
    state controls. `dereg_year=None` means never meaningfully regulated.
    """
    df = pd.DataFrame(_DEREG_TABLE, columns=["state_abbr", "state_name", "dereg_year", "citation"])
    df["dereg_year"] = df["dereg_year"].astype("Int64")
    return df
