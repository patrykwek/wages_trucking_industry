from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def load_fars_truck_crashes(years: Iterable[int]) -> pd.DataFrame:
    """NHTSA FARS fatal-crash micro-data restricted to heavy-truck involvement.

    FARS distributes annual SAS/CSV archives at ftp://ftp.nhtsa.dot.gov/FARS/. This
    loader is a placeholder; supply the archive parser in a follow-up PR or point it
    at a pre-parsed local copy.
    """
    raise NotImplementedError("FARS distributions require parsing the NHTSA archive; not implemented here.")
