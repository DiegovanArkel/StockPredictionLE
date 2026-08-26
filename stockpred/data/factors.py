"""Fama-French Europe factors via pandas-datareader.

Joins the ``Europe_3_Factors`` (Mkt-RF, SMB, HML, RF) and
``Europe_Mom_Factor`` (momentum) monthly tables from Ken French's data
library, converts from percent to decimal fractions, and aligns dates to
month-end timestamps.
"""

from __future__ import annotations

import logging

import pandas as pd
import pandas_datareader.data as web

logger = logging.getLogger(__name__)

_TIDY_COLUMNS = ["date", "mkt_rf", "smb", "hml", "mom", "rf"]
_RENAME = {"Mkt-RF": "mkt_rf", "SMB": "smb", "HML": "hml", "RF": "rf"}


def fetch_famafrench(start: str = "2000-01-01") -> pd.DataFrame:
    """Fetch and join the Europe 3-factor and momentum tables.

    Returns a tidy monthly frame ``[date, mkt_rf, smb, hml, mom, rf]`` with
    values as decimal fractions (not percent) and ``date`` set to the
    calendar month-end.
    """
    three_factors = web.DataReader("Europe_3_Factors", "famafrench", start=start)[0].copy()
    mom_factor = web.DataReader("Europe_Mom_Factor", "famafrench", start=start)[0].copy()

    mom_col = "WML" if "WML" in mom_factor.columns else "Mom"
    mom_factor = mom_factor.rename(columns={mom_col: "mom"})

    combined = three_factors.join(mom_factor[["mom"]], how="inner")
    combined = combined.rename(columns=_RENAME)
    combined = combined[["mkt_rf", "smb", "hml", "mom", "rf"]].astype(float) / 100.0

    combined.index = combined.index.to_timestamp(how="end").normalize()
    combined.index.name = "date"

    return combined.reset_index()[_TIDY_COLUMNS]
