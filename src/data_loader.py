"""
IST data loading.

Responsible only for:
1. reading the raw CSV from data/raw/IST_corrected.csv
2. reporting basic shape / sanity checks
3. caching a parquet copy of the cleaned dataset (written by preprocess.py)

Cleaning and feature engineering live in preprocess.py — keep concerns separate.
"""
from __future__ import annotations

import logging

import pandas as pd

from . import config

log = logging.getLogger(__name__)


def load_raw_ist() -> pd.DataFrame:
    """Read the raw IST CSV. Fails loudly if the file is missing."""
    path = config.IST_RAW_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"IST raw file not found at {path}.\n"
            "Download IST_corrected.csv from "
            "https://datashare.ed.ac.uk/handle/10283/124 "
            f"and place it in {config.DATA_RAW}/"
        )
    df = pd.read_csv(path, low_memory=False)
    log.info("Loaded IST raw: %d rows, %d cols", *df.shape)
    return df


def load_clean_ist() -> pd.DataFrame:
    """Load the cleaned, analysis-ready dataset.

    Uses the cached parquet if available. Otherwise preprocesses the raw CSV
    in-memory and returns the result directly — safe on read-only filesystems
    like Streamlit Cloud. The caller's @st.cache_data handles in-memory caching.
    """
    path = config.IST_CLEAN_PATH
    if path.exists():
        return pd.read_parquet(path)
    log.info("Parquet not found — preprocessing from raw CSV.")
    from .preprocess import preprocess as _preprocess
    return _preprocess(load_raw_ist())


def quick_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One-row-per-column summary: dtype, n_missing, n_unique, sample values."""
    rows = []
    for col in df.columns:
        s = df[col]
        rows.append(
            {
                "column": col,
                "dtype": str(s.dtype),
                "n_missing": int(s.isna().sum()),
                "pct_missing": round(s.isna().mean() * 100, 2),
                "n_unique": int(s.nunique(dropna=True)),
                "sample": s.dropna().unique()[:5].tolist(),
            }
        )
    return pd.DataFrame(rows)
