"""
IST preprocessing pipeline.

Reads raw CSV -> applies cleaning rules -> writes parquet.

Run as a script:
    python -m src.preprocess

Cleaning steps:
1. Standardise Y/N text columns to {1, 0, NaN}
2. Coerce numerics
3. Build the primary composite outcome (death or dependence at 6 months)
4. Drop rows with missing treatment assignment or missing primary outcome
5. Persist to parquet for fast downstream loading

Design choice: we keep this conservative and *transparent*. No imputation here —
imputation belongs inside the modelling pipeline (sklearn ColumnTransformer)
so that it is fit on training folds only and never leaks.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import config
from .data_loader import load_raw_ist

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _yn_to_binary(s: pd.Series) -> pd.Series:
    """Map Y/N (and lowercase / whitespace variants) to 1/0; everything else -> NaN."""
    # is_string_dtype covers both object (old pandas) and Arrow-backed strings (pandas 3+)
    if not pd.api.types.is_string_dtype(s):
        return s
    mapped = (
        s.astype(str)
        .str.strip()
        .str.upper()
        .map({"Y": 1, "N": 0})
    )
    return mapped.astype("Float64")  # nullable float so NaN survives


def _coerce_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


# --------------------------------------------------------------------------- #
# Outcome construction
# --------------------------------------------------------------------------- #
def build_primary_outcome(df: pd.DataFrame) -> pd.Series:
    """
    Composite: death OR dependence at 6 months.

    FDEAD: dead at 6-month follow-up (Y/N)
    FDENNIS: dependent at 6-month follow-up (Y/N)

    Returns a 0/1 Int64 Series; NaN if BOTH source flags are missing.
    """
    fdead = _yn_to_binary(df[config.COL_FDEAD])
    fdennis = _yn_to_binary(df[config.COL_FDENNIS])

    composite = ((fdead == 1) | (fdennis == 1)).astype("Int64")
    composite[fdead.isna() & fdennis.isna()] = pd.NA
    return composite


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    n0 = len(df)

    # 1. Treatment assignment (aspirin) -> binary
    df["treatment"] = _yn_to_binary(df[config.COL_TRT_ASPIRIN]).astype("Int64")

    # 2. Numeric covariates
    for col in config.BASELINE_NUMERIC:
        if col in df.columns:
            df[col] = _coerce_numeric(df[col])

    # 3. Y/N covariates -> 0/1 (nullable)
    yn_cols = [
        "RVISINF", "RHEP24", "RASP3", "RATRIAL", "RCT",
        "RDEF1", "RDEF2", "RDEF3", "RDEF4",
        "RDEF5", "RDEF6", "RDEF7", "RDEF8",
    ]
    for col in yn_cols:
        if col in df.columns:
            df[col] = _yn_to_binary(df[col])

    # 4. Categorical: keep as 'category' dtype
    for col in ["SEX", "RCONSC", "STYPE"]:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # 5. Primary outcome
    df["outcome"] = build_primary_outcome(df)

    # 6. Drop rows with missing treatment or missing primary outcome
    before = len(df)
    df = df.dropna(subset=["treatment", "outcome"])
    log.info("Dropped %d rows with missing treatment/outcome", before - len(df))

    # 7. Cast to plain ints now that NaNs are gone in these two cols
    df["treatment"] = df["treatment"].astype(int)
    df["outcome"] = df["outcome"].astype(int)

    log.info("Final cleaned shape: %d rows (started with %d)", len(df), n0)
    return df


def main() -> None:
    raw = load_raw_ist()
    clean = preprocess(raw)

    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    clean.to_parquet(config.IST_CLEAN_PATH, index=False)
    log.info("Wrote cleaned dataset -> %s", config.IST_CLEAN_PATH)

    # Quick treatment x outcome cross-tab for sanity
    ct = pd.crosstab(clean["treatment"], clean["outcome"], margins=True)
    log.info("Treatment x Outcome:\n%s", ct.to_string())

    # Marginal event rates by arm
    rates = clean.groupby("treatment")["outcome"].agg(["count", "mean"]).round(4)
    rates.columns = ["n", "event_rate"]
    log.info("Event rates by arm:\n%s", rates.to_string())


if __name__ == "__main__":
    main()
