"""
Central configuration for the PATH-HTE pipeline.

Single source of truth for:
- file paths
- IST column groupings (baseline covariates, outcomes, treatments)
- analytical choices (random seed, n_bootstrap, n_strata)

Edit ONLY here when the schema or design changes; everything else imports
from this module.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
REPORTS = PROJECT_ROOT / "reports"
FIGURES = REPORTS / "figures"

IST_RAW_FILENAME = "IST_corrected.CSV"
IST_RAW_PATH = DATA_RAW / IST_RAW_FILENAME
IST_CLEAN_PATH = DATA_PROCESSED / "ist_clean.parquet"

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
RANDOM_SEED = 42

# --------------------------------------------------------------------------- #
# IST schema  (see Sandercock et al. 2011, Trials 12:101)
# --------------------------------------------------------------------------- #
# IST has a 2x2 factorial design: aspirin (yes/no) x heparin (none/low/medium).
# For an HTE analysis we usually start with ASPIRIN as the treatment of interest
# (binary, simpler) and treat heparin allocation as a covariate / sensitivity.

# --- Treatment columns ---------------------------------------------------- #
COL_TRT_ASPIRIN = "RXASP"   # 'Y' = aspirin allocated, 'N' = not allocated
COL_TRT_HEPARIN = "RXHEP"   # 'M' medium, 'L' low, 'N' none

# --- Baseline covariates (collected at randomisation) --------------------- #
# Demographics
COL_AGE = "AGE"
COL_SEX = "SEX"             # 'M' / 'F'

# Pre-randomisation clinical state
BASELINE_BINARY = [
    "RCONSC",    # conscious state at randomisation: F=alert, D=drowsy, U=unconscious
    "RDELAY",    # numeric: hours from stroke onset to randomisation (kept numeric below)
    "RVISINF",   # visible infarct on CT: Y/N
    "RHEP24",    # heparin within 24h pre-randomisation: Y/N
    "RASP3",     # aspirin within 3 days pre-randomisation: Y/N
    "RSBP",      # systolic BP at randomisation (numeric)
    "RDEF1",     # face deficit: Y/N/C(can't assess)
    "RDEF2",     # arm/hand deficit
    "RDEF3",     # leg/foot deficit
    "RDEF4",     # dysphasia
    "RDEF5",     # hemianopia
    "RDEF6",     # visuospatial disorder
    "RDEF7",     # brainstem/cerebellar signs
    "RDEF8",     # other deficit
    "STYPE",     # stroke subtype (TACS/PACS/POCS/LACS/OTH)
    "RATRIAL",   # atrial fibrillation Y/N (only collected in pilot; high missingness)
    "RCT",       # CT before randomisation Y/N
]

BASELINE_NUMERIC = [
    COL_AGE,
    "RDELAY",    # hours since stroke onset
    "RSBP",      # systolic BP
]

BASELINE_CATEGORICAL = [
    COL_SEX,
    "RCONSC",
    "STYPE",
    "RVISINF",
    "RHEP24",
    "RASP3",
    "RDEF1", "RDEF2", "RDEF3", "RDEF4",
    "RDEF5", "RDEF6", "RDEF7", "RDEF8",
    "RATRIAL",
    "RCT",
]

# --- Outcomes ------------------------------------------------------------- #
# 14-day outcomes
COL_DEAD_14D = "ID14"        # dead within 14 days: Y/N (derived; preferred over OCCODE alone)

# 6-month outcomes
COL_DEAD_6M = "OCCODE"       # 1=dead, 2=dependent, 3=not recovered, 4=recovered, ...
COL_FDEAD = "FDEAD"          # dead at 6m follow-up: Y/N
COL_FDENNIS = "FDENNIS"      # dependent at 6m: Y/N

# Primary composite outcome for PATH analysis (configurable):
#   "death_or_dependence_6m" = FDEAD == 'Y' OR FDENNIS == 'Y'
PRIMARY_OUTCOME = "death_or_dependence_6m"

# --------------------------------------------------------------------------- #
# Analysis hyperparameters (PATH Statement defaults)
# --------------------------------------------------------------------------- #
N_RISK_STRATA = 4            # quartiles of predicted baseline risk
N_BOOTSTRAP = 500            # for CIs around stratum-specific effects
TEST_SIZE = 0.25             # holdout for risk-model validation
CV_FOLDS = 5                 # for internal validation of risk model
