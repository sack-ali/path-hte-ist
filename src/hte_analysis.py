"""
Risk-stratified HTE analysis (PATH Statement core).

Given:
  - per-patient baseline (control-arm) risk
  - treatment assignment
  - observed outcome

Procedure:
  1. Stratify patients into N quantile bins by baseline risk.
  2. Within each stratum, compute:
       - control event rate
       - treated event rate
       - Absolute Risk Difference (ARR = control - treated; positive = benefit)
       - Relative Risk
       - Number Needed to Treat (NNT = 1/ARR)
  3. Bootstrap the whole pipeline to get 95% CIs that respect within-stratum
     correlation.

Notes:
  - We bootstrap *patients* with replacement, then re-stratify on each replicate.
    This is more conservative (and correct) than fixing strata and bootstrapping
    inside each.
  - For very small bootstrap event counts, ARR CIs can be wide. That's honest
    uncertainty, not a bug.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config


@dataclass
class StratumResult:
    stratum: int
    n: int
    n_treated: int
    n_control: int
    risk_min: float
    risk_max: float
    risk_mean: float
    event_rate_control: float
    event_rate_treated: float
    arr: float           # absolute risk reduction (control - treated)
    rr: float            # relative risk (treated / control)
    nnt: float           # number needed to treat (1 / arr); inf if arr <= 0


def _stratify(risk: np.ndarray, n_strata: int) -> np.ndarray:
    """Return integer stratum labels 0..n_strata-1 by quantile of risk."""
    quantiles = np.linspace(0, 1, n_strata + 1)
    edges = np.quantile(risk, quantiles)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return np.digitize(risk, edges[1:-1], right=False)


def _stratum_effect(
    risk: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    stratum_labels: np.ndarray,
    s: int,
) -> StratumResult:
    mask = stratum_labels == s
    r_s = risk[mask]
    t_s = treatment[mask]
    y_s = outcome[mask]

    n = int(mask.sum())
    n_t = int(t_s.sum())
    n_c = n - n_t

    if n_c == 0 or n_t == 0:
        return StratumResult(
            stratum=s, n=n, n_treated=n_t, n_control=n_c,
            risk_min=float(r_s.min()) if n else np.nan,
            risk_max=float(r_s.max()) if n else np.nan,
            risk_mean=float(r_s.mean()) if n else np.nan,
            event_rate_control=np.nan, event_rate_treated=np.nan,
            arr=np.nan, rr=np.nan, nnt=np.nan,
        )

    er_c = float(y_s[t_s == 0].mean())
    er_t = float(y_s[t_s == 1].mean())
    arr = er_c - er_t
    rr = (er_t / er_c) if er_c > 0 else np.nan
    nnt = (1.0 / arr) if arr > 0 else np.inf

    return StratumResult(
        stratum=s, n=n, n_treated=n_t, n_control=n_c,
        risk_min=float(r_s.min()), risk_max=float(r_s.max()),
        risk_mean=float(r_s.mean()),
        event_rate_control=er_c, event_rate_treated=er_t,
        arr=arr, rr=rr, nnt=nnt,
    )


def stratified_effects(
    risk: pd.Series,
    treatment: pd.Series,
    outcome: pd.Series,
    n_strata: int = config.N_RISK_STRATA,
) -> pd.DataFrame:
    """Point estimates of stratum-specific effects."""
    r = risk.to_numpy()
    t = treatment.to_numpy()
    y = outcome.to_numpy()

    labels = _stratify(r, n_strata)
    rows = [_stratum_effect(r, t, y, labels, s) for s in range(n_strata)]
    return pd.DataFrame([row.__dict__ for row in rows])


def bootstrap_effects(
    risk: pd.Series,
    treatment: pd.Series,
    outcome: pd.Series,
    n_strata: int = config.N_RISK_STRATA,
    n_boot: int = config.N_BOOTSTRAP,
    random_state: int = config.RANDOM_SEED,
) -> pd.DataFrame:
    """
    Bootstrap stratum-specific ARR (and RR, NNT). Returns one row per
    stratum with point estimate plus 2.5/97.5 percentile CI.
    """
    rng = np.random.default_rng(random_state)
    n = len(risk)

    r = risk.to_numpy()
    t = treatment.to_numpy()
    y = outcome.to_numpy()

    arr_draws = np.full((n_boot, n_strata), np.nan)
    rr_draws = np.full((n_boot, n_strata), np.nan)

    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        rb, tb, yb = r[idx], t[idx], y[idx]
        labels = _stratify(rb, n_strata)
        for s in range(n_strata):
            res = _stratum_effect(rb, tb, yb, labels, s)
            arr_draws[b, s] = res.arr
            rr_draws[b, s] = res.rr

    point = stratified_effects(risk, treatment, outcome, n_strata=n_strata)
    point["arr_lo"] = np.nanpercentile(arr_draws, 2.5, axis=0)
    point["arr_hi"] = np.nanpercentile(arr_draws, 97.5, axis=0)
    point["rr_lo"] = np.nanpercentile(rr_draws, 2.5, axis=0)
    point["rr_hi"] = np.nanpercentile(rr_draws, 97.5, axis=0)
    return point
