"""
Synthetic-data smoke test for the full pipeline.

Generates a fake trial that *deliberately* has HTE: high-risk patients get
~bigger absolute benefit. Runs the full pipeline and prints the stratum
table. If the bottom and top strata show different ARRs, the pipeline works.

Run:
    python -m tests.test_pipeline_synth
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.hte_analysis import bootstrap_effects
from src.risk_model import fit_risk_model


def make_synthetic_trial(n: int = 5000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age = rng.normal(70, 12, n).clip(40, 95)
    sbp = rng.normal(150, 25, n).clip(80, 240)
    sex = rng.choice(["M", "F"], size=n)
    consc = rng.choice(["F", "D", "U"], size=n, p=[0.8, 0.15, 0.05])

    # True baseline log-odds depends on age, sbp, consciousness
    logit_baseline = (
        -3.0
        + 0.04 * (age - 70)
        + 0.005 * (sbp - 150)
        + np.where(consc == "D", 0.6, 0)
        + np.where(consc == "U", 1.5, 0)
    )
    p_baseline = 1 / (1 + np.exp(-logit_baseline))

    treatment = rng.integers(0, 2, n)

    # HTE: treatment effect on log-odds is bigger for higher-risk patients
    # (matching PATH motivation: greater absolute benefit at higher baseline risk)
    trt_effect = -0.3 - 0.5 * (p_baseline > np.median(p_baseline)).astype(float)
    logit = logit_baseline + treatment * trt_effect
    p = 1 / (1 + np.exp(-logit))
    outcome = rng.binomial(1, p)

    return pd.DataFrame(
        {
            "AGE": age, "RSBP": sbp, "SEX": sex, "RCONSC": consc,
            "treatment": treatment, "outcome": outcome,
        }
    )


def main() -> None:
    df = make_synthetic_trial(n=8000)

    rm = fit_risk_model(
        df,
        numeric_features=["AGE", "RSBP"],
        categorical_features=["SEX", "RCONSC"],
    )
    print(f"Risk model CV AUC:   {rm.cv_auc:.3f}")
    print(f"Risk model CV Brier: {rm.cv_brier:.4f}")

    df = df.assign(baseline_risk=rm.baseline_risk)

    out = bootstrap_effects(
        df["baseline_risk"], df["treatment"], df["outcome"],
        n_strata=4, n_boot=200,
    )
    cols = ["stratum", "n", "risk_mean",
            "event_rate_control", "event_rate_treated",
            "arr", "arr_lo", "arr_hi"]
    print("\nStratum-specific effects:")
    print(out[cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
