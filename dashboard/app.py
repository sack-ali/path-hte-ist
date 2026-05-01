"""
PATH-HTE Streamlit dashboard.

Run:
    streamlit run dashboard/app.py

Shows:
  - cohort summary
  - baseline-risk distribution histogram
  - risk-stratified ARR forest plot (the 'PATH plot')
  - per-stratum table

This is the v0 dashboard. It loads the cleaned dataset, fits the risk model
on the fly (cached), and renders results. Once we have v1 of the analysis
locked, we'll persist model + bootstrap output to disk and load them here
instead of refitting on every launch.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow `from src.*` imports when run from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config
from src.data_loader import load_clean_ist
from src.hte_analysis import bootstrap_effects
from src.risk_model import fit_risk_model

st.set_page_config(page_title="PATH-HTE: IST", layout="wide")

st.title("Predictive HTE Analysis — International Stroke Trial")
st.caption(
    "Risk-stratified treatment effects following the PATH Statement "
    "(Kent et al. 2020). Treatment: aspirin vs no-aspirin. Outcome: "
    "death or dependence at 6 months."
)


# --------------------------------------------------------------------------- #
# Data + model (cached)
# --------------------------------------------------------------------------- #
@st.cache_data
def _load_data() -> pd.DataFrame:
    return load_clean_ist()


@st.cache_resource
def _fit_model(df: pd.DataFrame):
    return fit_risk_model(
        df,
        numeric_features=["AGE", "RSBP", "RDELAY"],
        categorical_features=["SEX", "RCONSC", "STYPE"],
    )


@st.cache_data
def _bootstrap(risk: pd.Series, treatment: pd.Series, outcome: pd.Series,
               n_strata: int, n_boot: int) -> pd.DataFrame:
    return bootstrap_effects(risk, treatment, outcome,
                             n_strata=n_strata, n_boot=n_boot)


# --------------------------------------------------------------------------- #
# Sidebar controls
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Analysis settings")
    n_strata = st.slider("Number of risk strata", 2, 10, 4)
    n_boot = st.slider("Bootstrap replicates", 100, 2000, 500, step=100)

# --------------------------------------------------------------------------- #
# Try to load real data; fall back to a friendly message
# --------------------------------------------------------------------------- #
try:
    df = _load_data()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

# --------------------------------------------------------------------------- #
# Cohort summary
# --------------------------------------------------------------------------- #
c1, c2, c3, c4 = st.columns(4)
c1.metric("Patients", f"{len(df):,}")
c2.metric("Treated (aspirin)", f"{int(df['treatment'].sum()):,}")
c3.metric("Events (death/dep. 6m)", f"{int(df['outcome'].sum()):,}")
c4.metric("Marginal event rate", f"{df['outcome'].mean():.1%}")

# --------------------------------------------------------------------------- #
# Risk model
# --------------------------------------------------------------------------- #
with st.spinner("Fitting baseline risk model..."):
    rm = _fit_model(df)

st.subheader("Risk model performance (cross-validated)")
c1, c2 = st.columns(2)
c1.metric("AUC", f"{rm.cv_auc:.3f}")
c2.metric("Brier", f"{rm.cv_brier:.4f}")

df = df.assign(baseline_risk=rm.baseline_risk.values)

st.subheader("Distribution of predicted baseline risk")
fig = go.Figure(go.Histogram(x=df["baseline_risk"], nbinsx=50))
fig.update_layout(xaxis_title="Predicted baseline risk", yaxis_title="Patients",
                  height=300, margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------- #
# HTE
# --------------------------------------------------------------------------- #
st.subheader(f"Risk-stratified treatment effect ({n_strata} strata)")

with st.spinner(f"Bootstrapping ({n_boot} replicates)..."):
    eff = _bootstrap(df["baseline_risk"], df["treatment"], df["outcome"],
                     n_strata=n_strata, n_boot=n_boot)

# Forest-plot-style ARR
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=eff["arr"], y=eff["stratum"],
        mode="markers",
        error_x=dict(
            type="data", symmetric=False,
            array=eff["arr_hi"] - eff["arr"],
            arrayminus=eff["arr"] - eff["arr_lo"],
        ),
        marker=dict(size=10),
        name="ARR",
    )
)
fig.add_vline(x=0, line_dash="dash", line_color="gray")
fig.update_layout(
    xaxis_title="Absolute Risk Reduction (control - treated)",
    yaxis_title="Risk stratum (0 = lowest)",
    yaxis=dict(autorange="reversed"),
    height=350, margin=dict(l=10, r=10, t=10, b=10),
)
st.plotly_chart(fig, use_container_width=True)

st.dataframe(
    eff[
        ["stratum", "n", "risk_min", "risk_max", "risk_mean",
         "event_rate_control", "event_rate_treated",
         "arr", "arr_lo", "arr_hi"]
    ].round(4),
    use_container_width=True,
)

st.caption(
    "Bars show 95% bootstrap CIs. Strata are quantiles of out-of-fold predicted "
    "baseline risk. NNT is omitted when the point estimate ARR <= 0."
)
