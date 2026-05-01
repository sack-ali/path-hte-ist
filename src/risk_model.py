"""
Baseline outcome risk model.

PATH Statement methodology:
- Fit a model that predicts the OUTCOME using BASELINE COVARIATES on the full
  cohort, with treatment included as a covariate.
- For each patient, predict their risk under control (treatment=0) — this is
  their "baseline risk".
- Stratify by baseline risk -> examine treatment effect within each stratum.

Why include treatment in the model and then predict at treatment=0?
- Using only the placebo arm for risk-model fitting wastes ~half the data.
- Including treatment with no interaction terms gives an unbiased baseline-risk
  estimate (under the assumption of no HTE) while doubling sample size.
- This is the "constant relative effect" risk model recommended by Kent et al.

The model is intentionally simple and interpretable: penalised logistic
regression. We can swap in gradient boosting later via the same interface.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


@dataclass
class RiskModelResult:
    """Container for everything you'd want from a fitted risk model."""

    pipeline: Pipeline
    feature_names: list[str]
    cv_auc: float
    cv_brier: float
    baseline_risk: pd.Series   # predicted P(Y=1 | X, treatment=0), one per patient


def _build_preprocessor(
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
) -> ColumnTransformer:
    """Median impute + scale for numerics; mode impute + one-hot for categoricals."""
    numeric_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first")),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_pipe, list(numeric_features)),
            ("cat", categorical_pipe, list(categorical_features)),
        ],
        remainder="drop",
    )


def fit_risk_model(
    df: pd.DataFrame,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    treatment_col: str = "treatment",
    outcome_col: str = "outcome",
    random_state: int = config.RANDOM_SEED,
) -> RiskModelResult:
    """
    Fit a penalised logistic regression risk model.

    Returns out-of-fold predicted baseline risks for every patient (computed
    by setting treatment=0 in each held-out fold and predicting).
    """
    feature_cols = list(numeric_features) + list(categorical_features) + [treatment_col]
    X = df[feature_cols].copy()
    y = df[outcome_col].astype(int).values

    # Treatment is a numeric 0/1 column already
    pre = _build_preprocessor(
        numeric_features=list(numeric_features) + [treatment_col],
        categorical_features=list(categorical_features),
    )
    clf = LogisticRegressionCV(
        Cs=10,
        cv=config.CV_FOLDS,
        penalty="l2",
        scoring="neg_brier_score",
        max_iter=2000,
        random_state=random_state,
    )
    pipe = Pipeline([("pre", pre), ("clf", clf)])

    # Out-of-fold predictions for honest performance metrics
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=random_state)
    oof_pred = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
    cv_auc = roc_auc_score(y, oof_pred)
    cv_brier = brier_score_loss(y, oof_pred)

    # Refit on full data (final model)
    pipe.fit(X, y)

    # Compute baseline risk: same X but treatment forced to 0
    X_baseline = X.copy()
    X_baseline[treatment_col] = 0
    baseline_risk = pd.Series(
        pipe.predict_proba(X_baseline)[:, 1],
        index=df.index,
        name="baseline_risk",
    )

    # Recover post-transformation feature names for inspection
    feat_names = pipe.named_steps["pre"].get_feature_names_out().tolist()

    return RiskModelResult(
        pipeline=pipe,
        feature_names=feat_names,
        cv_auc=float(cv_auc),
        cv_brier=float(cv_brier),
        baseline_risk=baseline_risk,
    )
