"""RCT analysis + CUPED. True randomization means simple difference-in-means is
already unbiased — the one scenario in this repo where the estimator's job
isn't to fix a bias, it's to tighten the interval around an already-correct
answer. CUPED (Controlled-experiment Using Pre-Experiment Data, Deng et al.
2013) uses a pre-treatment covariate to strip out some of the outcome's
variance that has nothing to do with the treatment — valid specifically
*because* the covariate is pre-treatment, so it can't itself be affected by
random assignment, and adjusting for it can't introduce bias, only reduce
noise. Variance reduction is approximately corr(covariate, outcome)^2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

Z_90 = 1.6448536269514722


@dataclass(frozen=True)
class RCTResult:
    effect: float
    se: float
    lo: float
    hi: float
    n_treated: int
    n_control: int


def simple_diff_in_means(
    df: pd.DataFrame, outcome_col: str, treatment_col: str, alpha: float = 0.10
) -> RCTResult:
    treated = df.loc[df[treatment_col] == 1, outcome_col].to_numpy()
    control = df.loc[df[treatment_col] == 0, outcome_col].to_numpy()
    effect = float(treated.mean() - control.mean())
    se = float(np.sqrt(treated.var(ddof=1) / len(treated) + control.var(ddof=1) / len(control)))
    return RCTResult(
        effect=effect,
        se=se,
        lo=effect - Z_90 * se,
        hi=effect + Z_90 * se,
        n_treated=len(treated),
        n_control=len(control),
    )


def cuped(
    df: pd.DataFrame, outcome_col: str, treatment_col: str, covariate_col: str, alpha: float = 0.10
) -> RCTResult:
    y = df[outcome_col].to_numpy()
    x = df[covariate_col].to_numpy()
    theta = np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1)

    adjusted = df.copy()
    adjusted["_y_cuped"] = y - theta * (x - x.mean())
    return simple_diff_in_means(adjusted, "_y_cuped", treatment_col, alpha=alpha)


def variance_reduction(df: pd.DataFrame, outcome_col: str, covariate_col: str) -> float:
    """Approximate fraction of variance CUPED removes: corr(covariate, outcome)^2."""
    corr = df[[outcome_col, covariate_col]].corr().iloc[0, 1]
    return float(corr**2)
