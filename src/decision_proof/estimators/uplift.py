"""Uplift modeling / CATE: a T-learner.

Fits two separate outcome models — one on treated customers, one on control —
then predicts both for every customer and takes the difference:
CATE(x) = E[Y|X=x, T=1] - E[Y|X=x, T=0]. Simplest member of the meta-learner
family (vs. S-learner, X-learner, R-learner), used here with a linear model on
each arm (statsmodels OLS, already a dependency) to keep the mechanism
transparent — a linear T-learner recovers heterogeneity that's linear in the
covariate, which is what the simulated DGP has. A fully nonparametric
T-learner (e.g. gradient-boosted trees per arm) would generalize further but
adds a dependency and a black box this repo has otherwise avoided.

Valid under randomization (same as Q8's RCT, no confounding to separately
correct for) — the question here isn't "is the average effect real," it's
"whose effect is bigger."
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm


@dataclass
class FittedTLearner:
    m1: object
    m0: object
    covariate_col: str

    def predict_cate(self, covariate_values: np.ndarray) -> np.ndarray:
        x = sm.add_constant(np.asarray(covariate_values), has_constant="add")
        return self.m1.predict(x) - self.m0.predict(x)

    @property
    def cate_slope(self) -> float:
        return float(self.m1.params[1] - self.m0.params[1])

    @property
    def cate_intercept(self) -> float:
        return float(self.m1.params[0] - self.m0.params[0])

    def segment_cate(self, df: pd.DataFrame, n_segments: int = 3) -> pd.DataFrame:
        """Average predicted CATE by quantile segment of the covariate — "which
        segment is most price-sensitive," directly."""
        x = df[self.covariate_col].to_numpy()
        preds = self.predict_cate(x)
        labels = [f"tier_{i + 1}" for i in range(n_segments)]
        segments = pd.qcut(x, n_segments, labels=labels)
        out = pd.DataFrame({"segment": segments, "cate": preds})
        return out.groupby("segment", observed=True)["cate"].mean().reset_index()


@dataclass
class TLearner:
    name: str = "t_learner"

    def fit(self, df: pd.DataFrame, outcome_col: str, treatment_col: str, covariate_col: str) -> FittedTLearner:
        treated = df[df[treatment_col] == 1]
        control = df[df[treatment_col] == 0]

        x1 = sm.add_constant(treated[covariate_col].to_numpy())
        m1 = sm.OLS(treated[outcome_col].to_numpy(), x1).fit()

        x0 = sm.add_constant(control[covariate_col].to_numpy())
        m0 = sm.OLS(control[outcome_col].to_numpy(), x0).fit()

        return FittedTLearner(m1=m1, m0=m0, covariate_col=covariate_col)
