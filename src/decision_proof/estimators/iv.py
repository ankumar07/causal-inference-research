"""Instrumental variables (2SLS): recovers the causal effect of price on
revenue despite price being endogenous (confounded with unobserved demand),
using an exogenous cost shock as the instrument.

Implemented as two plain OLS regressions rather than a dedicated IV package
call, so the mechanism stays visible:
  stage 1: price ~ cost_shock                 (how much the instrument moves price)
  stage 2: revenue ~ price_hat (fitted from stage 1)   (price's effect, purged of
                                                          the confounding that rode
                                                          along with the *actual*
                                                          price, since price_hat only
                                                          contains the cost-shock-driven
                                                          variation)
Standard errors are the standard 2SLS ones (using the second-stage residuals
against the *actual* price, not price_hat — a common correction; using the
stage-2 OLS's own reported SE on price_hat understates uncertainty).

Also exposes naive_ols(), the biased comparison: regressing revenue directly
on price, which stays confounded by the latent demand variable the instrument
was built to route around. Same "show the naive method fails, show the fix"
pattern as did.py/synth.py vs synth_conservation.py and did.py vs
staggered_did.py elsewhere in this repo.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

Z_90 = 1.6448536269514722


@dataclass(frozen=True)
class RegressionResult:
    coef: float
    se: float
    lo: float
    hi: float
    n: int


def _ols_coef_on(y: np.ndarray, x: np.ndarray) -> tuple[float, float, int]:
    design = sm.add_constant(x.reshape(-1, 1))
    model = sm.OLS(y, design).fit()
    return float(model.params[1]), float(model.bse[1]), len(y)


def naive_ols(df: pd.DataFrame, outcome_col: str, price_col: str, alpha: float = 0.10) -> RegressionResult:
    """Biased by construction: price is confounded with the unobserved demand
    that also drives the outcome directly."""
    y = df[outcome_col].to_numpy()
    x = df[price_col].to_numpy()
    coef, se, n = _ols_coef_on(y, x)
    return RegressionResult(coef=coef, se=se, lo=coef - Z_90 * se, hi=coef + Z_90 * se, n=n)


@dataclass
class InstrumentalVariables:
    name: str = "iv_2sls"
    alpha: float = 0.10

    def first_stage_f_stat(self, df: pd.DataFrame, price_col: str, instrument_col: str) -> float:
        """Classic weak-instrument diagnostic: the first stage's F-statistic on the
        instrument. Rule of thumb (Staiger & Stock): F < 10 means the instrument is
        too weakly related to the endogenous variable to trust the 2SLS estimate,
        regardless of how clean the exclusion restriction is.
        """
        price = df[price_col].to_numpy()
        z = df[instrument_col].to_numpy()
        design = sm.add_constant(z.reshape(-1, 1))
        model = sm.OLS(price, design).fit()
        return float(model.fvalue)

    def first_stage(self, df: pd.DataFrame, price_col: str, instrument_col: str) -> RegressionResult:
        price = df[price_col].to_numpy()
        z = df[instrument_col].to_numpy()
        coef, se, n = _ols_coef_on(price, z)
        return RegressionResult(coef=coef, se=se, lo=coef - Z_90 * se, hi=coef + Z_90 * se, n=n)

    def fit(
        self, df: pd.DataFrame, outcome_col: str, price_col: str, instrument_col: str
    ) -> RegressionResult:
        y = df[outcome_col].to_numpy()
        x = df[price_col].to_numpy()
        z = df[instrument_col].to_numpy()
        n = len(df)

        z_design = sm.add_constant(z.reshape(-1, 1))
        stage1 = sm.OLS(x, z_design).fit()
        x_hat = stage1.predict(z_design)

        stage2_design = sm.add_constant(x_hat.reshape(-1, 1))
        stage2 = sm.OLS(y, stage2_design).fit()
        coef = float(stage2.params[1])

        # proper 2SLS SE: residuals from y vs actual x (not x_hat), scaled by the
        # fitted-value design's (X'X)^-1 - the standard correction for using OLS
        # machinery to compute 2SLS, since stage2's own reported SE (on x_hat) is
        # too small.
        resid = y - (stage2.params[0] + coef * x)
        sigma2 = float(np.sum(resid**2) / (n - 2))
        xhat_design = stage2_design
        xtx_inv = np.linalg.inv(xhat_design.T @ xhat_design)
        se = float(np.sqrt(sigma2 * xtx_inv[1, 1]))

        return RegressionResult(coef=coef, se=se, lo=coef - Z_90 * se, hi=coef + Z_90 * se, n=n)
