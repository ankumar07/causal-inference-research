"""Marketing mix modeling: adstock + saturation transforms, grid search over
their parameters, OLS for the linear coefficients given each grid point.

A simplified, lightweight version — real production MMM (Meta's Robyn,
Google's LightweightMMM) uses Bayesian priors / hierarchical models and far
more careful handling of multicollinearity between simultaneously-run
channels. Here: grid search (decay, saturation k) x OLS, picking the
combination with the best R² — transparent, not state-of-the-art, and
documented as such rather than implying more rigor than it has.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm

from decision_proof.simulate.mmm import adstock, saturate


@dataclass(frozen=True)
class MMMEstimate:
    decay: float
    saturation_k: float
    price_coef: float
    price_coef_se: float
    promo_scale: float  # coefficient on the saturated-adstock promo term
    promo_scale_se: float
    r2: float
    fitted_promo_response: np.ndarray  # promo_scale * saturate(adstock(spend, decay), k), per period


@dataclass
class MarketingMixModel:
    name: str = "mmm"
    decay_grid: tuple = (0.1, 0.3, 0.5, 0.7, 0.9)
    k_grid_fracs: tuple = (0.25, 0.5, 1.0, 2.0)  # candidate k as a fraction of max adstocked spend

    def fit(self, df: pd.DataFrame, outcome_col: str, price_col: str, spend_col: str) -> MMMEstimate:
        spend = df[spend_col].to_numpy()
        price = df[price_col].to_numpy()
        y = df[outcome_col].to_numpy()

        best = None
        for decay in self.decay_grid:
            adstocked = adstock(spend, decay)
            max_adstocked = adstocked.max() or 1.0
            for k_frac in self.k_grid_fracs:
                k = k_frac * max_adstocked
                sat = saturate(adstocked, k)
                design = sm.add_constant(np.column_stack([price, sat]))
                model = sm.OLS(y, design).fit()
                if best is None or model.rsquared > best[0]:
                    best = (model.rsquared, decay, k, model, sat)

        r2, decay, k, model, sat = best
        return MMMEstimate(
            decay=decay,
            saturation_k=k,
            price_coef=float(model.params[1]),
            price_coef_se=float(model.bse[1]),
            promo_scale=float(model.params[2]),
            promo_scale_se=float(model.bse[2]),
            r2=float(r2),
            fitted_promo_response=float(model.params[2]) * sat,
        )
