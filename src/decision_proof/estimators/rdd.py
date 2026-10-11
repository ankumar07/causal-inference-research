"""Regression discontinuity: Daybreak Coffee's Gold-tier loyalty threshold.

Local linear regression on each side of the cutoff, within a bandwidth —
standard practice, rather than fitting one regression across the whole range
(which would mix in the baseline relationship's curvature far from the cutoff
and bias the estimate). The discontinuity estimate is the right-side fitted
value at the cutoff minus the left-side fitted value at the cutoff.

Bandwidth here is a simple rule of thumb (a fixed fraction of the running
variable's observed range on each side), not the full Imbens-Kalyanaraman or
Calonico-Cattaneo-Titiunik data-driven bandwidth-selection literature —
documented as a lightweight approximation, not a claim to that rigor.

Result type is deliberately its own dataclass, not EffectEstimate: RDD
estimates a *local* effect right at the cutoff, not a cumulative dollar effect
over a post-period, and forcing it into EffectEstimate's shape would misstate
what the number means.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

Z_90 = 1.6448536269514722


@dataclass(frozen=True)
class RDDEstimate:
    cutoff: float
    bandwidth: float
    effect: float
    effect_lo: float
    effect_hi: float
    n_left: int
    n_right: int
    left_intercept: float
    right_intercept: float
    diagnostics: dict


def _local_linear_fit(x: np.ndarray, y: np.ndarray, cutoff: float) -> tuple[float, float, int]:
    """OLS of y on (x - cutoff); returns (fitted value at the cutoff, its SE, n)."""
    xc = (x - cutoff).reshape(-1, 1)
    design = sm.add_constant(xc)
    model = sm.OLS(y, design).fit()
    return float(model.params[0]), float(model.bse[0]), len(y)


@dataclass
class RegressionDiscontinuity:
    name: str = "regression_discontinuity"
    bandwidth_frac: float = 0.3  # fraction of the running variable's half-range, used on each side
    alpha: float = 0.10

    def fit(self, df: pd.DataFrame, running_var_col: str, outcome_col: str, cutoff: float) -> RDDEstimate:
        x_all = df[running_var_col].to_numpy()
        bandwidth = self.bandwidth_frac * (x_all.max() - x_all.min()) / 2

        left_mask = (x_all >= cutoff - bandwidth) & (x_all < cutoff)
        right_mask = (x_all >= cutoff) & (x_all <= cutoff + bandwidth)

        x_left, y_left = x_all[left_mask], df.loc[left_mask, outcome_col].to_numpy()
        x_right, y_right = x_all[right_mask], df.loc[right_mask, outcome_col].to_numpy()
        if len(x_left) < 5 or len(x_right) < 5:
            raise ValueError(f"too few observations within bandwidth {bandwidth:.1f} of cutoff {cutoff}")

        left_b0, left_se, n_left = _local_linear_fit(x_left, y_left, cutoff)
        right_b0, right_se, n_right = _local_linear_fit(x_right, y_right, cutoff)

        effect = right_b0 - left_b0
        se = (left_se**2 + right_se**2) ** 0.5
        effect_lo, effect_hi = effect - Z_90 * se, effect + Z_90 * se

        return RDDEstimate(
            cutoff=cutoff,
            bandwidth=bandwidth,
            effect=effect,
            effect_lo=effect_lo,
            effect_hi=effect_hi,
            n_left=n_left,
            n_right=n_right,
            left_intercept=left_b0,
            right_intercept=right_b0,
            diagnostics={"bandwidth_frac": self.bandwidth_frac, "se": se},
        )

    def placebo_cutoffs(
        self, df: pd.DataFrame, running_var_col: str, outcome_col: str, real_cutoff: float, offsets: list[float]
    ) -> dict[float, RDDEstimate]:
        """Refit at fake cutoffs away from the real one — same ethos as the rest of
        the repo's refutation suite: a real discontinuity shouldn't also show up
        at an arbitrary nearby point."""
        results = {}
        for offset in offsets:
            fake_cutoff = real_cutoff + offset
            try:
                results[fake_cutoff] = self.fit(df, running_var_col, outcome_col, fake_cutoff)
            except ValueError:
                continue
        return results
