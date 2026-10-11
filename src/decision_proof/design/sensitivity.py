"""Price sensitivity (dose-response): fit price-change-% against the resulting
revenue-effect-% across a batch of experiments run at different price points,
using only the ones that survived refutation. The slope is the headline "price
sensitivity" number — how much revenue moves, in percentage-point terms, per
percentage point of price change. A verdict of NOT_PROVEN keeps an experiment
out of the fit entirely, so one bad experiment can't quietly skew the range —
same principle as B5's conservation constraint, applied at the batch level.

This is the dose-response method the README's M6 describes, not the fuller
power-by-simulation experiment designer (that's a separate piece of M6).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from decision_proof.contracts import EffectEstimate, RefutationReport


@dataclass(frozen=True)
class SensitivityPoint:
    decision_id: str
    price_change_pct: float
    effect_pct: float
    effect_pct_lo: float
    effect_pct_hi: float
    verdict: str


@dataclass(frozen=True)
class SensitivityResult:
    slope: float  # % revenue effect per 1 percentage point of price change
    slope_lo: float
    slope_hi: float
    intercept: float
    n_points: int
    n_excluded_not_proven: int
    points: list[SensitivityPoint] = field(default_factory=list)

    @property
    def is_fitted(self) -> bool:
        return self.n_points >= 2 and not np.isnan(self.slope)


def fit_price_sensitivity(
    experiments: list[tuple[float, EffectEstimate, RefutationReport, str]],
    alpha: float = 0.10,
) -> SensitivityResult:
    """experiments: list of (price_change_pct, EffectEstimate, RefutationReport, decision_id).
    Only experiments with verdict != NOT_PROVEN are used for the fit; all of
    them (including excluded ones) are kept in `points` for display.
    """
    all_points = []
    used_x, used_y = [], []
    n_excluded = 0

    for price_pct, est, report, decision_id in experiments:
        point = SensitivityPoint(
            decision_id=decision_id,
            price_change_pct=price_pct,
            effect_pct=est.pct_effect,
            effect_pct_lo=est.pct_lo,
            effect_pct_hi=est.pct_hi,
            verdict=report.verdict,
        )
        all_points.append(point)
        if report.verdict == "NOT_PROVEN":
            n_excluded += 1
            continue
        used_x.append(price_pct * 100)  # work in percentage-point units
        used_y.append(est.pct_effect)

    if len(used_x) < 2:
        return SensitivityResult(
            slope=float("nan"),
            slope_lo=float("nan"),
            slope_hi=float("nan"),
            intercept=float("nan"),
            n_points=len(used_x),
            n_excluded_not_proven=n_excluded,
            points=all_points,
        )

    x = np.array(used_x)
    y = np.array(used_y)
    fit = stats.linregress(x, y)

    df = len(x) - 2
    if df > 0 and fit.stderr is not None:
        tval = stats.t.ppf(1 - alpha / 2, df)
        margin = tval * fit.stderr
        slope_lo, slope_hi = fit.slope - margin, fit.slope + margin
    else:
        slope_lo, slope_hi = float("-inf"), float("inf")

    return SensitivityResult(
        slope=float(fit.slope),
        slope_lo=float(slope_lo),
        slope_hi=float(slope_hi),
        intercept=float(fit.intercept),
        n_points=len(used_x),
        n_excluded_not_proven=n_excluded,
        points=all_points,
    )


def fit_price_sensitivity_from_rows(rows: list[dict], alpha: float = 0.10) -> SensitivityResult:
    """Same as fit_price_sensitivity, but for plain dict rows (e.g. from a cached
    app-layer query) instead of full EffectEstimate/RefutationReport objects — the
    fit only ever needed price_change_pct, pct_effect, pct_lo, pct_hi, verdict, and
    decision_id, so callers that already have exactly those fields shouldn't have
    to reconstruct pydantic objects just to call this.
    """

    class _Est:
        def __init__(self, row: dict):
            self.pct_effect = row["pct_effect"]
            self.pct_lo = row["pct_lo"]
            self.pct_hi = row["pct_hi"]

    class _Report:
        def __init__(self, row: dict):
            self.verdict = row["verdict"]

    experiments = [(r["price_change_pct"], _Est(r), _Report(r), r["decision_id"]) for r in rows]
    return fit_price_sensitivity(experiments, alpha=alpha)
