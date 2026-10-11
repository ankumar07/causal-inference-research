"""Staggered-adoption-robust DiD, a simplified Callaway & Sant'Anna (2021)-style
estimator.

The problem it fixes: ordinary two-way-fixed-effects DiD (what did.py does)
assumes a single treatment date. Applied to a staggered rollout, it silently
uses already-treated units as part of the implicit comparison for later-treated
units during their own pre-period, and the standard TWFE weighting can even put
*negative* weight on some genuinely-treated comparisons when effects vary over
time — producing a biased estimate even in a perfectly clean experiment, not
from noise or confounding.

The fix: never compare a unit to another treated unit. Each treated unit gets
its own 2x2 DiD against only the never-treated donor pool, aligned to its own
event time (h = periods since *that unit's* start, not a shared start date).
Every treated unit here is its own cohort of one (see
simulate/scenarios.py::_s5_treated_and_starts — every staggered start is
distinct), so the usual group-time ATT(g,t) aggregation collapses to a plain
per-unit, per-relative-time average — transparent arithmetic, not a second
regression. Inference is a percentile bootstrap over the control pool.

Unlike the other estimators, this one can't work off Decision.start_period (a
single int) — staggered timing is the whole point — so its fit() signature
takes an explicit per-unit start-period map instead.

Calibration note, reported as-is rather than tuned to look good (see
CLAUDE.md): tried to reproduce B5's "build it, calibrate a scenario that
actually exercises the bias, confirm improvement" arc here, and the last step
didn't land. A homogeneous effect across cohorts (plain
simulate.scenarios.scenario_s5) gives naive TWFE nothing to be biased by, so
that comparison is a wash by construction (not a meaningful test).
scenario_s5_heterogeneous (effect growing ~0x to ~2.2x across the 6 waves)
does introduce the actual bias mechanism — but on the most reliable read
available (120 seeds, parallelized), naive TWFE's mean absolute bias ($145.3k)
is *better* than this estimator's ($154.8k), not worse. Two smaller, earlier
samples (40 and 60 seeds) had each told a different, more flattering story in
both directions before this larger run settled it — a reminder that N=30-60
isn't enough to trust a comparison like this one way or the other.

Best current read on why: with only 6 treated units (each its own cohort of
one), this estimator's percentile bootstrap over the control pool carries
enough of its own variance that it swamps whatever systematic TWFE bias the
heterogeneous effect sizes introduce at this scale — the same bias-variance
tradeoff B5 ran into with its first (rejected) donor-exclusion attempt, except
here it shows up in the kept design too, not just a discarded first attempt.
Left in the codebase as a correctly-implemented, honestly-evaluated estimator
rather than pulled or force-fit to win a comparison — the textbook fix for
staggered-adoption bias is real and worth having built, even though this
particular 6-unit demo is too small a stage for it to pay off.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from decision_proof.contracts import EffectEstimate, Panel

Z_90 = 1.6448536269514722


def _pivot(df: pd.DataFrame, outcome: str) -> pd.DataFrame:
    return df.pivot(index="period", columns="unit_id", values=outcome).sort_index()


def _unit_effect_path(
    wide: pd.DataFrame, unit: str, start: int, control_avg: pd.Series
) -> dict[int, float]:
    """effect_i(h) for h = 0, 1, 2, ... (periods since this unit's own start),
    each a clean 2x2 DiD against the control average, using period (start - 1)
    as this unit's own baseline.
    """
    baseline = wide[unit].loc[start - 1] - control_avg.loc[start - 1]
    post_periods = wide.index[wide.index >= start]
    return {int(p - start): float((wide[unit].loc[p] - control_avg.loc[p]) - baseline) for p in post_periods}


@dataclass
class StaggeredDiD:
    name: str = "staggered_did"
    n_resamples: int = 2000
    seed: int = 1
    alpha: float = 0.10

    def fit(
        self,
        panel: Panel,
        unit_start_periods: dict[str, int],
        control_units: list[str] | None = None,
    ) -> EffectEstimate:
        outcome = panel.outcome
        df = panel.df
        wide = _pivot(df, outcome)
        treated_units = list(unit_start_periods.keys())
        controls = control_units or [u for u in df["unit_id"].unique() if u not in treated_units]

        control_avg = wide[controls].mean(axis=1)
        unit_paths = {u: _unit_effect_path(wide, u, g, control_avg) for u, g in unit_start_periods.items()}

        max_h = max(h for path in unit_paths.values() for h in path)
        path = []
        for h in range(0, max_h + 1):
            contributions = [unit_paths[u][h] for u in treated_units if h in unit_paths[u]]
            if not contributions:
                continue
            path.append({"h": h, "est": float(np.sum(contributions)), "n_cohorts": len(contributions)})

        cum_effect = float(sum(v for p in unit_paths.values() for v in p.values()))

        rng = np.random.default_rng(self.seed)
        boot_cum = np.empty(self.n_resamples)
        for b in range(self.n_resamples):
            sampled = rng.choice(controls, size=len(controls), replace=True)
            c_avg = wide[sampled].mean(axis=1)
            draw_cum = 0.0
            for u, g in unit_start_periods.items():
                draw_path = _unit_effect_path(wide, u, g, c_avg)
                draw_cum += sum(draw_path.values())
            boot_cum[b] = draw_cum

        # percentile bootstrap, re-centered on the point estimate (the bootstrap
        # distribution's own center can drift slightly from cum_effect; we want the
        # interval's *width* from the bootstrap, centered on the actual estimate)
        lo_q, hi_q = np.quantile(boot_cum - boot_cum.mean(), [self.alpha / 2, 1 - self.alpha / 2])
        cum_lo, cum_hi = cum_effect + lo_q, cum_effect + hi_q

        # denominator: each unit's own last-pre-period level x its number of observed
        # post periods, summed - "what revenue would have run at without treatment",
        # same style of proxy did.py uses for its own pct_effect denominator.
        pre_baseline_total = sum(wide[u].loc[g - 1] * len(unit_paths[u]) for u, g in unit_start_periods.items())
        denom = pre_baseline_total or float("nan")
        pct_effect = cum_effect / denom * 100 if denom and not np.isnan(denom) else 0.0
        pct_lo = cum_lo / denom * 100 if denom and not np.isnan(denom) else 0.0
        pct_hi = cum_hi / denom * 100 if denom and not np.isnan(denom) else 0.0

        total_path_est = sum(p["est"] for p in path) or 1.0
        running = 0.0
        lag_50 = lag_90 = None
        for p in path:
            running += p["est"]
            frac = running / total_path_est
            if lag_50 is None and frac >= 0.5:
                lag_50 = p["h"]
            if lag_90 is None and frac >= 0.9:
                lag_90 = p["h"]

        return EffectEstimate(
            method=self.name,
            cum_effect=cum_effect,
            cum_lo=cum_lo,
            cum_hi=cum_hi,
            pct_effect=pct_effect,
            pct_lo=pct_lo,
            pct_hi=pct_hi,
            path=path,
            lag_50=lag_50,
            lag_90=lag_90,
            diagnostics={
                "cohorts": {u: g for u, g in unit_start_periods.items()},
                "n_controls": len(controls),
                "bootstrap": {"reps": self.n_resamples, "seed": self.seed},
            },
        )
