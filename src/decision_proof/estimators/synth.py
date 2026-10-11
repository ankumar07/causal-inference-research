"""Synthetic control, fit per treated unit and aggregated.

Donor weights are constrained non-negative and sum to 1, fit on the pre-period
only (cvxpy QP). Inference is placebo-in-space: refit the same procedure with
every untreated donor as a fake "treated" unit, and compare the real unit's
post/pre RMSPE ratio against that placebo distribution. The 90% interval on the
aggregated effect is a permutation-style interval built from the same placebo
units' cumulative (and per-week) effects, under the null that they have no true
effect.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable

import cvxpy as cp
import numpy as np
import pandas as pd
from scipy import stats

from decision_proof.contracts import Decision, EffectEstimate, Panel


def _pivot(df: pd.DataFrame, outcome: str) -> pd.DataFrame:
    """period x unit_id matrix, sorted by period."""
    return df.pivot(index="period", columns="unit_id", values=outcome).sort_index()


def _fit_weights(donor_pre: np.ndarray, target_pre: np.ndarray) -> np.ndarray:
    """Non-negative, sum-to-1 donor weights minimizing pre-period squared error.

    CLARABEL is used instead of OSQP (cvxpy's default QP solver): on this problem's
    scale (revenue in the tens of thousands) OSQP reports infeasible and a naive SCS
    fallback can converge to a solution that violates sum(w)==1 without flagging an
    error — CLARABEL solves it cleanly.
    """
    n_donors = donor_pre.shape[1]
    w = cp.Variable(n_donors, nonneg=True)
    objective = cp.Minimize(cp.sum_squares(donor_pre @ w - target_pre))
    constraints = [cp.sum(w) == 1]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.CLARABEL)
    if w.value is None or problem.status != cp.OPTIMAL:
        # rare fallback for an ill-conditioned donor subset; SCS still converges to a
        # validly-constrained (if not fully tolerance-tight) solution, which we renormalize below.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            problem.solve(solver=cp.SCS, eps=1e-9)
    if w.value is None:
        raise RuntimeError("synthetic control weight optimization failed to converge")
    weights = np.clip(np.asarray(w.value).ravel(), 0, None)
    return weights / weights.sum()


def _unit_effect(
    wide: pd.DataFrame, target: str, donors: list[str], pre_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (effect series over all periods, donor weights)."""
    target_series = wide[target].to_numpy()
    donor_matrix = wide[donors].to_numpy()
    # scale by the target's own pre-period mean before solving: the optimal weights are
    # scale-invariant, but the solver's numerical conditioning isn't — donor units can sit
    # at a very different revenue scale than the target, which otherwise occasionally makes
    # CLARABEL (and even the SCS fallback) fail to converge.
    scale = target_series[pre_mask].mean() or 1.0
    w = _fit_weights(donor_matrix[pre_mask] / scale, target_series[pre_mask] / scale)
    synthetic = donor_matrix @ w
    effect = target_series - synthetic
    return effect, w


def quick_cum_effect(panel: Panel, decision: Decision, exclude_donors: tuple[str, ...] = ()) -> float:
    """Point-estimate-only cumulative effect (no placebo-in-space loop). For refutation
    checks (leave-one-out, pre-window sensitivity) that need many cheap refits and only
    care about the point estimate, not full inference.
    """
    outcome = panel.outcome
    df = panel.df
    start = decision.start_period
    treated_units = list(decision.treated_units)
    donor_pool = [u for u in df["unit_id"].unique() if u not in treated_units and u not in exclude_donors]

    wide = _pivot(df, outcome)
    periods = wide.index.to_numpy()
    pre_mask = periods < start
    post_mask = ~pre_mask

    cum = 0.0
    for unit in treated_units:
        effect, _ = _unit_effect(wide, unit, donor_pool, pre_mask)
        cum += effect[post_mask].sum()
    return float(cum)


def _rmspe_ratio(effect: np.ndarray, pre_mask: np.ndarray, post_mask: np.ndarray) -> float:
    pre_rmspe = np.sqrt(np.mean(effect[pre_mask] ** 2))
    post_rmspe = np.sqrt(np.mean(effect[post_mask] ** 2))
    return float(post_rmspe / pre_rmspe) if pre_rmspe > 0 else np.inf


@dataclass
class SyntheticControl:
    name: str = "synthetic_control"
    n_resamples: int = 2000
    seed: int = 1
    alpha: float = 0.10
    # Given the unit(s) being fit (the real treated units, or a single placebo unit playing
    # that role), returns donor ids to exclude from the pool. None = exclude nothing (plain
    # SC). Subclassed by ConservationConstrainedSyntheticControl to exclude known-adjacent
    # donors that could be contaminated by spillover from the unit(s) being fit.
    exclude_fn: Callable[[list[str]], set[str]] | None = field(default=None, repr=False, compare=False)

    def fit(self, panel: Panel, decision: Decision) -> EffectEstimate:
        outcome = panel.outcome
        df = panel.df
        start = decision.start_period
        treated_units = list(decision.treated_units)
        base_pool = [u for u in df["unit_id"].unique() if u not in treated_units]
        excluded_for_treated = self.exclude_fn(treated_units) if self.exclude_fn else set()
        donor_pool = [u for u in base_pool if u not in excluded_for_treated]

        wide = _pivot(df, outcome)
        periods = wide.index.to_numpy()
        pre_mask = periods < start
        post_mask = ~pre_mask
        post_periods = periods[post_mask]

        # --- real treated units ---
        treated_effects = {}
        donor_weights = {}
        pre_fit_quality = {}
        real_ratios = {}
        for unit in treated_units:
            effect, w = _unit_effect(wide, unit, donor_pool, pre_mask)
            treated_effects[unit] = effect
            donor_weights[unit] = {d: float(wi) for d, wi in zip(donor_pool, w) if wi > 1e-3}
            pre_rmspe = np.sqrt(np.mean(effect[pre_mask] ** 2))
            pre_fit_quality[unit] = float(pre_rmspe / wide[unit].to_numpy()[pre_mask].mean())
            real_ratios[unit] = _rmspe_ratio(effect, pre_mask, post_mask)

        # --- placebo-in-space: every donor, treated as a fake-treated unit against the rest.
        # Candidates are the full base pool (not the treated units' reduced donor_pool) — each
        # placebo unit gets its own exclusion applied symmetrically to how a real treated unit
        # would be fit, so the inference stays apples-to-apples. ---
        placebo_effects = {}
        placebo_ratios = {}
        for unit in base_pool:
            sub_donors = [d for d in base_pool if d != unit]
            excluded_for_unit = self.exclude_fn([unit]) if self.exclude_fn else set()
            sub_donors = [d for d in sub_donors if d not in excluded_for_unit]
            effect, _ = _unit_effect(wide, unit, sub_donors, pre_mask)
            placebo_effects[unit] = effect
            placebo_ratios[unit] = _rmspe_ratio(effect, pre_mask, post_mask)

        placebo_pvalues = {}
        all_ratios = np.array(list(placebo_ratios.values()))
        for unit, ratio in real_ratios.items():
            extreme = int(np.sum(all_ratios >= ratio))
            placebo_pvalues[unit] = (extreme + 1) / (len(all_ratios) + 1)
        # Fisher's method to combine per-unit placebo p-values: a Bonferroni min-p
        # correction is nearly un-passable here, since with ~20 donors the smallest
        # achievable single-unit p-value (~1/21) alone exceeds typical thresholds
        # once multiplied by n_treated.
        if placebo_pvalues:
            ps = np.clip(list(placebo_pvalues.values()), 1e-10, 1.0)
            fisher_stat = -2 * np.sum(np.log(ps))
            combined_pvalue = float(1 - stats.chi2.cdf(fisher_stat, df=2 * len(ps)))
        else:
            combined_pvalue = 1.0

        # --- aggregate across treated units ---
        agg_effect = np.sum([treated_effects[u] for u in treated_units], axis=0)
        agg_synthetic_post = np.sum(
            [wide[u].to_numpy()[post_mask] - treated_effects[u][post_mask] for u in treated_units], axis=0
        )
        cum_effect = float(agg_effect[post_mask].sum())
        denom = float(agg_synthetic_post.sum())
        pct_effect = cum_effect / denom * 100 if denom else 0.0

        # --- permutation-style 90% interval from the placebo units' noise ---
        rng = np.random.default_rng(self.seed)
        placebo_cum = np.array([placebo_effects[u][post_mask].sum() for u in base_pool])
        draws = rng.choice(placebo_cum, size=(self.n_resamples, len(treated_units)), replace=True).sum(axis=1)
        lo_q, hi_q = np.quantile(draws, [self.alpha / 2, 1 - self.alpha / 2])
        cum_lo, cum_hi = cum_effect + lo_q, cum_effect + hi_q
        pct_lo = cum_lo / denom * 100 if denom else 0.0
        pct_hi = cum_hi / denom * 100 if denom else 0.0

        placebo_by_h = np.array([placebo_effects[u][post_mask] for u in base_pool])  # n_candidates x n_post
        path = []
        for i, p in enumerate(post_periods):
            point = float(agg_effect[post_mask][i])
            h_draws = rng.choice(placebo_by_h[:, i], size=(self.n_resamples, len(treated_units)), replace=True).sum(
                axis=1
            )
            h_lo, h_hi = np.quantile(h_draws, [self.alpha / 2, 1 - self.alpha / 2])
            path.append({"h": int(p - start), "est": point, "lo": point + h_lo, "hi": point + h_hi})

        total_path = sum(r["est"] for r in path) or 1.0
        running = 0.0
        lag_50 = lag_90 = None
        for r in path:
            running += r["est"]
            frac = running / total_path
            if lag_50 is None and frac >= 0.5:
                lag_50 = r["h"]
            if lag_90 is None and frac >= 0.9:
                lag_90 = r["h"]

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
                "donor_weights": donor_weights,
                "pre_fit_quality": pre_fit_quality,
                "placebo_rmspe_ratios": {"treated": real_ratios, "donors": placebo_ratios},
                "placebo_pvalues": placebo_pvalues,
                "placebo_pvalue_combined": combined_pvalue,
                "n_donors": len(donor_pool),
                "n_excluded_from_treated_pool": len(excluded_for_treated),
            },
        )
