"""Refutation checks. Each returns a RefutationCheck; none of them ever see the
simulator's ground truth — only what a real analyst would have: the panel, the
decision, and the already-fit estimates.

Critical checks are a hard PASS/FAIL gate (any FAIL -> NOT_PROVEN downstream).
Non-critical checks are a soft PASS/WARN signal. Thresholds are judgment calls,
documented in docs/METHODS.md, and overridable via the `thresholds` argument.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from decision_proof.contracts import Decision, EffectEstimate, Panel, RefutationCheck
from decision_proof.estimators.synth import SyntheticControl, quick_cum_effect

DEFAULT_THRESHOLDS = {
    "pre_trend_pvalue": 0.10,  # critical: FAIL if joint pre-trend p-value < this
    "sc_pre_fit_rmspe_rel": 0.15,  # critical: FAIL if relative pre-period RMSPE exceeds this
    "placebo_pvalue": 0.10,  # critical: FAIL if combined placebo-in-space p-value > this
    "placebo_in_time_min_pre_periods": 6,  # minimum periods needed on each side to run the check
    "leave_one_out_pct_move": 0.50,  # non-critical: WARN if a left-out donor swings the estimate more than this
    "pre_window_pct_move": 0.50,  # non-critical: WARN if a shorter pre-window swings the estimate more than this
}


def methods_agree(estimates: dict[str, EffectEstimate]) -> bool:
    """Same-sign point estimates with overlapping 90% intervals."""
    ests = list(estimates.values())
    if len(ests) < 2:
        return True
    signs = {np.sign(e.cum_effect) for e in ests}
    if len(signs) > 1:
        return False
    los = [e.cum_lo for e in ests]
    his = [e.cum_hi for e in ests]
    return max(los) <= min(his)


def pre_trend_check(did_estimate: EffectEstimate, threshold: float) -> RefutationCheck:
    leads = did_estimate.diagnostics.get("leads", [])
    if not leads:
        return RefutationCheck(
            name="pre_trend_test",
            statistic=None,
            threshold=threshold,
            status="WARN",
            explanation="No pre-period leads available (no pre-treatment history) to test for a pre-trend.",
            critical=True,
        )
    # joint Wald-style test that all lead coefficients are zero, ignoring their (unestimated) covariance
    chi2_stat = float(sum((r["est"] / r["se"]) ** 2 for r in leads if r["se"] > 0))
    df = len(leads)
    pvalue = float(1 - stats.chi2.cdf(chi2_stat, df=df))
    status = "FAIL" if pvalue < threshold else "PASS"
    return RefutationCheck(
        name="pre_trend_test",
        statistic=pvalue,
        threshold=threshold,
        status=status,
        explanation=(
            f"Joint test on {df} pre-period lead(s): p={pvalue:.3f}. "
            + ("A significant pre-trend exists before the decision." if status == "FAIL" else "No significant pre-trend.")
        ),
        critical=True,
    )


def sc_pre_fit_check(sc_estimate: EffectEstimate, threshold: float) -> RefutationCheck:
    quality = sc_estimate.diagnostics.get("pre_fit_quality", {})
    if not quality:
        return RefutationCheck(
            name="sc_pre_fit_quality",
            statistic=None,
            threshold=threshold,
            status="WARN",
            explanation="Synthetic control pre-fit diagnostics unavailable.",
            critical=True,
        )
    worst_unit, worst_rel_rmspe = max(quality.items(), key=lambda kv: kv[1])
    status = "FAIL" if worst_rel_rmspe > threshold else "PASS"
    return RefutationCheck(
        name="sc_pre_fit_quality",
        statistic=worst_rel_rmspe,
        threshold=threshold,
        status=status,
        explanation=(
            f"Worst pre-period fit is {worst_unit} at {worst_rel_rmspe:.1%} of its own scale "
            f"(threshold {threshold:.0%}). "
            + ("The synthetic control doesn't track the treated unit well before the decision." if status == "FAIL" else "")
        ),
        critical=True,
    )


def placebo_in_space_check(sc_estimate: EffectEstimate, threshold: float) -> RefutationCheck:
    combined_p = sc_estimate.diagnostics.get("placebo_pvalue_combined")
    if combined_p is None:
        return RefutationCheck(
            name="placebo_in_space",
            statistic=None,
            threshold=threshold,
            status="WARN",
            explanation="Placebo-in-space diagnostics unavailable.",
            critical=True,
        )
    status = "PASS" if combined_p <= threshold else "FAIL"
    return RefutationCheck(
        name="placebo_in_space",
        statistic=combined_p,
        threshold=threshold,
        status=status,
        explanation=(
            f"Combined placebo rank p-value is {combined_p:.3f} (threshold {threshold:.2f}). "
            + (
                "The treated unit's post/pre RMSPE ratio is unusually extreme among placebo units."
                if status == "PASS"
                else "The treated unit's effect is not distinguishable from a placebo donor's."
            )
        ),
        critical=True,
    )


def placebo_in_time_check(panel: Panel, decision: Decision, threshold: float, min_periods: int) -> RefutationCheck:
    df = panel.df
    n_pre = int(df.loc[df["unit_id"] == decision.treated_units[0], "period"].lt(decision.start_period).sum())
    pseudo_start = decision.start_period - n_pre // 2

    if n_pre // 2 < min_periods or (n_pre - n_pre // 2) < min_periods:
        return RefutationCheck(
            name="placebo_in_time",
            statistic=None,
            threshold=threshold,
            status="WARN",
            explanation=f"Only {n_pre} pre-period weeks available; too little history to test an earlier pseudo-decision.",
            critical=True,
        )

    pre_only = df[df["period"] < decision.start_period]
    pseudo_panel = Panel(df=pre_only, outcome=panel.outcome, period_freq=panel.period_freq)
    pseudo_decision = Decision(
        decision_id="placebo_in_time",
        description="Pseudo-decision at an earlier date (no real change occurred)",
        treated_units=decision.treated_units,
        start_period=pseudo_start,
        outcome=decision.outcome,
    )
    pseudo_estimate = SyntheticControl(name="placebo_in_time_sc", alpha=threshold).fit(pseudo_panel, pseudo_decision)
    significant = not (pseudo_estimate.cum_lo <= 0 <= pseudo_estimate.cum_hi)
    status = "FAIL" if significant else "PASS"
    return RefutationCheck(
        name="placebo_in_time",
        statistic=pseudo_estimate.cum_effect,
        threshold=threshold,
        status=status,
        explanation=(
            f"Pretending the decision happened at period {pseudo_start} (nothing really changed): "
            f"estimated effect ${pseudo_estimate.cum_effect:,.0f} [{pseudo_estimate.cum_lo:,.0f}, {pseudo_estimate.cum_hi:,.0f}]. "
            + ("A significant effect appears where none should exist." if status == "FAIL" else "No significant effect, as expected.")
        ),
        critical=True,
    )


def leave_one_out_check(panel: Panel, decision: Decision, sc_estimate: EffectEstimate, threshold: float) -> RefutationCheck:
    donor_pool = [u for u in panel.df["unit_id"].unique() if u not in decision.treated_units]
    if len(donor_pool) < 3:
        return RefutationCheck(
            name="leave_one_out_donors",
            statistic=None,
            threshold=threshold,
            status="WARN",
            explanation="Too few donors to meaningfully leave one out.",
            critical=False,
        )

    base = sc_estimate.cum_effect
    unstable = 0
    for donor in donor_pool:
        cum_effect = quick_cum_effect(panel, decision, exclude_donors=(donor,))
        sign_flip = np.sign(cum_effect) != np.sign(base)
        moved = abs(cum_effect - base) / abs(base) > threshold if base else False
        if sign_flip or moved:
            unstable += 1

    frac_unstable = unstable / len(donor_pool)
    status = "WARN" if frac_unstable > 0 else "PASS"
    return RefutationCheck(
        name="leave_one_out_donors",
        statistic=frac_unstable,
        threshold=threshold,
        status=status,
        explanation=(
            f"Dropping each of {len(donor_pool)} donors in turn: {unstable} run(s) flipped sign or moved "
            f"more than {threshold:.0%}."
        ),
        critical=False,
    )


def pre_window_sensitivity_check(panel: Panel, decision: Decision, sc_estimate: EffectEstimate, threshold: float) -> RefutationCheck:
    df = panel.df
    n_pre = int(df.loc[df["unit_id"] == decision.treated_units[0], "period"].lt(decision.start_period).sum())
    half_pre_start = decision.start_period - n_pre // 2
    if n_pre // 2 < 4:
        return RefutationCheck(
            name="pre_window_sensitivity",
            statistic=None,
            threshold=threshold,
            status="WARN",
            explanation="Too little pre-period history to test a shorter window.",
            critical=False,
        )

    short_df = df[df["period"] >= half_pre_start]
    short_panel = Panel(df=short_df, outcome=panel.outcome, period_freq=panel.period_freq)
    cum_effect = quick_cum_effect(short_panel, decision)

    base = sc_estimate.cum_effect
    moved = abs(cum_effect - base) / abs(base) if base else np.inf
    status = "WARN" if moved > threshold else "PASS"
    return RefutationCheck(
        name="pre_window_sensitivity",
        statistic=moved,
        threshold=threshold,
        status=status,
        explanation=(
            f"Re-estimated with half the pre-period ({n_pre} -> {n_pre - n_pre // 2} weeks): "
            f"effect moved {moved:.0%} (threshold {threshold:.0%})."
        ),
        critical=False,
    )


def method_agreement_check(estimates: dict[str, EffectEstimate]) -> RefutationCheck:
    agree = methods_agree(estimates)
    status = "PASS" if agree else "WARN"
    names = list(estimates.keys())
    return RefutationCheck(
        name="method_agreement",
        statistic=None,
        threshold=None,
        status=status,
        explanation=(
            f"{' vs '.join(names)}: "
            + ("intervals overlap and point estimates agree in sign." if agree else "intervals don't overlap, or point estimates disagree in sign.")
        ),
        critical=False,
    )


def run_checks(
    panel: Panel,
    decision: Decision,
    estimates: dict[str, EffectEstimate],
    thresholds: dict | None = None,
) -> list[RefutationCheck]:
    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    did_estimate = estimates.get("did_event_study")
    sc_estimate = estimates.get("synthetic_control")

    checks = []
    if did_estimate is not None:
        checks.append(pre_trend_check(did_estimate, t["pre_trend_pvalue"]))
    if sc_estimate is not None:
        checks.append(sc_pre_fit_check(sc_estimate, t["sc_pre_fit_rmspe_rel"]))
        checks.append(placebo_in_space_check(sc_estimate, t["placebo_pvalue"]))
        checks.append(
            placebo_in_time_check(panel, decision, t["placebo_pvalue"], t["placebo_in_time_min_pre_periods"])
        )
        checks.append(leave_one_out_check(panel, decision, sc_estimate, t["leave_one_out_pct_move"]))
        checks.append(pre_window_sensitivity_check(panel, decision, sc_estimate, t["pre_window_pct_move"]))
    if len(estimates) >= 2:
        checks.append(method_agreement_check(estimates))
    return checks
