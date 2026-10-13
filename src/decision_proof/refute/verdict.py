"""Verdict engine.

- BOARD_READY: every critical check passes, the methods agree, and the interval
  excludes zero.
- DIRECTIONAL: critical checks pass, but the interval includes zero or the
  methods only partly agree.
- NOT_PROVEN: any critical check fails. "Not proven" is a valid product output,
  not an error state.
"""

from __future__ import annotations

from decision_proof.contracts import Decision, EffectEstimate, Panel, RefutationCheck, RefutationReport
from decision_proof.refute.checks import methods_agree, run_checks


def decide_verdict(checks: list[RefutationCheck], estimates: dict[str, EffectEstimate]) -> RefutationReport:
    critical_fails = [c for c in checks if c.critical and c.status == "FAIL"]

    if critical_fails:
        reasons = [f"{c.name}: {c.explanation}" for c in critical_fails]
        return RefutationReport(
            checks=checks,
            verdict="NOT_PROVEN",
            reasons=["Failed critical check(s):"] + reasons,
        )

    agree = methods_agree(estimates)
    all_exclude_zero = all(not (e.cum_lo <= 0 <= e.cum_hi) for e in estimates.values()) if estimates else False

    if agree and all_exclude_zero:
        return RefutationReport(
            checks=checks,
            verdict="BOARD_READY",
            reasons=[
                "All critical checks passed.",
                "Methods agree (same sign, overlapping intervals).",
                "Every method's interval excludes zero.",
            ],
        )

    reasons = ["All critical checks passed, but the effect isn't board-ready yet:"]
    if not agree:
        reasons.append("Methods disagree (different sign, or non-overlapping intervals).")
    if not all_exclude_zero:
        reasons.append("At least one method's interval includes zero.")
    return RefutationReport(checks=checks, verdict="DIRECTIONAL", reasons=reasons)


def refute(
    panel: Panel,
    decision: Decision,
    estimates: dict[str, EffectEstimate],
    thresholds: dict | None = None,
) -> RefutationReport:
    """Run the full refutation suite and decide a verdict. Thresholds are
    documented in docs/METHODS.md and overridable per call."""
    checks = run_checks(panel, decision, estimates, thresholds)
    return decide_verdict(checks, estimates)
