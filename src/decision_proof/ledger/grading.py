"""Grade a logged decision once the actual outcome is available: re-estimate on the
extended window (we never observe a true counterfactual, even after the fact — the
best available proxy for "what really happened" is the same estimation method, matured
with more post-period data), then check whether that matured estimate falls inside the
originally predicted range.
"""

from __future__ import annotations

from decision_proof.contracts import Decision, Panel
from decision_proof.ledger.store import Ledger


def interval_score(lo: float, hi: float, realized: float, alpha: float = 0.10) -> float:
    """Gneiting & Raftery's interval score for a (1-alpha) interval: rewards narrow
    intervals, penalizes missing the realized value proportionally to how far outside
    and how narrow alpha is. Lower is better.
    """
    width = hi - lo
    penalty = 0.0
    if realized < lo:
        penalty = (2 / alpha) * (lo - realized)
    elif realized > hi:
        penalty = (2 / alpha) * (realized - hi)
    return width + penalty


def grade_decision(
    ledger: Ledger,
    decision_id: str,
    extended_panel: Panel,
    decision: Decision,
    estimator,
    alpha: float = 0.10,
) -> dict:
    """Re-estimate on `extended_panel` (more post-period data than was available when
    the decision was logged) and grade the original prediction against it."""
    record = ledger.get_decision(decision_id)
    if record is None:
        raise ValueError(f"decision_id {decision_id!r} not found")
    if record["status"] == "graded":
        raise ValueError(f"decision_id {decision_id!r} is already graded")

    matured_estimate = estimator.fit(extended_panel, decision)
    realized = matured_estimate.cum_effect

    inside = record["pred_cum_lo"] <= realized <= record["pred_cum_hi"]
    score = interval_score(record["pred_cum_lo"], record["pred_cum_hi"], realized, alpha=alpha)

    ledger.record_grade(decision_id, realized_cum_effect=realized, inside_range=bool(inside), interval_score=score)

    return {
        "decision_id": decision_id,
        "realized_cum_effect": realized,
        "inside_range": bool(inside),
        "interval_score": score,
    }
