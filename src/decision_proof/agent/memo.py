"""Claude drafts the CFO memo. It explains the results; it never produces them —
every number it's allowed to use was already computed by the estimators and the
refutation suite before Claude ever sees the decision.

Three guardrails, per the README:
1. Verdict-gated templates: a NOT_PROVEN verdict can only produce a "we can't yet
   attribute this, here's what would settle it" memo — enforced by using a
   different system prompt per verdict, not by asking nicely inside one prompt.
2. Numeric grounding check: every number in the memo must match a value from the
   analysis, within rounding (agent/grounding.py). A memo that fails is rejected
   and regenerated, not silently shipped.
3. Ranges required: the system prompts explicitly forbid stating a point estimate
   without its 90% interval.

Model is set via ANTHROPIC_MODEL; the API key is read from ANTHROPIC_API_KEY by the
Anthropic SDK itself — never passed or logged here. This module is the only place in
the codebase that makes a network call (see CLAUDE.md).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from decision_proof.agent.grounding import check_grounding, grounded_values
from decision_proof.contracts import Decision, EffectEstimate, RefutationReport

DEFAULT_MODEL = "claude-sonnet-5"
MAX_ATTEMPTS = 3

_COMMON_RULES = """
You are drafting a short internal memo about a pricing decision, for a CFO and board audience.
Hard rules, no exceptions:
- Use ONLY the numbers given to you below. Never introduce a number (a dollar figure, a
  percentage, a week count) that isn't stated in the analysis context. If you want to
  describe something you don't have an exact number for, describe it qualitatively instead.
- Never state a point estimate without its 90% interval alongside it.
- Keep it to one short paragraph (4-6 sentences). No headers, no bullet lists.
"""

SYSTEM_PROMPTS = {
    "BOARD_READY": _COMMON_RULES
    + """
The verdict is BOARD_READY: the effect is real and has survived refutation. Write a confident
memo stating the effect range, the lag before it took effect, and that it's ready to inform a
board-level rollout decision.
""",
    "DIRECTIONAL": _COMMON_RULES
    + """
The verdict is DIRECTIONAL: critical checks passed, but the interval includes zero or the
methods only partly agree. Write a memo that states the estimated range honestly, is explicit
that this is directional (not board-ready) evidence, and names the specific reason it falls
short (from the reasons given below) rather than overstating confidence.
""",
    "NOT_PROVEN": _COMMON_RULES
    + """
The verdict is NOT_PROVEN: a critical refutation check failed. You may ONLY write a memo in
the form "we can't yet attribute this to the decision, and here's what would settle it." Do
NOT state an effect range as if it were credible — name which check failed and why, and say
what evidence (more history, a cleaner control group, a different window) would be needed to
reach a verdict. Never imply the decision worked or didn't work; the honest answer right now
is "not proven."
""",
}


class MemoRejected(Exception):
    """Raised when a memo still contains ungrounded numbers after MAX_ATTEMPTS."""


@dataclass
class MemoResult:
    text: str
    attempts: int
    verdict: str


def _build_context(decision: Decision, estimate: EffectEstimate, report: RefutationReport) -> str:
    lines = [
        f"Decision: {decision.description}",
        f"Treated units: {len(decision.treated_units)}",
        f"Method: {estimate.method}",
        f"Verdict: {report.verdict}",
        f"Cumulative effect: ${estimate.cum_effect:,.0f} (90% interval: ${estimate.cum_lo:,.0f} to ${estimate.cum_hi:,.0f})",
        f"Percent effect: {estimate.pct_effect:.1f}% (90% interval: {estimate.pct_lo:.1f}% to {estimate.pct_hi:.1f}%)",
    ]
    if estimate.lag_50 is not None:
        lines.append(f"Lag to 50% of the effect: {estimate.lag_50} weeks")
    if estimate.lag_90 is not None:
        lines.append(f"Lag to 90% of the effect: {estimate.lag_90} weeks")
    lines.append("Refutation reasons: " + "; ".join(report.reasons))
    failed = [c.name for c in report.checks if c.critical and c.status == "FAIL"]
    if failed:
        lines.append("Failed critical checks: " + ", ".join(failed))
    return "\n".join(lines)


def draft_memo(
    decision: Decision,
    estimate: EffectEstimate,
    report: RefutationReport,
    model: str | None = None,
    client=None,
) -> MemoResult:
    model = model or os.environ.get("ANTHROPIC_MODEL", DEFAULT_MODEL)
    if client is None:
        import anthropic

        client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

    system = SYSTEM_PROMPTS[report.verdict]
    allowed = grounded_values(decision, estimate, report)
    context = _build_context(decision, estimate, report)
    user_message = f"Analysis context:\n{context}\n\nDraft the memo now."

    feedback = ""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        response = client.messages.create(
            model=model,
            max_tokens=500,
            system=system,
            messages=[{"role": "user", "content": user_message + feedback}],
        )
        text = "".join(block.text for block in response.content if block.type == "text")

        ok, ungrounded = check_grounding(text, allowed)
        if ok:
            return MemoResult(text=text, attempts=attempt, verdict=report.verdict)

        feedback = (
            f"\n\nYour previous draft used number(s) not present in the analysis context: "
            f"{ungrounded}. Revise the memo using ONLY the numbers given above."
        )

    raise MemoRejected(
        f"memo rejected after {MAX_ATTEMPTS} attempts; last draft still contained ungrounded numbers"
    )
