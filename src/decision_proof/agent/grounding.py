"""Numeric grounding check: every number in a memo must match a value that actually
came out of the analysis, within rounding. No network calls here — pure functions,
so this is fully unit-testable without an API key.
"""

from __future__ import annotations

import re

from decision_proof.contracts import Decision, EffectEstimate, RefutationReport

# small counts (weeks, unit counts, list markers, ordinals) are not a hallucination
# risk worth flagging — only larger, "substantive" numbers need to trace back to the
# analysis output.
SMALL_NUMBER_EXEMPTION = 12

# fixed statistical constants used throughout the product (the 90% interval, its
# complement), not analysis outputs — always allowed, never a hallucination risk.
STANDARD_ALLOWED = {90.0, 10.0}

_NUMBER_RE = re.compile(r"-?\$?\(?\d[\d,]*\.?\d*\)?%?(?:[KkMm](?![a-zA-Z]))?")
_SUFFIX_MULT = {"k": 1_000, "m": 1_000_000}


def _parse_number(raw: str) -> float | None:
    s = raw.strip()
    negative = s.startswith("(") and s.endswith(")")
    s = s.strip("()")
    is_pct = s.endswith("%")
    s = s.rstrip("%")
    mult = 1.0
    if s and s[-1].lower() in _SUFFIX_MULT:
        mult = _SUFFIX_MULT[s[-1].lower()]
        s = s[:-1]
    s = s.replace("$", "").replace(",", "").strip()
    if s in ("", "-", "."):
        return None
    try:
        value = float(s) * mult
    except ValueError:
        return None
    if negative:
        value = -abs(value)
    return value if not is_pct else value  # percentages are matched against pct_* values directly


def extract_numbers(text: str) -> list[float]:
    """All numeric values mentioned in `text`, parsed from $/%/K/M-formatted substrings."""
    found = []
    for match in _NUMBER_RE.finditer(text):
        value = _parse_number(match.group())
        if value is not None:
            found.append(value)
    return found


def grounded_values(decision: Decision, estimate: EffectEstimate, report: RefutationReport) -> set[float]:
    """Every number legitimately derivable from the analysis output — the only
    numbers a memo is allowed to state (within rounding tolerance)."""
    values: set[float] = set()

    for v in (
        estimate.cum_effect, estimate.cum_lo, estimate.cum_hi,
        estimate.pct_effect, estimate.pct_lo, estimate.pct_hi,
    ):
        values.add(round(v, 2))
        values.add(round(v))
        values.add(round(v / 1000, 1))  # thousands, e.g. "$170.8K"
        values.add(round(v / 1_000_000, 2))  # millions

    for lag in (estimate.lag_50, estimate.lag_90):
        if lag is not None:
            values.add(float(lag))

    for row in estimate.path:
        for key in ("est", "lo", "hi"):
            v = row.get(key)
            if v is not None:
                values.add(round(v))
                values.add(round(v / 1000, 1))

    values.add(float(len(decision.treated_units)))

    for check in report.checks:
        if check.statistic is not None:
            values.add(round(check.statistic, 3))
            values.add(round(check.statistic * 100, 1))  # e.g. a p-value stated as a percent
        if check.threshold is not None:
            values.add(round(check.threshold, 3))
            values.add(round(check.threshold * 100, 1))

    return values


def check_grounding(
    text: str, allowed: set[float], rel_tol: float = 0.01, abs_tol: float = 1.0
) -> tuple[bool, list[float]]:
    """Returns (is_grounded, [numbers in `text` that don't match any allowed value])."""
    ungrounded = []
    for value in extract_numbers(text):
        if abs(value) <= SMALL_NUMBER_EXEMPTION:
            continue
        if value in STANDARD_ALLOWED:
            continue
        if any(abs(value - a) <= max(abs_tol, rel_tol * max(abs(a), 1)) for a in allowed):
            continue
        ungrounded.append(value)
    return len(ungrounded) == 0, ungrounded
