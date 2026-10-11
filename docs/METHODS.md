# Methods

## Refutation thresholds

All thresholds below are judgment calls (`refute/checks.py::DEFAULT_THRESHOLDS`), not
universal truths, and are overridable per call. B4's calibration lab is where they
get checked against actual false-positive/coverage rates rather than hand-picked.

| Threshold | Default | Used by | Meaning |
|---|---|---|---|
| `pre_trend_pvalue` | 0.10 | pre_trend_test (critical) | FAIL if the joint test on pre-period DiD leads has p < this (a real pre-trend exists) |
| `sc_pre_fit_rmspe_rel` | 0.15 | sc_pre_fit_quality (critical) | FAIL if the worst treated unit's pre-period RMSPE, relative to its own pre-period mean, exceeds this |
| `placebo_pvalue` | 0.10 | placebo_in_space, placebo_in_time (both critical) | placebo_in_space: FAIL if the Fisher-combined p-value across treated units exceeds this. placebo_in_time: also used as the synthetic control's own CI alpha for the pseudo-decision check |
| `placebo_in_time_min_pre_periods` | 6 | placebo_in_time (critical) | if fewer than this many periods exist on either side of the pseudo-decision point, skip the check (WARN) rather than fail or fabricate a result |
| `leave_one_out_pct_move` | 0.50 | leave_one_out_donors (non-critical) | WARN if dropping any single donor flips the effect's sign or moves it more than this fraction |
| `pre_window_pct_move` | 0.50 | pre_window_sensitivity (non-critical) | WARN if halving the pre-period window moves the effect more than this fraction |

## Combining placebo-in-space across multiple treated units

With `n` treated units, each gets its own placebo rank p-value from the donor pool. A
Bonferroni correction (`min(p_i) * n`) was tried first and rejected: with ~20 donors,
the smallest *achievable* single-unit p-value is already `1/21 ≈ 0.048`, so Bonferroni
with `n=4` treated units blows past a 0.10 threshold even when every treated unit is
maximally extreme — the check becomes nearly impossible to pass regardless of signal
strength. Combining via **Fisher's method** (`-2 * sum(ln(p_i)) ~ chi2(2n)`) instead
gives a standard, less conservative joint test for "the treated units collectively
show a placebo-rare pattern," which is the question we actually want answered.

## Known finding: S2 (short history) lands NOT_PROVEN, not DIRECTIONAL

The original build plan scripted S2 (same decision as S1, only 12 weeks of
pre-period history) as degrading to a DIRECTIONAL verdict. What the suite actually
finds, as of B3, is NOT_PROVEN — specifically, `placebo_in_space` fails.

Why: with 20 donors and only 12 pre-period observations, the synthetic control's
donor-weight optimization is underdetermined for *every* unit, including placebo
donors — not just the treated one. Every unit, real or placebo, gets a near-perfect
pre-period fit, which erases the discriminating signal placebo-in-space depends on
(the treated unit is supposed to look unusually well-explained *relative to* the
placebo distribution; when everything fits perfectly, nothing stands out).

This is left as an honest result rather than patched by loosening a threshold or
changing the scenario's 12-week parameter to hit the scripted outcome — see
`CLAUDE.md`: "Report calibration numbers as they come out; never tune to look good."
Candidate real fixes (not yet done): cap the donor pool size relative to the
pre-period length for short-history decisions, or revisit this during B4's
calibration pass, where coverage and false-positive rate get checked by simulation
rather than by eyeballing one seed.
