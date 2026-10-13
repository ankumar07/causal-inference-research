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

## Conservation-constrained synthetic control (B5)

**The mechanism it targets.** In the simulator, spillover only shows up *after* a
decision starts — lost transactions at a treated metro are zero before treatment, so
a donor's pre-period values are never contaminated. Plain synthetic control fits
weights on clean pre-period data, so the fitting step itself isn't biased. The bias
comes from what happens next: if a donor that goes on to absorb redirected demand was
given nonzero weight, the *synthetic counterfactual* inflates right when the real
effect is supposed to show up, shrinking (treated − synthetic) and understating the
effect. Demand within a shared market is roughly conserved — a price rise shifts
transactions to the nearest shop-cluster rather than destroying them — so a
demand-conserving neighbor's post-period values can't be trusted as independent,
untreated signal.

**First attempt (rejected): exclude known-adjacent donors from the pool entirely.**
Tested via the calibration lab on S4 at its original, gentler parameters
(effect +7%, elasticity −0.6, spillover 0.7; n=150): plain SC bias was already
−$3.0k (essentially unbiased), and exclusion made it *worse*: +$31.2k. Losing 4 of
20 donors costs more in donor-pool diversity than it saves in avoided contamination,
especially when the contamination at hand is modest.

**Second attempt (kept): decontaminate instead of excluding.** Keep the neighbor in
the pool — its pre-period values are clean — but before using it to build the
treated unit's synthetic counterfactual, replace its *post*-period values with its
own synthetic control estimate, fit from a donor pool that excludes both the treated
units and the neighbor itself. This preserves the neighbor's genuine pre-period fit
information and only launders the specific values conservation theory says can't be
trusted. Implemented as a panel transform (`synth_conservation.decontaminate_panel`)
feeding into the ordinary `SyntheticControl.fit()` — which means placebo-in-space is
computed on the same decontaminated data too, for free.

**Why S4's parameters were recalibrated.** At the original gentler parameters, there
wasn't enough contamination for the fix to have anything meaningful to correct — and
the decontamination step has its own cost: replacing a donor's values with a model
estimate introduces that estimate's own error, which (per-seed) is roughly a
*fixed, scenario-independent* noise term — it depends on the random donor data, not
on the true effect size. So decontamination is a net win only once the contamination
it's correcting for is large relative to that fixed cost, and a net loss otherwise.
S4 was recalibrated (effect +10%, elasticity −1.5, spillover 1.0) specifically to be
a regime where contamination dominates — which also means demand turns out more
elastic than expected and the decision is net revenue-*negative* on average, not a
clean win like S1. That's an intentional, realistic complication, not a mistake: a
board needs a system that catches a backfiring price increase, not just one that
confirms decisions that worked.

**Result** (N=200, seeds 4000-4199): plain SC mean absolute error $150.2k vs
conservation-constrained $118.5k — a ~21% reduction. Plain SC's bias is strongly
negative (−$116.0k, consistent with the "contaminated donor inflates the synthetic,
shrinking the estimated effect" mechanism); the conservation-constrained method's
bias is smaller in magnitude but overshoots in the other direction (+$16.3k) rather
than landing at zero — it trades one bias for a smaller one, not a clean fix. Interval
coverage also drops (88.5% → 72.5%): the decontamination step adds its own estimation
uncertainty that the current permutation interval doesn't fully account for, worth
revisiting if this becomes a formally reported production interval rather than a demo.

## Staggered-adoption DiD (Q5): a correction that didn't beat the thing it corrects

Built a simplified Callaway & Sant'Anna (2021)-style estimator (`estimators/staggered_did.py`)
to fix a real, textbook problem: naive two-way-fixed-effects DiD (`estimators/did.py`) is
biased under staggered treatment timing when the effect size varies by cohort, because it
implicitly uses already-treated units as part of the comparison for later-treated ones.

`simulate.scenarios.scenario_s5` (the original staggered scenario) has a *homogeneous*
effect across all 6 waves, which gives naive TWFE nothing to be biased by — confirmed that
comparison is a wash by construction. Added `scenario_s5_heterogeneous`, where effect size
grows ~0x to ~2.2x across the waves, to actually introduce the bias mechanism.

Result, on the most reliable read available (120 seeds, parallelized): naive TWFE's mean
absolute bias ($145.3k) is *better* than the staggered estimator's ($154.8k) — the fix
didn't pay off here. Two smaller, earlier samples (40 and 60 seeds) had each told a
different, more flattering story in both directions before the larger run settled it — a
reminder that n=30-60 isn't enough to trust a comparison like this one way or the other, and
a caution against reporting an early favorable-looking sample as the result.

Best current read on why: with only 6 treated units (each its own cohort of one), the
estimator's percentile bootstrap over the control pool carries enough of its own variance
to swamp whatever systematic TWFE bias the heterogeneous effects introduce at this scale —
the same bias-variance tradeoff B5's first (rejected) attempt ran into, except here it shows
up in the kept design, not a discarded one. Left in the codebase as a correctly-implemented,
honestly-evaluated estimator rather than pulled or re-tuned to win the comparison it didn't
win (per `CLAUDE.md`: "never tune to look good") — the textbook fix for staggered-adoption
bias is real and worth having built, even though this particular 6-unit demo is too small a
stage for it to pay off. Candidate follow-up, not yet done: more treated units per cohort
(so cohort-level averaging reduces the bootstrap's variance) would be the natural next thing
to try before concluding the method itself doesn't work, rather than just this demo's scale.

## Instrumental variables (Q7): a weak-instrument near-miss, caught before shipping

First calibration of `simulate.iv.generate_iv_data` used `INSTRUMENT_STRENGTH = 0.40` against
a confounding term on price with an implied standard deviation of ~15 — the instrument's
contribution to price's variance was swamped roughly 20-to-1 by the confound. First-stage
F-statistic: **1.3**, far under the Staiger-Stock rule-of-thumb threshold of 10. The 2SLS
estimate still happened to contain the true effect at that calibration, but only because its
confidence interval was so wide from the weak first stage that it would have covered almost
anything — a fragile, not-reproducible "win," not a real one. Recalibrated
(`INSTRUMENT_STRENGTH = 3.0`, confound-on-price reduced, n=500): first-stage F-stat ~826.
Across 60 seeds at the recalibrated settings: naive OLS covers the true effect **0%** of the
time (confidently wrong, not just noisy — mean bias +$528), 2SLS covers it **90%** of the
time, matching the target, with mean bias of only -$12. Caught via the same discipline this
whole suite is built around — compute the diagnostic (first-stage F) before trusting the
headline number, not after.
