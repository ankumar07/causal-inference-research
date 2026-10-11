"""Conservation-constrained synthetic control — the original-methods contribution.

Plain synthetic control fits donor weights on clean pre-period data, so spillover
(which only shows up post-treatment, once a decision is live) doesn't contaminate
the *fitting* step. The bias instead comes from what happens *after* fitting: if a
donor that happens to pick up redirected demand post-treatment was given nonzero
weight, the synthetic counterfactual itself inflates right when the real effect is
supposed to show up, shrinking (treated - synthetic) and understating the effect.

This matters because demand within a shared market is roughly conserved: a price
rise can shift transactions to the nearest shop-cluster rather than destroying
them, so a demand-conserving neighbor's post-period values can't be trusted as
independent, untreated signal.

First attempt (not kept): exclude known-adjacent donors from the pool entirely.
Tested via the calibration lab on S4 (150 reps) and rejected — it didn't reduce
bias, and sometimes made it worse (conservation_constrained_sc bias +$31.2k vs
plain SC's -$3.0k). Reasoning in hindsight: only ~4 of 20 donors are even
potentially contaminated, and the contamination is often small relative to each
donor's own noise, so losing 4 reasonably-fitting donors costs more in donor-pool
diversity than it saves in avoided contamination.

Second attempt (kept): decontaminate instead of excluding. Keep the neighbor in
the pool — its pre-period values are never contaminated in this simulator (lost
transactions are zero before the decision starts) — but before using it to build
the treated unit's synthetic counterfactual, replace its *post*-period values with
its own synthetic control estimate, built from a donor pool that excludes both the
treated units and the neighbor itself. This keeps the neighbor's genuine pre-period
fit information instead of discarding it, and only launders the specific post-
period values that conservation theory says can't be trusted.

Implemented as a panel transform: decontaminate the known-adjacent donors' observed
values, then run the ordinary SyntheticControl.fit() on the corrected panel — which
also means placebo-in-space is computed on the same decontaminated data, for free.
"""

from __future__ import annotations

from dataclasses import dataclass

from decision_proof.contracts import Decision, Panel
from decision_proof.estimators.synth import SyntheticControl, _pivot, _unit_effect
from decision_proof.simulate.dgp import neighbor_of


def ring_neighbor_exclusion(units: list[str]) -> set[str]:
    """Demand-conserving neighbors of `units`, per the simulator's ring topology."""
    return {neighbor_of(u) for u in units}


def decontaminate_panel(panel: Panel, decision: Decision, conservation_group) -> Panel:
    """Replace each demand-conserving neighbor's post-period values with its own
    synthetic control estimate (fit from donors outside the conservation group)."""
    outcome = panel.outcome
    df = panel.df
    treated_units = list(decision.treated_units)
    start = decision.start_period
    all_units = df["unit_id"].unique().tolist()

    neighbors = conservation_group(treated_units) & set(all_units) - set(treated_units)
    if not neighbors:
        return panel

    wide = _pivot(df, outcome)
    periods = wide.index.to_numpy()
    pre_mask = periods < start
    post_mask = ~pre_mask
    clean_pool = [u for u in all_units if u not in treated_units and u not in neighbors]

    corrected = wide.copy()
    for donor in neighbors:
        effect, _ = _unit_effect(wide, donor, clean_pool, pre_mask)
        decontaminated = wide[donor].to_numpy() - effect  # donor's own clean-donor synthetic estimate
        corrected.loc[wide.index[post_mask], donor] = decontaminated[post_mask]

    other_cols = [c for c in df.columns if c not in ("unit_id", "period", outcome)]
    long = corrected.reset_index().melt(id_vars="period", var_name="unit_id", value_name=outcome)
    merged = long.merge(df[["unit_id", "period"] + other_cols], on=["unit_id", "period"], how="left")
    return Panel(df=merged, outcome=outcome, period_freq=panel.period_freq)


@dataclass
class ConservationConstrainedSyntheticControl(SyntheticControl):
    name: str = "conservation_constrained_sc"

    def fit(self, panel: Panel, decision: Decision):
        decontaminated = decontaminate_panel(panel, decision, ring_neighbor_exclusion)
        return super().fit(decontaminated, decision)
