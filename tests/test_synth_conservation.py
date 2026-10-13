import numpy as np

from decision_proof.estimators.synth import SyntheticControl
from decision_proof.estimators.synth_conservation import (
    ConservationConstrainedSyntheticControl,
    decontaminate_panel,
    ring_neighbor_exclusion,
)
from decision_proof.simulate.dgp import neighbor_of
from decision_proof.simulate.scenarios import generate_scenario


def _s4_errors(n=40, base_seed=7000):
    plain_errors, cons_errors = [], []
    for i in range(n):
        panel, decision, truth = generate_scenario("S4", seed=base_seed + i)
        plain = SyntheticControl().fit(panel, decision)
        cons = ConservationConstrainedSyntheticControl().fit(panel, decision)
        plain_errors.append(plain.cum_effect - truth.true_cum_effect)
        cons_errors.append(cons.cum_effect - truth.true_cum_effect)
    return np.array(plain_errors), np.array(cons_errors)


def test_s4_conservation_reduces_mean_absolute_bias():
    """Exit test for B5: on S4 (spillover, calibrated so contamination dominates), the
    conservation-constrained method's mean absolute bias beats plain synthetic control's,
    averaged over many replications — not a single-seed fluke in either direction."""
    plain_errors, cons_errors = _s4_errors()

    assert np.abs(cons_errors).mean() < np.abs(plain_errors).mean()


def test_decontaminate_panel_only_changes_post_period_neighbor_values():
    panel, decision, truth = generate_scenario("S4", seed=42)
    corrected = decontaminate_panel(panel, decision, ring_neighbor_exclusion)

    neighbors = ring_neighbor_exclusion(decision.treated_units) - set(decision.treated_units)
    assert neighbors, "S4 should have at least one demand-conserving neighbor to decontaminate"

    orig = panel.df.set_index(["unit_id", "period"])["revenue"]
    corr = corrected.df.set_index(["unit_id", "period"])["revenue"]

    for unit in panel.df["unit_id"].unique():
        pre = orig.loc[unit][orig.loc[unit].index < decision.start_period]
        pre_corr = corr.loc[unit][corr.loc[unit].index < decision.start_period]
        assert np.allclose(pre.to_numpy(), pre_corr.to_numpy()), f"{unit}: pre-period should be untouched"

        post = orig.loc[unit][orig.loc[unit].index >= decision.start_period]
        post_corr = corr.loc[unit][corr.loc[unit].index >= decision.start_period]
        if unit in neighbors:
            assert not np.allclose(post.to_numpy(), post_corr.to_numpy()), f"{unit}: post-period should be decontaminated"
        else:
            assert np.allclose(post.to_numpy(), post_corr.to_numpy()), f"{unit}: non-neighbor post-period should be untouched"


def test_ring_neighbor_exclusion_matches_dgp_topology():
    assert ring_neighbor_exclusion(["metro_00"]) == {neighbor_of("metro_00")}
    assert ring_neighbor_exclusion(["metro_00", "metro_05"]) == {neighbor_of("metro_00"), neighbor_of("metro_05")}
