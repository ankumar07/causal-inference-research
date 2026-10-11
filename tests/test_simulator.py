import numpy as np
import pytest

from decision_proof.simulate.scenarios import SCENARIOS, generate_scenario


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_scenario_generates(name):
    panel, decision, truth = generate_scenario(name, seed=42)

    assert not panel.df.isna().any().any()
    assert (panel.df["revenue"] > 0).all()
    assert (panel.df["transactions"] > 0).all()
    assert set(truth.treated_units) == set(decision.treated_units)
    assert len(truth.true_effect_path) > 0

    for unit_id, group in panel.df.groupby("unit_id"):
        periods = group["period"].to_numpy()
        assert np.array_equal(periods, np.sort(periods))
        assert len(set(periods)) == len(periods)


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_scenario_reproducible(name):
    panel_a, _, truth_a = generate_scenario(name, seed=42)
    panel_b, _, truth_b = generate_scenario(name, seed=42)

    pd_assert_frame_equal(panel_a.df, panel_b.df)
    assert truth_a.true_effect_path == truth_b.true_effect_path
    assert truth_a.true_cum_effect == truth_b.true_cum_effect


def pd_assert_frame_equal(a, b):
    import pandas.testing as pdt

    pdt.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))


def test_s1_true_effect_is_positive_and_ramps():
    _, _, truth = generate_scenario("S1", seed=42)

    assert truth.true_cum_effect > 0
    assert truth.true_pct_effect > 0
    # ramp: effect in week 1 post-treatment should be smaller than once the ramp (4 weeks) completes
    assert truth.true_effect_path[0] < truth.true_effect_path[5]


def test_s2_is_a_truncated_s1():
    panel_s1, decision_s1, truth_s1 = generate_scenario("S1", seed=42)
    panel_s2, decision_s2, truth_s2 = generate_scenario("S2", seed=42)

    assert decision_s1.treated_units == decision_s2.treated_units
    assert truth_s1.true_cum_effect == truth_s2.true_cum_effect
    assert panel_s2.df["period"].min() > panel_s1.df["period"].min()
    assert panel_s2.df["period"].max() == panel_s1.df["period"].max()


def test_s3_has_zero_true_effect():
    _, _, truth = generate_scenario("S3", seed=42)

    assert truth.true_cum_effect == 0
    assert all(e == 0 for e in truth.true_effect_path)


def test_s4_spillover_contaminates_neighbor():
    from decision_proof.simulate.dgp import make_units, neighbor_of, simulate_baseline

    panel, decision, truth = generate_scenario("S4", seed=42)
    units = make_units(seed=42)
    baseline = simulate_baseline(units, len(panel.df["period"].unique()), seed=42)

    treated_unit = decision.treated_units[0]
    neighbor = neighbor_of(treated_unit)
    assert neighbor not in decision.treated_units

    post_mask = panel.df["period"] >= decision.start_period
    observed_neighbor_tx = panel.df[(panel.df["unit_id"] == neighbor) & post_mask]["transactions"].sum()
    baseline_neighbor_tx = baseline[(baseline["unit_id"] == neighbor) & (baseline["period"] >= decision.start_period)][
        "transactions"
    ].sum()

    assert observed_neighbor_tx > baseline_neighbor_tx


def test_s5_staggered_starts_differ():
    panel, decision, truth = generate_scenario("S5", seed=42)

    assert len(decision.treated_units) > 1
    # at least two distinct true-effect contributions should turn on at different periods
    assert truth.true_effect_path[0] != truth.true_effect_path[-1]
