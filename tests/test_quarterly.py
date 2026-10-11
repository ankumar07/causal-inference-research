import numpy as np

from decision_proof.simulate.quarterly import (
    AGE_BANDS,
    EXPERIMENT_QUARTERS,
    INCOME_TIERS,
    N_METROS,
    PRICE_POINTS,
    build_experiment,
    experiments_for_quarter,
    generate_quarterly_world,
    plan_quarter_experiments,
    quarter_start_period,
)


def test_generate_quarterly_world_is_reproducible():
    baseline_a, demo_a, plans_a = generate_quarterly_world(seed=42)
    baseline_b, demo_b, plans_b = generate_quarterly_world(seed=42)

    import pandas.testing as pdt

    pdt.assert_frame_equal(baseline_a, baseline_b)
    assert demo_a == demo_b
    assert plans_a == plans_b


def test_every_metro_has_a_demographic_profile():
    _, demographics, _ = generate_quarterly_world(seed=42)
    assert len(demographics) == N_METROS
    for profile in demographics.values():
        assert profile["age_band"] in AGE_BANDS
        assert profile["income_tier"] in INCOME_TIERS


def test_plan_has_four_price_points_per_experimental_quarter():
    plans = plan_quarter_experiments()
    for q in EXPERIMENT_QUARTERS:
        q_plans = [p for p in plans if p.quarter == q]
        assert len(q_plans) == len(PRICE_POINTS)
        assert sorted(p.price_change_pct for p in q_plans) == sorted(PRICE_POINTS)
        assert all(p.start_period == quarter_start_period(q) for p in q_plans)


def test_treated_groups_are_disjoint_within_a_quarter():
    plans = plan_quarter_experiments()
    for q in EXPERIMENT_QUARTERS:
        q_plans = [p for p in plans if p.quarter == q]
        all_units = [u for p in q_plans for u in p.treated_units]
        assert len(all_units) == len(set(all_units))


def test_build_experiment_excludes_concurrent_quarter_experiments():
    baseline, demographics, plans = generate_quarterly_world(seed=42)
    q3_plans = [p for p in plans if p.quarter == 3]
    panel, decision, truth = build_experiment(baseline, q3_plans[0], plans)

    other_treated = {u for p in q3_plans[1:] for u in p.treated_units}
    panel_units = set(panel.df["unit_id"].unique())
    assert not (other_treated & panel_units)
    assert set(decision.treated_units).issubset(panel_units)


def test_true_effect_scales_with_price_point():
    """Higher price-change experiments should produce a larger true cumulative
    effect, within the same quarter - the whole premise of fitting a sensitivity
    curve across them."""
    baseline, demographics, plans = generate_quarterly_world(seed=42)
    exps = experiments_for_quarter(baseline, plans, 3)
    exps_sorted = sorted(exps, key=lambda e: e[1].decision_id)  # p3, p6, p9, p12 lexically sorted by price

    pct_effects = []
    for panel, decision, truth in exps:
        price_pct = float(decision.description.split("+")[1].split("%")[0]) / 100
        pct_effects.append((price_pct, truth.true_pct_effect))
    pct_effects.sort()

    effects_only = [e for _, e in pct_effects]
    assert effects_only == sorted(effects_only)  # monotonically increasing with price point


def test_experiments_for_quarter_returns_all_price_points():
    baseline, demographics, plans = generate_quarterly_world(seed=42)
    for q in EXPERIMENT_QUARTERS:
        exps = experiments_for_quarter(baseline, plans, q)
        assert len(exps) == len(PRICE_POINTS)
        for panel, decision, truth in exps:
            assert not panel.df.isna().any().any()
            assert (panel.df["revenue"] > 0).all()
