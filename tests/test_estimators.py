import pytest

from decision_proof.estimators.did import DiDEventStudy
from decision_proof.estimators.synth import SyntheticControl
from decision_proof.simulate.scenarios import generate_scenario

ESTIMATORS = [DiDEventStudy(), SyntheticControl()]


def _assert_valid_estimate(est, panel, decision):
    assert est.cum_lo <= est.cum_effect <= est.cum_hi
    assert est.pct_lo <= est.pct_effect <= est.pct_hi
    assert len(est.path) > 0
    post_periods = sorted(panel.df.loc[panel.df["period"] >= decision.start_period, "period"].unique())
    assert [p["h"] for p in est.path] == [p - decision.start_period for p in post_periods]


@pytest.mark.parametrize("estimator", ESTIMATORS, ids=lambda e: e.name)
def test_s1_interval_contains_truth(estimator):
    """Exit test for B2: on S1, each estimator's 90% interval contains the true effect."""
    panel, decision, truth = generate_scenario("S1", seed=42)
    est = estimator.fit(panel, decision)

    _assert_valid_estimate(est, panel, decision)
    assert est.cum_lo <= truth.true_cum_effect <= est.cum_hi


def test_s1_did_event_study_shows_the_ramp():
    panel, decision, truth = generate_scenario("S1", seed=42)
    est = DiDEventStudy().fit(panel, decision)

    by_h = {p["h"]: p["est"] for p in est.path}
    # the effect ramps over the first 4 weeks (TreatmentSpec.ramp_weeks in S1) then holds roughly steady;
    # the ramp should show up as growth over that window, not be flat from week 0.
    assert by_h[0] < max(by_h[h] for h in range(4, 8))


@pytest.mark.parametrize("name", ["S1", "S2", "S3", "S4", "S5"])
@pytest.mark.parametrize("estimator", ESTIMATORS, ids=lambda e: e.name)
def test_estimator_runs_on_every_scenario(estimator, name):
    panel, decision, truth = generate_scenario(name, seed=42)
    est = estimator.fit(panel, decision)
    _assert_valid_estimate(est, panel, decision)


def test_synthetic_control_donor_weights_are_valid():
    panel, decision, truth = generate_scenario("S1", seed=42)
    est = SyntheticControl().fit(panel, decision)

    for unit, weights in est.diagnostics["donor_weights"].items():
        assert unit in decision.treated_units
        assert all(w > 0 for w in weights.values())
        assert all(d not in decision.treated_units for d in weights)
