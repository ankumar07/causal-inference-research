from decision_proof.estimators.staggered_did import StaggeredDiD
from decision_proof.simulate.scenarios import scenario_s5_heterogeneous, scenario_s5_with_starts


def test_staggered_did_runs_on_s5():
    panel, decision, truth, starts = scenario_s5_with_starts(seed=42)
    est = StaggeredDiD().fit(panel, starts)

    assert est.cum_lo <= est.cum_effect <= est.cum_hi
    assert len(est.path) > 0
    assert set(est.diagnostics["cohorts"]) == set(starts)


def test_staggered_did_path_h_zero_is_each_units_own_start():
    panel, decision, truth, starts = scenario_s5_with_starts(seed=42)
    est = StaggeredDiD().fit(panel, starts)

    hs = [p["h"] for p in est.path]
    assert hs[0] == 0
    assert hs == sorted(hs)


def test_staggered_did_runs_on_heterogeneous_effects_and_stays_in_a_sane_range():
    """Not an "our method wins" test - see staggered_did.py's module docstring.
    A 120-seed parallel comparison found naive TWFE's mean absolute bias ($145.3k)
    actually *beats* this estimator's ($154.8k) on the heterogeneous-effect
    scenario built to exercise the bias it's meant to fix - two smaller, earlier
    samples (40 and 60 seeds) had each suggested the opposite before that larger
    run settled it. Reported as-is rather than re-tuned to flip the result (see
    CLAUDE.md). This test only checks the estimator runs correctly and its errors
    stay within a broad sanity bound, not that it wins a comparison it didn't win.
    """
    for seed in range(6000, 6008):
        panel, decision, truth, starts = scenario_s5_heterogeneous(seed=seed)
        stag = StaggeredDiD().fit(panel, starts)
        assert stag.cum_lo <= stag.cum_effect <= stag.cum_hi
        # within an order of magnitude of the true effect - a basic sanity bound,
        # not a precision claim
        assert abs(stag.cum_effect - truth.true_cum_effect) < 10 * abs(truth.true_cum_effect) + 1e6


def test_scenario_s5_heterogeneous_has_cohort_varying_true_effect():
    panel, decision, truth, starts = scenario_s5_heterogeneous(seed=42)
    assert len(starts) == 6
    assert len(set(starts.values())) == 6  # every cohort has a distinct start
