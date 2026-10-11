import pytest

from decision_proof.contracts import Panel
from decision_proof.estimators.synth import SyntheticControl
from decision_proof.ledger.grading import grade_decision, interval_score
from decision_proof.ledger.store import Ledger
from decision_proof.refute.verdict import refute
from decision_proof.simulate.scenarios import generate_scenario


@pytest.fixture
def s1():
    return generate_scenario("S1", seed=42)


def _early_panel(panel, decision, post_weeks=8):
    df = panel.df[panel.df["period"] <= decision.start_period + post_weeks - 1]
    return Panel(df=df, outcome=panel.outcome, period_freq=panel.period_freq)


def test_log_decision_then_grade_end_to_end(s1):
    panel, decision, truth = s1
    early_panel = _early_panel(panel, decision)
    estimator = SyntheticControl()
    early_estimate = estimator.fit(early_panel, decision)
    report = refute(early_panel, decision, {"synthetic_control": early_estimate})

    ledger = Ledger(":memory:")
    ledger.log_decision(decision, early_estimate, report)

    open_decisions = ledger.list_decisions(status="open")
    assert len(open_decisions) == 1
    assert open_decisions.iloc[0]["status"] == "open"
    assert open_decisions.iloc[0]["decision_id"] == decision.decision_id

    result = grade_decision(ledger, decision.decision_id, panel, decision, estimator)

    assert result["decision_id"] == decision.decision_id
    assert isinstance(result["inside_range"], bool)
    assert result["interval_score"] >= 0

    graded = ledger.get_decision(decision.decision_id)
    assert graded["status"] == "graded"
    assert graded["realized_cum_effect"] == result["realized_cum_effect"]
    assert len(ledger.list_decisions(status="open")) == 0


def test_log_decision_rejects_duplicate_id(s1):
    panel, decision, truth = s1
    estimator = SyntheticControl()
    estimate = estimator.fit(panel, decision)
    report = refute(panel, decision, {"synthetic_control": estimate})

    ledger = Ledger(":memory:")
    ledger.log_decision(decision, estimate, report)
    with pytest.raises(ValueError):
        ledger.log_decision(decision, estimate, report)


def test_grade_decision_rejects_unknown_id(s1):
    panel, decision, truth = s1
    estimator = SyntheticControl()
    ledger = Ledger(":memory:")
    with pytest.raises(ValueError):
        grade_decision(ledger, "nonexistent", panel, decision, estimator)


def test_grade_decision_rejects_already_graded(s1):
    panel, decision, truth = s1
    estimator = SyntheticControl()
    early_panel = _early_panel(panel, decision)
    early_estimate = estimator.fit(early_panel, decision)
    report = refute(early_panel, decision, {"synthetic_control": early_estimate})

    ledger = Ledger(":memory:")
    ledger.log_decision(decision, early_estimate, report)
    grade_decision(ledger, decision.decision_id, panel, decision, estimator)

    with pytest.raises(ValueError):
        grade_decision(ledger, decision.decision_id, panel, decision, estimator)


def test_track_record_running_share_inside(s1):
    panel, decision, truth = s1
    estimator = SyntheticControl()
    ledger = Ledger(":memory:")

    for i, name in enumerate(["S1", "S3"]):
        p, d, t = generate_scenario(name, seed=42)
        d = d.model_copy(update={"decision_id": f"{name.lower()}-{i}"})
        est = estimator.fit(p, d)
        rep = refute(p, d, {"synthetic_control": est})
        ledger.log_decision(d, est, rep)
        grade_decision(ledger, d.decision_id, p, d, estimator)

    track = ledger.track_record()
    assert len(track) == 2
    assert track["running_share_inside"].iloc[-1] == track["inside_range"].mean()


def test_interval_score_zero_width_penalizes_miss():
    assert interval_score(lo=0, hi=10, realized=5) == 10  # inside: just the width
    missed = interval_score(lo=0, hi=10, realized=20, alpha=0.10)
    assert missed > 10  # outside: width plus a penalty
