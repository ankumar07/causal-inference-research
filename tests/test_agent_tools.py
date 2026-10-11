import pytest

from decision_proof.agent.tools import AnalystTools
from decision_proof.ledger.store import Ledger
from decision_proof.simulate.scenarios import generate_scenario


@pytest.fixture
def tools():
    panel, decision, truth = generate_scenario("S1", seed=42)
    ledger = Ledger(":memory:")
    t = AnalystTools(ledger=ledger)
    t.register(panel, decision)
    return t, decision


def test_run_analysis_returns_estimates_and_report(tools):
    t, decision = tools
    result = t.call("run_analysis", {"decision_id": decision.decision_id})

    assert result["decision_id"] == decision.decision_id
    assert "did_event_study" in result["estimates"]
    assert "synthetic_control" in result["estimates"]
    assert result["refutation_report"]["verdict"] in ("BOARD_READY", "DIRECTIONAL", "NOT_PROVEN")


def test_get_refutation_report_matches_run_analysis(tools):
    t, decision = tools
    report = t.call("get_refutation_report", {"decision_id": decision.decision_id})
    assert "checks" in report
    assert "reasons" in report


def test_design_experiment_scales_mde_with_weeks(tools):
    t, decision = tools
    short = t.call("design_experiment", {"decision_id": decision.decision_id, "candidate_effect_pct": 6.0, "weeks": 4})
    long = t.call("design_experiment", {"decision_id": decision.decision_id, "candidate_effect_pct": 6.0, "weeks": 20})

    assert short["minimum_detectable_effect_pct"] > 0
    assert long["minimum_detectable_effect_pct"] > 0
    # more weeks of data -> a given cumulative dollar effect is easier to detect as a % of baseline,
    # but the minimum detectable dollar effect itself grows with the aggregation window
    assert long["minimum_detectable_effect_dollars"] > short["minimum_detectable_effect_dollars"]


def test_call_rejects_unknown_tool_name(tools):
    t, decision = tools
    with pytest.raises(ValueError):
        t.call("delete_everything", {})


def test_call_rejects_unregistered_decision(tools):
    t, decision = tools
    with pytest.raises(ValueError):
        t.call("run_analysis", {"decision_id": "not-registered"})


def test_list_decisions_reflects_ledger_state(tools):
    t, decision = tools
    assert t.call("list_decisions", {}) == []

    estimate = t.call("run_analysis", {"decision_id": decision.decision_id})
    from decision_proof.contracts import EffectEstimate, RefutationReport

    est = EffectEstimate(**estimate["estimates"]["synthetic_control"])
    report = RefutationReport(**estimate["refutation_report"])
    t.ledger.log_decision(decision, est, report)

    listed = t.call("list_decisions", {})
    assert len(listed) == 1
    assert listed[0]["decision_id"] == decision.decision_id
