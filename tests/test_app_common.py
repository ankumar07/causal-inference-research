import decision_proof.app.common as common
from decision_proof.simulate.scenarios import SCENARIOS


def _reset(monkeypatch, tmp_path):
    monkeypatch.setattr(common, "_TOOLS", None)
    monkeypatch.setattr(common, "DB_PATH", str(tmp_path / "ledger.duckdb"))


def test_get_tools_returns_the_same_instance(monkeypatch, tmp_path):
    _reset(monkeypatch, tmp_path)
    a = common.get_tools()
    b = common.get_tools()
    assert a is b


def test_ensure_demo_seeded_logs_all_scenarios_with_distinct_ids(monkeypatch, tmp_path):
    _reset(monkeypatch, tmp_path)
    tools = common.get_tools()
    common.ensure_demo_seeded(tools)

    df = tools.ledger.list_decisions()
    ids = set(df["decision_id"])
    expected = {name.lower() for name in SCENARIOS} | {"s1-early-example"}
    assert ids == expected

    for name in SCENARIOS:
        assert name.lower() in tools.panels


def test_ensure_demo_seeded_is_idempotent(monkeypatch, tmp_path):
    _reset(monkeypatch, tmp_path)
    tools = common.get_tools()
    common.ensure_demo_seeded(tools)
    common.ensure_demo_seeded(tools)  # should not raise or duplicate

    df = tools.ledger.list_decisions()
    assert len(df) == len(set(df["decision_id"]))


def test_s1_and_s2_get_distinct_panels_despite_shared_underlying_decision_id(monkeypatch, tmp_path):
    """Regression test: scenario_s2() deliberately reuses S1's exact Decision object
    (same decision_id) to represent "the same decision, less history" - the app layer
    must rename it to its own id, or S2 silently collides with S1 and never appears.
    """
    _reset(monkeypatch, tmp_path)
    tools = common.get_tools()
    common.ensure_demo_seeded(tools)

    s1_panel, s1_decision = tools.panels["s1"]
    s2_panel, s2_decision = tools.panels["s2"]

    assert s1_decision.decision_id == "s1"
    assert s2_decision.decision_id == "s2"
    assert len(s2_panel.df["period"].unique()) < len(s1_panel.df["period"].unique())


def test_analyze_decision_returns_both_estimators_and_registers_panel(monkeypatch, tmp_path):
    _reset(monkeypatch, tmp_path)
    tools = common.get_tools()
    from decision_proof.simulate.scenarios import generate_scenario

    panel, decision, truth = generate_scenario("S1", seed=42)
    estimates, report = common.analyze_decision(tools, panel, decision)

    assert set(estimates) == {"did_event_study", "synthetic_control"}
    assert report.verdict in ("BOARD_READY", "DIRECTIONAL", "NOT_PROVEN")
    assert decision.decision_id in tools.panels
