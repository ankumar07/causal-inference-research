"""Shared app state: one Ledger and one AnalystTools instance *per server process*,
not per browser session — DuckDB's file-based connection holds an exclusive write
lock, so two sessions each opening their own connection to the same file race and
fail (confirmed while testing this page: a second session's seed attempt hit
"decision_id 's1' is already logged" because it ran concurrently with the first
session's own seeding).

st.cache_resource looked like the standard fix, but its own deduplication turned out
to be racy under this app's load: two sessions requesting the cached object before
the first construction finished each got their own Ledger AND their own Lock
instance (also cached, also stomped), so the lock briefly didn't serialize anything
— and with estimation genuinely running on two threads at once, S2 silently vanished
from the seeded demo entirely (its numba-jitted wild-bootstrap step is not
thread-safe for concurrent first compiles). Switched to a plain module-level
singleton instead: Python's import system itself only ever executes a module's
top-level code once, so `_LOCK` below is guaranteed to be the one and only Lock
instance in this process from the moment the module is first imported — no
framework-level caching involved, nothing left to race.
"""

from __future__ import annotations

import threading

from decision_proof.agent.tools import AnalystTools
from decision_proof.contracts import Decision, EffectEstimate, Panel, RefutationReport
from decision_proof.estimators.did import DiDEventStudy
from decision_proof.estimators.synth import SyntheticControl
from decision_proof.ledger.store import Ledger
from decision_proof.refute.verdict import refute
from decision_proof.simulate.scenarios import SCENARIOS, generate_scenario

DB_PATH = "data/ledger.duckdb"
DEFAULT_SEED = 42

ESTIMATOR_FACTORIES = {
    "did_event_study": DiDEventStudy,
    "synthetic_control": SyntheticControl,
}

_LOCK = threading.Lock()
_TOOLS: AnalystTools | None = None


def _lock() -> threading.Lock:
    return _LOCK


def get_tools() -> AnalystTools:
    global _TOOLS
    if _TOOLS is None:
        with _LOCK:
            if _TOOLS is None:
                _TOOLS = AnalystTools(ledger=Ledger(DB_PATH))
    return _TOOLS


def analyze_decision(
    tools: AnalystTools, panel: Panel, decision: Decision
) -> tuple[dict[str, EffectEstimate], RefutationReport]:
    """Run both estimators + refutation, register the panel for later pages."""
    estimates = {name: cls().fit(panel, decision) for name, cls in ESTIMATOR_FACTORIES.items()}
    report = refute(panel, decision, estimates)
    with _lock():
        tools.register(panel, decision)
    return estimates, report


def ensure_demo_seeded(tools: AnalystTools) -> None:
    """Idempotent: safe to call on every page load, from any session, any number of
    times concurrently. Registers panels (needed every session, even if the DuckDB
    file already has the rows from an earlier one) and logs each demo decision.

    The lock plus a pre-check narrows the race, but isn't airtight on its own: an
    st.cache_resource factory can still run more than once if two sessions request it
    before the first call finishes constructing and caching the object (observed while
    testing this page — two such calls each got their own Ledger/Lock instance, so the
    lock briefly didn't actually serialize them). Rather than fight that startup
    window, log_decision's own duplicate-id error is caught and treated as "someone
    else already seeded this" — correct regardless of why two inserts raced.
    """
    with _lock():
        existing = tools.ledger.list_decisions()
        existing_ids = set(existing["decision_id"]) if not existing.empty else set()

        for name in SCENARIOS:
            panel, decision, truth = generate_scenario(name, seed=DEFAULT_SEED)
            # S2 deliberately reuses S1's exact Decision (same decision_id) to represent
            # "the same decision, viewed with less history" (see simulate/scenarios.py) -
            # rename to the scenario's own id here so each scenario gets its own
            # navigable ledger row and panel registration, instead of S2 silently
            # colliding with (and getting skipped as a duplicate of) S1.
            app_id = name.lower()
            if decision.decision_id != app_id:
                decision = decision.model_copy(update={"decision_id": app_id})
            tools.register(panel, decision)
            if decision.decision_id in existing_ids:
                continue
            estimates = {n: cls().fit(panel, decision) for n, cls in ESTIMATOR_FACTORIES.items()}
            report = refute(panel, decision, estimates)
            try:
                tools.ledger.log_decision(decision, estimates["synthetic_control"], report)
            except ValueError as e:
                if "already logged" not in str(e):
                    raise  # a genuine bug, not the expected race - never swallow this

        # One decision taken through the full open -> graded lifecycle, so the Lab
        # page's track record has something to show out of the box: logged from an
        # early, partial view of S1 (as an analyst would have at the time), then
        # graded against the full panel (as if enough time had since passed).
        demo_id = "s1-early-example"
        if demo_id not in existing_ids:
            full_panel, decision, truth = generate_scenario("S1", seed=DEFAULT_SEED)
            decision = decision.model_copy(
                update={
                    "decision_id": demo_id,
                    "description": decision.description + " (logged early, 8 weeks post)",
                }
            )
            tools.register(full_panel, decision)
            try:
                early_df = full_panel.df[full_panel.df["period"] <= decision.start_period + 7]
                early_panel = Panel(df=early_df, outcome=full_panel.outcome, period_freq=full_panel.period_freq)
                early_estimate = SyntheticControl().fit(early_panel, decision)
                early_report = refute(early_panel, decision, {"synthetic_control": early_estimate})
                tools.ledger.log_decision(decision, early_estimate, early_report)

                from decision_proof.ledger.grading import grade_decision

                grade_decision(tools.ledger, demo_id, full_panel, decision, SyntheticControl())
            except ValueError as e:
                if "already logged" not in str(e) and "already graded" not in str(e):
                    raise  # a genuine bug, not the expected race - never swallow this
