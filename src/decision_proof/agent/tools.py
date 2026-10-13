"""Tools Claude calls to explain results. Claude never produces an estimate, a
refutation verdict, or an experiment design itself — it only reads what these
tools return and writes about it. Each tool takes only JSON-friendly arguments
(e.g. a decision_id), matching what a real tool-use call can pass; the actual
Panel/Decision objects live in a registry this class holds, not in the LLM's
context.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from decision_proof.contracts import Decision, Panel
from decision_proof.estimators.did import DiDEventStudy
from decision_proof.estimators.synth import SyntheticControl
from decision_proof.ledger.store import Ledger
from decision_proof.refute.verdict import refute

ESTIMATORS = {
    "did_event_study": DiDEventStudy,
    "synthetic_control": SyntheticControl,
}

Z_90_TWO_SIDED = 1.6448536269514722
Z_80_POWER = stats.norm.ppf(0.80)

TOOL_NAMES = {"list_decisions", "run_analysis", "get_refutation_report", "design_experiment"}


@dataclass
class AnalystTools:
    ledger: Ledger
    panels: dict[str, tuple[Panel, Decision]] = field(default_factory=dict)

    def register(self, panel: Panel, decision: Decision) -> None:
        self.panels[decision.decision_id] = (panel, decision)

    def _lookup(self, decision_id: str) -> tuple[Panel, Decision]:
        if decision_id not in self.panels:
            raise ValueError(f"no panel registered for decision_id {decision_id!r}")
        return self.panels[decision_id]

    def list_decisions(self, status: str | None = None) -> list[dict]:
        df = self.ledger.list_decisions(status=status)
        return df.to_dict(orient="records")

    def run_analysis(self, decision_id: str, methods: list[str] | None = None) -> dict:
        panel, decision = self._lookup(decision_id)
        methods = methods or list(ESTIMATORS)
        estimates = {m: ESTIMATORS[m]().fit(panel, decision) for m in methods}
        report = refute(panel, decision, estimates)
        return {
            "decision_id": decision_id,
            "estimates": {m: e.model_dump() for m, e in estimates.items()},
            "refutation_report": report.model_dump(),
        }

    def get_refutation_report(self, decision_id: str) -> dict:
        return self.run_analysis(decision_id)["refutation_report"]

    def design_experiment(self, decision_id: str, candidate_effect_pct: float, weeks: int) -> dict:
        """Minimum detectable effect at 80% power for `weeks` of data, estimated from
        the decision's own pre-period noise. A lightweight analytical approximation —
        not the full power-by-simulation design in M6 (not yet built); flags that
        explicitly in the result rather than implying more rigor than it has.
        """
        panel, decision = self._lookup(decision_id)
        df = panel.df
        pre = df[(df["unit_id"].isin(decision.treated_units)) & (df["period"] < decision.start_period)]
        if pre.empty:
            raise ValueError(f"no pre-period history available for decision_id {decision_id!r}")

        weekly_std = pre.groupby("unit_id")[panel.outcome].std().mean()
        n_treated = len(decision.treated_units)
        baseline_weekly = pre.groupby("unit_id")[panel.outcome].mean().sum()

        agg_std = weekly_std * np.sqrt(n_treated * weeks)
        mde_cum = (Z_90_TWO_SIDED + Z_80_POWER) * agg_std
        mde_pct = mde_cum / (baseline_weekly * weeks) * 100 if baseline_weekly else None

        candidate_cum = candidate_effect_pct / 100 * baseline_weekly * weeks
        detectable = abs(candidate_cum) >= mde_cum

        return {
            "decision_id": decision_id,
            "weeks": weeks,
            "n_treated_units": n_treated,
            "minimum_detectable_effect_dollars": float(mde_cum),
            "minimum_detectable_effect_pct": float(mde_pct) if mde_pct is not None else None,
            "candidate_effect_pct": candidate_effect_pct,
            "candidate_likely_detectable": bool(detectable),
            "method": "analytical approximation from pre-period noise, not simulation-based",
        }

    def tool_schemas(self) -> list[dict]:
        return [
            {
                "name": "list_decisions",
                "description": "List decisions in the ledger, optionally filtered by status ('open' or 'graded').",
                "input_schema": {
                    "type": "object",
                    "properties": {"status": {"type": "string", "enum": ["open", "graded"]}},
                },
            },
            {
                "name": "run_analysis",
                "description": "Run the estimators and refutation suite for a decision and return the full results.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "decision_id": {"type": "string"},
                        "methods": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["decision_id"],
                },
            },
            {
                "name": "get_refutation_report",
                "description": "Get the refutation report (checks and verdict) for a decision.",
                "input_schema": {
                    "type": "object",
                    "properties": {"decision_id": {"type": "string"}},
                    "required": ["decision_id"],
                },
            },
            {
                "name": "design_experiment",
                "description": "Minimum detectable effect at 80% power for a candidate decision, given a number of weeks.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "decision_id": {"type": "string"},
                        "candidate_effect_pct": {"type": "number"},
                        "weeks": {"type": "integer"},
                    },
                    "required": ["decision_id", "candidate_effect_pct", "weeks"],
                },
            },
        ]

    def call(self, name: str, tool_input: dict):
        if name not in TOOL_NAMES:
            raise ValueError(f"unknown tool {name!r}")
        return getattr(self, name)(**tool_input)
