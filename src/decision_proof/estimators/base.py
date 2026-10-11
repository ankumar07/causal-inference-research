"""Shared estimator interface. Every estimator exposes `fit(panel, decision) -> EffectEstimate`."""

from __future__ import annotations

from typing import Protocol

from decision_proof.contracts import Decision, EffectEstimate, Panel


class Estimator(Protocol):
    name: str

    def fit(self, panel: Panel, decision: Decision) -> EffectEstimate: ...
