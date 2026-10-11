"""Cached helpers for the Quarterly Pricing page. Pure, deterministic functions
keyed by their arguments — safe for st.cache_data (unlike st.cache_resource,
there's no shared-mutable-object race here, see app/common.py's docstring for
why that distinction mattered).
"""

from __future__ import annotations

import streamlit as st

from decision_proof.estimators.synth import SyntheticControl
from decision_proof.refute.verdict import refute
from decision_proof.simulate.quarterly import build_experiment, generate_quarterly_world

SEED = 42


@st.cache_data
def run_all_quarter_experiments() -> list[dict]:
    """One row per experiment across every experimental quarter. Cached once;
    the page filters by quarter selection afterward rather than re-running
    estimators on every interaction.
    """
    baseline, demographics, plans = generate_quarterly_world(seed=SEED)
    rows = []
    for plan in plans:
        panel, decision, truth = build_experiment(baseline, plan, plans)
        est = SyntheticControl().fit(panel, decision)
        report = refute(panel, decision, {"synthetic_control": est})

        treated_demo = [demographics[u] for u in decision.treated_units]
        age_bands = [d["age_band"] for d in treated_demo]
        income_tiers = [d["income_tier"] for d in treated_demo]
        dominant_age = max(set(age_bands), key=age_bands.count)
        dominant_income = max(set(income_tiers), key=income_tiers.count)

        rows.append(
            {
                "quarter": plan.quarter,
                "decision_id": decision.decision_id,
                "description": decision.description,
                "price_change_pct": plan.price_change_pct,
                "verdict": report.verdict,
                "cum_effect": est.cum_effect,
                "cum_lo": est.cum_lo,
                "cum_hi": est.cum_hi,
                "pct_effect": est.pct_effect,
                "pct_lo": est.pct_lo,
                "pct_hi": est.pct_hi,
                "dominant_age_band": dominant_age,
                "dominant_income_tier": dominant_income,
            }
        )
    return rows
