"""DiD event study, via pyfixest.

With only a handful of treated units, ordinary clustered standard errors are
unreliable (few-treated-cluster problem), so the headline cumulative effect's
interval comes from a wild cluster bootstrap (percentile-t, null imposed) on a
single static post*treated coefficient. The per-week path uses the same panel's
dynamic event-study coefficients with analytic cluster-robust (CRV1) intervals —
an approximation for visualization, not the headline number.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyfixest as pf

from decision_proof.contracts import Decision, EffectEstimate, Panel

Z_90 = 1.6448536269514722  # two-sided 90% normal critical value


def _prep(panel: Panel, decision: Decision) -> pd.DataFrame:
    df = panel.df.copy()
    treated = set(decision.treated_units)
    start = decision.start_period
    df["treated"] = df["unit_id"].isin(treated)
    df["unit_code"] = pd.Categorical(df["unit_id"]).codes
    df["period"] = df["period"].astype(int)
    return df


def _bootstrap_ci(
    model, param: str, cluster: str, reps: int, seed: int, alpha: float = 0.10
) -> tuple[float, float]:
    """Percentile-t wild cluster bootstrap CI for one coefficient (null imposed)."""
    coef = float(model.coef()[param])
    se = float(model.se()[param])
    _, tboot = model.wildboottest(
        param=param, reps=reps, cluster=cluster, seed=seed, return_bootstrapped_t_stats=True
    )
    tq_lo, tq_hi = np.quantile(tboot, [alpha / 2, 1 - alpha / 2])
    return coef - se * tq_hi, coef - se * tq_lo


@dataclass
class DiDEventStudy:
    name: str = "did_event_study"
    n_leads_detail: int = 12
    reps: int = 199
    seed: int = 1
    alpha: float = 0.10

    def fit(self, panel: Panel, decision: Decision) -> EffectEstimate:
        outcome = panel.outcome
        df = _prep(panel, decision)
        start = decision.start_period
        n_treated = len(decision.treated_units)

        # --- headline cumulative effect: single static post*treated coefficient, wild-bootstrapped ---
        df["post_treated"] = (df["treated"] & (df["period"] >= start)).astype(int)
        static_fml = f"{outcome} ~ post_treated | unit_id + period"
        static_model = pf.feols(static_fml, data=df, vcov={"CRV1": "unit_code"})

        avg_effect = float(static_model.coef()["post_treated"])
        avg_lo, avg_hi = _bootstrap_ci(static_model, "post_treated", "unit_code", self.reps, self.seed, self.alpha)

        n_post_periods = int(df.loc[df["period"] >= start, "period"].nunique())
        cum_effect = avg_effect * n_treated * n_post_periods
        cum_lo = avg_lo * n_treated * n_post_periods
        cum_hi = avg_hi * n_treated * n_post_periods

        pre_treated = df[df["treated"] & (df["period"] < start)]
        baseline_level = pre_treated.groupby("unit_id")[outcome].mean().mean() if len(pre_treated) else np.nan
        denom = baseline_level * n_treated * n_post_periods if pd.notna(baseline_level) and baseline_level else np.nan
        pct_effect = float(cum_effect / denom * 100) if denom else 0.0
        pct_lo = float(cum_lo / denom * 100) if denom else 0.0
        pct_hi = float(cum_hi / denom * 100) if denom else 0.0

        # --- dynamic event-study path, for visualization + the pre-trend check ---
        df["rel_time"] = -1
        mask = df["treated"]
        df.loc[mask, "rel_time"] = df.loc[mask, "period"] - start
        df["rel_time_b"] = df["rel_time"].clip(lower=-self.n_leads_detail)

        dyn_fml = f"{outcome} ~ C(rel_time_b, Treatment(reference=-1)) | unit_id + period"
        dyn_model = pf.feols(dyn_fml, data=df, vcov={"CRV1": "unit_code"})
        coefs = dyn_model.coef()
        ses = dyn_model.se()

        def _h_of(name: str) -> int:
            return int(name.split("[T.")[1].rstrip("]"))

        leads = []
        path = []
        for cname in coefs.index:
            h = _h_of(cname)
            est = float(coefs[cname])
            se = float(ses[cname])
            lo, hi = est - Z_90 * se, est + Z_90 * se
            if h < 0:
                leads.append({"h": h, "est": est, "lo": lo, "hi": hi, "se": se})
            else:
                path.append({"h": h, "est": est * n_treated, "lo": lo * n_treated, "hi": hi * n_treated})
        path.sort(key=lambda r: r["h"])
        leads.sort(key=lambda r: r["h"])

        total_path = sum(r["est"] for r in path) or 1.0
        running = 0.0
        lag_50 = lag_90 = None
        for r in path:
            running += r["est"]
            frac = running / total_path
            if lag_50 is None and frac >= 0.5:
                lag_50 = r["h"]
            if lag_90 is None and frac >= 0.9:
                lag_90 = r["h"]

        return EffectEstimate(
            method=self.name,
            cum_effect=cum_effect,
            cum_lo=cum_lo,
            cum_hi=cum_hi,
            pct_effect=pct_effect,
            pct_lo=pct_lo,
            pct_hi=pct_hi,
            path=path,
            lag_50=lag_50,
            lag_90=lag_90,
            diagnostics={
                "n_treated": n_treated,
                "n_post_periods": n_post_periods,
                "n_pre_periods": int(pre_treated["period"].nunique()) if len(pre_treated) else 0,
                "leads": leads,
                "window": {"n_leads_detail": self.n_leads_detail},
                "bootstrap": {"reps": self.reps, "seed": self.seed},
            },
        )
