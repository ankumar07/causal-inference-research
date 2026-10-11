"""Aggregate raw calibration replications into docs/LAB_REPORT.md: bias, RMSE,
90% interval coverage, false-positive rate (S3), and interval width, plus a
coverage chart. Reports the numbers that come out — never tuned to look good.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SCENARIO_NOTES = {
    "S1": "Price +6%, 60 weeks of history. The clean case.",
    "S2": "Same decision, 12 weeks of history. Short-history degradation.",
    "S3": "No true effect, timed to a seasonal spike. True effect is 0 — `significant_rate` here *is* the false-positive rate.",
    "S4": "Price +10%, elastic demand (net revenue-negative), with spillover into a neighboring donor. Naive synthetic control is expected to be biased here.",
    "S5": "Staggered rollout across 6 metros over 8 weeks.",
}


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    def agg(group: pd.DataFrame) -> pd.Series:
        error = group["est_cum_effect"] - group["true_cum_effect"]
        return pd.Series(
            {
                "n": len(group),
                "bias": error.mean(),
                "rmse": (error**2).mean() ** 0.5,
                "coverage_90": group["covered"].mean(),
                "avg_interval_width": (group["est_cum_hi"] - group["est_cum_lo"]).mean(),
                "significant_rate": group["significant"].mean(),
                "true_cum_effect": group["true_cum_effect"].mean(),  # varies by seed; this is the average
            }
        )

    return raw.groupby(["scenario", "method"], sort=True).apply(agg, include_groups=False).reset_index()


def _coverage_chart(summary: pd.DataFrame, path: str) -> None:
    scenarios = sorted(summary["scenario"].unique())
    methods = sorted(summary["method"].unique())
    width = 0.8 / len(methods)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, method in enumerate(methods):
        rows = summary[summary["method"] == method].set_index("scenario").reindex(scenarios)
        xs = [j + i * width for j in range(len(scenarios))]
        ax.bar(xs, rows["coverage_90"], width=width, label=method)

    ax.axhline(0.90, color="black", linestyle="--", linewidth=1, label="target (90%)")
    ax.set_xticks([j + width * (len(methods) - 1) / 2 for j in range(len(scenarios))])
    ax.set_xticklabels(scenarios)
    ax.set_ylabel("90% interval coverage")
    ax.set_ylim(0, 1.05)
    ax.set_title("Calibration: does the 90% interval cover the truth 90% of the time?")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_report(raw: pd.DataFrame, out_path: str, n: int, elapsed_seconds: float) -> None:
    summary = summarize(raw)
    chart_path = os.path.join(os.path.dirname(out_path) or ".", "lab_coverage.png")
    _coverage_chart(summary, chart_path)
    chart_name = os.path.basename(chart_path)

    lines = [
        "# Lab report",
        "",
        f"{n} replications per scenario per method, {len(raw)} total fits, {elapsed_seconds:.0f}s wall clock.",
        "Regenerate with `uv run python -m decision_proof.calibrate.harness --n 200`.",
        "",
        "Numbers below are whatever the harness produced on this run — not tuned to hit a target.",
        "",
        f"![90% interval coverage by scenario and method]({chart_name})",
        "",
        "## Coverage, bias, RMSE, interval width",
        "",
        "| Scenario | Method | n | Avg true effect | Bias | RMSE | 90% coverage | Avg interval width | Significant rate |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    for _, row in summary.iterrows():
        lines.append(
            f"| {row['scenario']} | {row['method']} | {row['n']:.0f} | "
            f"${row['true_cum_effect']:,.0f} | ${row['bias']:,.0f} | ${row['rmse']:,.0f} | "
            f"{row['coverage_90']:.0%} | ${row['avg_interval_width']:,.0f} | {row['significant_rate']:.0%} |"
        )

    lines += [
        "",
        "## Where each method breaks",
        "",
    ]
    for scenario in sorted(summary["scenario"].unique()):
        lines.append(f"**{scenario}**: {SCENARIO_NOTES.get(scenario, '')}")
        for _, row in summary[summary["scenario"] == scenario].iterrows():
            if scenario == "S3":
                lines.append(
                    f"- {row['method']}: false-positive rate {row['significant_rate']:.0%} "
                    f"(found a 'significant' effect when the true effect is $0)"
                )
            else:
                lines.append(
                    f"- {row['method']}: {row['coverage_90']:.0%} coverage "
                    f"(target 90%), bias ${row['bias']:,.0f}, power (detected as significant) {row['significant_rate']:.0%}"
                )
        lines.append("")

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
