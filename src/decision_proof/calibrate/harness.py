"""Calibration harness: run each estimator on N simulated replications per scenario,
in parallel, and record whether the 90% interval actually behaves like a 90% interval.

Usage: `uv run python -m decision_proof.calibrate.harness --n 200`
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict, dataclass

import pandas as pd
from joblib import Parallel, delayed

from decision_proof.estimators.did import DiDEventStudy
from decision_proof.estimators.synth import SyntheticControl
from decision_proof.simulate.scenarios import SCENARIOS, generate_scenario

ESTIMATORS = {
    "did_event_study": DiDEventStudy,
    "synthetic_control": SyntheticControl,
}


@dataclass
class ReplicationResult:
    scenario: str
    method: str
    seed: int
    true_cum_effect: float
    est_cum_effect: float
    est_cum_lo: float
    est_cum_hi: float
    covered: bool  # true effect inside the 90% interval
    significant: bool  # interval excludes zero


def _run_one(scenario: str, method: str, seed: int) -> ReplicationResult:
    panel, decision, truth = generate_scenario(scenario, seed=seed)
    estimator = ESTIMATORS[method]()
    est = estimator.fit(panel, decision)
    covered = est.cum_lo <= truth.true_cum_effect <= est.cum_hi
    significant = not (est.cum_lo <= 0 <= est.cum_hi)
    return ReplicationResult(
        scenario=scenario,
        method=method,
        seed=seed,
        true_cum_effect=truth.true_cum_effect,
        est_cum_effect=est.cum_effect,
        est_cum_lo=est.cum_lo,
        est_cum_hi=est.cum_hi,
        covered=covered,
        significant=significant,
    )


def run_calibration(
    scenarios: list[str] | None = None,
    methods: list[str] | None = None,
    n: int = 200,
    base_seed: int = 1000,
    n_jobs: int = -1,
) -> pd.DataFrame:
    scenarios = scenarios or list(SCENARIOS)
    methods = methods or list(ESTIMATORS)
    seeds = [base_seed + i for i in range(n)]
    tasks = [(s, m, seed) for s in scenarios for m in methods for seed in seeds]

    results = Parallel(n_jobs=n_jobs, verbose=5)(delayed(_run_one)(s, m, seed) for s, m, seed in tasks)
    return pd.DataFrame([asdict(r) for r in results])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Decision Proof calibration lab")
    parser.add_argument("--n", type=int, default=200, help="replications per scenario")
    parser.add_argument("--scenarios", nargs="*", default=None, help="subset of scenarios (default: all)")
    parser.add_argument("--n-jobs", type=int, default=-1, help="joblib worker count (-1 = all cores)")
    parser.add_argument("--out", default="docs/LAB_REPORT.md", help="output report path")
    args = parser.parse_args()

    t0 = time.time()
    raw = run_calibration(scenarios=args.scenarios, n=args.n, n_jobs=args.n_jobs)
    elapsed = time.time() - t0

    from decision_proof.calibrate.report import write_report

    write_report(raw, args.out, n=args.n, elapsed_seconds=elapsed)
    print(f"Wrote {args.out} from {len(raw)} replications in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
