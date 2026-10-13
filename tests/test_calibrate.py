import os

from decision_proof.calibrate.harness import ESTIMATORS, run_calibration
from decision_proof.calibrate.report import summarize, write_report


def test_run_calibration_shape():
    raw = run_calibration(scenarios=["S1"], methods=["synthetic_control"], n=2, n_jobs=1)

    assert len(raw) == 2
    assert set(raw.columns) == {
        "scenario", "method", "seed", "true_cum_effect",
        "est_cum_effect", "est_cum_lo", "est_cum_hi", "covered", "significant",
    }
    assert (raw["scenario"] == "S1").all()
    assert (raw["method"] == "synthetic_control").all()
    assert raw["seed"].nunique() == 2


def test_summarize_one_row_per_scenario_method():
    raw = run_calibration(scenarios=["S1", "S3"], methods=list(ESTIMATORS), n=2, n_jobs=1)
    summary = summarize(raw)

    assert len(summary) == 2 * len(ESTIMATORS)  # 2 scenarios x all registered methods
    assert (summary["n"] == 2).all()
    assert (summary["coverage_90"].between(0, 1)).all()
    assert (summary["significant_rate"].between(0, 1)).all()
    assert (summary["avg_interval_width"] >= 0).all()


def test_write_report_produces_file_and_chart(tmp_path):
    raw = run_calibration(scenarios=["S3"], methods=list(ESTIMATORS), n=2, n_jobs=1)
    out_path = tmp_path / "LAB_REPORT.md"

    write_report(raw, str(out_path), n=2, elapsed_seconds=1.2)

    assert out_path.exists()
    content = out_path.read_text()
    assert "Lab report" in content
    assert "false-positive rate" in content
    assert os.path.exists(tmp_path / "lab_coverage.png")
