"""Decision ledger: a DuckDB table of logged decisions, open until graded.

One row per decision: what was predicted (method, verdict, predicted range) and,
once graded, what actually happened (realized effect, whether it fell inside the
predicted range, and an interval score). This is how the system's credibility gets
earned rather than claimed — see ledger/grading.py for how a decision gets graded.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd

from decision_proof.contracts import Decision, EffectEstimate, RefutationReport

DEFAULT_DB_PATH = "data/ledger.duckdb"

SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    decision_id VARCHAR PRIMARY KEY,
    description VARCHAR,
    owner VARCHAR,
    decision_date DATE,
    treated_units VARCHAR[],
    start_period INTEGER,
    outcome VARCHAR,
    method VARCHAR,
    verdict VARCHAR,
    pred_cum_lo DOUBLE,
    pred_cum_effect DOUBLE,
    pred_cum_hi DOUBLE,
    status VARCHAR DEFAULT 'open',
    created_at TIMESTAMP,
    realized_cum_effect DOUBLE,
    inside_range BOOLEAN,
    interval_score DOUBLE,
    graded_at TIMESTAMP
);
"""


class Ledger:
    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(db_path)
        self.con.execute(SCHEMA)

    def log_decision(
        self,
        decision: Decision,
        estimate: EffectEstimate,
        report: RefutationReport,
        decision_date: date | None = None,
    ) -> None:
        """Log a decision as 'open'. Raises if decision_id already exists."""
        existing = self.con.execute(
            "SELECT 1 FROM decisions WHERE decision_id = ?", [decision.decision_id]
        ).fetchone()
        if existing:
            raise ValueError(f"decision_id {decision.decision_id!r} is already logged")

        self.con.execute(
            """
            INSERT INTO decisions (
                decision_id, description, owner, decision_date, treated_units,
                start_period, outcome, method, verdict,
                pred_cum_lo, pred_cum_effect, pred_cum_hi, status, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open', ?)
            """,
            [
                decision.decision_id,
                decision.description,
                decision.owner,
                decision_date or date.today(),
                list(decision.treated_units),
                decision.start_period,
                decision.outcome,
                estimate.method,
                report.verdict,
                estimate.cum_lo,
                estimate.cum_effect,
                estimate.cum_hi,
                datetime.now(timezone.utc),
            ],
        )

    def record_grade(
        self,
        decision_id: str,
        realized_cum_effect: float,
        inside_range: bool,
        interval_score: float,
    ) -> None:
        if self.get_decision(decision_id) is None:
            raise ValueError(f"decision_id {decision_id!r} not found")
        self.con.execute(
            """
            UPDATE decisions
            SET status = 'graded', realized_cum_effect = ?, inside_range = ?,
                interval_score = ?, graded_at = ?
            WHERE decision_id = ?
            """,
            [realized_cum_effect, inside_range, interval_score, datetime.now(timezone.utc), decision_id],
        )

    def get_decision(self, decision_id: str) -> dict | None:
        df = self.con.execute("SELECT * FROM decisions WHERE decision_id = ?", [decision_id]).df()
        if df.empty:
            return None
        return df.iloc[0].to_dict()

    def list_decisions(self, status: str | None = None) -> pd.DataFrame:
        if status is None:
            return self.con.execute("SELECT * FROM decisions ORDER BY created_at").df()
        return self.con.execute(
            "SELECT * FROM decisions WHERE status = ? ORDER BY created_at", [status]
        ).df()

    def track_record(self) -> pd.DataFrame:
        """One row per graded decision, in graded order, with a running share-inside-range."""
        df = self.con.execute(
            "SELECT * FROM decisions WHERE status = 'graded' ORDER BY graded_at"
        ).df()
        if df.empty:
            return df
        df["running_share_inside"] = df["inside_range"].expanding().mean()
        return df

    def close(self) -> None:
        self.con.close()
