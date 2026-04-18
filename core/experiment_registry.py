from __future__ import annotations
import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional
from utils.logger import get_logger

logger = get_logger("experiment_registry")


@dataclass
class ExperimentRecord:
    run_id: str
    config_hash: str
    config_snapshot: dict
    status: str
    start_time: str
    end_time: Optional[str] = None
    final_map50: float = 0.0
    final_map50_95: float = 0.0
    best_epoch: int = 0
    notes: str = ""

    @classmethod
    def from_row(cls, row: tuple) -> "ExperimentRecord":
        return cls(
            run_id=row[0],
            config_hash=row[1],
            config_snapshot=json.loads(row[2]),
            status=row[3],
            start_time=row[4],
            end_time=row[5],
            final_map50=row[6],
            final_map50_95=row[7],
            best_epoch=row[8],
            notes=row[9],
        )


class ExperimentRegistry:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    run_id TEXT PRIMARY KEY,
                    config_hash TEXT NOT NULL,
                    config_snapshot TEXT NOT NULL,
                    status TEXT NOT NULL,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    final_map50 REAL DEFAULT 0.0,
                    final_map50_95 REAL DEFAULT 0.0,
                    best_epoch INTEGER DEFAULT 0,
                    notes TEXT DEFAULT ''
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_config_hash ON experiments(config_hash)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON experiments(status)")
            conn.commit()

    @staticmethod
    def compute_config_hash(training_config: dict) -> str:
        relevant_keys = [
            "model", "data", "epochs", "imgsz", "batch", "lr0", "lrf",
            "momentum", "weight_decay", "mosaic", "mixup", "copy_paste",
            "dropout", "hsv_h", "hsv_s", "hsv_v", "degrees", "translate",
            "scale", "shear", "flipud", "fliplr",
        ]
        subset = {k: training_config.get(k) for k in relevant_keys if k in training_config}
        serialized = json.dumps(subset, sort_keys=True)
        return hashlib.sha256(serialized.encode()).hexdigest()[:16]

    def is_duplicate(self, training_config: dict) -> Optional[ExperimentRecord]:
        config_hash = self.compute_config_hash(training_config)
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM experiments WHERE config_hash = ? AND status IN ('running', 'completed')",
                (config_hash,),
            ).fetchone()
        if row:
            return ExperimentRecord.from_row(row)
        return None

    def register(self, run_id: str, config: dict) -> str:
        config_hash = self.compute_config_hash(config.get("training", config))
        record = ExperimentRecord(
            run_id=run_id,
            config_hash=config_hash,
            config_snapshot=config,
            status="pending",
            start_time=datetime.now().isoformat(),
        )
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO experiments
                   (run_id, config_hash, config_snapshot, status, start_time, end_time,
                    final_map50, final_map50_95, best_epoch, notes)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    record.run_id, record.config_hash,
                    json.dumps(record.config_snapshot), record.status,
                    record.start_time, record.end_time,
                    record.final_map50, record.final_map50_95,
                    record.best_epoch, record.notes,
                ),
            )
            conn.commit()
        logger.info(f"Registered experiment [bold]{run_id}[/bold] (hash: {config_hash})")
        return config_hash

    def update_status(self, run_id: str, status: str, **kwargs) -> None:
        allowed = {"end_time", "final_map50", "final_map50_95", "best_epoch", "notes"}
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if status == "completed" and "end_time" not in updates:
            updates["end_time"] = datetime.now().isoformat()

        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [status, run_id]
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                f"UPDATE experiments SET {set_clause}, status = ? WHERE run_id = ?",
                values,
            )
            conn.commit()

    def get_all(self, status: Optional[str] = None) -> list[ExperimentRecord]:
        with sqlite3.connect(self.db_path) as conn:
            if status:
                rows = conn.execute(
                    "SELECT * FROM experiments WHERE status = ? ORDER BY start_time DESC", (status,)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM experiments ORDER BY start_time DESC"
                ).fetchall()
        return [ExperimentRecord.from_row(r) for r in rows]

    def get_best(self, top_n: int = 5) -> list[ExperimentRecord]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM experiments WHERE status = 'completed' ORDER BY final_map50 DESC LIMIT ?",
                (top_n,),
            ).fetchall()
        return [ExperimentRecord.from_row(r) for r in rows]

    def summary(self) -> dict:
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
            by_status = dict(
                conn.execute("SELECT status, COUNT(*) FROM experiments GROUP BY status").fetchall()
            )
            best_map = conn.execute(
                "SELECT MAX(final_map50) FROM experiments WHERE status = 'completed'"
            ).fetchone()[0] or 0.0
        return {"total": total, "by_status": by_status, "best_map50": best_map}
