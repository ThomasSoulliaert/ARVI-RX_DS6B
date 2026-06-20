from __future__ import annotations

import json
import sqlite3
from pathlib import Path

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "sql" / "schema.sql"
DEFAULT_DB_PATH = Path("data") / "predictions.sqlite"


def connect(db_path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str | Path = DEFAULT_DB_PATH) -> None:
    conn = connect(db_path)
    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    conn.commit(); conn.close()


def insert_run(db_path: str | Path, case_id: str, image_path: str, prediction: dict) -> None:
    init_db(db_path)
    conn = connect(db_path)
    conn.execute(
        """
        INSERT INTO runs(case_id, image_path, model_name, prompt_version, prediction_json, predicted_class, confidence, latency_ms)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            case_id,
            image_path,
            prediction.get("model_name"),
            prediction.get("prompt_version"),
            json.dumps(prediction, ensure_ascii=False),
            prediction.get("predicted_class"),
            float(prediction.get("confidence", 0.0)),
            int(prediction.get("latency_ms", 0)),
        ),
    )
    conn.commit(); conn.close()


def fetch_recent_runs(
    db_path: str | Path = DEFAULT_DB_PATH,
    limit: int = 20,
) -> list[dict]:
    db_file = Path(db_path)
    if not db_file.exists():
        return []

    init_db(db_file)
    conn = connect(db_file)
    rows = conn.execute(
        """
        SELECT id, case_id, image_path, model_name, prompt_version,
               prediction_json, predicted_class, confidence, latency_ms, created_at
        FROM runs
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    conn.close()

    results = []
    for row in rows:
        item = dict(row)
        try:
            item["prediction"] = json.loads(item.get("prediction_json") or "{}")
        except json.JSONDecodeError:
            item["prediction"] = {}
        results.append(item)
    return results


def summarize_runs(db_path: str | Path = DEFAULT_DB_PATH) -> dict:
    db_file = Path(db_path)
    if not db_file.exists():
        return {
            "total": 0,
            "average_confidence": 0.0,
            "average_latency_ms": 0.0,
            "class_counts": {},
        }

    init_db(db_file)
    conn = connect(db_file)
    summary_row = conn.execute(
        """
        SELECT COUNT(*) AS total,
               AVG(confidence) AS average_confidence,
               AVG(latency_ms) AS average_latency_ms
        FROM runs
        """
    ).fetchone()
    class_rows = conn.execute(
        """
        SELECT predicted_class, COUNT(*) AS count
        FROM runs
        GROUP BY predicted_class
        ORDER BY count DESC
        """
    ).fetchall()
    conn.close()

    return {
        "total": int(summary_row["total"] or 0),
        "average_confidence": round(float(summary_row["average_confidence"] or 0.0), 3),
        "average_latency_ms": round(float(summary_row["average_latency_ms"] or 0.0), 1),
        "class_counts": {
            row["predicted_class"] or "unknown": int(row["count"])
            for row in class_rows
        },
    }

