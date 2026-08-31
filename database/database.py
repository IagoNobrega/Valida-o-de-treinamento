from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any


class Database:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def initialize(self) -> None:
        with self.connect() as conn:
            conn.executescript("""
            CREATE TABLE IF NOT EXISTS models (id INTEGER PRIMARY KEY, model_name TEXT NOT NULL, model_path TEXT UNIQUE NOT NULL, file_name TEXT NOT NULL, created_at TEXT NOT NULL, class_count INTEGER NOT NULL, classes_json TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS comparisons (id TEXT PRIMARY KEY, old_model_id INTEGER NOT NULL, new_model_id INTEGER NOT NULL, dataset_path TEXT NOT NULL, dataset_name TEXT NOT NULL, split TEXT NOT NULL, conf REAL NOT NULL, iou REAL NOT NULL, imgsz INTEGER NOT NULL, device TEXT NOT NULL, created_at TEXT NOT NULL, result_status TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS metrics (id INTEGER PRIMARY KEY, comparison_id TEXT NOT NULL, model_role TEXT NOT NULL, precision REAL, recall REAL, map50 REAL, map50_95 REAL, preprocess_ms REAL, inference_ms REAL, postprocess_ms REAL);
            CREATE TABLE IF NOT EXISTS class_metrics (id INTEGER PRIMARY KEY, comparison_id TEXT NOT NULL, model_role TEXT NOT NULL, class_name TEXT NOT NULL, precision REAL, recall REAL, map50 REAL, map50_95 REAL);
            """)

    def _save_model(self, conn: sqlite3.Connection, path: Path, classes: list[str]) -> int:
        now = datetime.now().isoformat(timespec="seconds")
        conn.execute("INSERT OR IGNORE INTO models (model_name,model_path,file_name,created_at,class_count,classes_json) VALUES (?,?,?,?,?,?)", (path.stem, str(path), path.name, now, len(classes), json.dumps(classes)))
        return conn.execute("SELECT id FROM models WHERE model_path=?", (str(path),)).fetchone()[0]

    def save_comparison(self, payload: dict[str, Any]) -> None:
        with self.connect() as conn:
            old_id = self._save_model(conn, Path(payload["old_model"]), payload["old_classes"])
            new_id = self._save_model(conn, Path(payload["new_model"]), payload["new_classes"])
            config = payload["config"]
            conn.execute("INSERT INTO comparisons VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", (payload["comparison_id"], old_id, new_id, config["data"], Path(config["data"]).name, config["split"], config["conf"], config["iou"], config["imgsz"], config["device"], payload["created_at"], payload["status"]))
            for role in ("OLD", "NEW"):
                values = payload["metrics"][role.lower()]
                conn.execute("INSERT INTO metrics (comparison_id,model_role,precision,recall,map50,map50_95,preprocess_ms,inference_ms,postprocess_ms) VALUES (?,?,?,?,?,?,?,?,?)", (payload["comparison_id"], role, *(values[key] for key in ("precision","recall","map50","map50_95","preprocess_ms","inference_ms","postprocess_ms"))))
                conn.executemany("INSERT INTO class_metrics (comparison_id,model_role,class_name,precision,recall,map50,map50_95) VALUES (?,?,?,?,?,?,?)", [(payload["comparison_id"], role, row["class_name"], row.get("precision"), row.get("recall"), row.get("map50"), row.get("map50_95")) for row in payload["class_metrics"][role.lower()]])

    def list_comparisons(self) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute("SELECT c.id,c.created_at,o.file_name AS old_model,n.file_name AS new_model,c.dataset_name,c.result_status FROM comparisons c JOIN models o ON o.id=c.old_model_id JOIN models n ON n.id=c.new_model_id ORDER BY c.created_at DESC").fetchall()
        return [dict(row) for row in rows]

    def load_comparison(self, comparison_id: str) -> dict | None:
        with self.connect() as conn:
            comparison = conn.execute("SELECT * FROM comparisons WHERE id=?", (comparison_id,)).fetchone()
            if not comparison: return None
            metrics = conn.execute("SELECT * FROM metrics WHERE comparison_id=?", (comparison_id,)).fetchall()
            classes = conn.execute("SELECT * FROM class_metrics WHERE comparison_id=?", (comparison_id,)).fetchall()
        return {"comparison": dict(comparison), "metrics": [dict(row) for row in metrics], "class_metrics": [dict(row) for row in classes]}
