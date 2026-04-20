import os
import sqlite3
from typing import Literal, Optional


Label = Literal["normal", "anomaly"]


class AnnotationStore:
    def __init__(self, db_path: str):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self._init()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init(self) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS template_annotations (
                  cluster_id INTEGER PRIMARY KEY,
                  label TEXT NOT NULL CHECK(label IN ('normal','anomaly')),
                  note TEXT,
                  updated_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def upsert(self, cluster_id: int, label: Label, note: Optional[str], updated_at: str) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO template_annotations(cluster_id, label, note, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(cluster_id) DO UPDATE SET
                  label=excluded.label,
                  note=excluded.note,
                  updated_at=excluded.updated_at
                """,
                (cluster_id, label, note, updated_at),
            )
            conn.commit()

    def get(self, cluster_id: int):
        with self._conn() as conn:
            row = conn.execute(
                "SELECT cluster_id, label, note, updated_at FROM template_annotations WHERE cluster_id=?",
                (cluster_id,),
            ).fetchone()
        return dict(row) if row else None

    def list_all(self):
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT cluster_id, label, note, updated_at FROM template_annotations ORDER BY updated_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

