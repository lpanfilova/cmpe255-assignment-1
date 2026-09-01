"""SQLite adapter for Workflow and Run persistence."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


class Repository:
    def __init__(self, path: str | Path):
        self.path = str(path)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS workflows (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL, version INTEGER NOT NULL,
                    definition TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY, workflow_id TEXT NOT NULL, created_at TEXT NOT NULL,
                    result TEXT NOT NULL
                );
            """)

    def save_workflow(self, workflow: dict[str, Any]) -> dict[str, Any]:
        workflow_id = workflow.get("id") or uuid4().hex[:12]
        existing = self.get_workflow(workflow_id)
        saved = {**workflow, "id": workflow_id, "version": (existing or {}).get("version", 0) + 1}
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO workflows VALUES (?, ?, ?, ?, ?)",
                (workflow_id, saved.get("name", "Untitled workflow"), saved["version"], json.dumps(saved), now),
            )
        return saved

    def list_workflows(self) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute("SELECT definition FROM workflows ORDER BY updated_at DESC").fetchall()
        return [json.loads(row["definition"]) for row in rows]

    def get_workflow(self, workflow_id: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute("SELECT definition FROM workflows WHERE id = ?", (workflow_id,)).fetchone()
        return json.loads(row["definition"]) if row else None

    def save_run(self, workflow_id: str, result: dict[str, Any]) -> dict[str, Any]:
        run_record = {**result, "id": uuid4().hex[:12], "workflow_id": workflow_id, "created_at": datetime.now(timezone.utc).isoformat()}
        with self._connect() as db:
            db.execute("INSERT INTO runs VALUES (?, ?, ?, ?)", (run_record["id"], workflow_id, run_record["created_at"], json.dumps(run_record)))
        return run_record

    def list_runs(self, workflow_id: str) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute("SELECT result FROM runs WHERE workflow_id = ? ORDER BY created_at DESC", (workflow_id,)).fetchall()
        return [json.loads(row["result"]) for row in rows]

