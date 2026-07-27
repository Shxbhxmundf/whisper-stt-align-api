"""SQLite-backed job store for whisper_api.

One connection guarded by a lock (accessed from request threads and the single
worker thread). Rows live in jobs/jobs.db; each job's files live in jobs/<id>/.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sqlite3
import threading
import time
import uuid

log = logging.getLogger("whisper_api.jobs")

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id          TEXT PRIMARY KEY,
    task        TEXT NOT NULL,          -- transcribe | align
    status      TEXT NOT NULL,          -- queued | running | done | failed | cancelled
    stage       TEXT DEFAULT '',
    progress    REAL DEFAULT 0.0,
    params      TEXT NOT NULL,          -- JSON
    audio_path  TEXT,
    result_path TEXT,
    error       TEXT,
    attempts    INTEGER DEFAULT 0,
    audio_duration REAL,
    created_at  REAL NOT NULL,
    started_at  REAL,
    finished_at REAL
);
CREATE INDEX IF NOT EXISTS idx_jobs_status_created ON jobs(status, created_at);
"""

TERMINAL = ("done", "failed", "cancelled")


class JobStore:
    def __init__(self, root: str):
        self.root = root
        os.makedirs(root, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            os.path.join(root, "jobs.db"), check_same_thread=False
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def job_dir(self, job_id: str) -> str:
        d = os.path.join(self.root, job_id)
        os.makedirs(d, exist_ok=True)
        return d

    def new_id(self) -> str:
        return uuid.uuid4().hex[:12]

    def enqueue(self, job_id: str, task: str, params: dict, audio_path: str):
        with self._lock:
            self._conn.execute(
                "INSERT INTO jobs (id, task, status, params, audio_path, created_at)"
                " VALUES (?, ?, 'queued', ?, ?, ?)",
                (job_id, task, json.dumps(params), audio_path, time.time()),
            )
            self._conn.commit()

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return dict(row) if row else None

    def list_jobs(self, limit: int = 50) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, task, status, stage, progress, error, audio_duration,"
                " created_at, started_at, finished_at FROM jobs"
                " ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def queue_position(self, job_id: str) -> int | None:
        """Jobs ahead of this queued job (earlier-queued + currently running) + 1."""
        with self._lock:
            row = self._conn.execute(
                "SELECT status, created_at FROM jobs WHERE id=?", (job_id,)
            ).fetchone()
            if not row or row["status"] != "queued":
                return None
            ahead = self._conn.execute(
                "SELECT (SELECT COUNT(*) FROM jobs WHERE status='queued' AND created_at < ?)"
                " + (SELECT COUNT(*) FROM jobs WHERE status='running')",
                (row["created_at"],),
            ).fetchone()[0]
        return ahead + 1

    def counts(self) -> dict:
        with self._lock:
            rows = self._conn.execute(
                "SELECT status, COUNT(*) c FROM jobs WHERE status IN ('queued','running')"
                " GROUP BY status"
            ).fetchall()
        out = {"queued": 0, "running": 0}
        out.update({r["status"]: r["c"] for r in rows})
        return out

    def claim_next(self) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM jobs WHERE status='queued' ORDER BY created_at LIMIT 1"
            ).fetchone()
            if not row:
                return None
            self._conn.execute(
                "UPDATE jobs SET status='running', started_at=?, attempts=attempts+1,"
                " stage='starting', progress=0 WHERE id=?",
                (time.time(), row["id"]),
            )
            self._conn.commit()
        return dict(row)

    def update_stage(self, job_id: str, stage: str, progress: float = 0.0):
        with self._lock:
            self._conn.execute(
                "UPDATE jobs SET stage=?, progress=? WHERE id=?",
                (stage, round(float(progress), 3), job_id),
            )
            self._conn.commit()

    def set_duration(self, job_id: str, duration: float | None):
        if duration is None:
            return
        with self._lock:
            self._conn.execute(
                "UPDATE jobs SET audio_duration=? WHERE id=?", (duration, job_id)
            )
            self._conn.commit()

    def finish(self, job_id: str, result_path: str | None = None, error: str | None = None):
        status = "failed" if error else "done"
        with self._lock:
            self._conn.execute(
                "UPDATE jobs SET status=?, result_path=?, error=?, finished_at=?,"
                " stage=?, progress=? WHERE id=?",
                (status, result_path, error, time.time(),
                 "failed" if error else "done", 0.0 if error else 1.0, job_id),
            )
            self._conn.commit()

    def cancel(self, job_id: str) -> str | None:
        """Cancel a queued job. Returns the job's (new) status, or None if unknown."""
        with self._lock:
            row = self._conn.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()
            if not row:
                return None
            if row["status"] == "queued":
                self._conn.execute(
                    "UPDATE jobs SET status='cancelled', finished_at=? WHERE id=?",
                    (time.time(), job_id),
                )
                self._conn.commit()
                return "cancelled"
            return row["status"]

    def delete(self, job_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM jobs WHERE id=?", (job_id,))
            self._conn.commit()
        shutil.rmtree(os.path.join(self.root, job_id), ignore_errors=True)
        return cur.rowcount > 0

    def recover_on_boot(self):
        """Requeue jobs interrupted by a restart (once); fail them on the second interruption."""
        with self._lock:
            running = self._conn.execute("SELECT id, attempts FROM jobs WHERE status='running'").fetchall()
            for r in running:
                if r["attempts"] >= 2:
                    self._conn.execute(
                        "UPDATE jobs SET status='failed', finished_at=?,"
                        " error='interrupted by server restart twice' WHERE id=?",
                        (time.time(), r["id"]),
                    )
                    log.warning("job %s failed: interrupted twice", r["id"])
                else:
                    self._conn.execute(
                        "UPDATE jobs SET status='queued', stage='', progress=0 WHERE id=?",
                        (r["id"],),
                    )
                    log.info("job %s requeued after restart", r["id"])
            self._conn.commit()

    def purge_older_than(self, ttl_hours: float) -> int:
        cutoff = time.time() - ttl_hours * 3600
        with self._lock:
            rows = self._conn.execute(
                "SELECT id FROM jobs WHERE created_at < ? AND status IN (?,?,?)",
                (cutoff, *TERMINAL),
            ).fetchall()
            ids = [r["id"] for r in rows]
            if ids:
                self._conn.executemany("DELETE FROM jobs WHERE id=?", [(i,) for i in ids])
                self._conn.commit()
        for i in ids:
            shutil.rmtree(os.path.join(self.root, i), ignore_errors=True)
        if ids:
            log.info("purged %d expired jobs", len(ids))
        return len(ids)
