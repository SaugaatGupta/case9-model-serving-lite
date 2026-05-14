"""SQLite event store. Raw text is NEVER persisted — only a sha256 hash."""
import hashlib
import sqlite3
import time
from contextlib import contextmanager
from typing import Optional

from .config import DB_PATH


SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    request_id TEXT PRIMARY KEY,
    ts REAL NOT NULL,
    text_hash TEXT NOT NULL,
    text_len INTEGER NOT NULL,
    text_lang TEXT,
    label TEXT NOT NULL,
    score REAL NOT NULL,
    latency_ms REAL NOT NULL,
    model_version TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_pred_ts ON predictions(ts);
"""


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@contextmanager
def _conn(path: Optional[str] = None):
    p = path or DB_PATH
    c = sqlite3.connect(p)
    c.row_factory = sqlite3.Row
    try:
        yield c
        c.commit()
    finally:
        c.close()


def init_db(path: Optional[str] = None) -> None:
    with _conn(path) as c:
        c.executescript(SCHEMA)


def insert_event(
    request_id: str,
    text_hash: str,
    text_len: int,
    text_lang: str,
    label: str,
    score: float,
    latency_ms: float,
    model_version: str,
    path: Optional[str] = None,
) -> None:
    with _conn(path) as c:
        c.execute(
            """INSERT OR REPLACE INTO predictions
               (request_id, ts, text_hash, text_len, text_lang, label, score, latency_ms, model_version)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                request_id,
                time.time(),
                text_hash,
                text_len,
                text_lang,
                label,
                score,
                latency_ms,
                model_version,
            ),
        )


def recent_events(limit: int = 20, path: Optional[str] = None) -> list[dict]:
    with _conn(path) as c:
        rows = c.execute(
            "SELECT * FROM predictions ORDER BY ts DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


def window_events(window: int = 500, path: Optional[str] = None) -> list[dict]:
    with _conn(path) as c:
        rows = c.execute(
            "SELECT * FROM predictions ORDER BY ts DESC LIMIT ?", (window,)
        ).fetchall()
        return [dict(r) for r in rows]
