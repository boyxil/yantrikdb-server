"""Urge queue — persistent priority queue backed by SQLite."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional

import uuid

from yantrikdb.agent.config import UrgeConfig
from yantrikdb.agent.instincts.protocol import UrgeSpec

log = logging.getLogger("yantrik.urges")

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS urges (
    urge_id TEXT PRIMARY KEY,
    instinct_name TEXT NOT NULL,
    reason TEXT NOT NULL,
    urgency REAL NOT NULL,
    suggested_message TEXT DEFAULT '',
    action TEXT,
    context TEXT DEFAULT '{}',
    cooldown_key TEXT NOT NULL,
    status TEXT DEFAULT 'pending',
    created_at REAL NOT NULL,
    delivered_at REAL,
    expires_at REAL,
    boost_count INTEGER DEFAULT 0
)
"""

_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_urges_status ON urges(status)",
    "CREATE INDEX IF NOT EXISTS idx_urges_cooldown ON urges(cooldown_key, status)",
    "CREATE INDEX IF NOT EXISTS idx_urges_urgency ON urges(status, urgency DESC)",
]


class UrgeQueue:
    """Persistent urge queue using YantrikDB's SQLite connection."""

    def __init__(self, db_conn, config: UrgeConfig):
        self._conn = db_conn
        self._config = config
        self._ensure_table()

    def _ensure_table(self):
        cursor = self._conn.execute(_CREATE_TABLE)
        for idx_sql in _CREATE_INDEXES:
            self._conn.execute(idx_sql)

    def push(self, spec: UrgeSpec) -> Optional[str]:
        """Add an urge. Returns urge_id, or None if boosted an existing one."""
        # Check for active urge with same cooldown key
        existing = self._conn.execute(
            "SELECT urge_id, urgency, boost_count FROM urges "
            "WHERE cooldown_key = ? AND status IN ('pending', 'delivered') "
            "ORDER BY created_at DESC LIMIT 1",
            (spec.cooldown_key,),
        ).fetchone()

        if existing:
            new_urgency = min(1.0, existing["urgency"] + self._config.boost_increment)
            self._conn.execute(
                "UPDATE urges SET urgency = ?, boost_count = boost_count + 1 WHERE urge_id = ?",
                (new_urgency, existing["urge_id"]),
            )
            log.debug("Boosted urge %s to %.2f", existing["urge_id"], new_urgency)
            return None

        # Check max pending
        count = self._conn.execute(
            "SELECT COUNT(*) as c FROM urges WHERE status = 'pending'"
        ).fetchone()["c"]
        if count >= self._config.max_pending:
            # Expire lowest urgency pending urge
            self._conn.execute(
                "UPDATE urges SET status = 'expired' WHERE urge_id = ("
                "  SELECT urge_id FROM urges WHERE status = 'pending' "
                "  ORDER BY urgency ASC LIMIT 1"
                ")"
            )

        urge_id = str(uuid.uuid4())
        now = time.time()
        expires_at = now + self._config.expiry_hours * 3600

        self._conn.execute(
            "INSERT INTO urges (urge_id, instinct_name, reason, urgency, "
            "suggested_message, action, context, cooldown_key, status, "
            "created_at, expires_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)",
            (
                urge_id, spec.instinct_name, spec.reason, spec.urgency,
                spec.suggested_message, spec.action,
                json.dumps(spec.context), spec.cooldown_key,
                now, expires_at,
            ),
        )
        log.info("New urge %s from %s: %s (urgency=%.2f)",
                 urge_id, spec.instinct_name, spec.reason[:60], spec.urgency)
        return urge_id

    def pop_for_interaction(self, limit: int = 2) -> list[dict]:
        """Get top urges by urgency for the next interaction. Marks as delivered."""
        rows = self._conn.execute(
            "SELECT * FROM urges WHERE status = 'pending' "
            "ORDER BY urgency DESC LIMIT ?",
            (limit,),
        ).fetchall()
        if not rows:
            return []

        # One UPDATE for the whole batch instead of one per row — the
        # row set is already fixed by the SELECT above, so the loop
        # only multiplied round-trips.
        now = time.time()
        ids = [row["urge_id"] for row in rows]
        placeholders = ",".join("?" * len(ids))
        self._conn.execute(
            f"UPDATE urges SET status = 'delivered', delivered_at = ? "
            f"WHERE urge_id IN ({placeholders})",
            (now, *ids),
        )
        return [dict(row) for row in rows]

    def pop_due(self, threshold: float = 0.0) -> list[dict]:
        """Get all pending urges above urgency threshold."""
        rows = self._conn.execute(
            "SELECT * FROM urges WHERE status = 'pending' AND urgency >= ? "
            "ORDER BY urgency DESC",
            (threshold,),
        ).fetchall()
        return [dict(r) for r in rows]

    def suppress(self, urge_id: str) -> bool:
        cursor = self._conn.execute(
            "UPDATE urges SET status = 'suppressed' WHERE urge_id = ? AND status IN ('pending', 'delivered')",
            (urge_id,),
        )
        return cursor.rowcount > 0

    def expire_old(self) -> int:
        now = time.time()
        cursor = self._conn.execute(
            "UPDATE urges SET status = 'expired' WHERE status = 'pending' AND expires_at < ?",
            (now,),
        )
        if cursor.rowcount:
            log.info("Expired %d old urges", cursor.rowcount)
        return cursor.rowcount

    def get_pending(self, limit: int = 20) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM urges WHERE status = 'pending' ORDER BY urgency DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def count_pending(self) -> int:
        return self._conn.execute(
            "SELECT COUNT(*) as c FROM urges WHERE status = 'pending'"
        ).fetchone()["c"]
