"""UrgeQueue tests — pop_for_interaction batching and ordering."""

import sqlite3
import sys
import types

# Unit scope: the SQLite-backed urge queue only. Two package __init__s
# pull far more than this test needs:
#   * `yantrikdb/__init__` imports the compiled extension (built by
#     maturin, absent in a plain pytest env) — stub it; __getattr__
#     resolves ANY imported name to a placeholder.
#   * `yantrikdb/agent/__init__` imports CompanionService and with it
#     the whole LLM/service stack — replace the package module with a
#     bare one carrying the real __path__, so `yantrikdb.agent.config`
#     and friends import from disk without running the heavy __init__.
import pathlib


class _RustStub(types.ModuleType):
    def __getattr__(self, name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        return object


sys.modules.setdefault(
    "yantrikdb._yantrikdb_rust", _RustStub("yantrikdb._yantrikdb_rust")
)

_agent_pkg = types.ModuleType("yantrikdb.agent")
_agent_pkg.__path__ = [
    str(pathlib.Path(__file__).resolve().parents[1] / "src" / "yantrikdb" / "agent")
]
sys.modules.setdefault("yantrikdb.agent", _agent_pkg)

from yantrikdb.agent.config import UrgeConfig
from yantrikdb.agent.instincts.protocol import UrgeSpec
from yantrikdb.agent.urges import UrgeQueue


def _queue() -> tuple[UrgeQueue, sqlite3.Connection]:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    return UrgeQueue(conn, UrgeConfig()), conn


def _spec(name: str, urgency: float, cooldown_key: str = "") -> UrgeSpec:
    return UrgeSpec(
        instinct_name=name,
        reason=f"why {name}",
        urgency=urgency,
        cooldown_key=cooldown_key or f"key-{name}",
    )


def test_pop_returns_top_by_urgency_and_marks_delivered():
    q, conn = _queue()
    q.push(_spec("low", 0.2))
    q.push(_spec("high", 0.9))
    q.push(_spec("mid", 0.5))

    popped = q.pop_for_interaction(limit=2)
    assert [p["instinct_name"] for p in popped] == ["high", "mid"]

    statuses = dict(
        conn.execute(
            "SELECT instinct_name, status FROM urges"
        ).fetchall()
    )
    assert statuses == {"high": "delivered", "mid": "delivered", "low": "pending"}
    # Every delivered row got a timestamp.
    delivered = conn.execute(
        "SELECT COUNT(*) AS c FROM urges "
        "WHERE status = 'delivered' AND delivered_at IS NOT NULL"
    ).fetchone()["c"]
    assert delivered == 2


def test_pop_respects_limit_and_empties():
    q, _ = _queue()
    q.push(_spec("a", 0.3))
    q.push(_spec("b", 0.4))

    first = q.pop_for_interaction(limit=1)
    assert [p["instinct_name"] for p in first] == ["b"]

    second = q.pop_for_interaction(limit=1)
    assert [p["instinct_name"] for p in second] == ["a"]

    assert q.pop_for_interaction(limit=1) == []


def test_pop_on_empty_queue_returns_empty():
    q, conn = _queue()
    assert q.pop_for_interaction() == []
    # No phantom delivered rows.
    assert conn.execute("SELECT COUNT(*) AS c FROM urges").fetchone()["c"] == 0
