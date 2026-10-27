"""UrgeQueue tests — pop_for_interaction batching and ordering.

The heavy imports (maturin-built extension, agent package __init__ with
CompanionService) are neutered INSIDE a module-scoped fixture rather
than at import time, and everything the fixture added is purged from
sys.modules on teardown — so no other test module in the same pytest
session can ever observe the stubs, regardless of collection order.
"""

import pathlib
import sqlite3
import sys
import types
from types import SimpleNamespace

import pytest

_AGENT_DIR = pathlib.Path(__file__).resolve().parents[1] / "src" / "yantrikdb" / "agent"


class _RustStub(types.ModuleType):
    """Resolves ANY name imported from the compiled extension."""

    def __getattr__(self, name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        return object


@pytest.fixture(scope="module")
def agent():
    """Import yantrikdb.agent.{config,instincts.protocol,urges}.

    If the compiled extension is available (CI python job), the real
    `yantrikdb` import is used and only the heavy agent __init__ is
    bypassed. Otherwise the extension is stubbed for the duration of
    this module's tests. Teardown purges every `yantrikdb*` entry from
    sys.modules so later imports always see the real thing.
    """
    import importlib

    try:
        import yantrikdb  # noqa: F401 — real import when the extension exists
    except Exception:
        sys.modules["yantrikdb._yantrikdb_rust"] = _RustStub(
            "yantrikdb._yantrikdb_rust"
        )
        import yantrikdb  # noqa: F401

    if "yantrikdb.agent" not in sys.modules:
        pkg = types.ModuleType("yantrikdb.agent")
        pkg.__path__ = [str(_AGENT_DIR)]
        sys.modules["yantrikdb.agent"] = pkg

    yield SimpleNamespace(
        urges=importlib.import_module("yantrikdb.agent.urges"),
        config=importlib.import_module("yantrikdb.agent.config"),
        protocol=importlib.import_module("yantrikdb.agent.instincts.protocol"),
    )

    for key in [k for k in list(sys.modules) if k.startswith("yantrikdb")]:
        del sys.modules[key]


def _queue(agent):
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    return agent.urges.UrgeQueue(conn, agent.config.UrgeConfig()), conn


def _spec(agent, name: str, urgency: float, cooldown_key: str = ""):
    return agent.protocol.UrgeSpec(
        instinct_name=name,
        reason=f"why {name}",
        urgency=urgency,
        cooldown_key=cooldown_key or f"key-{name}",
    )


def test_pop_returns_top_by_urgency_and_marks_delivered(agent):
    q, conn = _queue(agent)
    q.push(_spec(agent, "low", 0.2))
    q.push(_spec(agent, "high", 0.9))
    q.push(_spec(agent, "mid", 0.5))

    popped = q.pop_for_interaction(limit=2)
    assert [p["instinct_name"] for p in popped] == ["high", "mid"]

    statuses = dict(
        conn.execute("SELECT instinct_name, status FROM urges").fetchall()
    )
    assert statuses == {"high": "delivered", "mid": "delivered", "low": "pending"}
    # Every delivered row got a timestamp.
    delivered = conn.execute(
        "SELECT COUNT(*) AS c FROM urges "
        "WHERE status = 'delivered' AND delivered_at IS NOT NULL"
    ).fetchone()["c"]
    assert delivered == 2


def test_pop_respects_limit_and_empties(agent):
    q, _ = _queue(agent)
    q.push(_spec(agent, "a", 0.3))
    q.push(_spec(agent, "b", 0.4))

    first = q.pop_for_interaction(limit=1)
    assert [p["instinct_name"] for p in first] == ["b"]

    second = q.pop_for_interaction(limit=1)
    assert [p["instinct_name"] for p in second] == ["a"]

    assert q.pop_for_interaction(limit=1) == []


def test_pop_on_empty_queue_returns_empty(agent):
    q, conn = _queue(agent)
    assert q.pop_for_interaction() == []
    # No phantom delivered rows.
    assert conn.execute("SELECT COUNT(*) AS c FROM urges").fetchone()["c"] == 0


def test_pop_marks_only_the_popped_rows_not_their_instinct(agent):
    """The batched UPDATE must key on urge_id. Production instincts
    (follow_up, reminder, pattern_surfacing) keep several pending urges
    sharing one instinct_name — keying the UPDATE by instinct_name (or
    any other coincidentally-unique fixture column) would mark strangers
    delivered."""
    q, conn = _queue(agent)
    # Same instinct_name, different cooldown keys (so push does not
    # boost-merge them), different urgencies for a deterministic pick.
    q.push(_spec(agent, "reminder", 0.9, cooldown_key="task-1"))
    q.push(_spec(agent, "reminder", 0.4, cooldown_key="task-2"))

    popped = q.pop_for_interaction(limit=1)
    assert len(popped) == 1
    assert popped[0]["cooldown_key"] == "task-1"

    rows = conn.execute(
        "SELECT cooldown_key, status FROM urges ORDER BY cooldown_key"
    ).fetchall()
    assert dict(rows) == {"task-1": "delivered", "task-2": "pending"}
