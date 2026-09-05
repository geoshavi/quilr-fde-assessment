"""Task 4 -- TokenRateLimiter: per-tenant sliding-window token admission.

Backed by an on-disk SQLite file (never ':memory:' -- state must persist
across separate TokenRateLimiter instances/processes pointed at the same
path). Concurrency safety comes from SQLite's own locking, not an
in-process asyncio.Lock: every admission opens its own short-lived
connection and takes an immediate write lock (`BEGIN IMMEDIATE`) around a
single read-modify-write transaction, so two limiter instances (or many
concurrent asyncio tasks sharing one instance) racing to admit against the
same tenant are correctly serialized by SQLite itself -- an in-process
lock could never protect a second instance/connection.

Rolling-window boundary (assessment decision, documented explicitly):
the window is the half-open interval (now - window_seconds, now]. An
event timestamped exactly `now - window_seconds` has already expired and
does not count toward current usage.

Token-accounting contract (assessment decision, documented explicitly):
no tokenizer is implemented or assumed. `requested_tokens` is a
caller-supplied, deterministic positive integer representing the full
cost of the request; it is reserved (committed to the ledger) atomically
at admission time, before any provider is ever called.

Reservation/refund contract (assessment decision, documented explicitly):
once a request is admitted, its reserved tokens remain charged for the
rest of the rolling window even if the eventual provider call fails, the
fallback also fails, or the caller ultimately receives an error. No
refund or reconciliation is performed -- the assessment specifies no
usage-based (post-hoc) accounting, and admission is explicitly required
to happen before any provider is contacted.
"""

from __future__ import annotations

import asyncio
import os
import sqlite3
import time
from typing import Callable

RATE_LIMIT_WINDOW_SECONDS = float(os.environ.get("TASK4_RATE_LIMIT_WINDOW_SECONDS", "60"))
RATE_LIMIT_MAX_TOKENS = int(os.environ.get("TASK4_RATE_LIMIT_MAX_TOKENS", "50000"))
DEFAULT_DB_PATH = os.environ.get("TASK4_RATE_LIMIT_DB_PATH", "./task4_model_router/rate_limit.db")

# How long a connection waits for another writer's BEGIN IMMEDIATE to
# release before raising "database is locked", instead of failing instantly.
_BUSY_TIMEOUT_SECONDS = 5.0

_SCHEMA = """
CREATE TABLE IF NOT EXISTS usage (
    tenant_key TEXT NOT NULL,
    ts REAL NOT NULL,
    tokens INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_usage_tenant_ts ON usage (tenant_key, ts);
"""


def validate_admission_request(tenant_key: object, requested_tokens: object) -> None:
    """Raise ValueError for any input that must never reach SQLite or a provider.

    `bool` is deliberately rejected even though `isinstance(True, int)` is
    True in Python -- otherwise `True`/`False` would silently be accepted
    as token counts of 1/0, a real accounting bypass.
    """
    if not isinstance(tenant_key, str) or not tenant_key:
        raise ValueError("tenant_key must be a non-empty string")
    if isinstance(requested_tokens, bool) or not isinstance(requested_tokens, int):
        raise ValueError("requested_tokens must be an int")
    if requested_tokens <= 0:
        raise ValueError("requested_tokens must be > 0")


class TokenRateLimiter:
    """One rolling-window token ledger, shared across instances via `db_path`."""

    def __init__(
        self,
        db_path: str = DEFAULT_DB_PATH,
        *,
        max_tokens: int = RATE_LIMIT_MAX_TOKENS,
        window_seconds: float = RATE_LIMIT_WINDOW_SECONDS,
        now: Callable[[], float] | None = None,
    ) -> None:
        self._db_path = db_path
        self._max_tokens = max_tokens
        self._window_seconds = window_seconds
        self._now = now if now is not None else time.time
        conn = sqlite3.connect(self._db_path, timeout=_BUSY_TIMEOUT_SECONDS, isolation_level=None)
        try:
            conn.executescript(_SCHEMA)
        finally:
            conn.close()

    async def admit(self, tenant_key: str, requested_tokens: int) -> bool:
        """Atomically admit or reject `requested_tokens` for `tenant_key`.

        Returns True (and reserves the tokens) if the tenant's rolling-window
        usage, including this request, would not exceed `max_tokens`; False
        (reserving nothing) otherwise. Raises ValueError for invalid input,
        validated up front so it never reaches SQLite.
        """
        validate_admission_request(tenant_key, requested_tokens)
        return await asyncio.to_thread(self._admit_sync, tenant_key, requested_tokens)

    def _admit_sync(self, tenant_key: str, requested_tokens: int) -> bool:
        now = self._now()
        cutoff = now - self._window_seconds
        conn = sqlite3.connect(self._db_path, timeout=_BUSY_TIMEOUT_SECONDS, isolation_level=None)
        try:
            conn.execute("BEGIN IMMEDIATE")
            # Half-open window (cutoff, now]: a row timestamped exactly at
            # cutoff (now - window_seconds) has already expired.
            conn.execute("DELETE FROM usage WHERE tenant_key = ? AND ts <= ?", (tenant_key, cutoff))
            row = conn.execute(
                "SELECT COALESCE(SUM(tokens), 0) FROM usage WHERE tenant_key = ? AND ts > ?",
                (tenant_key, cutoff),
            ).fetchone()
            current_usage = row[0]
            if current_usage + requested_tokens > self._max_tokens:
                conn.execute("COMMIT")
                return False
            conn.execute(
                "INSERT INTO usage (tenant_key, ts, tokens) VALUES (?, ?, ?)",
                (tenant_key, now, requested_tokens),
            )
            conn.execute("COMMIT")
            return True
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

    async def current_usage(self, tenant_key: str) -> int:
        """Observe a tenant's current (non-expired) usage without reserving anything."""
        return await asyncio.to_thread(self._current_usage_sync, tenant_key)

    def _current_usage_sync(self, tenant_key: str) -> int:
        now = self._now()
        cutoff = now - self._window_seconds
        conn = sqlite3.connect(self._db_path, timeout=_BUSY_TIMEOUT_SECONDS, isolation_level=None)
        try:
            row = conn.execute(
                "SELECT COALESCE(SUM(tokens), 0) FROM usage WHERE tenant_key = ? AND ts > ?",
                (tenant_key, cutoff),
            ).fetchone()
            return row[0]
        finally:
            conn.close()
