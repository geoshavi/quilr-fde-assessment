"""Task 4 rate limiter tests.

TokenRateLimiter is tested directly (no ModelRouter involved) against a
real on-disk SQLite file under `tmp_path` -- never ':memory:', since
persistence-across-instances and true multi-connection concurrency are
both explicit requirements. A small injectable clock replaces
`time.time` so rolling-window expiration is tested deterministically,
with no real sleeps.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from task4_model_router.rate_limiter import TokenRateLimiter, validate_admission_request


class FakeClock:
    """A mutable injectable clock: calling it returns `.t`, advanced by tests."""

    def __init__(self, start: float = 1_000_000.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t


def _db_path(tmp_path: Path) -> str:
    return str(tmp_path / "rate_limit.db")


# Tests that exist specifically to prove the assessment's 50,000-token /
# 60-second contract pin those values explicitly, rather than inheriting
# whatever TASK4_RATE_LIMIT_MAX_TOKENS / TASK4_RATE_LIMIT_WINDOW_SECONDS the
# ambient environment happens to set -- otherwise these tests would silently
# stop proving the assessment requirement under a populated .env.
REQUIRED_MAX_TOKENS = 50_000
REQUIRED_WINDOW_SECONDS = 60


def _new_limiter(tmp_path: Path, *, now: FakeClock | None = None, db_name: str = "rate_limit.db") -> TokenRateLimiter:
    return TokenRateLimiter(
        str(tmp_path / db_name),
        max_tokens=REQUIRED_MAX_TOKENS,
        window_seconds=REQUIRED_WINDOW_SECONDS,
        now=now if now is not None else FakeClock(),
    )


# ---------------------------------------------------------------------------
# 1-3: boundary admission at exactly 50,000 tokens
# ---------------------------------------------------------------------------


async def test_49999_tokens_allowed(tmp_path: Path) -> None:
    limiter = _new_limiter(tmp_path)
    assert await limiter.admit("tenant-a", 49_999) is True


async def test_exactly_50000_tokens_allowed(tmp_path: Path) -> None:
    limiter = _new_limiter(tmp_path)
    assert await limiter.admit("tenant-a", 50_000) is True


async def test_50001_tokens_rejected(tmp_path: Path) -> None:
    limiter = _new_limiter(tmp_path)
    assert await limiter.admit("tenant-a", 50_001) is False


async def test_admission_accumulates_toward_the_same_cap(tmp_path: Path) -> None:
    limiter = _new_limiter(tmp_path)
    assert await limiter.admit("tenant-a", 30_000) is True
    assert await limiter.admit("tenant-a", 20_000) is True  # exactly fills to 50,000
    assert await limiter.admit("tenant-a", 1) is False  # any further request now rejected


# ---------------------------------------------------------------------------
# 4-5: rolling-window expiration, exact half-open boundary
# ---------------------------------------------------------------------------


async def test_rolling_window_expiration_frees_capacity(tmp_path: Path) -> None:
    clock = FakeClock(1_000.0)
    limiter = _new_limiter(tmp_path, now=clock)
    assert await limiter.admit("tenant-a", 50_000) is True
    assert await limiter.admit("tenant-a", 1) is False  # window still full

    clock.t = 1_000.0 + 61  # fully past the 60s window
    assert await limiter.admit("tenant-a", 50_000) is True  # old usage has expired


async def test_usage_exactly_at_window_boundary_is_expired(tmp_path: Path) -> None:
    """Half-open window (now - window, now]: a timestamp of exactly
    `now - window_seconds` has already expired and does not count.
    """
    clock = FakeClock(1_000.0)
    limiter = TokenRateLimiter(_db_path(tmp_path), now=clock, window_seconds=60, max_tokens=100)
    assert await limiter.admit("tenant-a", 100) is True  # usage recorded at t=1000.0

    clock.t = 1_060.0  # now - 60 == 1000.0 == the old event's timestamp, exactly
    assert await limiter.admit("tenant-a", 100) is True  # only succeeds if 1000.0 is expired


async def test_usage_one_instant_before_boundary_still_counts(tmp_path: Path) -> None:
    """Mirror case: a timestamp fractionally inside the window still counts."""
    clock = FakeClock(1_000.0)
    limiter = TokenRateLimiter(_db_path(tmp_path), now=clock, window_seconds=60, max_tokens=100)
    assert await limiter.admit("tenant-a", 100) is True

    clock.t = 1_059.999999
    assert await limiter.admit("tenant-a", 1) is False  # old usage still counts, no room left


# ---------------------------------------------------------------------------
# 6: independent tenants
# ---------------------------------------------------------------------------


async def test_tenants_have_independent_limits(tmp_path: Path) -> None:
    limiter = _new_limiter(tmp_path)
    assert await limiter.admit("tenant-a", 50_000) is True
    assert await limiter.admit("tenant-a", 1) is False
    assert await limiter.admit("tenant-b", 50_000) is True  # unaffected by tenant-a


# ---------------------------------------------------------------------------
# 7-8: concurrency invariant -- proves atomicity, not just completion
# ---------------------------------------------------------------------------


async def test_concurrent_admissions_never_exceed_cap(tmp_path: Path) -> None:
    """20 concurrent contenders x 10,000 tokens against a 50,000 cap: exactly
    5 must be admitted and 15 rejected. A non-atomic check-then-insert would
    admit more than 5 under this contention (each reads a stale usage count
    before any insert lands), so this fails loudly under that bug rather
    than merely proving the calls completed.
    """
    limiter = _new_limiter(tmp_path)
    results = await asyncio.gather(*(limiter.admit("tenant-a", 10_000) for _ in range(20)))
    assert sum(results) == 5
    assert await limiter.current_usage("tenant-a") == 50_000


async def test_concurrent_admissions_across_two_limiter_instances(tmp_path: Path) -> None:
    """Two separate TokenRateLimiter instances (separate connections, separate
    Python objects) racing against the same on-disk DB file must still be
    serialized correctly -- proof that SQLite itself enforces the invariant,
    not an in-process asyncio.Lock, which cannot protect a second instance.
    """
    path = _db_path(tmp_path)
    limiter_a = TokenRateLimiter(path, max_tokens=REQUIRED_MAX_TOKENS, window_seconds=REQUIRED_WINDOW_SECONDS, now=FakeClock())
    limiter_b = TokenRateLimiter(path, max_tokens=REQUIRED_MAX_TOKENS, window_seconds=REQUIRED_WINDOW_SECONDS, now=FakeClock())
    calls = [
        (limiter_a if i % 2 == 0 else limiter_b).admit("tenant-a", 10_000) for i in range(20)
    ]
    results = await asyncio.gather(*calls)
    assert sum(results) == 5
    assert await limiter_a.current_usage("tenant-a") == 50_000


# ---------------------------------------------------------------------------
# 9: persistence across separate instances
# ---------------------------------------------------------------------------


async def test_usage_persists_across_separate_limiter_instances(tmp_path: Path) -> None:
    path = _db_path(tmp_path)
    clock = FakeClock(2_000.0)
    limiter_1 = TokenRateLimiter(path, max_tokens=REQUIRED_MAX_TOKENS, window_seconds=REQUIRED_WINDOW_SECONDS, now=clock)
    assert await limiter_1.admit("tenant-a", 50_000) is True

    limiter_2 = TokenRateLimiter(path, max_tokens=REQUIRED_MAX_TOKENS, window_seconds=REQUIRED_WINDOW_SECONDS, now=clock)  # brand-new instance, same DB file
    assert await limiter_2.admit("tenant-a", 1) is False  # sees limiter_1's committed usage


# ---------------------------------------------------------------------------
# 10-13: input validation -- invalid input never reaches SQLite
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_tokens", [0, -1, -50_000])
async def test_non_positive_token_count_rejected(tmp_path: Path, bad_tokens: int) -> None:
    limiter = TokenRateLimiter(_db_path(tmp_path), now=FakeClock())
    with pytest.raises(ValueError):
        await limiter.admit("tenant-a", bad_tokens)
    assert await limiter.current_usage("tenant-a") == 0


@pytest.mark.parametrize("bad_tokens", [True, False, 1.5, "50000", None])
async def test_non_integer_token_count_rejected(tmp_path: Path, bad_tokens: object) -> None:
    """A bool must not silently count as an int, even though
    `isinstance(True, int)` is True in Python -- a real accounting bypass
    if left unchecked.
    """
    limiter = TokenRateLimiter(_db_path(tmp_path), now=FakeClock())
    with pytest.raises(ValueError):
        await limiter.admit("tenant-a", bad_tokens)  # type: ignore[arg-type]
    assert await limiter.current_usage("tenant-a") == 0


@pytest.mark.parametrize("bad_tenant", ["", None, 123])
async def test_invalid_tenant_key_rejected(tmp_path: Path, bad_tenant: object) -> None:
    limiter = TokenRateLimiter(_db_path(tmp_path), now=FakeClock())
    with pytest.raises(ValueError):
        await limiter.admit(bad_tenant, 1_000)  # type: ignore[arg-type]


def test_validate_admission_request_rejects_bool_tokens_directly() -> None:
    with pytest.raises(ValueError):
        validate_admission_request("tenant-a", True)


def test_validate_admission_request_accepts_well_formed_input() -> None:
    validate_admission_request("tenant-a", 1)  # must not raise
