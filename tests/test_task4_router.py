"""Task 4 router tests.

ModelRouter integrates rate-limit admission with primary/secondary
provider routing behind one entry point. Providers are plain injectable
async callables from `providers_mock.MockProvider` -- no real network
calls, no FastAPI/ASGI transport needed, since the provider contract here
is just `async (prompt, requested_tokens) -> ProviderResponse`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from task4_model_router.errors import ErrorCode, RouterError, to_response
from task4_model_router.providers_mock import MockProvider
from task4_model_router.rate_limiter import TokenRateLimiter
from task4_model_router.router import ModelRouter, ProviderResponse, _default_primary_timeout_ms

TENANT = "tenant-a"

# Tests use a tiny effective deadline so a "timeout" scenario resolves in a
# fraction of a second, never a real 3-second sleep. The production/default
# behavior (no override passed) remains exactly TASK4_PRIMARY_TIMEOUT_MS.
# The margin between this and the "finishes before deadline" delay below is
# kept wide (30x) to stay robust against scheduler/timer jitter in CI/VM
# environments rather than racing a razor-thin boundary.
TEST_TIMEOUT_MS = 300


def _limiter(tmp_path: Path, max_tokens: int = 50_000) -> TokenRateLimiter:
    return TokenRateLimiter(str(tmp_path / "router.db"), max_tokens=max_tokens)


# ---------------------------------------------------------------------------
# 14: rate-limit rejection calls no provider at all
# ---------------------------------------------------------------------------


async def test_rate_limit_rejection_calls_no_provider(tmp_path: Path) -> None:
    limiter = _limiter(tmp_path, max_tokens=100)
    primary = MockProvider()
    secondary = MockProvider()
    router = ModelRouter(limiter, primary, secondary, primary_timeout_ms=TEST_TIMEOUT_MS)

    with pytest.raises(RouterError) as exc_info:
        await router.generate(TENANT, "prompt", 1_000)  # exceeds the 100-token cap
    assert exc_info.value.code == ErrorCode.RATE_LIMITED
    assert primary.call_count == 0
    assert secondary.call_count == 0


async def test_rate_limiter_internal_failure_is_sanitized_and_calls_no_provider(tmp_path: Path) -> None:
    """An unexpected failure inside the rate limiter itself (e.g. a real
    SQLite I/O error) must map to a sanitized INTERNAL_ERROR, never leak the
    raw exception (which could name the on-disk DB path), and never reach
    either provider.
    """

    class _BrokenLimiter:
        async def admit(self, tenant_key: str, requested_tokens: int) -> bool:
            raise OSError(f"disk I/O error opening {tmp_path / 'super-secret.db'}")

    primary = MockProvider()
    secondary = MockProvider()
    router = ModelRouter(_BrokenLimiter(), primary, secondary, primary_timeout_ms=TEST_TIMEOUT_MS)

    with pytest.raises(RouterError) as exc_info:
        await router.generate(TENANT, "prompt", 100)
    assert exc_info.value.code == ErrorCode.INTERNAL_ERROR
    assert primary.call_count == 0
    assert secondary.call_count == 0

    serialized = json.dumps(to_response(exc_info.value.code))
    assert "super-secret.db" not in serialized
    assert str(tmp_path) not in serialized


# ---------------------------------------------------------------------------
# 15/18: primary success / finishes before deadline -- secondary never called
# ---------------------------------------------------------------------------


async def test_primary_success_secondary_never_called(tmp_path: Path) -> None:
    limiter = _limiter(tmp_path)
    primary = MockProvider(status_code=200, body="primary-ok")
    secondary = MockProvider()
    router = ModelRouter(limiter, primary, secondary, primary_timeout_ms=TEST_TIMEOUT_MS)

    response = await router.generate(TENANT, "prompt", 100)
    assert response.body == "primary-ok"
    assert primary.call_count == 1
    assert secondary.call_count == 0


async def test_primary_finishes_before_deadline_secondary_not_called(tmp_path: Path) -> None:
    limiter = _limiter(tmp_path)
    primary = MockProvider(status_code=200, delay_seconds=0.01)  # well under TEST_TIMEOUT_MS
    secondary = MockProvider()
    router = ModelRouter(limiter, primary, secondary, primary_timeout_ms=TEST_TIMEOUT_MS)

    await router.generate(TENANT, "prompt", 100)
    assert secondary.call_count == 0


# ---------------------------------------------------------------------------
# 16/23: primary 429 -- secondary called exactly once, its result returned
# ---------------------------------------------------------------------------


async def test_primary_429_falls_back_to_secondary_once(tmp_path: Path) -> None:
    limiter = _limiter(tmp_path)
    call_log: list[str] = []
    primary = MockProvider(status_code=429, name="primary", call_log=call_log)
    secondary = MockProvider(status_code=200, body="secondary-ok", name="secondary", call_log=call_log)
    router = ModelRouter(limiter, primary, secondary, primary_timeout_ms=TEST_TIMEOUT_MS)

    response = await router.generate(TENANT, "prompt", 100)
    assert response.body == "secondary-ok"
    assert primary.call_count == 1
    assert secondary.call_count == 1
    assert call_log == ["primary", "secondary"]


# ---------------------------------------------------------------------------
# 17/24: primary exceeds the router deadline -- secondary called exactly once
# ---------------------------------------------------------------------------


async def test_primary_timeout_falls_back_to_secondary_once(tmp_path: Path) -> None:
    limiter = _limiter(tmp_path)
    primary = MockProvider(status_code=200, delay_seconds=1.0)  # far past TEST_TIMEOUT_MS
    secondary = MockProvider(status_code=200, body="secondary-ok")
    router = ModelRouter(limiter, primary, secondary, primary_timeout_ms=TEST_TIMEOUT_MS)

    response = await router.generate(TENANT, "prompt", 100)
    assert response.body == "secondary-ok"
    assert primary.call_count == 1
    assert secondary.call_count == 1


def test_documented_default_primary_timeout_is_3000ms(monkeypatch: pytest.MonkeyPatch) -> None:
    """The documented production default (TASK4_PRIMARY_TIMEOUT_MS unset) is
    exactly 3000ms -- proven independent of whatever TASK4_PRIMARY_TIMEOUT_MS
    happens to be set to in the ambient environment, without reloading the
    already-imported router module (which would desync RouterError/ModelRouter
    identity from names other tests already imported) and without weakening
    the real env-override support in production code.
    """
    monkeypatch.delenv("TASK4_PRIMARY_TIMEOUT_MS", raising=False)
    assert _default_primary_timeout_ms() == 3000


# ---------------------------------------------------------------------------
# 19/20: primary 500 / other 4xx -- no fallback
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status", [500, 502, 503])
async def test_primary_5xx_no_fallback(tmp_path: Path, status: int) -> None:
    limiter = _limiter(tmp_path)
    primary = MockProvider(status_code=status)
    secondary = MockProvider()
    router = ModelRouter(limiter, primary, secondary, primary_timeout_ms=TEST_TIMEOUT_MS)

    with pytest.raises(RouterError) as exc_info:
        await router.generate(TENANT, "prompt", 100)
    assert exc_info.value.code == ErrorCode.PROVIDER_ERROR
    assert secondary.call_count == 0


@pytest.mark.parametrize("status", [400, 401, 403, 404])
async def test_primary_non_429_4xx_no_fallback(tmp_path: Path, status: int) -> None:
    limiter = _limiter(tmp_path)
    primary = MockProvider(status_code=status)
    secondary = MockProvider()
    router = ModelRouter(limiter, primary, secondary, primary_timeout_ms=TEST_TIMEOUT_MS)

    with pytest.raises(RouterError) as exc_info:
        await router.generate(TENANT, "prompt", 100)
    assert exc_info.value.code == ErrorCode.PROVIDER_ERROR
    assert secondary.call_count == 0


# ---------------------------------------------------------------------------
# 21: arbitrary primary exception -- no fallback
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("exc", [RuntimeError("boom"), ValueError("bad"), ConnectionError("down")])
async def test_primary_arbitrary_exception_no_fallback(tmp_path: Path, exc: BaseException) -> None:
    limiter = _limiter(tmp_path)
    primary = MockProvider(exception=exc)
    secondary = MockProvider()
    router = ModelRouter(limiter, primary, secondary, primary_timeout_ms=TEST_TIMEOUT_MS)

    with pytest.raises(RouterError) as exc_info:
        await router.generate(TENANT, "prompt", 100)
    assert exc_info.value.code == ErrorCode.PROVIDER_ERROR
    assert secondary.call_count == 0


async def test_primary_malformed_response_no_fallback(tmp_path: Path) -> None:
    """A provider callable that returns something without `.status_code`
    (e.g. a bad integration) must be treated as a hard failure, not silently
    routed into fallback.
    """
    limiter = _limiter(tmp_path)

    async def malformed_primary(prompt: str, requested_tokens: int) -> str:
        return "not a ProviderResponse"

    secondary = MockProvider()
    router = ModelRouter(limiter, malformed_primary, secondary, primary_timeout_ms=TEST_TIMEOUT_MS)

    with pytest.raises(RouterError) as exc_info:
        await router.generate(TENANT, "prompt", 100)
    assert exc_info.value.code == ErrorCode.PROVIDER_ERROR
    assert secondary.call_count == 0


# ---------------------------------------------------------------------------
# C/D: non-int status_code on primary must not raise a raw TypeError
# (independent-review findings: status_code="500" and status_code=None)
# ---------------------------------------------------------------------------


async def test_primary_string_status_code_no_raw_typeerror(tmp_path: Path) -> None:
    """Test C: ProviderResponse(status_code="500", ...) must classify as a
    malformed response (PROVIDER_ERROR, no fallback) -- never let a bare
    TypeError from `status_code >= 400` escape to the caller.
    """
    limiter = _limiter(tmp_path)

    class StringStatusResponse:
        status_code = "500"
        body = "secret"

    async def primary(prompt: str, requested_tokens: int) -> StringStatusResponse:
        return StringStatusResponse()

    secondary = MockProvider()
    router = ModelRouter(limiter, primary, secondary, primary_timeout_ms=TEST_TIMEOUT_MS)

    with pytest.raises(RouterError) as exc_info:
        await router.generate(TENANT, "prompt", 100)
    assert exc_info.value.code == ErrorCode.PROVIDER_ERROR
    assert secondary.call_count == 0


async def test_primary_none_status_code_no_raw_typeerror(tmp_path: Path) -> None:
    """Test D: ProviderResponse(status_code=None, ...) must classify as a
    malformed response (PROVIDER_ERROR, no fallback) -- never a raw TypeError.
    """
    limiter = _limiter(tmp_path)

    class NoneStatusResponse:
        status_code = None
        body = "secret"

    async def primary(prompt: str, requested_tokens: int) -> NoneStatusResponse:
        return NoneStatusResponse()

    secondary = MockProvider()
    router = ModelRouter(limiter, primary, secondary, primary_timeout_ms=TEST_TIMEOUT_MS)

    with pytest.raises(RouterError) as exc_info:
        await router.generate(TENANT, "prompt", 100)
    assert exc_info.value.code == ErrorCode.PROVIDER_ERROR
    assert secondary.call_count == 0


async def test_primary_bool_status_code_not_treated_as_int(tmp_path: Path) -> None:
    """`bool` is an `int` subclass in Python -- status_code=True must not be
    silently accepted as status 1 (a "success"). It must classify as
    malformed, exactly like any other non-genuine-int status.
    """
    limiter = _limiter(tmp_path)
    primary = MockProvider(status_code=True)  # type: ignore[arg-type]
    secondary = MockProvider()
    router = ModelRouter(limiter, primary, secondary, primary_timeout_ms=TEST_TIMEOUT_MS)

    with pytest.raises(RouterError) as exc_info:
        await router.generate(TENANT, "prompt", 100)
    assert exc_info.value.code == ErrorCode.PROVIDER_ERROR
    assert secondary.call_count == 0


# ---------------------------------------------------------------------------
# 22: secondary failure after permitted fallback -- sanitized error
# ---------------------------------------------------------------------------


async def test_secondary_failure_after_429_is_sanitized(tmp_path: Path) -> None:
    limiter = _limiter(tmp_path)
    primary = MockProvider(status_code=429)
    secondary = MockProvider(exception=RuntimeError("secondary is down"))
    router = ModelRouter(limiter, primary, secondary, primary_timeout_ms=TEST_TIMEOUT_MS)

    with pytest.raises(RouterError) as exc_info:
        await router.generate(TENANT, "prompt", 100)
    assert exc_info.value.code == ErrorCode.PROVIDER_UNAVAILABLE


async def test_secondary_failure_after_timeout_is_sanitized(tmp_path: Path) -> None:
    limiter = _limiter(tmp_path)
    primary = MockProvider(delay_seconds=1.0)
    secondary = MockProvider(exception=RuntimeError("secondary is down"))
    router = ModelRouter(limiter, primary, secondary, primary_timeout_ms=TEST_TIMEOUT_MS)

    with pytest.raises(RouterError) as exc_info:
        await router.generate(TENANT, "prompt", 100)
    assert exc_info.value.code == ErrorCode.PROVIDER_UNAVAILABLE


# ---------------------------------------------------------------------------
# BLOCKER fix regression tests (independent review): a secondary error
# RESPONSE (not just a raised exception) must never be returned verbatim.
# ---------------------------------------------------------------------------

_LEAKY_SENTINELS = (
    "SECRET_SENTINEL",
    "internal-provider.localhost",
    "rate_limit.db",
)


def _assert_no_leakage(serialized: str) -> None:
    for sentinel in _LEAKY_SENTINELS:
        assert sentinel not in serialized
    assert "Traceback" not in serialized
    assert "ProviderResponse" not in serialized
    assert "RuntimeError" not in serialized
    assert "Exception" not in serialized


async def test_A_secondary_500_response_after_429_is_sanitized_not_forwarded(tmp_path: Path) -> None:
    """Test A: primary 429 -> secondary RETURNS (does not raise) a 500 whose
    body carries a secret + internal URL + DB-path-shaped sentinel. The raw
    ProviderResponse must never reach the caller.
    """
    limiter = _limiter(tmp_path)
    primary = MockProvider(status_code=429, name="primary")
    secondary = MockProvider(
        status_code=500,
        body="SECRET_SENTINEL http://internal-provider.localhost C:\\private\\rate_limit.db",
        name="secondary",
    )
    router = ModelRouter(limiter, primary, secondary, primary_timeout_ms=TEST_TIMEOUT_MS)

    with pytest.raises(RouterError) as exc_info:
        await router.generate(TENANT, "prompt", 100)
    assert exc_info.value.code == ErrorCode.PROVIDER_UNAVAILABLE
    assert secondary.call_count == 1

    serialized = json.dumps(to_response(exc_info.value.code))
    _assert_no_leakage(serialized)
    # Also prove the RouterError object itself (not just to_response) carries
    # nothing forwardable -- a caller serializing str(exc_info.value) is
    # exactly as safe as one calling to_response(exc_info.value.code).
    _assert_no_leakage(str(exc_info.value))


async def test_B_secondary_401_response_after_timeout_is_sanitized_not_forwarded(tmp_path: Path) -> None:
    """Test B: primary times out -> secondary RETURNS a 401 with a
    secret-bearing body. Must not leak."""
    limiter = _limiter(tmp_path)
    primary = MockProvider(delay_seconds=1.0)
    secondary = MockProvider(status_code=401, body="SECRET_SENTINEL bad api key for internal-provider.localhost")
    router = ModelRouter(limiter, primary, secondary, primary_timeout_ms=TEST_TIMEOUT_MS)

    with pytest.raises(RouterError) as exc_info:
        await router.generate(TENANT, "prompt", 100)
    assert exc_info.value.code == ErrorCode.PROVIDER_UNAVAILABLE
    assert secondary.call_count == 1

    serialized = json.dumps(to_response(exc_info.value.code))
    _assert_no_leakage(serialized)


async def test_E_secondary_malformed_response_after_429_is_sanitized(tmp_path: Path) -> None:
    """Test E: secondary returns something with a missing/wrong-typed
    status_code -- must classify as a failure (PROVIDER_UNAVAILABLE), not
    a raw TypeError and not a passthrough success.
    """
    limiter = _limiter(tmp_path)
    primary = MockProvider(status_code=429)

    async def malformed_secondary(prompt: str, requested_tokens: int) -> str:
        return "not a ProviderResponse, no status_code at all"

    router = ModelRouter(limiter, primary, malformed_secondary, primary_timeout_ms=TEST_TIMEOUT_MS)

    with pytest.raises(RouterError) as exc_info:
        await router.generate(TENANT, "prompt", 100)
    assert exc_info.value.code == ErrorCode.PROVIDER_UNAVAILABLE


async def test_secondary_bool_status_code_not_treated_as_int_success(tmp_path: Path) -> None:
    """secondary status_code=True must not be silently treated as a
    successful int status (1) -- must classify as malformed/failed."""
    limiter = _limiter(tmp_path)
    primary = MockProvider(status_code=429)
    secondary = MockProvider(status_code=True)  # type: ignore[arg-type]
    router = ModelRouter(limiter, primary, secondary, primary_timeout_ms=TEST_TIMEOUT_MS)

    with pytest.raises(RouterError) as exc_info:
        await router.generate(TENANT, "prompt", 100)
    assert exc_info.value.code == ErrorCode.PROVIDER_UNAVAILABLE


async def test_secondary_429_response_has_no_tertiary_fallback(tmp_path: Path) -> None:
    """A secondary that ALSO returns 429 is still just a failed fallback --
    there is no tertiary provider to try, so it must sanitize to
    PROVIDER_UNAVAILABLE, not be treated as an admissible status."""
    limiter = _limiter(tmp_path)
    primary = MockProvider(status_code=429)
    secondary = MockProvider(status_code=429, body="rate limited upstream too")
    router = ModelRouter(limiter, primary, secondary, primary_timeout_ms=TEST_TIMEOUT_MS)

    with pytest.raises(RouterError) as exc_info:
        await router.generate(TENANT, "prompt", 100)
    assert exc_info.value.code == ErrorCode.PROVIDER_UNAVAILABLE


async def test_secondary_success_response_is_still_returned(tmp_path: Path) -> None:
    """Regression guard: the fix must not over-correct into never returning
    a secondary's response at all -- a genuine secondary success must still
    reach the caller."""
    limiter = _limiter(tmp_path)
    primary = MockProvider(status_code=429)
    secondary = MockProvider(status_code=200, body="secondary-ok")
    router = ModelRouter(limiter, primary, secondary, primary_timeout_ms=TEST_TIMEOUT_MS)

    response = await router.generate(TENANT, "prompt", 100)
    assert isinstance(response, ProviderResponse)
    assert response.status_code == 200
    assert response.body == "secondary-ok"


# ---------------------------------------------------------------------------
# Optional small regression tests: reservation/refund semantics the
# independent review verified by hand but found untested.
# ---------------------------------------------------------------------------


async def test_reservation_stays_charged_after_primary_500(tmp_path: Path) -> None:
    limiter = _limiter(tmp_path)
    router = ModelRouter(limiter, MockProvider(status_code=500), MockProvider(),
                         primary_timeout_ms=TEST_TIMEOUT_MS)
    with pytest.raises(RouterError):
        await router.generate(TENANT, "prompt", 1_000)
    assert await limiter.current_usage(TENANT) == 1_000


async def test_reservation_stays_charged_after_429_then_secondary_exception(tmp_path: Path) -> None:
    limiter = _limiter(tmp_path)
    router = ModelRouter(limiter, MockProvider(status_code=429),
                         MockProvider(exception=RuntimeError("down")), primary_timeout_ms=TEST_TIMEOUT_MS)
    with pytest.raises(RouterError):
        await router.generate(TENANT, "prompt", 2_500)
    assert await limiter.current_usage(TENANT) == 2_500


async def test_rejected_request_adds_no_reservation(tmp_path: Path) -> None:
    limiter = _limiter(tmp_path, max_tokens=1_000)
    router = ModelRouter(limiter, MockProvider(), MockProvider(), primary_timeout_ms=TEST_TIMEOUT_MS)
    await router.generate(TENANT, "prompt", 600)
    with pytest.raises(RouterError) as exc_info:
        await router.generate(TENANT, "prompt", 600)  # would exceed the 1,000 cap
    assert exc_info.value.code == ErrorCode.RATE_LIMITED
    assert await limiter.current_usage(TENANT) == 600  # not 1200


async def test_timeout_fallback_creates_only_one_reservation(tmp_path: Path) -> None:
    limiter = _limiter(tmp_path)
    router = ModelRouter(limiter, MockProvider(delay_seconds=1.0), MockProvider(body="sec"),
                         primary_timeout_ms=TEST_TIMEOUT_MS)
    await router.generate(TENANT, "prompt", 3_000)
    assert await limiter.current_usage(TENANT) == 3_000


# ---------------------------------------------------------------------------
# Invalid input never reaches a provider
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tenant,tokens",
    [("", 100), ("tenant-a", 0), ("tenant-a", -5), ("tenant-a", True)],
)
async def test_invalid_input_calls_no_provider(tmp_path: Path, tenant: object, tokens: object) -> None:
    limiter = _limiter(tmp_path)
    primary = MockProvider()
    secondary = MockProvider()
    router = ModelRouter(limiter, primary, secondary, primary_timeout_ms=TEST_TIMEOUT_MS)

    with pytest.raises(RouterError) as exc_info:
        await router.generate(tenant, "prompt", tokens)  # type: ignore[arg-type]
    assert exc_info.value.code == ErrorCode.INVALID_REQUEST
    assert primary.call_count == 0
    assert secondary.call_count == 0


# ---------------------------------------------------------------------------
# Error sanitization -- hostile sentinel data must never surface
# ---------------------------------------------------------------------------

SENTINELS = [
    "tenant-secret-api-key-DEADBEEF",
    "sk-provider-secret-CAFEF00D",
    "http://internal-provider.localhost:9999/v1/generate",
    "internal-host.corp.example",
]


def _sentinel_db_path(tmp_path: Path) -> str:
    return str(tmp_path / "super-secret-rate-limit-file.db")


@pytest.mark.parametrize(
    "exc",
    [
        RuntimeError(
            "Failed POST http://internal-provider.localhost:9999/v1/generate "
            "using key sk-provider-secret-CAFEF00D for tenant tenant-secret-api-key-DEADBEEF"
        ),
        ConnectionError("could not reach internal-host.corp.example"),
    ],
)
async def test_primary_error_response_never_leaks_sentinels(tmp_path: Path, exc: BaseException) -> None:
    db_path = _sentinel_db_path(tmp_path)
    limiter = TokenRateLimiter(db_path)
    primary = MockProvider(exception=exc)
    secondary = MockProvider()
    router = ModelRouter(limiter, primary, secondary, primary_timeout_ms=TEST_TIMEOUT_MS)

    with pytest.raises(RouterError) as exc_info:
        await router.generate(TENANT, "prompt", 100)

    serialized = json.dumps(to_response(exc_info.value.code))
    for sentinel in [*SENTINELS, db_path, str(exc)]:
        assert sentinel not in serialized
    assert "Traceback" not in serialized
    assert "RuntimeError" not in serialized
    assert "ConnectionError" not in serialized


async def test_secondary_failure_error_never_leaks_sentinels(tmp_path: Path) -> None:
    db_path = _sentinel_db_path(tmp_path)
    limiter = TokenRateLimiter(db_path)
    primary = MockProvider(status_code=429)
    secret_exc = RuntimeError(
        "secondary auth failed with key sk-provider-secret-CAFEF00D against "
        "http://internal-provider.localhost:9999/v1/generate"
    )
    secondary = MockProvider(exception=secret_exc)
    router = ModelRouter(limiter, primary, secondary, primary_timeout_ms=TEST_TIMEOUT_MS)

    with pytest.raises(RouterError) as exc_info:
        await router.generate(TENANT, "prompt", 100)

    serialized = json.dumps(to_response(exc_info.value.code))
    for sentinel in [*SENTINELS, db_path, str(secret_exc)]:
        assert sentinel not in serialized
    assert "Traceback" not in serialized
    assert "RuntimeError" not in serialized


def test_error_schema_is_stable() -> None:
    for code in ErrorCode:
        payload = to_response(code)
        assert set(payload.keys()) == {"error"}
        assert set(payload["error"].keys()) == {"code", "message"}
        assert payload["error"]["code"] == code.value
        assert isinstance(payload["error"]["message"], str) and payload["error"]["message"]
