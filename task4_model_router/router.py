"""Task 4 -- ModelRouter: rate-limit admission + primary/secondary provider
routing behind one orchestration entry point.

Request flow:
    validate tenant/requested_tokens
        -> await rate_limiter.admit(...)
        -> if rejected: raise RouterError(RATE_LIMITED); NO provider is ever called
        -> call primary, enforcing the primary deadline ourselves
        -> fall back to secondary only for the two permitted reasons below
        -> return the provider's response, or raise a sanitized RouterError

Fallback matrix (the ONLY two conditions that call secondary):
    - primary responds with HTTP-style status 429
    - the router-enforced primary deadline (TASK4_PRIMARY_TIMEOUT_MS,
      default 3000ms) expires

Everything else from primary -- a non-429 4xx, a 5xx, a malformed
response, or any raised exception -- maps directly to a sanitized
RouterError and never calls secondary. This is deliberately NOT a broad
`except Exception: fallback()`: only the timeout and the 429 status are
classified as fallback-eligible; every other failure is caught narrowly
for sanitization only and is never routed into the fallback branch.

Secondary's result is classified with the SAME `_classify_response` used
for primary: a secondary success (status < 400) is returned to the
caller, but ANY secondary status >= 400 (429 included -- there is no
tertiary fallback), a malformed/missing/non-int/bool status_code, or a
raised exception all map to a sanitized `PROVIDER_UNAVAILABLE`. A raw
provider response is never returned to the client except on a genuine,
well-formed success -- a failing secondary's status/body is never
forwarded, since that body may itself carry a URL, secret, or other
provider-controlled internal detail.

The 3000ms default is enforced HERE via `asyncio.wait_for`, not left to
whatever timeout an HTTP client or mock transport happens to apply --
this keeps the boundary deterministic and directly testable (tests
inject a much smaller timeout against a controllable mock delay instead
of waiting a real 3 seconds; the module-level default remains exactly
3000ms). `asyncio.wait_for` itself cancels the primary call and awaits
its cleanup before raising the timeout, so no additional cleanup code is
needed here.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Awaitable, Callable

from task4_model_router.errors import ErrorCode, RouterError
from task4_model_router.rate_limiter import TokenRateLimiter, validate_admission_request

logger = logging.getLogger("task4_model_router")

def _default_primary_timeout_ms() -> int:
    """The documented production default: exactly 3000ms, unless overridden
    via TASK4_PRIMARY_TIMEOUT_MS. Kept as its own function (rather than only
    a module-level constant) so tests can prove the documented fallback
    value directly, independent of whatever the ambient environment happens
    to set -- without weakening or bypassing the real env-override support.
    """
    return int(os.environ.get("TASK4_PRIMARY_TIMEOUT_MS", "3000"))


# Single named constant for the assessment-required boundary, converted to
# seconds once here rather than scattered as a magic number.
PRIMARY_TIMEOUT_MS = _default_primary_timeout_ms()


@dataclass(frozen=True)
class ProviderResponse:
    status_code: int
    body: str = ""


ProviderCallable = Callable[[str, int], Awaitable[ProviderResponse]]
"""async (prompt, requested_tokens) -> ProviderResponse.

May raise any exception to signal a hard failure. Only an HTTP-style 429
`ProviderResponse` and the router's own enforced timeout are
fallback-eligible; every other exception or status maps straight to a
sanitized error (see module docstring)."""


@dataclass(frozen=True)
class _PrimaryOutcome:
    response: ProviderResponse | None
    fallback: bool


def _classify_response(response: object) -> int | None:
    """The response's usable integer status_code, or None if malformed.

    Shared by both primary and secondary classification so neither path can
    diverge: missing `status_code`, a non-int value, and `bool` (which is an
    `int` subclass in Python, so `True`/`False` must be excluded explicitly)
    are all treated identically as "malformed", never as a status number.
    """
    status_code = getattr(response, "status_code", None)
    if isinstance(status_code, bool) or not isinstance(status_code, int):
        return None
    return status_code


class ModelRouter:
    """Integrates per-tenant admission with primary/secondary provider routing."""

    def __init__(
        self,
        rate_limiter: TokenRateLimiter,
        primary: ProviderCallable,
        secondary: ProviderCallable,
        *,
        primary_timeout_ms: int = PRIMARY_TIMEOUT_MS,
    ) -> None:
        self._rate_limiter = rate_limiter
        self._primary = primary
        self._secondary = secondary
        self._primary_timeout_s = primary_timeout_ms / 1000.0

    async def generate(self, tenant_key: str, prompt: str, requested_tokens: int) -> ProviderResponse:
        try:
            validate_admission_request(tenant_key, requested_tokens)
        except ValueError:
            raise RouterError(ErrorCode.INVALID_REQUEST) from None

        try:
            admitted = await self._rate_limiter.admit(tenant_key, requested_tokens)
        except Exception:
            # An unexpected failure in the rate limiter itself (e.g. a real
            # SQLite I/O error) must not leak a raw exception -- which could
            # name the on-disk DB path -- straight to the caller. This is
            # sanitization only: it never calls a provider, so it does not
            # touch the fallback-eligibility rules below.
            logger.error("rate limiter admission failed unexpectedly")
            raise RouterError(ErrorCode.INTERNAL_ERROR) from None
        if not admitted:
            raise RouterError(ErrorCode.RATE_LIMITED)

        outcome = await self._call_primary(prompt, requested_tokens)
        if not outcome.fallback:
            assert outcome.response is not None
            return outcome.response

        logger.info("primary unavailable (429 or timeout); attempting secondary once")
        return await self._call_secondary(prompt, requested_tokens)

    async def _call_primary(self, prompt: str, requested_tokens: int) -> _PrimaryOutcome:
        try:
            response = await asyncio.wait_for(
                self._primary(prompt, requested_tokens), timeout=self._primary_timeout_s
            )
        except asyncio.TimeoutError:
            logger.info("primary exceeded %dms deadline; falling back", int(self._primary_timeout_s * 1000))
            return _PrimaryOutcome(response=None, fallback=True)
        except Exception:
            # Any other primary failure -- connection error, RuntimeError,
            # malformed response missing status_code, etc. -- is explicitly
            # NOT fallback-eligible.
            logger.warning("primary provider call failed (non-retryable); no fallback")
            raise RouterError(ErrorCode.PROVIDER_ERROR) from None

        status_code = _classify_response(response)
        if status_code is None:
            # Malformed result (missing/non-int/bool status_code) is a hard
            # failure, exactly like a raised exception -- never fallback-eligible.
            logger.warning("primary provider returned a malformed response; no fallback")
            raise RouterError(ErrorCode.PROVIDER_ERROR)
        if status_code == 429:
            return _PrimaryOutcome(response=None, fallback=True)
        if status_code >= 400:
            raise RouterError(ErrorCode.PROVIDER_ERROR)
        return _PrimaryOutcome(response=response, fallback=False)

    async def _call_secondary(self, prompt: str, requested_tokens: int) -> ProviderResponse:
        """The single, terminal fallback attempt -- there is no tertiary fallback.

        A raw secondary response/body is returned to the caller ONLY on a
        well-formed success (status < 400). Any status >= 400 (429
        included), a malformed response, or a raised exception all map to
        the same sanitized PROVIDER_UNAVAILABLE -- the secondary's status
        and body are never forwarded to the client on failure, since a
        provider-controlled error body may itself carry a URL, secret, or
        other internal detail.
        """
        try:
            response = await self._secondary(prompt, requested_tokens)
        except Exception:
            logger.warning("secondary provider call failed; no further fallback")
            raise RouterError(ErrorCode.PROVIDER_UNAVAILABLE) from None

        status_code = _classify_response(response)
        if status_code is None or status_code >= 400:
            logger.warning("secondary provider returned an error or malformed response; no further fallback")
            raise RouterError(ErrorCode.PROVIDER_UNAVAILABLE)
        return response
