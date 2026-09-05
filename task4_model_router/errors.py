"""Task 4 -- Sanitized, closed-set error envelope.

Client-facing errors are built ONLY from a small fixed set of safe
(code, message) pairs. Raw exception text/repr, tracebacks, provider
response bodies, provider/database URLs and paths, and tenant API keys
must never be serialized into a client-facing error. An original
exception may be used internally only for control-flow classification
(e.g. `except asyncio.TimeoutError` in router.py) -- never echoed into
the response built here.
"""

from __future__ import annotations

from enum import Enum
from typing import Any


class ErrorCode(str, Enum):
    INVALID_REQUEST = "INVALID_REQUEST"
    RATE_LIMITED = "RATE_LIMITED"
    PROVIDER_ERROR = "PROVIDER_ERROR"
    PROVIDER_UNAVAILABLE = "PROVIDER_UNAVAILABLE"
    INTERNAL_ERROR = "INTERNAL_ERROR"


_MESSAGES: dict[ErrorCode, str] = {
    ErrorCode.INVALID_REQUEST: "The request was invalid.",
    ErrorCode.RATE_LIMITED: "Rate limit exceeded for this tenant.",
    ErrorCode.PROVIDER_ERROR: "The model provider returned an error.",
    ErrorCode.PROVIDER_UNAVAILABLE: "The model provider is unavailable.",
    ErrorCode.INTERNAL_ERROR: "An internal error occurred.",
}


class RouterError(Exception):
    """Carries a classified, sanitized error code across the routing layer.

    `code` is the only thing ever used to build the client-facing response
    (see `to_response`) -- nothing else about how or why this was raised is
    exposed to the caller.
    """

    def __init__(self, code: ErrorCode) -> None:
        super().__init__(code.value)
        self.code = code


def to_response(code: ErrorCode) -> dict[str, Any]:
    """The fixed, sanitized client-facing error envelope for `code`."""
    return {"error": {"code": code.value, "message": _MESSAGES[code]}}
