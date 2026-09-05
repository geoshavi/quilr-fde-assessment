"""Task 2 -- MCP Security Gateway: an HTTP/JSON-RPC reverse proxy.

Sits between an AI agent client and a downstream MCP server. Every request is
authenticated via `Authorization: Bearer <token>` (resolved to a role by
task2_mcp_gateway.auth). `tools/list` (and any method other than `tools/call`)
is forwarded to downstream unchanged. `tools/call` is inspected: a tool name
starting with `admin_` requires role == "admin"; otherwise the gateway returns
a JSON-RPC error (code -32001, "Unauthorized Tool Call") itself and the
request never reaches downstream.

Design decisions (also documented in README):
- Missing/malformed/unknown bearer auth is an HTTP-layer concern, not a
  JSON-RPC one -- we haven't parsed (or trusted) a JSON-RPC body yet, so
  there's no `id` to echo and no `error` envelope to build. It is rejected
  with a plain HTTP 401 before the request body is even read.
- Once past auth, this endpoint always returns HTTP 200 for both JSON-RPC
  successes and JSON-RPC-level errors (including our own -32001) -- JSON-RPC
  error/success is an application-level distinction carried in the body's
  `error`/`result` field, not the HTTP status. This mirrors typical JSON-RPC
  over HTTP conventions.
- Standard JSON-RPC codes are used for our own protocol-layer errors: -32700
  Parse error (invalid JSON), -32600 Invalid Request (valid JSON, not a valid
  JSON-RPC request shape), -32602 Invalid params (valid request, but
  `tools/call` without a usable `params.name`), -32603 Internal error
  (downstream unreachable, returned a 4xx/5xx status, or returned something
  unusable). -32001 is the assessment-specified "Unauthorized Tool Call" code.
- The `id` from the client's request is preserved in every gateway-generated
  error where it can be recovered; per JSON-RPC convention it is `null` only
  when the body could not be parsed as JSON at all.
- Every method other than `tools/call` is forwarded transparently (not just
  `tools/list`) -- the assessment defines guarded behavior only for
  `tools/call`, and treats `tools/list` passthrough as the general default,
  not a special case with its own logic path.
- The client's own bearer token is never forwarded to downstream: downstream
  is a trust boundary the gateway sits in front of, not behind.
- Downstream failures never leak exception text, stack traces, or the
  downstream URL to the client -- only a generic sanitized message.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Literal

import httpx
from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from task2_mcp_gateway.auth import resolve_role

logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("task2_mcp_gateway")

DOWNSTREAM_URL = os.environ.get("TASK2_DOWNSTREAM_URL", "http://localhost:8100")
ADMIN_TOOL_PREFIX = "admin_"

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
UNAUTHORIZED_TOOL_CALL = -32001


class JsonRpcRequest(BaseModel):
    model_config = ConfigDict(extra="allow")  # unknown extra fields on the client's request are not our concern

    jsonrpc: Literal["2.0"]
    method: str = Field(min_length=1)
    params: dict[str, Any] | None = None
    id: int | str | None = None


# A single shared client for real (non-test) runs. Tests replace this entirely
# via app.dependency_overrides[get_http_client], so this instance is never
# used -- and never opens a real connection -- during the test suite.
_shared_http_client = httpx.AsyncClient(base_url=DOWNSTREAM_URL, timeout=5.0)


def get_http_client() -> httpx.AsyncClient:
    return _shared_http_client


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Close the real (non-test) shared client on shutdown.

    Tests never trigger ASGI lifespan events (they call the app directly via
    httpx.ASGITransport and always override get_http_client), so this has no
    effect on the test suite -- it only matters for `uvicorn ...:app` runs.
    """
    yield
    await _shared_http_client.aclose()


app = FastAPI(title="Quilr Task 2 MCP Security Gateway", lifespan=_lifespan)


def _jsonrpc_error(request_id: int | str | None, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def _extract_id(payload: object) -> int | str | None:
    if isinstance(payload, dict):
        candidate = payload.get("id")
        if candidate is None or isinstance(candidate, (int, str)):
            return candidate
    return None


@app.post("/")
async def handle_jsonrpc(
    request: Request, http_client: httpx.AsyncClient = Depends(get_http_client)
) -> JSONResponse:
    role = resolve_role(request.headers.get("authorization"))
    if role is None:
        return JSONResponse(
            status_code=401,
            content={"error": "Unauthorized: missing or invalid bearer token"},
            headers={"WWW-Authenticate": "Bearer"},
        )

    raw_body = await request.body()
    try:
        payload = json.loads(raw_body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return JSONResponse(content=_jsonrpc_error(None, PARSE_ERROR, "Parse error: invalid JSON"))

    request_id = _extract_id(payload)
    try:
        parsed = JsonRpcRequest.model_validate(payload)
    except ValidationError:
        return JSONResponse(content=_jsonrpc_error(request_id, INVALID_REQUEST, "Invalid Request"))

    if parsed.method == "tools/call":
        name = parsed.params.get("name") if isinstance(parsed.params, dict) else None
        if not isinstance(name, str) or not name:
            return JSONResponse(
                content=_jsonrpc_error(
                    parsed.id, INVALID_PARAMS, "Invalid params: tools/call requires a string params.name"
                )
            )
        if name.startswith(ADMIN_TOOL_PREFIX) and role != "admin":
            logger.info("Rejected unauthorized tools/call to %r for role %r", name, role)
            return JSONResponse(content=_jsonrpc_error(parsed.id, UNAUTHORIZED_TOOL_CALL, "Unauthorized Tool Call"))

    # Authorized: tools/list, any other method, or a tools/call this role may make.
    try:
        downstream_response = await http_client.post("/", json=payload)
    except httpx.HTTPError:
        logger.exception("Downstream request failed")
        return JSONResponse(content=_jsonrpc_error(parsed.id, INTERNAL_ERROR, "Downstream MCP server unavailable"))

    if downstream_response.is_error:
        # A 4xx/5xx downstream response may carry an arbitrary body (framework
        # error pages, stack traces, internal paths) that is not ours to trust
        # or forward -- sanitize it exactly like a connection failure, logging
        # only the status code server-side, never the body.
        logger.error("Downstream returned error status %s", downstream_response.status_code)
        return JSONResponse(
            content=_jsonrpc_error(parsed.id, INTERNAL_ERROR, "Downstream MCP server returned an error")
        )

    try:
        downstream_payload = downstream_response.json()
    except ValueError:
        logger.error("Downstream returned a non-JSON response")
        return JSONResponse(
            content=_jsonrpc_error(parsed.id, INTERNAL_ERROR, "Downstream MCP server returned an invalid response")
        )

    return JSONResponse(content=downstream_payload)
