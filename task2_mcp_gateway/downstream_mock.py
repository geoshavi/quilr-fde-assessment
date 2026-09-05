"""Mock downstream MCP server the gateway proxies to.

A minimal JSON-RPC responder over HTTP (FastAPI), standing in for a real MCP
server behind the gateway. Deterministic, no external calls. `call_log` records
every request body this app receives, in order -- tests use it to prove a
rejected tools/call never reached here.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request

app = FastAPI()

call_log: list[dict[str, Any]] = []
received_auth_headers: list[str | None] = []
"""Parallel to call_log: the Authorization header (if any) each request arrived with.

Lets tests prove the gateway never forwards the caller's bearer token downstream.
"""

MOCK_TOOLS = [
    {"name": "get_weather", "description": "Look up mock weather for a city."},
    {"name": "admin_delete_user", "description": "Mock admin-only user deletion."},
]


@app.post("/")
async def handle(request: Request) -> dict[str, Any]:
    body = await request.json()
    call_log.append(body)
    received_auth_headers.append(request.headers.get("authorization"))
    method = body.get("method")
    request_id = body.get("id")

    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": request_id, "result": {"tools": MOCK_TOOLS}}

    if method == "tools/call":
        params = body.get("params") or {}
        name = params.get("name")
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "content": [{"type": "text", "text": f"mock downstream result for {name}"}],
                "isError": False,
            },
        }

    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": -32601, "message": "Method not found"}}
