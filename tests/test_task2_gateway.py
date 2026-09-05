"""Task 2 tests: JSON-RPC parsing, bearer auth, tool-call authorization, proxying.

Both the gateway (`task2_mcp_gateway.proxy.app`) and the downstream mock
(`task2_mcp_gateway.downstream_mock.app`) are real FastAPI/Starlette ASGI apps,
driven in-process over `httpx.ASGITransport` -- no real sockets, no external
network, but genuine HTTP request/response handling through the actual FastAPI
routing/dependency layer (not hand-called Python functions).
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest_asyncio
from httpx import ASGITransport

from task2_mcp_gateway import downstream_mock, proxy

ADMIN_TOKEN = "admin-demo-token"
VIEWER_TOKEN = "viewer-demo-token"


def _auth_headers(token: str | None) -> dict[str, str]:
    return {} if token is None else {"Authorization": f"Bearer {token}"}


def _rpc(method: str, params: dict[str, Any] | None = None, id_: int | str = 1) -> dict[str, Any]:
    body: dict[str, Any] = {"jsonrpc": "2.0", "method": method, "id": id_}
    if params is not None:
        body["params"] = params
    return body


@pytest_asyncio.fixture
async def downstream_client():
    downstream_mock.call_log.clear()
    downstream_mock.received_auth_headers.clear()
    transport = ASGITransport(app=downstream_mock.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://downstream") as client:
        yield client


@pytest_asyncio.fixture
async def gateway_client(downstream_client: httpx.AsyncClient):
    proxy.app.dependency_overrides[proxy.get_http_client] = lambda: downstream_client
    transport = ASGITransport(app=proxy.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://gateway") as client:
        yield client
    proxy.app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# tools/list passthrough
# ---------------------------------------------------------------------------


async def test_tools_list_passthrough(gateway_client: httpx.AsyncClient):
    response = await gateway_client.post("/", json=_rpc("tools/list", id_=1), headers=_auth_headers(VIEWER_TOKEN))
    assert response.status_code == 200
    body = response.json()
    assert body["id"] == 1
    assert "error" not in body
    names = {tool["name"] for tool in body["result"]["tools"]}
    assert names == {"get_weather", "admin_delete_user"}
    assert len(downstream_mock.call_log) == 1
    assert downstream_mock.call_log[0]["method"] == "tools/list"


# ---------------------------------------------------------------------------
# tools/call authorization
# ---------------------------------------------------------------------------


async def test_viewer_can_call_normal_tool(gateway_client: httpx.AsyncClient):
    response = await gateway_client.post(
        "/",
        json=_rpc("tools/call", {"name": "get_weather", "arguments": {}}, id_=2),
        headers=_auth_headers(VIEWER_TOKEN),
    )
    body = response.json()
    assert "error" not in body
    assert "get_weather" in body["result"]["content"][0]["text"]
    assert len(downstream_mock.call_log) == 1


async def test_admin_can_call_admin_tool(gateway_client: httpx.AsyncClient):
    response = await gateway_client.post(
        "/",
        json=_rpc("tools/call", {"name": "admin_delete_user", "arguments": {}}, id_=3),
        headers=_auth_headers(ADMIN_TOKEN),
    )
    body = response.json()
    assert "error" not in body
    assert "admin_delete_user" in body["result"]["content"][0]["text"]
    assert len(downstream_mock.call_log) == 1


async def test_viewer_admin_tool_rejected(gateway_client: httpx.AsyncClient):
    response = await gateway_client.post(
        "/",
        json=_rpc("tools/call", {"name": "admin_delete_user", "arguments": {}}, id_=4),
        headers=_auth_headers(VIEWER_TOKEN),
    )
    assert response.status_code == 200
    body = response.json()
    assert body["id"] == 4
    assert body["error"]["code"] == -32001
    assert body["error"]["message"] == "Unauthorized Tool Call"


async def test_rejected_admin_call_never_reaches_downstream(gateway_client: httpx.AsyncClient):
    await gateway_client.post(
        "/",
        json=_rpc("tools/call", {"name": "admin_delete_user", "arguments": {}}, id_=5),
        headers=_auth_headers(VIEWER_TOKEN),
    )
    assert downstream_mock.call_log == []


async def test_viewer_calling_non_admin_tool_is_never_blocked(gateway_client: httpx.AsyncClient):
    """Explicit coverage of requirement 3: non-admin_ tools are open to any authenticated role."""
    response = await gateway_client.post(
        "/",
        json=_rpc("tools/call", {"name": "get_weather", "arguments": {}}, id_=6),
        headers=_auth_headers(VIEWER_TOKEN),
    )
    body = response.json()
    assert "error" not in body


# ---------------------------------------------------------------------------
# Bearer auth: missing / malformed / invalid
# ---------------------------------------------------------------------------


async def test_missing_authorization_header(gateway_client: httpx.AsyncClient):
    response = await gateway_client.post("/", json=_rpc("tools/list"))
    assert response.status_code == 401
    assert downstream_mock.call_log == []


async def test_malformed_bearer_header_variants(gateway_client: httpx.AsyncClient):
    for header_value in [
        "Bearer",  # scheme with no token
        "Bearer ",  # scheme with empty token
        "Basic admin-demo-token",  # wrong scheme
        "admin-demo-token",  # no scheme at all
        "BearerAdmin-demo-token",  # no separating space
    ]:
        response = await gateway_client.post(
            "/", json=_rpc("tools/list"), headers={"Authorization": header_value}
        )
        assert response.status_code == 401, f"expected 401 for header {header_value!r}"
    assert downstream_mock.call_log == []


async def test_invalid_token_rejected(gateway_client: httpx.AsyncClient):
    response = await gateway_client.post(
        "/", json=_rpc("tools/list"), headers={"Authorization": "Bearer not-a-real-token"}
    )
    assert response.status_code == 401
    assert downstream_mock.call_log == []


async def test_authorization_token_never_forwarded_downstream(gateway_client: httpx.AsyncClient):
    await gateway_client.post("/", json=_rpc("tools/list", id_=7), headers=_auth_headers(VIEWER_TOKEN))
    assert downstream_mock.received_auth_headers == [None]


# ---------------------------------------------------------------------------
# Malformed JSON-RPC
# ---------------------------------------------------------------------------


async def test_malformed_json_body_is_parse_error(gateway_client: httpx.AsyncClient):
    response = await gateway_client.post(
        "/", content=b"not json at all {{{", headers=_auth_headers(VIEWER_TOKEN)
    )
    assert response.status_code == 200
    body = response.json()
    assert body["error"]["code"] == -32700
    assert body["id"] is None
    assert downstream_mock.call_log == []


async def test_invalid_jsonrpc_shape_missing_method(gateway_client: httpx.AsyncClient):
    response = await gateway_client.post(
        "/", json={"jsonrpc": "2.0", "id": 9}, headers=_auth_headers(VIEWER_TOKEN)
    )
    body = response.json()
    assert body["error"]["code"] == -32600
    assert body["id"] == 9  # recoverable even though the request is otherwise invalid
    assert downstream_mock.call_log == []


async def test_invalid_jsonrpc_shape_wrong_version(gateway_client: httpx.AsyncClient):
    response = await gateway_client.post(
        "/",
        json={"jsonrpc": "1.0", "method": "tools/list", "id": 10},
        headers=_auth_headers(VIEWER_TOKEN),
    )
    body = response.json()
    assert body["error"]["code"] == -32600
    assert downstream_mock.call_log == []


async def test_invalid_jsonrpc_shape_method_wrong_type(gateway_client: httpx.AsyncClient):
    response = await gateway_client.post(
        "/",
        json={"jsonrpc": "2.0", "method": 123, "id": 11},
        headers=_auth_headers(VIEWER_TOKEN),
    )
    body = response.json()
    assert body["error"]["code"] == -32600
    assert downstream_mock.call_log == []


async def test_tools_call_missing_params_name(gateway_client: httpx.AsyncClient):
    response = await gateway_client.post(
        "/", json=_rpc("tools/call", id_=12), headers=_auth_headers(VIEWER_TOKEN)
    )
    body = response.json()
    assert body["error"]["code"] == -32602
    assert downstream_mock.call_log == []


async def test_tools_call_name_wrong_type(gateway_client: httpx.AsyncClient):
    response = await gateway_client.post(
        "/",
        json=_rpc("tools/call", {"name": 123}, id_=13),
        headers=_auth_headers(VIEWER_TOKEN),
    )
    body = response.json()
    assert body["error"]["code"] == -32602
    assert downstream_mock.call_log == []


# ---------------------------------------------------------------------------
# Downstream failure
# ---------------------------------------------------------------------------


async def test_downstream_unavailable_is_sanitized(gateway_client: httpx.AsyncClient):
    def _boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused to internal-mcp.svc.cluster.local:8100", request=request)

    broken_client = httpx.AsyncClient(transport=httpx.MockTransport(_boom), base_url="http://downstream")
    proxy.app.dependency_overrides[proxy.get_http_client] = lambda: broken_client
    try:
        response = await gateway_client.post(
            "/", json=_rpc("tools/list", id_=14), headers=_auth_headers(VIEWER_TOKEN)
        )
    finally:
        await broken_client.aclose()

    body = response.json()
    assert body["error"]["code"] == -32603
    assert body["id"] == 14
    # No leaked exception text, internal hostnames, or the real downstream URL.
    assert "Connection refused" not in response.text
    assert "internal-mcp" not in response.text
    assert proxy.DOWNSTREAM_URL not in response.text
    assert "Traceback" not in response.text


async def test_downstream_5xx_with_sensitive_body_is_sanitized(gateway_client: httpx.AsyncClient):
    """M2 review regression: a downstream that returns a *well-formed* JSON

    error body (not a connection failure) must still be sanitized -- the
    gateway previously forwarded such bodies verbatim, which would leak
    whatever a misbehaving downstream put in them.
    """

    def _leaky_downstream(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            500,
            json={"detail": "Traceback: DB_PASSWORD=hunter2 at /srv/internal/secret_module.py:42"},
            request=request,
        )

    leaky_client = httpx.AsyncClient(transport=httpx.MockTransport(_leaky_downstream), base_url="http://downstream")
    proxy.app.dependency_overrides[proxy.get_http_client] = lambda: leaky_client
    try:
        response = await gateway_client.post(
            "/", json=_rpc("tools/list", id_=15), headers=_auth_headers(VIEWER_TOKEN)
        )
    finally:
        await leaky_client.aclose()

    body = response.json()
    assert body["error"]["code"] == -32603
    assert body["id"] == 15
    assert "DB_PASSWORD" not in response.text
    assert "hunter2" not in response.text
    assert "secret_module" not in response.text
    assert "Traceback" not in response.text


async def test_downstream_4xx_is_sanitized(gateway_client: httpx.AsyncClient):
    def _not_found(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "no route matched /internal/admin-console"}, request=request)

    client_404 = httpx.AsyncClient(transport=httpx.MockTransport(_not_found), base_url="http://downstream")
    proxy.app.dependency_overrides[proxy.get_http_client] = lambda: client_404
    try:
        response = await gateway_client.post(
            "/", json=_rpc("tools/list", id_=16), headers=_auth_headers(VIEWER_TOKEN)
        )
    finally:
        await client_404.aclose()

    body = response.json()
    assert body["error"]["code"] == -32603
    assert body["id"] == 16
    assert "admin-console" not in response.text
    assert "no route matched" not in response.text


# ---------------------------------------------------------------------------
# Exact-match security semantics (no normalization) -- M2 review regression
# ---------------------------------------------------------------------------


async def test_method_with_trailing_space_is_not_tools_call(gateway_client: httpx.AsyncClient):
    """"tools/call " (trailing space) must NOT be treated as "tools/call".

    JSON-RPC method names are exact strings; the gateway must not normalize
    them. Per the existing "every method other than tools/call is forwarded
    transparently" design, this near-miss method is forwarded as-is (not
    intercepted by the admin_ guard, and not silently corrected) -- it is
    downstream's responsibility to reject an method it doesn't recognize,
    exactly as it does for any other unknown method.
    """
    response = await gateway_client.post(
        "/",
        json=_rpc("tools/call ", {"name": "admin_delete_user"}, id_=17),
        headers=_auth_headers(VIEWER_TOKEN),
    )
    body = response.json()
    # Must NOT be our own -32001 authorization rejection...
    assert body["error"]["code"] != -32001
    # ...because the admin_ guard was never entered for this method string;
    # the request reached downstream unmodified, exactly as sent.
    assert len(downstream_mock.call_log) == 1
    assert downstream_mock.call_log[0]["method"] == "tools/call "
    assert downstream_mock.call_log[0]["params"]["name"] == "admin_delete_user"
    # The mock's own exact-match dispatch rejects the unrecognized method name.
    assert body["error"]["code"] == -32601


async def test_mixed_case_admin_tool_name_is_not_treated_as_admin_prefixed(gateway_client: httpx.AsyncClient):
    """"AdMiN_delete_user" must NOT be silently rewritten/matched as "admin_delete_user".

    The assessment defines the guard as the literal, case-sensitive prefix
    "admin_". A differently-cased name simply does not match that prefix, so
    it is not subject to the admin_ guard at all -- same as any other
    non-admin_ tool name, a viewer may call it, and it is forwarded to
    downstream unmodified (not corrected to the "real" admin tool name).
    """
    response = await gateway_client.post(
        "/",
        json=_rpc("tools/call", {"name": "AdMiN_delete_user"}, id_=18),
        headers=_auth_headers(VIEWER_TOKEN),
    )
    body = response.json()
    assert "error" not in body
    assert len(downstream_mock.call_log) == 1
    assert downstream_mock.call_log[0]["params"]["name"] == "AdMiN_delete_user"
    assert "AdMiN_delete_user" in body["result"]["content"][0]["text"]


# ---------------------------------------------------------------------------
# id passthrough correctness
# ---------------------------------------------------------------------------


async def test_id_passthrough_string_and_int(gateway_client: httpx.AsyncClient):
    response_str = await gateway_client.post(
        "/", json=_rpc("tools/list", id_="request-abc-123"), headers=_auth_headers(VIEWER_TOKEN)
    )
    assert response_str.json()["id"] == "request-abc-123"

    response_int = await gateway_client.post(
        "/", json=_rpc("tools/list", id_=999999), headers=_auth_headers(VIEWER_TOKEN)
    )
    assert response_int.json()["id"] == 999999
