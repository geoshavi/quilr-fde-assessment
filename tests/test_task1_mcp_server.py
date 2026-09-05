"""Task 1 tests: input validation, MCP/JSON-RPC error semantics, stdio isolation.

Two layers, deliberately:

1. Handler-level tests call `_list_tools`/`_call_tool` directly. These are NOT
   mocks of the MCP layer -- `CallToolRequestParams`, `CallToolResult`,
   `MCPError`, and the Pydantic input models are all real SDK/pydantic types,
   exercised exactly as the SDK's own request dispatcher would use them. This
   layer gives fast, precise coverage of every boundary case.
2. A real stdio subprocess integration layer (`TestStdioProtocol`) spawns
   `python -m task1_mcp_server.server` as an actual child process and drives it
   over real OS pipes, building requests from `mcp_types` models (not
   hand-invented dicts) and validating every response line through the SDK's
   own `mcp_types.jsonrpc_message_adapter` -- the same validator the SDK's
   stdio transport uses on its read side. This is what proves stdout carries
   only protocol traffic and that the full handshake/tool-call round trip
   works end-to-end, per the assessment's explicit preference for a real
   integration test over a faked one.
"""

from __future__ import annotations

import subprocess
import sys
import threading
from pathlib import Path

import mcp_types as types
import pytest
from mcp.shared.exceptions import MCPError
from pydantic import ValidationError

from task1_mcp_server.models import TriggerRefundInput
from task1_mcp_server.server import _call_tool, _list_tools

REPO_ROOT = Path(__file__).resolve().parent.parent
VALID_REASON = "defective item arrived damaged"  # 30 chars, comfortably >= 10


async def _acall(name: str, arguments: dict[str, object]) -> types.CallToolResult:
    return await _call_tool(None, types.CallToolRequestParams(name=name, arguments=arguments))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 1-3: valid calls, including a customer_id boundary example
# ---------------------------------------------------------------------------


async def test_valid_get_customer_record():
    result = await _acall("get_customer_record", {"customer_id": "CUST-12345"})
    assert result.is_error is not True
    assert "CUST-12345" in result.content[0].text


async def test_valid_trigger_refund():
    result = await _acall(
        "trigger_refund",
        {"customer_id": "CUST-12345", "amount": 50.0, "reason": VALID_REASON},
    )
    assert result.is_error is not True
    assert "REFUND-CUST-12345-5000" in result.content[0].text


@pytest.mark.parametrize("customer_id", ["CUST-00000", "CUST-99999"])
async def test_valid_customer_id_boundary(customer_id: str):
    result = await _acall("get_customer_record", {"customer_id": customer_id})
    assert result.is_error is not True
    assert customer_id in result.content[0].text


# ---------------------------------------------------------------------------
# 4-9: customer_id format rejections -> genuine JSON-RPC error (MCPError)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "label,customer_id",
    [
        ("wrong_prefix", "CUSTOMER-12345"),
        ("lowercase_prefix", "cust-12345"),
        ("too_few_digits", "CUST-1234"),
        ("too_many_digits", "CUST-123456"),
        ("alphabetic_suffix", "CUST-ABCDE"),
        ("alphanumeric_suffix", "CUST-12A45"),
        ("leading_whitespace", " CUST-12345"),
        ("trailing_whitespace", "CUST-12345 "),
        ("internal_whitespace", "CUST 12345"),
        # M1 review regression: Unicode fullwidth digits (U+FF11..U+FF15) are
        # category-Nd "digits" that plain `\d` would match, and render as
        # near-indistinguishable from CUST-12345 in most fonts/terminals.
        ("fullwidth_unicode_digits", "CUST-" + "１２３４５"),
    ],
)
async def test_invalid_customer_id_rejected(label: str, customer_id: str):
    with pytest.raises(MCPError) as exc_info:
        await _acall("get_customer_record", {"customer_id": customer_id})
    assert exc_info.value.code == types.INVALID_PARAMS


# ---------------------------------------------------------------------------
# 10-12: amount validation
# ---------------------------------------------------------------------------


async def test_amount_zero_rejected():
    with pytest.raises(MCPError) as exc_info:
        await _acall(
            "trigger_refund",
            {"customer_id": "CUST-12345", "amount": 0, "reason": VALID_REASON},
        )
    assert exc_info.value.code == types.INVALID_PARAMS


async def test_amount_negative_rejected():
    with pytest.raises(MCPError) as exc_info:
        await _acall(
            "trigger_refund",
            {"customer_id": "CUST-12345", "amount": -5.0, "reason": VALID_REASON},
        )
    assert exc_info.value.code == types.INVALID_PARAMS


@pytest.mark.parametrize(
    "label,amount",
    [
        ("numeric_string", "50.0"),
        ("bool_true", True),
        ("bool_false", False),
        ("none", None),
        ("list", [50.0]),
    ],
)
async def test_amount_invalid_type_rejected(label: str, amount: object):
    with pytest.raises(MCPError) as exc_info:
        await _acall(
            "trigger_refund",
            {"customer_id": "CUST-12345", "amount": amount, "reason": VALID_REASON},
        )
    assert exc_info.value.code == types.INVALID_PARAMS


@pytest.mark.parametrize(
    "label,amount",
    [
        ("positive_infinity", float("inf")),
        ("negative_infinity", float("-inf")),
        ("nan", float("nan")),
    ],
)
async def test_amount_non_finite_rejected(label: str, amount: float):
    """M1 review regression: +inf previously passed `gt=0` (inf > 0 is True) and

    crashed mock_data.trigger_refund with an uncaught OverflowError instead of
    failing validation. All three non-finite values must now be rejected as a
    normal INVALID_PARAMS, not reach the mock business logic at all.
    """
    with pytest.raises(MCPError) as exc_info:
        await _acall(
            "trigger_refund",
            {"customer_id": "CUST-12345", "amount": amount, "reason": VALID_REASON},
        )
    assert exc_info.value.code == types.INVALID_PARAMS


async def test_amount_int_is_accepted():
    """int is a legitimate numeric type for a positive float amount, not a coercion loophole."""
    result = await _acall(
        "trigger_refund",
        {"customer_id": "CUST-12345", "amount": 50, "reason": VALID_REASON},
    )
    assert result.is_error is not True


# ---------------------------------------------------------------------------
# 13-15: reason validation
# ---------------------------------------------------------------------------


async def test_reason_length_9_rejected():
    with pytest.raises(MCPError) as exc_info:
        await _acall(
            "trigger_refund",
            {"customer_id": "CUST-12345", "amount": 10.0, "reason": "x" * 9},
        )
    assert exc_info.value.code == types.INVALID_PARAMS


async def test_reason_length_10_accepted():
    result = await _acall(
        "trigger_refund",
        {"customer_id": "CUST-12345", "amount": 10.0, "reason": "x" * 10},
    )
    assert result.is_error is not True


async def test_reason_invalid_type_rejected():
    with pytest.raises(MCPError) as exc_info:
        await _acall(
            "trigger_refund",
            {"customer_id": "CUST-12345", "amount": 10.0, "reason": 1234567890},
        )
    assert exc_info.value.code == types.INVALID_PARAMS


def test_reason_bytes_rejected_at_model_level():
    """M1 review regression: pydantic's lax `str` validator otherwise decodes

    bytes into str (e.g. b"x"*10 -> "xxxxxxxxxx"), silently accepting a type
    the assessment's "reason must be a string" requirement did not intend.
    """
    with pytest.raises(ValidationError) as exc_info:
        TriggerRefundInput(customer_id="CUST-12345", amount=10.0, reason=b"x" * 10)
    assert exc_info.value.errors()[0]["loc"] == ("reason",)


async def test_reason_bytes_rejected_at_handler_level():
    with pytest.raises(MCPError) as exc_info:
        await _acall(
            "trigger_refund",
            {"customer_id": "CUST-12345", "amount": 10.0, "reason": b"x" * 10},
        )
    assert exc_info.value.code == types.INVALID_PARAMS


# ---------------------------------------------------------------------------
# 16: unexpected extra fields (extra="forbid")
# ---------------------------------------------------------------------------


async def test_extra_fields_rejected_get_customer_record():
    with pytest.raises(MCPError) as exc_info:
        await _acall("get_customer_record", {"customer_id": "CUST-12345", "extra": "nope"})
    assert exc_info.value.code == types.INVALID_PARAMS


async def test_extra_fields_rejected_trigger_refund():
    with pytest.raises(MCPError) as exc_info:
        await _acall(
            "trigger_refund",
            {
                "customer_id": "CUST-12345",
                "amount": 10.0,
                "reason": VALID_REASON,
                "unexpected_field": True,
            },
        )
    assert exc_info.value.code == types.INVALID_PARAMS


# ---------------------------------------------------------------------------
# 17 (handler level): unknown tool name is a JSON-RPC error, not a crash
# ---------------------------------------------------------------------------


async def test_unknown_tool_name_rejected():
    with pytest.raises(MCPError) as exc_info:
        await _acall("delete_everything", {})
    assert exc_info.value.code == types.INVALID_PARAMS


async def test_list_tools_exposes_both_tools_with_schemas():
    result = await _list_tools(None, None)
    names = {tool.name for tool in result.tools}
    assert names == {"get_customer_record", "trigger_refund"}
    for tool in result.tools:
        assert tool.input_schema.get("additionalProperties") is False


# ---------------------------------------------------------------------------
# 18 + 17 (wire level): real stdio subprocess, protocol compliance, isolation
# ---------------------------------------------------------------------------


class _StdioSession:
    """Drives the real server subprocess over its actual stdin/stdout pipes."""

    def __init__(self) -> None:
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "task1_mcp_server.server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=str(REPO_ROOT),
        )
        self.stdout_lines: list[str] = []
        self._closed = False
        self._stderr_text = ""

    def send_raw(self, line: str) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def send(self, message: object) -> None:
        self.send_raw(message.model_dump_json(by_alias=True, exclude_none=True))  # type: ignore[attr-defined]

    def recv(self, timeout: float = 5.0) -> object:
        """Read one line, record it verbatim, and validate it via the SDK's own adapter."""
        assert self.proc.stdout is not None
        result: dict[str, str] = {}

        def _read() -> None:
            result["line"] = self.proc.stdout.readline()  # type: ignore[union-attr]

        t = threading.Thread(target=_read, daemon=True)
        t.start()
        t.join(timeout=timeout)
        if t.is_alive() or "line" not in result or not result["line"]:
            return None
        line = result["line"]
        self.stdout_lines.append(line)
        return types.jsonrpc_message_adapter.validate_json(line, by_name=False)

    def initialize(self) -> None:
        init_params = types.InitializeRequestParams(
            protocol_version="2025-11-25",
            capabilities=types.ClientCapabilities(),
            client_info=types.Implementation(name="task1-test-client", version="0.1.0"),
        ).model_dump(by_alias=True, exclude_none=True)
        self.send(types.JSONRPCRequest(jsonrpc="2.0", id=1, method="initialize", params=init_params))
        response = self.recv()
        assert response is not None, "server did not respond to initialize"
        self.send(types.JSONRPCNotification(jsonrpc="2.0", method="notifications/initialized"))

    def call_tool(self, request_id: int, name: str, arguments: dict[str, object]) -> object:
        params = types.CallToolRequestParams(name=name, arguments=arguments).model_dump(
            by_alias=True, exclude_none=True
        )
        self.send(types.JSONRPCRequest(jsonrpc="2.0", id=request_id, method="tools/call", params=params))
        return self.recv()

    def close(self) -> str:
        """Idempotent: safe to call from a test body and again from teardown."""
        if self._closed:
            return self._stderr_text
        assert self.proc.stdin is not None
        self.proc.stdin.close()
        self.proc.terminate()
        self.proc.wait(timeout=5)
        assert self.proc.stderr is not None
        self._stderr_text = self.proc.stderr.read()
        self._closed = True
        return self._stderr_text


@pytest.fixture
def stdio_session():
    session = _StdioSession()
    session.initialize()
    session.stdout_lines.clear()  # tests count only post-handshake traffic
    yield session
    session.close()


class TestStdioProtocol:
    def test_full_round_trip_and_stdout_isolation(self, stdio_session: _StdioSession):
        # tools/list over the real wire
        session = stdio_session
        session.send(types.JSONRPCRequest(jsonrpc="2.0", id=2, method="tools/list", params=None))
        list_response = session.recv()
        assert list_response is not None
        assert getattr(list_response, "error", None) is None

        # A valid tools/call
        ok_response = session.call_tool(3, "get_customer_record", {"customer_id": "CUST-12345"})
        assert ok_response is not None
        assert getattr(ok_response, "error", None) is None

        # An invalid tools/call -> genuine JSON-RPC error object on the wire
        bad_response = session.call_tool(4, "trigger_refund", {"customer_id": "not-valid", "amount": 1.0, "reason": VALID_REASON})
        assert bad_response is not None
        assert bad_response.error is not None
        assert bad_response.error.code == types.INVALID_PARAMS

        # Every line the subprocess wrote to stdout parsed as a genuine JSON-RPC
        # message (already proven per-line by recv(), which calls
        # jsonrpc_message_adapter.validate_json -- if any line had been stray
        # log/debug text, that call would have raised and failed the test
        # before we ever got here). Re-assert the count here as a guard against
        # a future refactor silently making recv() tolerant of bad lines.
        assert len(session.stdout_lines) == 3
        for line in session.stdout_lines:
            assert line.strip().startswith("{")
            assert '"jsonrpc"' in line

    def test_malformed_input_is_dropped_not_crashed(self, stdio_session: _StdioSession):
        """Malformed input on stdin must never surface as stray stdout text or a crash.

        Empirically verified SDK behavior (see server.py module docstring and
        README): a line that fails JSON-RPC message parsing is caught inside
        `mcp.server.stdio.stdio_server`'s reader and delivered to the
        dispatcher as a bare `Exception`, which the dispatcher only
        `logger.debug`s and drops -- no response is written for it at all. This
        differs from the JSON-RPC spec's usual expectation of a -32700 parse
        error reply; it is documented here and in the README rather than
        silently relied upon.
        """
        session = stdio_session
        session.send_raw("this is not json at all {{{")
        session.send_raw('{"no_jsonrpc_fields_here": true}')

        # The server must still be alive and correctly answer the next valid request.
        session.send(types.JSONRPCRequest(jsonrpc="2.0", id=5, method="tools/list", params=None))
        response = session.recv()
        assert response is not None, "server did not respond after malformed input; may have crashed"
        assert getattr(response, "error", None) is None
        assert response.id == 5

        # Only one line should have been produced for the one valid request.
        assert len(session.stdout_lines) == 1

    def test_stderr_carries_logs_not_stdout(self, stdio_session: _StdioSession):
        session = stdio_session
        bad_response = session.call_tool(6, "get_customer_record", {"customer_id": "bad"})
        assert bad_response is not None
        assert bad_response.error is not None

        stderr_text = session.close()
        assert "Rejected get_customer_record arguments" in stderr_text
        # And that log text never appeared on stdout.
        for line in session.stdout_lines:
            assert "Rejected" not in line
