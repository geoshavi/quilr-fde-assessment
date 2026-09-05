"""Task 1 MCP server: get_customer_record and trigger_refund over stdio.

SDK behavior discovered during M1 (verified by reading the installed `mcp` 2.1.1
source, not assumed from older SDK examples):

- The high-level `mcp.server.mcpserver.MCPServer` (the renamed FastMCP) catches
  *every* exception a tool raises -- including a `pydantic.ValidationError` from
  its own automatic argument binding -- and converts it into a
  `CallToolResult(is_error=True)`. That is a normal, successful JSON-RPC
  *response* whose payload happens to carry an error flag; it is NOT a
  JSON-RPC-level error object (see mcp/server/mcpserver/server.py
  `_handle_call_tool`, and mcp/server/mcpserver/tools/base.py where
  `ValidationError` is caught and re-raised as `ToolError`).
- This server therefore uses the low-level `mcp.server.lowlevel.Server` instead,
  registers `tools/call` with our own handler, and validates arguments
  ourselves with the Pydantic models in models.py. A validation failure is
  raised as `mcp.shared.exceptions.MCPError(code=INVALID_PARAMS, ...)`, which
  the low-level dispatcher's `_on_request` boundary re-raises untouched
  (mcp/shared/jsonrpc_dispatcher.py `handler_exception_to_error_data`) --
  producing a genuine top-level JSON-RPC error object (code -32602) on the wire,
  which is what "proper MCP / JSON-RPC error semantics" requires.
- `mcp.server.stdio.stdio_server()` in this SDK version already claims fd 1
  (stdout) for the wire and reroutes the process's stdout to stderr for the
  duration of the connection (see mcp/server/stdio.py `_claim_fd` /
  `stdio_server`), so an accidental `print()` in handler code would land on
  stderr rather than corrupt the protocol stream. This server does not rely on
  that safety net -- no `print()` is used anywhere in Task 1, and logging is
  explicitly configured to stderr below -- but it is worth recording since it
  changes what a "stdout contamination" bug would even look like on this SDK
  version versus older ones.
"""

from __future__ import annotations

import json
import logging
import sys

import anyio
import mcp_types as types
from mcp.server.context import ServerRequestContext
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server
from mcp.shared.exceptions import MCPError
from pydantic import ValidationError

from task1_mcp_server import mock_data
from task1_mcp_server.models import GetCustomerRecordInput, TriggerRefundInput

# Logging must never touch stdout: stdout is reserved exclusively for MCP
# protocol messages. Configuring the root logger's stream explicitly (rather
# than relying on logging's implicit default) keeps that true even if some
# other import path calls logging.basicConfig() first.
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("task1_mcp_server")


def _validation_error_data(tool_name: str, exc: ValidationError) -> dict[str, object]:
    """Field-level detail only: loc/type/msg, no echoed input values.

    Mirrors the SDK's own restraint in mcp/server/mcpserver/server.py
    (`_handle_call_tool`), which logs only rejected field names, not values.
    """
    return {
        "tool": tool_name,
        "errors": [
            {"loc": list(err["loc"]), "type": err["type"], "msg": err["msg"]}
            for err in exc.errors(include_url=False, include_context=True, include_input=False)
        ],
    }


async def _list_tools(
    ctx: ServerRequestContext[None], params: types.PaginatedRequestParams | None
) -> types.ListToolsResult:
    return types.ListToolsResult(
        tools=[
            types.Tool(
                name="get_customer_record",
                description="Look up a mock customer record by CUST-XXXXX id.",
                input_schema=GetCustomerRecordInput.model_json_schema(),
            ),
            types.Tool(
                name="trigger_refund",
                description="Trigger a mock refund for a customer.",
                input_schema=TriggerRefundInput.model_json_schema(),
            ),
        ]
    )


async def _call_tool(
    ctx: ServerRequestContext[None], params: types.CallToolRequestParams
) -> types.CallToolResult:
    name = params.name
    arguments = params.arguments or {}

    if name == "get_customer_record":
        try:
            parsed = GetCustomerRecordInput.model_validate(arguments)
        except ValidationError as exc:
            logger.info("Rejected get_customer_record arguments: %s", [e["loc"] for e in exc.errors()])
            raise MCPError(
                code=types.INVALID_PARAMS,
                message="Invalid tool arguments",
                data=_validation_error_data(name, exc),
            ) from exc
        record = mock_data.get_customer_record(parsed.customer_id)
        return types.CallToolResult(content=[types.TextContent(type="text", text=_to_json(record))])

    if name == "trigger_refund":
        try:
            parsed = TriggerRefundInput.model_validate(arguments)
        except ValidationError as exc:
            logger.info("Rejected trigger_refund arguments: %s", [e["loc"] for e in exc.errors()])
            raise MCPError(
                code=types.INVALID_PARAMS,
                message="Invalid tool arguments",
                data=_validation_error_data(name, exc),
            ) from exc
        result = mock_data.trigger_refund(parsed.customer_id, parsed.amount, parsed.reason)
        return types.CallToolResult(content=[types.TextContent(type="text", text=_to_json(result))])

    logger.info("Rejected call to unknown tool: %r", name)
    raise MCPError(
        code=types.INVALID_PARAMS,
        message=f"Unknown tool: {name}",
        data={"tool": name},
    )


def _to_json(payload: dict[str, object]) -> str:
    return json.dumps(payload, sort_keys=True)


server: Server[None] = Server(
    name="quilr-task1-mcp-server",
    version="0.1.0",
    on_list_tools=_list_tools,
    on_call_tool=_call_tool,
)


async def _main() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main() -> None:
    anyio.run(_main)


if __name__ == "__main__":
    main()
