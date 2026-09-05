<img width="1664" height="928" alt="read me" src="https://github.com/user-attachments/assets/32b51151-5d6f-4311-898b-ef67ef72d6bb" />


# QuilrAI FDE / AI Solutions Engineer Assessment

## Overview

Four self-contained, independently testable components, each mapping to one
assessment task:

1. **Task 1 — MCP Server** (`task1_mcp_server/`): a stdio MCP server exposing
   two tools (`get_customer_record`, `trigger_refund`) with strict input
   validation and genuine JSON-RPC error semantics.
2. **Task 2 — MCP Security Gateway** (`task2_mcp_gateway/`): a FastAPI
   HTTP/JSON-RPC reverse proxy that authenticates callers via bearer token,
   blocks non-admin callers from `admin_*` tools, and sits in front of a mock
   downstream MCP server.
3. **Task 3 — Streaming PII Guardrail** (`task3_stream_guardrail/`): a FastAPI
   proxy that streams a mock LLM's response back to the caller while
   redacting emails, SSNs, and Luhn-valid card numbers in real time, safely
   across arbitrary chunk boundaries.
4. **Task 4 — Rate Limiter + Fallback Router** (`task4_model_router/`): a
   per-tenant, SQLite-backed, concurrency-safe token rate limiter combined
   with a primary/secondary provider router with a narrow, explicit fallback
   policy.

Each task has its own mock dependencies (mock customer data, a mock
downstream MCP server, a mock LLM upstream, mock providers) so the whole
suite runs offline, deterministically, with no real network calls or
external services.

## Tech stack

- Python 3.12
- MCP official Python SDK (`mcp`)
- FastAPI
- httpx
- Pydantic v2
- SQLite (via `sqlite3`, stdlib)
- asyncio
- pytest / pytest-asyncio

## Repository structure

```
## Repository structure

```text
docs/                    Static HTML summary page for reviewers (`index.html`)
task1_mcp_server/        Task 1 — MCP stdio server, tool input models, mock data
task2_mcp_gateway/       Task 2 — HTTP/JSON-RPC gateway, auth, mock downstream
task3_stream_guardrail/  Task 3 — streaming proxy, PII redactor, mock LLM upstream
task4_model_router/      Task 4 — rate limiter, router, error envelope, mock providers
tests/                   Pytest suite covering all four tasks (1,224 tests)
.env.example             Optional local-run configuration (placeholder values only)
.gitignore               Git exclusions for envs, caches, DB state, and secrets
pytest.ini               Pytest configuration
README.md                Project documentation
requirements.txt         Python dependencies
```

## Setup

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If you'd rather not activate the virtual environment, every command below
also works by calling the venv's `python.exe` directly:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Running tests

Full suite:

```powershell
.venv\Scripts\python.exe -m pytest -q
```

Task-specific:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_task1_mcp_server.py -q
.venv\Scripts\python.exe -m pytest tests/test_task2_gateway.py -q
.venv\Scripts\python.exe -m pytest tests/test_task3_redactor.py tests/test_task3_adjacency.py -q
.venv\Scripts\python.exe -m pytest tests/test_task4_rate_limiter.py tests/test_task4_router.py -q
```

## Task 1 — MCP Server

An MCP server (stdio transport, official low-level `mcp.server.lowlevel.Server`
API) exposing two tools:

- `get_customer_record(customer_id: str)` — looks up a deterministic mock
  customer record.
- `trigger_refund(customer_id: str, amount: float, reason: str)` — returns a
  deterministic mock refund result.

**Run it directly** (mainly to confirm it starts; a real MCP client normally
launches it as a subprocess with this repo's root as the working directory):

```powershell
.venv\Scripts\python.exe -m task1_mcp_server.server
```

It then waits for JSON-RPC messages on stdin.

**Validation:**

| Field | Rule |
|---|---|
| `customer_id` | Must match `CUST-` + exactly 5 **ASCII** decimal digits (`CUST-[0-9]{5}`), e.g. `CUST-12345`. Unicode digit lookalikes (e.g. fullwidth `１２３４５`) are rejected — the pattern is ASCII-only, not the Unicode-aware `\d`. |
| `amount` (refund only) | int or float, strictly > 0, and finite. Numeric strings, bools, `None`, `NaN`, `+Infinity`, `-Infinity` are all rejected, never coerced. |
| `reason` (refund only) | an actual string, minimum length 10. `bytes` and any other non-`str` type are rejected outright. |
| *(extra fields)* | rejected — both tools forbid unrecognized input fields. |

**stdout/stderr isolation:** stdout is reserved exclusively for MCP protocol
JSON-RPC traffic. No `print()` is used anywhere in Task 1; all logging is
explicitly configured to stderr. Verified by an integration test that spawns
the server as a real subprocess and asserts every stdout line parses as a
valid JSON-RPC message.

**JSON-RPC error behavior:** every invalid input produces a genuine top-level
JSON-RPC error (`code: -32602 INVALID_PARAMS`), not a "successful" response
with an error flag buried in the payload — this required using the SDK's
low-level `Server` API instead of its high-level convenience layer, which
swallows validation errors into `CallToolResult(is_error=True)` (a nominal
success on the wire).

## Task 2 — MCP Security Gateway

A FastAPI HTTP/JSON-RPC reverse proxy (`proxy.py`) sitting between a caller
and a downstream mock MCP server (`downstream_mock.py`). Every request must
carry `Authorization: Bearer <token>`, resolved to a role via a small
hardcoded mapping (`auth.py`, demo tokens only). `tools/list` (and any
method other than `tools/call`) is forwarded to downstream unchanged;
`tools/call` is inspected — a tool name starting with `admin_` requires role
`admin`.

**Run it** (two terminals):

```powershell
.venv\Scripts\python.exe -m uvicorn task2_mcp_gateway.downstream_mock:app --port 8100

# second terminal
$env:TASK2_DOWNSTREAM_URL = "http://127.0.0.1:8100"
.venv\Scripts\python.exe -m uvicorn task2_mcp_gateway.proxy:app --port 8200
```

```powershell
curl.exe -X POST http://127.0.0.1:8200/ `
  -H "Authorization: Bearer viewer-demo-token" -H "Content-Type: application/json" `
  -d '{"jsonrpc":"2.0","method":"tools/list","id":1}'
```

**Key behaviors:**

- Non-admin role calling an `admin_*` tool → JSON-RPC error `code: -32001,
  message: "Unauthorized Tool Call"`, returned by the gateway itself.
  Downstream is **never contacted** for a rejected call.
- The `admin_` prefix match is a literal, case-sensitive string comparison
  (`"admin_Foo"` is guarded, `"Admin_foo"`/`"ADMIN_foo"` are not).
- The caller's own bearer token is **never forwarded** to downstream —
  downstream is a trust boundary the gateway sits in front of, not behind.
- Downstream connection failures, non-2xx statuses, and malformed responses
  are all sanitized to a generic `-32603 Internal error` — the real
  downstream URL, exception text, stack trace, and any provider-controlled
  response body are never included in what's sent back to the caller.
- Missing/malformed/unknown bearer tokens are rejected with a plain HTTP
  `401` (before any JSON-RPC body is even parsed); all other outcomes
  (including the gateway's own JSON-RPC-level errors) are returned over HTTP
  200, per standard JSON-RPC-over-HTTP convention.

## Task 3 — Streaming PII Guardrail

A FastAPI proxy (`app.py`) that forwards a prompt to a local mock LLM
(`mock_llm.py`) and streams the response back to the caller through a
bounded, candidate-aware streaming redactor (`redactor.py`), replacing
detected PII with `[REDACTED]` as the response streams — never buffering the
whole response before redacting.

**Supported patterns** (documented explicitly — this is not a general-purpose
PII/RFC-compliant email or card parser):

- **Email addresses**: `local@domain.tld`-shaped strings within RFC 5321-derived
  length bounds (64-char local part, 254-char total address).
- **SSNs**: the fixed `DDD-DD-DDDD` form only.
- **Credit cards**: 13–19 contiguous digits (bare, or grouped with a single
  space/hyphen between digit blocks) that pass a **Luhn checksum**. A
  digit-shaped run that fails Luhn is left untouched, not assumed to be a
  card.

**Streaming guarantees:**

- **Chunk-safe**: PII split across arbitrary chunk boundaries (mid-username,
  around `@`, mid-digit, byte-split multi-byte UTF-8 characters) is still
  detected correctly and never partially leaked.
- **Bounded pending state**: the redactor retains only a small, format-derived
  amount of raw text between chunks (never proportional to total stream
  length), even under an adversarial all-digits stream.
- **No whole-response buffering**: plain, non-PII text is released
  incrementally as it arrives, not held until the stream ends.
- Upstream failure, client disconnect, `send()` failure, and generator
  cancellation all deterministically close the upstream connection.

**Run it** (two terminals):

```powershell
.venv\Scripts\python.exe -m uvicorn task3_stream_guardrail.mock_llm:app --port 8100

# second terminal
$env:TASK3_UPSTREAM_URL = "http://127.0.0.1:8100/generate"
.venv\Scripts\python.exe -m uvicorn task3_stream_guardrail.app:app --port 8200
```

```powershell
curl.exe -N -X POST http://127.0.0.1:8200/generate `
  -H "Content-Type: application/json" `
  -d '{"prompt":"contact me at john@example.com","chunk_size":4}'
```

## Task 4 — Rate Limiter + Fallback Router

**Rate limiter** (`rate_limiter.py`): per-tenant token admission on a rolling
60-second window (half-open interval `(now - 60s, now]`), capped at 50,000
tokens per tenant, backed by an on-disk SQLite file (state persists across
process restarts, not `:memory:`). Concurrency safety comes from SQLite's own
`BEGIN IMMEDIATE` write-lock semantics around a single read-modify-write
transaction per admission — not an in-process `asyncio.Lock`, since that
could never protect a second process/connection sharing the same file.

**Router** (`router.py`): tries the primary provider first; falls back to the
secondary **only** on:
- primary HTTP-style status `429`, or
- the router-enforced primary deadline (`TASK4_PRIMARY_TIMEOUT_MS`, default
  **3000ms**, enforced via `asyncio.wait_for`) expiring.

Everything else — a non-429 4xx, a 5xx, a malformed/missing status code, or
any raised exception from the primary — maps directly to a sanitized error
and **never** triggers fallback. This is deliberately not a broad
`except Exception: fallback()`. Secondary failures (of any kind, including a
second 429) are terminal — there is no tertiary fallback — and are sanitized
the same way: a failing provider's raw status/body is never forwarded to the
caller, since it may itself carry a URL, key, or other internal detail.

## Token accounting assumption

No tokenizer is implemented or assumed anywhere in this codebase. Every
caller of the rate limiter supplies `requested_tokens` as an explicit,
caller-provided, positive integer representing the full cost of the request.
That amount is reserved atomically at admission time, before any provider is
ever called, and is **not refunded or reconciled** afterward even if the
eventual provider call (primary or fallback) fails — the assessment
specifies no usage-based post-hoc accounting, and admission is required to
happen before any provider is contacted.

## Important design decisions / assumptions

- **Task 1 — `CUST-XXXXX` = exactly 5 ASCII digits.** The pattern is
  deliberately `[0-9]{5}`, not the Unicode-aware `\d` pydantic-core's regex
  engine would otherwise use, which accepts any Unicode category-Nd digit
  (e.g. fullwidth `CUST-１２３４５`) — a real identifier-confusion risk.
- **Task 1 — malformed raw stdin** (not valid JSON, or valid JSON that isn't
  a JSON-RPC shape) is silently dropped by the underlying `mcp` SDK's stdio
  transport — logged at `DEBUG`, no response sent — rather than answered with
  `-32700`. This is inherited SDK behavior, not something Task 1's own code
  controls, and is documented/tested rather than assumed.
- **Task 2 — method matching is exact and literal.** `tools/call` is the only
  specially-handled method; every other method name (including anything
  resembling `tools/list`) is forwarded transparently, and the `admin_`
  prefix check is a plain, case-sensitive `str.startswith`.
- **Task 3 — card validation is Luhn-based**, applied to every contiguous
  13–19 digit run (bare or single-separator-grouped) that could plausibly be
  a card; a run that is digit-shaped but fails Luhn is left as plain text.
- **Task 3 — email bounds are derived from RFC 5321's 254-octet total
  address / 64-octet local-part limits**, not arbitrary constants, so the
  redactor knows exactly how much raw text it must retain across a chunk
  boundary before it can safely release text.
- **Task 4 — the rolling window is half-open**, `(now - 60s, now]`: a request
  timestamped exactly at the 60-second-old boundary has already expired.
- **Task 4 — fallback is a closed set of exactly two conditions** (429,
  timeout); every other primary failure mode is intentionally excluded from
  fallback to avoid masking non-transient provider errors as if they were
  transient.

## Verification

- Full suite: **1,224 tests passed**, 0 failed, 0 skipped.
- An independent adversarial verification pass (M5) additionally exercised
  concurrency (race conditions under simultaneous rate-limiter admission),
  protocol edge cases (malformed JSON-RPC, Unicode confusables, NaN/Infinity,
  bool-as-int), streaming lifecycle (disconnect/cancellation/upstream
  failure cleanup), and information-leakage paths (sanitized error bodies,
  no Authorization forwarding) across all four tasks, with 0 BLOCKER and 0
  IMPORTANT findings.
- This means the implemented behavior matches its documented contract under
  the scenarios above — it is not a claim of exhaustive correctness or
  freedom from every possible defect.

## Security / hygiene

- No secrets, API keys, or credentials are committed; Task 2's demo bearer
  tokens are explicitly labeled as non-secret placeholders.
- Downstream/provider errors (Task 2 and Task 4) are sanitized before
  reaching the caller — raw bodies, exception text, stack traces, internal
  URLs, and file paths are never forwarded.
- The caller's own Authorization header is never forwarded downstream
  (Task 2).
- Task 1's stdout is reserved exclusively for MCP protocol traffic; all
  logging goes to stderr.
- `.gitignore` excludes virtual environments, caches, local SQLite state,
  and `.env` (only `.env.example`, containing placeholders, is committed).

## Submission notes

To evaluate locally:

1. `python -m venv .venv && .venv\Scripts\Activate.ps1 && python -m pip install -r requirements.txt`
2. `.venv\Scripts\python.exe -m pytest -q` — should report `1224 passed`.
3. Each task can also be exercised live using the per-task "Run it" commands
   above (Tasks 2–4 need two terminals: a mock upstream/downstream and the
   component under test; Task 1 is a single stdio process).
4. No external services, API keys, or network access are required — every
   upstream/downstream dependency is a local mock included in this repo.
