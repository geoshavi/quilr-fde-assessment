"""Task 3 -- LLM Gateway Streaming Guardrail.

Proxies a text-generation request to a local mock LLM upstream
(task3_stream_guardrail.mock_llm) and streams the response back to the caller
with PII redacted in real time via task3_stream_guardrail.redactor.

Design decisions:
- The upstream call uses httpx's streaming API (`client.stream(...)` +
  `response.aiter_text()`); the full upstream response is never read into
  memory before forwarding starts, and `aiter_text()`'s incremental UTF-8
  decoder (not manual `.decode()` per chunk) is what keeps multi-byte
  characters split across chunk boundaries from being corrupted.
- The upstream connection is held open via `async with http_client.stream(...)`
  for exactly the lifetime of the downstream generator. If the client
  disconnects mid-stream, Starlette closes this generator (raising
  `GeneratorExit` at the suspended `yield`), which unwinds the `async with`
  and closes the upstream connection -- no separate cancellation handling
  code is needed beyond using the context manager correctly.
- Nothing about the streamed text (safe or redacted) is ever logged; only
  generic lifecycle/error events are, and only at a level that never
  includes response content.
"""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import Depends, FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from starlette.types import Send

from task3_stream_guardrail.redactor import StreamingRedactor

logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("task3_stream_guardrail")

UPSTREAM_URL = os.environ.get("TASK3_UPSTREAM_URL", "http://localhost:8100/generate")
DEFAULT_CHUNK_SIZE = 24

# A single shared client for real (non-test) runs. Tests replace this entirely
# via app.dependency_overrides[get_http_client].
_shared_http_client = httpx.AsyncClient(timeout=30.0)


def get_http_client() -> httpx.AsyncClient:
    return _shared_http_client


class GenerateRequest(BaseModel):
    prompt: str
    chunk_size: int = Field(default=DEFAULT_CHUNK_SIZE, gt=0)


class _ClosingStreamingResponse(StreamingResponse):
    """StreamingResponse that deterministically closes its body iterator.

    Starlette's `stream_response()` iterates `body_iterator` but never closes
    it (see starlette/responses.py). So if `send()` raises -- the client
    vanished after we had already written some of the response -- or the
    surrounding task is cancelled, our generator is left suspended at its
    `yield` and the upstream httpx stream it is holding open is only released
    whenever the event loop eventually finalizes the abandoned async
    generator. Closing it here makes that release immediate and deterministic
    instead of depending on garbage collection timing.
    """

    async def stream_response(self, send: Send) -> None:
        try:
            await super().stream_response(send)
        finally:
            aclose = getattr(self.body_iterator, "aclose", None)
            if aclose is not None:
                await aclose()


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Close the shared upstream client on shutdown.

    Tests never trigger ASGI lifespan events (they call the app directly via
    httpx.ASGITransport and override get_http_client), so this only affects
    real `uvicorn ...:app` runs.

    Wrapped in try/finally: without it, an exception propagating through the
    lifespan context (e.g. a startup/serving failure) would skip the aclose()
    entirely, since a bare `yield` followed by a statement only runs that
    statement on the ordinary, non-raising path.
    """
    try:
        yield
    finally:
        await _shared_http_client.aclose()


app = FastAPI(title="Quilr Task 3 Streaming Guardrail", lifespan=_lifespan)


async def _redacted_stream(prompt: str, chunk_size: int, http_client: httpx.AsyncClient) -> AsyncIterator[bytes]:
    redactor = StreamingRedactor()
    try:
        async with http_client.stream(
            "POST", UPSTREAM_URL, json={"text": prompt, "chunk_size": chunk_size}
        ) as upstream:
            async for text_chunk in upstream.aiter_text():
                safe = redactor.feed(text_chunk)
                if safe:
                    yield safe.encode("utf-8")
    except httpx.HTTPError:
        logger.exception("Upstream stream failed")
        return
    tail = redactor.flush()
    if tail:
        yield tail.encode("utf-8")


@app.post("/generate")
async def generate(
    req: GenerateRequest, http_client: httpx.AsyncClient = Depends(get_http_client)
) -> StreamingResponse:
    return _ClosingStreamingResponse(
        _redacted_stream(req.prompt, req.chunk_size, http_client), media_type="text/plain"
    )
