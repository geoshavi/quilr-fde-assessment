"""Local mock LLM upstream: streams caller-supplied text back in small chunks.

No real provider, no network egress, no credentials -- a small FastAPI app the
gateway calls over real httpx streaming, so the gateway's streaming/redaction
pipeline is exercised end-to-end without needing any real LLM. `chunk_size`
lets tests drive arbitrary, even pathological (e.g. character-by-character)
chunk boundaries through the real HTTP layer.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

app = FastAPI(title="Quilr Task 3 Mock LLM Upstream")


class GenerateRequest(BaseModel):
    text: str
    chunk_size: int = Field(default=16, gt=0)


async def _stream_chunks(text: str, chunk_size: int) -> AsyncIterator[bytes]:
    for i in range(0, len(text), chunk_size):
        yield text[i : i + chunk_size].encode("utf-8")


@app.post("/generate")
async def generate(req: GenerateRequest) -> StreamingResponse:
    return StreamingResponse(_stream_chunks(req.text, req.chunk_size), media_type="text/plain")
