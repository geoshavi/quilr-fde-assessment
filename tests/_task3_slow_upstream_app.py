"""Test-only helper: a deliberately slow local FastAPI upstream.

Used exclusively by test_task3_redactor.py::test_real_process_streams_incrementally_not_buffered
to empirically prove the gateway forwards bytes as they arrive rather than
buffering the full upstream response. Not part of the Task 3 implementation
itself -- kept in tests/ and launched only as a subprocess by that one test.
"""

import asyncio
from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()


async def _slow_chunks() -> AsyncIterator[bytes]:
    for i in range(20):
        await asyncio.sleep(0.05)
        yield f"segment{i} ".encode()


@app.post("/generate")
async def generate() -> StreamingResponse:
    return StreamingResponse(_slow_chunks(), media_type="text/plain")
