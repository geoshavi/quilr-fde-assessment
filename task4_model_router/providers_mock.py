"""Task 4 -- Injectable mock providers.

Plain async-callable test doubles matching `router.ProviderCallable`'s
shape (`async (prompt, requested_tokens) -> ProviderResponse`). No real
network egress, no external LLM calls -- controllable success/status,
latency, and failure, with call count/order recorded for assertions.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from task4_model_router.router import ProviderResponse


@dataclass
class MockProvider:
    status_code: int = 200
    body: str = "ok"
    delay_seconds: float = 0.0
    exception: BaseException | None = None
    name: str = "provider"
    call_log: list[str] = field(default_factory=list)
    """Appended with `name` on every call. Pass the SAME list to two
    MockProvider instances (e.g. primary and secondary) to observe their
    relative call order, not just each one's own count."""

    async def __call__(self, prompt: str, requested_tokens: int) -> ProviderResponse:
        self.call_log.append(self.name)
        if self.delay_seconds:
            await asyncio.sleep(self.delay_seconds)
        if self.exception is not None:
            raise self.exception
        return ProviderResponse(status_code=self.status_code, body=self.body)

    @property
    def call_count(self) -> int:
        return sum(1 for entry in self.call_log if entry == self.name)
