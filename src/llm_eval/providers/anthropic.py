import asyncio
from typing import Any

from anthropic import AsyncAnthropic

from llm_eval.utils import backoff_sleep


def _extract_text(resp: Any) -> str:
    content = getattr(resp, "content", None) or []
    parts = []
    for item in content:
        text = getattr(item, "text", None)
        if text:
            parts.append(text)
    return "\n".join(parts).strip()


class AnthropicClient:
    def __init__(self, api_key: str, model: str, timeout_s: int, retries: int = 3) -> None:
        self._client = AsyncAnthropic(api_key=api_key, timeout=timeout_s)
        self._model = model
        self._timeout_s = timeout_s
        self._retries = retries

    async def generate(self, prompt: str, strict_message: str) -> str:
        last_err: Exception | None = None
        for attempt in range(self._retries + 1):
            try:
                resp = await asyncio.wait_for(
                    self._client.messages.create(
                        model=self._model,
                        max_tokens=8192,
                        temperature=0,
                        messages=[
                            {"role": "user", "content": prompt},
                            {"role": "user", "content": strict_message},
                        ],
                    ),
                    timeout=self._timeout_s,
                )
                return _extract_text(resp)
            except Exception as e:
                last_err = e
                if attempt < self._retries:
                    await backoff_sleep(attempt + 1)
                else:
                    raise last_err
