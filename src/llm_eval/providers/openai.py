import asyncio
from typing import Any

from openai import AsyncOpenAI

from llm_eval.utils import backoff_sleep


def _extract_text(resp: Any) -> str:
    text = getattr(resp, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text
    output = getattr(resp, "output", None) or []
    collected = []
    for msg in output:
        content = getattr(msg, "content", None) or []
        for item in content:
            t = getattr(item, "type", None) or (isinstance(item, dict) and item.get("type"))
            if t in {"output_text", "text"}:
                val = getattr(item, "text", None) or (isinstance(item, dict) and item.get("text"))
                if val:
                    collected.append(val)
    return "\n".join(collected).strip()


class OpenAIClient:
    def __init__(self, api_key: str, model: str, timeout_s: int, retries: int = 3) -> None:
        self._client = AsyncOpenAI(api_key=api_key, timeout=timeout_s)
        self._model = model
        self._timeout_s = timeout_s
        self._retries = retries

    async def generate(self, prompt: str, strict_message: str) -> str:
        last_err: Exception | None = None
        for attempt in range(self._retries + 1):
            try:
                resp = await asyncio.wait_for(
                    self._client.responses.create(
                        model=self._model,
                        input=[
                            {"role": "user", "content": prompt},
                            {"role": "user", "content": strict_message},
                        ],
                        temperature=0,
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
