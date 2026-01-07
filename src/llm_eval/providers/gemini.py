import asyncio
from typing import Any

from google import genai
from google.genai import types

from llm_eval.utils import backoff_sleep


def _extract_text(resp: Any) -> str:
    return resp.text or ""


class GeminiClient:
    def __init__(
        self,
        api_key: str,
        model: str,
        timeout_s: int,
        max_output_tokens: int,
        retries: int = 3,
    ) -> None:
        http_options = types.HttpOptions(async_client_args={})
        self._client = genai.Client(api_key=api_key, http_options=http_options)
        self._model = model
        self._timeout_s = timeout_s
        self._max_output_tokens = max_output_tokens
        self._retries = retries

    async def generate(self, prompt: str, strict_message: str) -> str:
        last_err: Exception | None = None
        for attempt in range(self._retries + 1):
            try:
                resp = await asyncio.wait_for(
                    self._client.aio.models.generate_content(
                        model=self._model,
                        contents=[prompt, strict_message],
                        config=types.GenerateContentConfig(
                            temperature=0,
                            max_output_tokens=self._max_output_tokens,
                            response_mime_type="application/json",
                        ),
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
