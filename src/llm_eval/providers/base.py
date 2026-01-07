from typing import Protocol


class ProviderClient(Protocol):
    async def generate(self, prompt: str, strict_message: str) -> str:
        ...
