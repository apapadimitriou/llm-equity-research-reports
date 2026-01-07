import asyncio
import json
from pathlib import Path
from typing import Any, Tuple


def ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def extract_json_block(text: str) -> str:
    if "```" not in text:
        return text
    parts = text.split("```")
    if len(parts) < 3:
        return text
    block = parts[1]
    if block.strip().startswith("json"):
        return block.split("\n", 1)[1] if "\n" in block else ""
    return block


def try_json(text: str) -> Tuple[bool, Any]:
    try:
        return True, json.loads(text)
    except json.JSONDecodeError:
        return False, text


async def backoff_sleep(attempt: int) -> None:
    await asyncio.sleep(min(2 ** attempt, 30) + 0.1 * attempt)
