from dataclasses import dataclass
from pathlib import Path
from typing import Dict


DEFAULT_MODELS: Dict[str, str] = {
    "openai": "gpt-5",
    "anthropic": "claude-sonnet-4.5",
    "gemini": "gemini-3-flash-preview",
}


@dataclass(frozen=True)
class EvalConfig:
    input_root: Path
    output_root: Path
    provider: str
    model: str
    max_concurrency: int
    morningstar_only: bool
    human_reports: bool
    timeout_s: int
    max_output_tokens: int
    ticker_allowlist_root: Path | None
    strict_message: str


def resolve_model(provider: str, model_override: str | None) -> str:
    if model_override:
        return model_override
    if provider not in DEFAULT_MODELS:
        raise ValueError(f"Unknown provider: {provider}")
    return DEFAULT_MODELS[provider]
