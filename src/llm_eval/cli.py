import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from llm_eval.config import EvalConfig, resolve_model
from llm_eval.providers import AnthropicClient, GeminiClient, OpenAIClient
from llm_eval.runner import run


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run equity report evaluations.")
    parser.add_argument("--provider", choices=["openai", "anthropic", "gemini"], required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--input-root", default="output/report")
    parser.add_argument("--output-root", default="output/evaluation")
    parser.add_argument("--max-concurrency", type=int, default=8)
    parser.add_argument("--timeout-s", type=int, default=3600)
    parser.add_argument("--all-sources", action="store_true")
    parser.add_argument(
        "--human-reports",
        action="store_true",
        help="Read PDF reports from output/human_reports (source inferred from filename).",
    )
    parser.add_argument(
        "--strict-message",
        default="Grade strictly and harshly.",
        help="Additional user instruction appended to every request.",
    )
    return parser


def _build_client(provider: str, model: str, timeout_s: int):
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        return OpenAIClient(api_key=api_key, model=model, timeout_s=timeout_s)
    if provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set.")
        return AnthropicClient(api_key=api_key, model=model, timeout_s=timeout_s)
    if provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set.")
        return GeminiClient(api_key=api_key, model=model, timeout_s=timeout_s)
    raise ValueError(f"Unknown provider: {provider}")


def main() -> None:
    load_dotenv()
    args = _build_arg_parser().parse_args()
    model = resolve_model(args.provider, args.model)
    morningstar_only = not args.all_sources
    if args.human_reports:
        morningstar_only = False
    config = EvalConfig(
        input_root=Path("output/human_reports" if args.human_reports else args.input_root),
        output_root=Path(args.output_root),
        provider=args.provider,
        model=model,
        max_concurrency=args.max_concurrency,
        morningstar_only=morningstar_only,
        human_reports=args.human_reports,
        timeout_s=args.timeout_s,
        strict_message=args.strict_message,
    )
    client = _build_client(args.provider, model, args.timeout_s)
    import asyncio

    asyncio.run(run(config, client))
