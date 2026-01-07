import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

from tqdm import tqdm

from llm_eval.config import EvalConfig
from llm_eval.prompts import EVALS
from llm_eval.providers.base import ProviderClient
from llm_eval.utils import ensure_dir, extract_json_block, try_json


@dataclass
class EvalTask:
    eval_key: str
    prompt: str
    ticker: str
    report_text: str
    out_path: Path
    source_path: Path


def _ticker_from_path(p: Path, human_reports: bool) -> str:
    if human_reports:
        return p.stem.split("_", 1)[0]
    return p.stem


def _build_ticker_allowlist(root: Path, human_reports: bool) -> set[str]:
    allowlist: set[str] = set()
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if human_reports and p.suffix.lower() != ".pdf":
            continue
        allowlist.add(_ticker_from_path(p, human_reports))
    return allowlist


def find_reports(
    root: Path,
    morningstar_only: bool,
    human_reports: bool,
    ticker_allowlist: set[str] | None,
) -> list[Path]:
    files = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if human_reports and p.suffix.lower() != ".pdf":
            continue
        if morningstar_only and "morningstar" not in p.parts:
            continue
        if ticker_allowlist is not None:
            if _ticker_from_path(p, human_reports) not in ticker_allowlist:
                continue
        files.append(p)
    return sorted(files)


def _extract_pdf_text(p: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(p))
    chunks = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text:
            chunks.append(text)
    return "\n".join(chunks)


def _infer_source_from_filename(p: Path) -> str:
    name = p.stem.lower()
    if "morningstar" in name:
        return "morningstar"
    if "argus" in name:
        return "argus"
    return "unknown"


async def read_report(p: Path, human_reports: bool) -> Tuple[Path, str | None]:
    try:
        if human_reports:
            text = await asyncio.to_thread(_extract_pdf_text, p)
        else:
            text = await asyncio.to_thread(p.read_text, errors="ignore")
        return p, text
    except Exception as e:
        print(f"Failed to read {p}: {e}")
        return p, None


def build_tasks(
    report_path: Path,
    report_text: str,
    input_root: Path,
    output_root: Path,
    provider: str,
    human_reports: bool,
) -> list[EvalTask]:
    ticker = report_path.stem
    rel_parent = report_path.parent.relative_to(input_root)
    if human_reports:
        ticker = report_path.stem.split("_", 1)[0]
        source = _infer_source_from_filename(report_path)
        rel_parent = rel_parent / source / "analyst"
    tasks: list[EvalTask] = []
    for key, prompt in EVALS.items():
        out_dir = output_root / provider / rel_parent
        out_path = out_dir / f"{ticker}_{key}_eval.json"
        if out_path.exists():
            continue
        tasks.append(
            EvalTask(
                eval_key=key,
                prompt=prompt,
                ticker=ticker,
                report_text=report_text,
                out_path=out_path,
                source_path=report_path,
            )
        )
    return tasks


def _expected_output_paths(
    report_path: Path,
    input_root: Path,
    output_root: Path,
    provider: str,
    human_reports: bool,
) -> list[Path]:
    ticker = report_path.stem
    rel_parent = report_path.parent.relative_to(input_root)
    if human_reports:
        ticker = report_path.stem.split("_", 1)[0]
        source = _infer_source_from_filename(report_path)
        rel_parent = rel_parent / source / "analyst"
    out_dir = output_root / provider / rel_parent
    return [out_dir / f"{ticker}_{key}_eval.json" for key in EVALS.keys()]


async def evaluate_task(
    task: EvalTask,
    client: ProviderClient,
    sem: asyncio.Semaphore,
    strict_message: str,
) -> Tuple[str, Path, bool, str]:
    ensure_dir(task.out_path)
    try:
        report_wrapped = f"<report>{task.report_text}</report>"
        full_prompt = f"{task.prompt}\n {report_wrapped}"
        async with sem:
            raw = await client.generate(full_prompt, strict_message)
        extracted = extract_json_block(raw).strip()
        ok, parsed = try_json(extracted)
        if not ok:
            repair_prompt = (
                f"{full_prompt}\n\n"
                "Return ONLY valid JSON that matches the required schema. "
                "Do not include markdown. Keep the response concise and within the "
                "word limits specified in the prompt."
            )
            async with sem:
                raw = await client.generate(repair_prompt, strict_message)
            extracted = extract_json_block(raw).strip()
            ok, parsed = try_json(extracted)
        if ok:
            with task.out_path.open("w") as f:
                json.dump(parsed, f, indent=4)
        else:
            with task.out_path.with_suffix(".raw.txt").open("w") as f:
                f.write(raw)
        return task.eval_key, task.out_path, ok, ""
    except Exception as e:
        err_path = task.out_path.with_suffix(".error.txt")
        with err_path.open("w") as f:
            f.write(str(e))
        return task.eval_key, err_path, False, str(e)


async def run(config: EvalConfig, client: ProviderClient) -> None:
    ticker_allowlist = None
    if config.ticker_allowlist_root:
        ticker_allowlist = _build_ticker_allowlist(
            config.ticker_allowlist_root, config.human_reports
        )
    files = find_reports(
        config.input_root, config.morningstar_only, config.human_reports, ticker_allowlist
    )
    if not files:
        scope = "Morningstar " if config.morningstar_only else ""
        print(f"No {scope}input files found under: {config.input_root}")
        return

    reports: Dict[Path, str] = {}
    to_read = []
    for p in files:
        expected = _expected_output_paths(
            p, config.input_root, config.output_root, config.provider, config.human_reports
        )
        if all(path.exists() for path in expected):
            continue
        to_read.append(p)

    if not to_read:
        print("All reports already evaluated. Nothing to do.")
        return

    read_tasks = [read_report(p, config.human_reports) for p in to_read]
    for coro in tqdm(asyncio.as_completed(read_tasks), total=len(read_tasks), desc="Reading reports"):
        p, txt = await coro
        if txt:
            reports[p] = txt

    all_tasks: list[EvalTask] = []
    for p, txt in reports.items():
        all_tasks.extend(
            build_tasks(
                p,
                txt,
                config.input_root,
                config.output_root,
                config.provider,
                config.human_reports,
            )
        )

    sem = asyncio.Semaphore(config.max_concurrency)
    eval_tasks = [evaluate_task(t, client, sem, config.strict_message) for t in all_tasks]
    results = []
    for coro in tqdm(asyncio.as_completed(eval_tasks), total=len(eval_tasks), desc="Evaluating"):
        results.append(await coro)

    ok_count = sum(1 for _, _, is_json, _ in results if is_json)
    total = len(results)
    print(f"\nDone. Parsed valid JSON for {ok_count}/{total} evaluations.")
    bad = [(k, str(p), err) for k, p, okj, err in results if not okj and err]
    if bad:
        print("\nErrors (first 10):")
        for k, path, err in bad[:10]:
            print(f" - {k} -> {path}: {err}")
