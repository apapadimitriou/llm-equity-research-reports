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


def find_reports(root: Path, morningstar_only: bool, human_reports: bool) -> list[Path]:
    files = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if human_reports and p.suffix.lower() != ".pdf":
            continue
        if morningstar_only and "morningstar" not in p.parts:
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
        rel_parent = rel_parent / source
    tasks: list[EvalTask] = []
    for key, prompt in EVALS.items():
        out_dir = output_root / provider / rel_parent
        out_path = out_dir / f"{ticker}_{key}_eval.json"
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
        raw = extract_json_block(raw).strip()
        ok, parsed = try_json(raw)
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
    files = find_reports(config.input_root, config.morningstar_only, config.human_reports)
    if not files:
        scope = "Morningstar " if config.morningstar_only else ""
        print(f"No {scope}input files found under: {config.input_root}")
        return

    reports: Dict[Path, str] = {}
    read_tasks = [read_report(p, config.human_reports) for p in files]
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
