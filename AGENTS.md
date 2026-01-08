# AGENTS.md - AI Coding Agent Guide

## Project Overview

**Purpose:** Multi-LLM evaluation system that grades equity research reports across 5 metrics using OpenAI (GPT-5), Anthropic (Claude), and Google Gemini providers. Also includes a separate tool for extracting financial forecasts from analyst PDFs.

**Tech Stack:** Python 3.11+, asyncio, OpenAI SDK, Anthropic SDK, Google GenAI SDK, pypdf

## Directory Structure

```
src/llm_eval/
├── cli.py                 # CLI entry point for evaluation pipeline
├── config.py              # EvalConfig dataclass, DEFAULT_MODELS dict
├── prompts.py             # 5 evaluation prompts (Depth, Coherence, Assumptions, Comprehensiveness, Originality)
├── runner.py              # Async orchestration: find_reports, build_tasks, evaluate_task
├── utils.py               # Helpers: extract_json_block, try_json, backoff_sleep
├── extract_forecasts.py   # Separate tool: extract forecasts from PDFs using GPT-5
└── providers/
    ├── __init__.py
    ├── base.py            # ProviderClient Protocol (interface)
    ├── openai.py          # OpenAIClient - uses responses.create() API
    ├── anthropic.py       # AnthropicClient - uses messages.create() API
    └── gemini.py          # GeminiClient - uses generate_content() API

output/
├── report/                # Input: LLM-generated text reports
├── human_reports/         # Input: Human analyst PDFs (Morningstar, Argus)
├── evaluation/            # Output: JSON evaluation results
└── human_forecasts/       # Output: Extracted forecast JSONs
```

## Key Files Reference

| File | Purpose | Key Functions/Classes |
|------|---------|----------------------|
| `cli.py` | Entry point | `main()`, `_build_client()` |
| `config.py` | Configuration | `EvalConfig`, `DEFAULT_MODELS`, `resolve_model()` |
| `prompts.py` | Evaluation prompts | `EVALS` dict with 5 prompts |
| `runner.py` | Pipeline orchestration | `run()`, `find_reports()`, `build_tasks()`, `evaluate_task()` |
| `utils.py` | Utilities | `extract_json_block()`, `try_json()`, `backoff_sleep()` |
| `extract_forecasts.py` | Forecast extraction | `ForecastExtractor`, `process_reports()`, `ForecastData` |
| `providers/base.py` | Provider interface | `ProviderClient` Protocol |
| `providers/openai.py` | OpenAI integration | `OpenAIClient`, `_extract_text()` |
| `providers/anthropic.py` | Anthropic integration | `AnthropicClient`, `_extract_text()` |
| `providers/gemini.py` | Gemini integration | `GeminiClient`, `_extract_text()` |

## Data Flow

### Evaluation Pipeline

```
CLI args → EvalConfig → runner.run()
  │
  ├─ find_reports(input_root) → list[Path]
  │    └─ Filters: morningstar_only, ticker_allowlist, human_reports
  │
  ├─ read_report(path) → text
  │    └─ PDF: pypdf extraction | Text: read_text()
  │
  ├─ build_tasks(report) → list[EvalTask]
  │    └─ Creates 5 tasks (one per metric), skips existing outputs
  │
  └─ evaluate_task(task, client) → JSON output
       ├─ Wraps report: <report>...</report>
       ├─ Sends: prompt + strict_message
       ├─ Parses JSON (with repair retry on failure)
       └─ Writes: {ticker}_{metric}_eval.json
```

### Forecast Extraction Pipeline

```
CLI args → process_reports()
  │
  ├─ Find PDFs in input_dir (filter by ticker if specified)
  │
  ├─ For each PDF:
  │    ├─ extract_pdf_text() via pypdf
  │    ├─ Infer source from filename (morningstar/argus)
  │    ├─ Select prompt (MORNINGSTAR_PROMPT or ARGUS_PROMPT)
  │    ├─ Call GPT-5 for extraction
  │    ├─ Convert units to full numbers (millions → actual)
  │    └─ Save: {ticker}_forecast.json
  │
  └─ Save combined: all_forecasts.json
```

## Commands

### Run Evaluation

```bash
# Activate venv
source .venv/bin/activate

# Evaluate with specific provider
python -m src.llm_eval.cli --provider openai
python -m src.llm_eval.cli --provider anthropic
python -m src.llm_eval.cli --provider gemini

# Include all sources (not just Morningstar)
python -m src.llm_eval.cli --provider openai --all-sources

# Evaluate human analyst PDFs
python -m src.llm_eval.cli --provider openai --human-reports

# Custom paths and concurrency
python -m src.llm_eval.cli --provider openai \
  --input-root output/report \
  --output-root output/evaluation \
  --max-concurrency 10
```

### Run Forecast Extraction

```bash
# Extract all forecasts
python -m src.llm_eval.extract_forecasts

# Specific quarter
python -m src.llm_eval.extract_forecasts --quarter 2025_q1

# Single ticker
python -m src.llm_eval.extract_forecasts --ticker AAPL

# Combined
python -m src.llm_eval.extract_forecasts --quarter 2025_q1 --ticker AAPL
```

## Provider Details

### Default Models (config.py)

```python
DEFAULT_MODELS = {
    "openai": "gpt-5",
    "anthropic": "claude-sonnet-4.5",
    "gemini": "gemini-3-flash-preview",
}
```

### Provider-Specific Behaviors

| Provider | API Method | JSON Mode | Temperature |
|----------|-----------|-----------|-------------|
| OpenAI | `responses.create()` | `response_format={"type": "json_object"}` | 0 |
| Anthropic | `messages.create()` | Prompt-based | 0 |
| Gemini | `generate_content()` | `response_mime_type="application/json"` | 0 |

### Environment Variables Required

```bash
# .env file
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
```

## Evaluation Metrics (prompts.py)

| Metric | EVALS Key | Measures |
|--------|-----------|----------|
| Depth | `depth` | Causal reasoning, quantified forecasts, counterpoints |
| Coherence | `coherence` | Section flow, terminology consistency, contradictions |
| Assumptions | `assumptions` | Explicitness, justification, stress-testing |
| Comprehensiveness | `comprehensiveness` | Coverage of sections, sector KPIs, citations |
| Originality | `originality` | Synthesis vs copying, company-specific insights |

**Grading Scale:** Poor, Fair, Good, Excellent

**Output Schema:** Each evaluation produces JSON with:
- `grade`: string
- `summary_reasoning`: string
- `checks`: dict of boolean checks
- `flags`: dict of hard-cap violations
- Evidence arrays (metric-specific)

## Code Patterns

### Provider Protocol (providers/base.py)

```python
class ProviderClient(Protocol):
    async def generate(self, prompt: str, strict_message: str) -> str: ...
```

All providers implement this interface for interchangeability.

### Async Concurrency Pattern (runner.py)

```python
sem = asyncio.Semaphore(config.max_concurrency)
async with sem:
    result = await client.generate(prompt, strict_message)
```

### JSON Extraction Pattern (utils.py)

```python
extracted = extract_json_block(raw)  # Remove markdown fences
ok, parsed = try_json(extracted)     # Safe parse
if not ok:
    # Retry with explicit JSON instruction
```

### Report Wrapping Pattern (runner.py)

```python
report_wrapped = f"<report>{task.report_text}</report>"
full_prompt = f"{task.prompt}\n {report_wrapped}"
```

## Output Directory Structure

### Evaluation Output

```
output/evaluation/{provider}/{period}/{source}/{llm_provider}/
  {TICKER}_{metric}_eval.json

# Example:
output/evaluation/openai/2025_q1/morningstar/openai/AAPL_depth_eval.json
```

### Human Report Evaluation Output

```
output/evaluation/{provider}/{period}/{source}/analyst/
  {TICKER}_{metric}_eval.json

# Example:
output/evaluation/openai/2025_q1/morningstar/analyst/AAPL_depth_eval.json
```

### Forecast Extraction Output

```
output/human_forecasts/{period}/{source}/
  {TICKER}_forecast.json
  all_forecasts.json

# Example:
output/human_forecasts/2025_q1/morningstar/AAPL_forecast.json
```

## Important Implementation Notes

1. **GPT-5 doesn't support temperature=0** - `extract_forecasts.py` conditionally adds temperature only for non-GPT-5 models

2. **OpenAI uses Responses API** - Not the standard Chat Completions API. See `_extract_text()` in `openai.py` for response parsing

3. **PDF text extraction limitations** - pypdf doesn't support OCR; scanned PDFs will fail

4. **Idempotent evaluation** - Existing output files are skipped; safe to re-run

5. **Strict message pattern** - Two separate user messages sent: main prompt + strict grading instruction

6. **Unit conversion in forecasts** - Values in millions/billions converted to full numbers; EPS preserved as-is

## Common Tasks

### Add a New Provider

1. Create `src/llm_eval/providers/newprovider.py`
2. Implement class with `async def generate(self, prompt: str, strict_message: str) -> str`
3. Add to `DEFAULT_MODELS` in `config.py`
4. Add client construction in `cli.py:_build_client()`

### Add a New Evaluation Metric

1. Add prompt constant in `prompts.py` (follow existing pattern)
2. Add to `EVALS` dict at bottom of `prompts.py`
3. Output files will automatically include new metric

### Modify Forecast Extraction

1. Edit prompts: `MORNINGSTAR_PROMPT` or `ARGUS_PROMPT` in `extract_forecasts.py`
2. Update `ForecastData` dataclass if schema changes
3. Update `convert_forecasts_to_full_numbers()` if new metrics need conversion

## Testing

```bash
# Test single ticker evaluation
python -m src.llm_eval.cli --provider openai --ticker-allowlist-root output/report/2025_q1

# Test single ticker forecast extraction
python -m src.llm_eval.extract_forecasts --ticker AAPL --quarter 2025_q1
```

## Dependencies

```
anthropic>=0.39.0
google-genai[aiohttp]>=1.56.0
openai>=1.60.0
pypdf>=4.3.1
python-dotenv>=1.0.1
tqdm>=4.66.0
```
