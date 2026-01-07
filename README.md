# LLM Equity Research Report Evaluations

Evaluate equity research reports across multiple providers with consistent grading prompts. Supports OpenAI, Anthropic, and Gemini, runs async, and mirrors input folder structure into outputs.

## What This Does

- Grades report quality across five metrics: depth, coherence, assumptions, comprehensiveness, originality
- Supports multiple providers/models with a shared CLI
- Reads text reports in `output/report` and PDF human reports in `output/human_reports`
- Writes evaluations to `output/evaluation/<provider>/...` mirroring the input structure

## Quick Start

1) Create your `.env`:

```
cp .env.example .env
```

2) Add API keys to `.env`.

3) Install dependencies (use your existing venv):

```
pip install -r requirements.txt
```

4) Run an evaluation:

```
python -m llm_eval --provider gemini
```

## Usage

```
python -m llm_eval --provider {openai|anthropic|gemini} [options]
```

Common options:

- `--input-root`: default `output/report`
- `--output-root`: default `output/evaluation`
- `--all-sources`: include non‑Morningstar sources (default is Morningstar only)
- `--human-reports`: read PDFs from `output/human_reports` and infer source from filename
- `--model`: override default model for the provider
- `--strict-message`: override strict grading instruction (default is harsh)
- `--max-concurrency`: default `8`

## Examples

Morningstar text reports (default):

```
python -m llm_eval --provider openai
```

All sources under `output/report`:

```
python -m llm_eval --provider gemini --all-sources
```

Human PDF reports:

```
python -m llm_eval --provider anthropic --human-reports
```

## Output Structure

Evaluations are saved to:

```
output/evaluation/<provider>/<period>/<source>/<provider>/TICKER_<metric>_eval.json
```

For human PDFs, `<source>` is inferred from the filename (e.g., `*_Morningstar.pdf`, `*_Argus.pdf`).

## Notes

- `.env` is ignored by git; use `.env.example` as a template.
- If a model returns non‑JSON output, a `.raw.txt` sibling file is written for debugging.
