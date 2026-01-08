"""
Extract financial forecasts from human analyst reports (Morningstar and Argus PDFs).
Uses GPT-5 to extract structured forecast data.
"""

import asyncio
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI
from pypdf import PdfReader
from tqdm import tqdm

load_dotenv()


@dataclass
class ForecastData:
    """Structured forecast data extracted from reports."""
    ticker: str
    source: str  # "morningstar" or "argus"
    report_date: str | None
    target_price: float | None  # Fair value estimate / target price
    currency: str
    forecasts: dict[str, dict[str, float | None]]  # year -> {metric: value}
    raw_extraction: dict[str, Any]  # Original extraction for debugging


MORNINGSTAR_PROMPT = """You are a financial data extraction expert. Extract the financial forecasts from this Morningstar equity research report.

Look for the "Morningstar Valuation Model Summary" section which contains forecast data.

Extract the following information:

1. **Fair Value Estimate** (this is the target price) - usually shown at the top of the report
2. **Currency** - the reporting currency (usually USD)
3. **Report Date** - when the report was created

4. **Financial Forecasts** for each available year (both historical and forecast years):
   - Revenue (in the original units shown, note the unit)
   - Operating Income
   - EBITDA
   - Net Income
   - EPS (Earnings Per Share)
   - Free Cash Flow (if available)

IMPORTANT:
- Note the units used (usually "USD Mil" meaning millions)
- Include both historical years and forecast years
- Mark forecast years distinctly from historical years

Return your response as a JSON object with this exact structure:
{
  "fair_value_estimate": <number or null>,
  "currency": "<string>",
  "report_date": "<string or null>",
  "units": "<string describing units, e.g., 'millions'>",
  "forecasts": {
    "<year>": {
      "is_forecast": <boolean>,
      "revenue": <number or null>,
      "operating_income": <number or null>,
      "ebitda": <number or null>,
      "net_income": <number or null>,
      "eps": <number or null>,
      "free_cash_flow": <number or null>
    }
  }
}

Return ONLY the JSON object, no other text."""

ARGUS_PROMPT = """You are a financial data extraction expert. Extract the financial forecasts from this Argus equity research report.

Look for the "Growth & Valuation Analysis" section which contains financial data.

Extract the following information:

1. **Target Price** - shown in the Key Statistics or Market Data section
2. **Currency** - the reporting currency (usually USD)
3. **Report Date** - when the report was created

4. **Financial Forecasts** for each available year (both historical and forecast years):
   - Revenue (note the unit, usually in Millions or Billions)
   - Operating Income
   - Net Income
   - EPS (Earnings Per Share)
   - Gross Profit (if available)

IMPORTANT:
- Note the units used (usually millions, sometimes billions for Revenue)
- Include both historical years and forecast years
- Mark forecast years distinctly from historical years
- Argus often shows quarterly data - look for annual totals

Return your response as a JSON object with this exact structure:
{
  "target_price": <number or null>,
  "currency": "<string>",
  "report_date": "<string or null>",
  "units": "<string describing units, e.g., 'millions'>",
  "forecasts": {
    "<year>": {
      "is_forecast": <boolean>,
      "revenue": <number or null>,
      "operating_income": <number or null>,
      "net_income": <number or null>,
      "eps": <number or null>,
      "gross_profit": <number or null>
    }
  }
}

Return ONLY the JSON object, no other text."""


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract text from a PDF file."""
    reader = PdfReader(str(pdf_path))
    chunks = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text:
            chunks.append(text)
    return "\n".join(chunks)


def infer_source_from_filename(path: Path) -> str:
    """Determine report source from filename."""
    name = path.stem.lower()
    if "morningstar" in name:
        return "morningstar"
    if "argus" in name:
        return "argus"
    return "unknown"


def extract_ticker_from_filename(path: Path) -> str:
    """Extract ticker symbol from filename."""
    # Format is usually TICKER_Source.pdf
    return path.stem.split("_", 1)[0]


def convert_to_full_numbers(value: float | None, units: str) -> float | None:
    """Convert values from millions/billions to full numbers."""
    if value is None:
        return None

    units_lower = units.lower()

    if "billion" in units_lower or "bil" in units_lower:
        return value * 1_000_000_000
    elif "million" in units_lower or "mil" in units_lower:
        return value * 1_000_000
    elif "thousand" in units_lower:
        return value * 1_000
    else:
        # Assume the value is already in full numbers
        return value


def convert_forecasts_to_full_numbers(
    forecasts: dict[str, dict[str, Any]],
    units: str
) -> dict[str, dict[str, float | None]]:
    """Convert all forecast values to full numbers."""
    converted = {}

    # Metrics that should NOT be converted (they're already in per-share terms)
    skip_conversion = {"eps", "is_forecast"}

    for year, metrics in forecasts.items():
        converted[year] = {}
        for metric, value in metrics.items():
            if metric in skip_conversion:
                converted[year][metric] = value
            elif isinstance(value, (int, float)):
                converted[year][metric] = convert_to_full_numbers(value, units)
            else:
                converted[year][metric] = value

    return converted


def extract_json_from_response(text: str) -> dict[str, Any]:
    """Extract JSON from LLM response, handling markdown code blocks."""
    text = text.strip()

    # Remove markdown code blocks if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last lines if they're code block markers
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    # Try to find JSON object in the text
    start_idx = text.find("{")
    end_idx = text.rfind("}") + 1

    if start_idx != -1 and end_idx > start_idx:
        json_str = text[start_idx:end_idx]
        return json.loads(json_str)

    raise ValueError(f"Could not extract JSON from response: {text[:200]}...")


class ForecastExtractor:
    """Extract forecasts from PDFs using GPT-5."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-5",
        timeout_s: int = 120,
        max_retries: int = 3,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")

        self._client = AsyncOpenAI(api_key=self.api_key, timeout=timeout_s)
        self._model = model
        self._timeout_s = timeout_s
        self._max_retries = max_retries

    async def _call_gpt(self, prompt: str, report_text: str) -> str:
        """Call GPT-5 with the extraction prompt."""
        full_prompt = f"{prompt}\n\n<report>\n{report_text}\n</report>"

        for attempt in range(self._max_retries):
            try:
                # Build API call parameters - GPT-5 doesn't support temperature=0
                api_params = {
                    "model": self._model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a financial data extraction expert. Extract data precisely and return valid JSON."
                        },
                        {"role": "user", "content": full_prompt}
                    ],
                    "response_format": {"type": "json_object"},
                }

                # Only add temperature for non-GPT-5 models
                if not self._model.startswith("gpt-5"):
                    api_params["temperature"] = 0

                response = await asyncio.wait_for(
                    self._client.chat.completions.create(**api_params),
                    timeout=self._timeout_s,
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                if attempt < self._max_retries - 1:
                    await asyncio.sleep(2 ** (attempt + 1))
                else:
                    raise e

        return ""

    async def extract_from_pdf(self, pdf_path: Path) -> ForecastData:
        """Extract forecast data from a single PDF."""
        source = infer_source_from_filename(pdf_path)
        ticker = extract_ticker_from_filename(pdf_path)

        # Extract text from PDF
        report_text = await asyncio.to_thread(extract_pdf_text, pdf_path)

        # Select appropriate prompt
        if source == "morningstar":
            prompt = MORNINGSTAR_PROMPT
        elif source == "argus":
            prompt = ARGUS_PROMPT
        else:
            raise ValueError(f"Unknown report source: {source}")

        # Call GPT-5 for extraction
        response = await self._call_gpt(prompt, report_text)

        # Parse the response
        raw_data = extract_json_from_response(response)

        # Get units for conversion
        units = raw_data.get("units", "millions")

        # Extract target price (different field names for different sources)
        target_price = raw_data.get("fair_value_estimate") or raw_data.get("target_price")

        # Convert forecasts to full numbers
        raw_forecasts = raw_data.get("forecasts", {})
        converted_forecasts = convert_forecasts_to_full_numbers(raw_forecasts, units)

        return ForecastData(
            ticker=ticker,
            source=source,
            report_date=raw_data.get("report_date"),
            target_price=target_price,
            currency=raw_data.get("currency", "USD"),
            forecasts=converted_forecasts,
            raw_extraction=raw_data,
        )


async def process_reports(
    input_dir: Path,
    output_dir: Path,
    model: str = "gpt-5",
    max_concurrency: int = 5,
    ticker: str | None = None,
) -> list[ForecastData]:
    """Process all PDF reports in a directory."""

    # Find all PDFs
    pdf_files = list(input_dir.rglob("*.pdf"))

    # Filter by ticker if specified
    if ticker:
        pdf_files = [p for p in pdf_files if extract_ticker_from_filename(p).upper() == ticker.upper()]

    if not pdf_files:
        print(f"No PDF files found in {input_dir}" + (f" for ticker {ticker}" if ticker else ""))
        return []

    print(f"Found {len(pdf_files)} PDF files to process")

    # Create extractor
    extractor = ForecastExtractor(model=model)

    # Process with concurrency limit
    sem = asyncio.Semaphore(max_concurrency)

    async def process_one(pdf_path: Path) -> tuple[Path, ForecastData | None, str]:
        async with sem:
            try:
                result = await extractor.extract_from_pdf(pdf_path)
                return pdf_path, result, ""
            except Exception as e:
                return pdf_path, None, str(e)

    # Process all files
    tasks = [process_one(p) for p in pdf_files]
    results = []

    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Extracting forecasts"):
        pdf_path, data, error = await coro
        if data:
            results.append(data)

            # Save individual result
            rel_path = pdf_path.relative_to(input_dir)
            source = infer_source_from_filename(pdf_path)
            ticker = extract_ticker_from_filename(pdf_path)
            out_file = output_dir / rel_path.parent / source / f"{ticker}_forecast.json"
            out_file.parent.mkdir(parents=True, exist_ok=True)

            with open(out_file, "w") as f:
                json.dump(asdict(data), f, indent=2)
        else:
            print(f"Error processing {pdf_path}: {error}")

    # Save combined results
    combined_output = output_dir / "all_forecasts.json"
    with open(combined_output, "w") as f:
        json.dump([asdict(d) for d in results], f, indent=2)

    print(f"\nProcessed {len(results)}/{len(pdf_files)} files successfully")
    print(f"Results saved to {output_dir}")

    return results


def main():
    """Main entry point for CLI usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract financial forecasts from human analyst reports"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("output/human_reports"),
        help="Directory containing PDF reports",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/human_forecasts"),
        help="Directory to save extracted forecasts",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5",
        help="OpenAI model to use for extraction",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=5,
        help="Maximum concurrent API calls",
    )
    parser.add_argument(
        "--quarter",
        type=str,
        default=None,
        help="Specific quarter to process (e.g., '2025_q1')",
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default=None,
        help="Specific ticker to process (e.g., 'AAPL')",
    )

    args = parser.parse_args()

    input_dir = args.input_dir
    if args.quarter:
        input_dir = input_dir / args.quarter

    output_dir = args.output_dir
    if args.quarter:
        output_dir = output_dir / args.quarter

    output_dir.mkdir(parents=True, exist_ok=True)

    asyncio.run(
        process_reports(
            input_dir=input_dir,
            output_dir=output_dir,
            model=args.model,
            max_concurrency=args.max_concurrency,
            ticker=args.ticker,
        )
    )


if __name__ == "__main__":
    main()
