from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Any

from dotenv import load_dotenv
from tqdm import tqdm
from openai import OpenAI

load_dotenv()

# =========================
# -------- PROMPTS --------
# =========================
DEPTH_PROMPT = """
<instruction>
Role
You are an impartial evaluator. Your sole task is to grade the analytical depth of a single equity research report.

Definition: “Analytical Depth”
Analytical depth measures whether the report goes beyond description to provide causal reasoning, explicit assumptions, quantified forecasts, counterpoints/uncertainty, and decision-relevant implications.

A report with analytical depth includes:

Causal Explanation (why results occur, not just what happened)

Inference Quality (forecasts or hypotheses tied to explicit drivers)

Use of Data (quantified assumptions, benchmarks, sanity checks, clear distinction between correlation vs causation)

Counterpoints & Uncertainty (scenarios, sensitivities, ranges, limitations, alternative explanations)

Actionable Implications (clear, conditional takeaways relevant for investor decisions)

What NOT to do

Do not grade persuasiveness, writing style, or investment recommendation.

Do not summarize the entire report. Evaluate analytical depth only.

Do not invent facts. Use only what appears in the report.

Evidence you must extract
Identify and record the strongest and weakest examples of reasoning. Quote 3–6 short verbatim excerpts (≤10 words) from the report that illustrate causal reasoning, assumptions, sensitivities, or lack thereof.

Examples:

“NII was $14.6B, driven by higher yielding assets” (causal)

“Operating costs grow slower than revenue due to scale” (causal mechanism)

“If Fed cuts rates, NIM could compress” (counterpoint)

“WACC ~9%, terminal growth ~2.5%” (assumption, but no sensitivity)

“Projected steadily rising NII” (unsupported generalization)

Hard Caps (override any other impressions)

If no clear mechanisms or assumptions are given ⇒ max grade = “Poor”.

If assumptions are listed but not benchmarked or stress-tested ⇒ max grade = “Fair”.

If report contains explicit assumptions and causal reasoning but no quantified sensitivity/scenario ⇒ max grade = “Good”.

Only reports with mechanism-rich reasoning, quantified scenarios/ranges, and actionable implications ⇒ “Excellent”.

Grading Scale (choose exactly one)

Poor

Mostly descriptive. No mechanisms, little or no assumptions, no quantified sensitivity.

Example: Lists “EPS grows” without linking to costs, margins, or rates.

Fair

Some mechanisms/assumptions, but shallow, unsupported, or generic.

Limited quantification. Counterpoints mentioned but not explored. Implications vague.

Example: “NII up on higher yields” but no deposit beta or rate sensitivity.

Good

Clear mechanisms tied to forecasts and valuation.

Explicit assumptions quantified and benchmarked.

Some scenario/counterpoint analysis but not extensive.

Example: “100 bp cut lowers NIM ~20 bps, EPS down ~5%.”

Excellent

Comprehensive causal reasoning with explicit, benchmarked assumptions.

Multiple quantified scenarios or ranges.

Strong treatment of uncertainty. Clear, conditional implications.

Example: “Base case ROTE 14% vs COE 9.5%; if loan losses +50 bps, EPS –8% and FV drops $50→$44.”

Decision Rules (apply in order)

If no mechanisms/assumptions ⇒ Poor.

If mechanisms/assumptions exist but no benchmarks/sensitivity ⇒ Fair.

If mechanisms and assumptions are benchmarked but no scenario analysis ⇒ Good.

If all of the above + scenario/sensitivity analysis and decision-relevant implications ⇒ Excellent.
</instruction>

<output_format>
Required Output (JSON only; follow the schema exactly)

{
  "grade": "Poor | Fair | Good | Excellent",
  "summary_reasoning": "<150–250 words covering causal explanation, inference quality, data use, counterpoints/uncertainty, and actionable implications. Include 3–6 short verbatim excerpts.>",
  "checks": {
    "causal_explanation_present": true,
    "assumptions_explicit": true,
    "assumptions_benchmarked": false,
    "quantification_used": true,
    "sensitivity_or_scenarios": false,
    "actionable_implications_present": true
  },
  "flags": {
    "missing_mechanisms": ["balance sheet repositioning → yield calibration not explained"],
    "unsupported_assumptions": ["terminal growth 2.5% with no industry benchmark"],
    "lack_of_sensitivity": ["NIM assumption fixed at 3.3% without range"]
  }
}
</output_format>

<calibration_hints>
Calibration Hints

Reports like the BAC example (lists assumptions, shallow causal links, no quantified sensitivity) ⇒ Fair.

Reports with explicit assumptions, benchmarks, and some causal reasoning but no scenarios ⇒ Good.

Reports with explicit assumptions, benchmarks, quantified scenarios, and decision-relevant implications ⇒ Excellent.
</calibration_hints>
"""

COHERENCE_PROMPT = """
<instruction>
Role

You are an impartial grader. Evaluate the coherence of a single equity research report (text + any included tables) using only the content provided. Do not add outside facts.

Coherence = how well the report reads as one connected argument: clear language, smooth local transitions, explicit cross-references among sections, and no internal contradictions among text, tables, assumptions, dates, and cited facts.

What To Read

All narrative sections (e.g., Analyst Note, Strategy/Outlook, Bulls/Bears, Moat, Valuation/Assumptions, Risks, Capital Allocation, ESG, Appendix, Sources).

All tables/figures embedded in the report.

Step-by-Step Procedure

Scan & Map: Identify section headers and the overall narrative arc (intro → analysis → conclusion). Note defined terms/jargon and whether they are used consistently.

Flow Check (Within & Across Sections):

Within: Do paragraphs transition logically, stating why the next idea follows?

Across: Do sections reference each other (e.g., assumptions ↔ valuation ↔ risks) rather than read like isolated lists?

Consistency Check: Find contradictions or unresolved tensions among:

Narrative vs tables (e.g., “capex flat” vs large recurring tech spend in the same period).

Assumptions vs Bulls/Bears vs base-case valuation (e.g., “NII steadily rising” vs rate-cut sensitivity).

Temporal claims vs dates/citations (e.g., calling an older event “recent”).

Terminology use (e.g., NIM vs NII misapplied).

Citations Alignment: If sources are cited, do they support the claim as written (at least at a basic face-value level)? If not, note the mismatch.

Apply the Rubric & Guardrails (below) and determine the final grade.

Write Reasoning (180–260 words): Explain the key drivers of your grade with direct references to sections/lines where possible.

Select 2–3 Evidence Bullets: Quote or concisely paraphrase the most decisive strengths/weaknesses with a section/page cue if available.

Grading Scale (choose exactly one)

poor

Frequent abrupt jumps; list-like paragraphs; key terms undefined or inconsistently used.

Multiple unresolved contradictions (narrative ↔ tables/assumptions) or time-inaccurate “recency” claims.

Citations/figures not aligned with the stated point.

Reader must assemble the argument themselves.

fair

Overall structure exists, but within-section flow is choppy; limited transitions.

Some contradictions or unexplained tensions remain; terms mostly clear but occasionally inconsistent.

Understandable with effort, but internally noisy.

good

Clear language and mostly smooth transitions; sections reinforce each other.

No material contradictions; dates/sources align.

Minor rough edges (e.g., a thin transition or lightly supported claim) are acceptable.

excellent

Seamless narrative with explicit connective tissue (“Because X in Q1, therefore Y in margins”).

Assumptions, tables, and narrative perfectly align; terminology consistent; temporal claims precise.

Proactively surfaces and resolves potential contradictions.

Guardrails (Auto-Downgrades)

Hard cap = fair if you find any unaddressed contradiction among assumptions, tables, and narrative.

Hard cap = fair if any “recent/now” claims are time-inaccurate relative to dates provided.

Minus one level if Bulls/Bears are not reconciled with the base-case valuation narrative.

Minus one level if ≥2 sections read like bullet dumps (no transitions).
</instruction>

<output_format>
Output Format (return only this JSON object)

{
  "grade": "poor | fair | good | excellent",
  "reasoning": "<180–260 words explaining the grade, touching on clarity, flow, structural coherence, consistency, and temporal/source alignment>",
  "evidence": [
    "<brief quote or paraphrase + section/page cue>",
    "<brief quote or paraphrase + section/page cue>",
    "<optional third bullet>"
  ],
  "flags": {
    "contradictions_found": true,
    "temporal_inaccuracy_found": true,
    "bulls_bears_unreconciled": true,
    "list_like_sections_count": 0,
    "auto_downgrade_applied": true
  }
}

Notes on Filling the JSON

grade: one of exactly poor, fair, good, excellent (lowercase).

reasoning: 180–260 words; no quotes longer than one sentence; no URLs.

evidence: 2–3 bullets max; each should point to the decisive items influencing the grade.

flags.auto_downgrade_applied: true if any hard cap or minus-one rule changed the final grade.
</output_format>

<calibration_hints>
Calibration Hints (so the weak example scores Poor/Fair; strong reports score Good/Excellent)

If you detect contradictions like “capex/share count flat” alongside repeated, large tech spend claims, or “NII steadily rising” while Bears emphasize rate-cut sensitivity without reconciliation, set contradictions_found = true and cap at fair.

If a “recent” event is actually older within the report’s own dates/sources, set temporal_inaccuracy_found = true and cap at fair.

If the narrative is polished but jumps topic (earnings → tech → stress tests) with thin transitions, count those sections toward list_like_sections_count and consider a fair grade unless other strengths clearly outweigh.
</calibration_hints>
"""

ASSUMPTIONS_PROMPT = """
<instruction>
Role

You are an impartial evaluator. Your sole task is to grade the quality of assumptions in a single equity research report.

Definition: “Quality of Assumptions”

Quality of assumptions measures whether the report’s key inputs that drive forecasts, valuations, and conclusions are:

Explicit (clearly stated near the forecasts they inform),

Justified (supported by multi-year history, peer/industry benchmarks, or cited sources),

Specific (magnitudes, units, time horizons, and causal drivers),

Consistent (internally coherent with the report’s own tables and conclusions), and

Stress-tested (quantified ranges or scenarios for material drivers, not only a single point).

What NOT to do

Do not grade writing quality, persuasion, or investment merit.

Do not summarize the whole report. Evaluate assumptions only.

Do not invent facts. Use only what appears in the report.

Evidence you must extract

Identify and quote (verbatim, short excerpts) the top 3–8 assumptions that materially drive the model (e.g., revenue/NII growth, margins/efficiency ratio, credit losses, capex, tax rate, WACC/COE, terminal growth, share count). For each, record where it appears (section header and page if available).

Hard Caps (override any other impressions)

No sensitivity/scenario analysis on material drivers ⇒ max grade = “Fair”.

Any unresolved numeric contradiction between text and tables/figures ⇒ max grade = “Fair”.

Assumptions mostly unstated/opaque ⇒ “Poor”.

Use of figures that clearly conflict with recent history/peers without justification ⇒ max grade = “Fair”.

Two or more distinct contradictions (or one contradiction plus no sensitivities) ⇒ “Poor”.

Grading Scale (choose exactly one)

Poor

Assumptions largely unstated/opaque or scattered without context.

Little/no justification; conflicts with facts or the report’s own tables.

Vague magnitudes (no units, horizons, or drivers).

No sensitivities/scenarios; outputs feel black-box.

Often includes internal contradictions (e.g., “flat share count” while committing to heavy ongoing tech spend).

Fair

Key assumptions are listed but weakly justified; some are selective/hand-wavy.

Partial specificity (some units/timeframes) with minor inconsistencies.

Limited/qualitative sensitivity (mentions risk without quantified ranges).

Good

Assumptions are explicit and proximal to the forecasts they drive; units, horizons, and drivers are clear.

Justified with history/peers/sources; numbers reconcile to tables and narrative.

Quantified sensitivity for at least 2–3 material drivers (e.g., rates/NIM, credit cost, margin).

Excellent

Assumptions are comprehensive, traceable, auditable across narrative ↔ tables ↔ conclusions.

Evidence-rich (multi-year history, peer comps, and/or cited sources) for each major driver.

Robust, quantified sensitivities/scenarios for all material levers with valuation/target impact.

Zero contradictions; figures, units, and time windows reconcile throughout.

Decision Rules (apply in order)

If assumptions are mostly unstated/opaque ⇒ Poor.

Else, count contradictions (text vs table; claim vs clearly implied historical fact in the report).

≥2 contradictions ⇒ Poor.

1 contradiction ⇒ cap at Fair (unless fully reconciled in text).

Check sensitivities/scenarios on material drivers.

None/qualitative only ⇒ cap at Fair.

Check justification (history, peers, sources) and specificity (units/horizons/drivers).

Weak/partial ⇒ Fair.

Solid with some quantified sensitivities ⇒ Good.

Comprehensive + robust sensitivities on all material levers ⇒ Excellent.
</instruction>

<output_format>
Required Output (JSON only; follow the schema exactly)
{
  "grade": "Poor | Fair | Good | Excellent",
  "summary_reasoning": "<150–250 words focusing on the five pillars: explicitness, justification, specificity, consistency, sensitivity. No fluff.>",
  "assumptions_extracted": [
    {
      "quote": "<verbatim short excerpt>",
      "location": {"section": "<header>", "page": "<number or 'unknown'>"},
      "driver_type": "<e.g., revenue_growth | NII/NIM | margin/efficiency | credit_cost | tax_rate | capex | share_count | WACC/COE | terminal_growth | other>"
    }
  ],
  "checks": {
    "explicitness": true,
    "justification_with_evidence": "none | weak | partial | solid | comprehensive",
    "specificity_units_horizon": "none | weak | partial | solid | comprehensive",
    "internal_consistency": "clean | minor_issues | contradiction_found",
    "sensitivities_present": false,
    "sensitivities_quality": "none | qualitative_only | partial_quant | robust_quant",
    "material_drivers_covered": ["NII/NIM","margin/efficiency","WACC/COE","terminal_growth","share_count"]
  },
  "flags": {
    "contradictions": [
      {"description": "<what conflicts with what>", "locations": ["<where A>", "<where B>"]}
    ],
    "missing_or_opaque_assumptions": ["<driver types missing>"],
    "unjustified_parameters": ["<e.g., WACC 9% without source>"]
  }
}
</output_format>

<calibration_hints>
Calibration Hints (to guide borderline calls)

Reports that list assumptions but include incorrect/unsupported figures, internal contradictions, and no quantified sensitivities should end up Poor/Fair (the “leniency trap” is to over-reward the list itself—don’t).

Reports that state and justify inputs (e.g., efficiency ratio path, expense CAGR, credit losses, NII/NIM bands, WACC tied to market yields) and offer quantified scenarios for major levers should be Good/Excellent.

Final constraints

Keep the summary_reasoning within 150–250 words.

Return valid JSON matching the schema.

If a required field is unavailable, use "unknown" or an empty array as appropriate—do not invent content.
</calibration_hints>
"""

COMPREHENSIVENESS_PROMPT = """
<instruction>
Role
You are an impartial evaluator. Your sole task is to grade the comprehensiveness of a single equity research report.

Definition: “Comprehensiveness”
Comprehensiveness measures whether the report fully covers the expected content areas for equity research, with proportional depth, sector-appropriate KPIs, consistent use of evidence, and minimal redundancy.

A comprehensive report includes:

Cover block & contents

Analyst Note / Executive Summary

Business Description

Business Strategy & Outlook

Bulls / Bears (or equivalent upside/downside drivers)

Competitive Positioning / Economic Moat

Valuation & Profit Drivers (explicit link from drivers to valuation assumptions)

Risk & Uncertainty

Capital Allocation (balance sheet, reinvestment, M&A, buybacks/dividends)

Financials Snapshot (multi-year history + forecasts, with sector-specific KPIs)

ESG/Controversies (if material)

Appendix/Glossary and Sources (with sources actually cited in-text)

Sector-specific KPIs should be present (e.g., ARR/NRR for SaaS, same-store sales for Retail, production volumes for Energy, NII/NIM for Banks, pipeline/trial data for Biopharma, etc.).

What NOT to do

Do not grade persuasiveness, style, or investment recommendation.

Do not summarize the entire report. Evaluate comprehensiveness only.

Do not invent facts. Use only what appears in the report.

Evidence you must extract
Identify and record the key content areas covered or missing. For each, note presence/absence and level of depth.

Examples:

“Financials Snapshot includes revenue and EPS but omits sector KPIs (NIM, provisions)”

“Risk section repeats stress test results already covered in Analyst Note”

“Peer benchmarking absent”

Hard Caps (override any other impressions)

If ≥2 core sector KPIs are missing in the Financials Snapshot ⇒ max grade = “Fair”.

If ≥2 material claims are uncited or based on outdated info ⇒ max grade = “Fair”.

If valuation section does not link operating drivers to valuation outcome ⇒ max grade = “Fair”.

If most expected sections are missing/opaque ⇒ grade = “Poor”.

If the same talking point is repeated across ≥2 sections with no new depth ⇒ downgrade one level.

Grading Scale (choose exactly one)

Poor

Many essential sections absent or skeletal.

Sector KPIs missing.

Assertions uncited, outdated, or repetitive.

Report feels incomplete or superficial.

Fair

Headline sections present, but details are thin.

Financials Snapshot is generic or missing multiple sector KPIs.

Sourcing inconsistent; limited peer/industry context.

Some redundancy across sections.

Good

All major sections included with sector-appropriate KPIs.

Valuation assumptions explicit and linked to drivers.

Sources consistent; minimal redundancy.

Risks, peers, and capital allocation addressed at reasonable depth.

Excellent

Exhaustive coverage across all sections.

Every major claim tied to data/tables with sources.

Valuation bridges clearly from operating drivers to fair value.

Peer benchmarking and scenario/sensitivity analysis included.

Zero superficial repetition.

Decision Rules (apply in order)

If most expected sections absent/opaque ⇒ Poor.

If ≥2 contradictions/uncited claims/major KPI omissions ⇒ cap at Fair.

If valuation lacks explicit linkage from drivers to value ⇒ cap at Fair.

If peer/industry/regulatory context absent ⇒ downgrade one level.

If all sections, KPIs, evidence, and linkages present but no sensitivities ⇒ Good.

If comprehensive + scenario/sensitivity analysis + peer benchmarking ⇒ Excellent.
</instruction>

<output_format>
Required Output (JSON only; follow the schema exactly)

{
  "grade": "Poor | Fair | Good | Excellent",
  "summary_reasoning": "<150–250 words covering: coverage of key sections, proportionality, evidence integration, scope vs redundancy, and gaps & impact.>",
  "content_checks": {
    "sections_present": ["Analyst Note","Business Description","Valuation","Risk","Financials Snapshot"],
    "sections_missing": ["Peer Benchmarking","Scenario Analysis"],
    "sector_kpis_present": ["Revenue","EPS"],
    "sector_kpis_missing": ["NIM","Loan Loss Provisions","CET1"]
  },
  "checks": {
    "evidence_citations_consistent": true,
    "valuation_linked_to_drivers": false,
    "peer_context_present": false,
    "redundancy_detected": true,
    "scenario_analysis_present": false
  },
  "flags": {
    "contradictions": [
      {"description": "Tax rate assumption (18%) inconsistent with industry norms", "locations": ["Valuation","Appendix"]}
    ],
    "missing_kpis": ["NIM","Provisions","CET1"],
    "uncited_claims": ["Repurchased $9.5M shares in Q1 with no source"]
  }
}
</output_format>

<calibration_hints>
Calibration Hints

Reports with complete contents page but thin KPI tables, repeated stress-test mentions, and missing peer analysis ⇒ Fair.

Reports with robust KPI coverage, explicit valuation linkage, and consistent sourcing but no scenarios ⇒ Good.

Reports with exhaustive KPI coverage, peer benchmarking, and scenario/sensitivity analysis ⇒ Excellent.
</calibration_hints>
"""

ORIGINALITY_PROMPT = """
<instruction>
Role

You are an impartial evaluator. Your sole task is to grade the originality of insights in a single equity research report.

Definition: “Originality of Insights”

Originality measures whether the report provides unique, value-added perspectives that go beyond paraphrasing sources. A report with high originality:

Synthesizes multiple data points into new conclusions (e.g., linking rate path + deposit betas → margin trajectory).

Presents non-obvious, decision-relevant theses (specific, testable, with catalysts/timelines).

Provides peer- or company-specific angles not available in public filings or news.

Uses sources as evidence but adds interpretive commentary that goes further.

Reports with low originality:

Repeat public headlines or boilerplate (“scale & diversification,” “competition from fintech”).

Present standard DCF assumptions with no new drivers.

Copy or lightly rephrase phrases from cited sources.

Offer generic insights applicable to any bank/sector.

What NOT to do

Do not grade persuasiveness, writing style, or accuracy of forecasts.

Do not summarize the entire report. Evaluate originality only.

Do not infer insights beyond what is written.

Evidence you must extract

Identify and record:

Candidate insights (3–8 statements presented as analysis).

For each: classify as Copied/Paraphrased, Generic Restatement, Synthesis, or Original Thesis.

Note whether each is decision-relevant (specific, actionable, company-distinct).

List any red flags: boilerplate phrasing, generic DCF, stress-test recaps without deeper angle.

Hard Caps (override any other impressions)

If ≥3 insights are copied/paraphrased or generic ⇒ max grade = “Fair.”

If no synthesis or original thesis appears ⇒ grade = “Poor.”

If all insights are generic to the sector (not company-specific) ⇒ max grade = “Fair.”

If valuation section lacks any novel driver or mechanism ⇒ cap at “Fair.”

Grading Scale (choose exactly one)

Poor

All or nearly all insights are copied or generic.

No synthesis or unique thesis.

Zero decision relevance.
Example: Low-quality BAC report with recycled Reuters/Morningstar lines and standard DCF.

Fair

Some company detail but mostly restated.

At most one weak synthesis, vague or generic.

Low actionability.
Example: BAC report citing NII growth and stress tests but adding no unique angle.

Good

At least two clear syntheses or one original thesis with mechanism + numbers + catalyst/timing.

Distinct to the company.

Actionable and decision-relevant.

Excellent

Three or more decision-relevant syntheses or multiple original theses.

Explicit causal links, quantified implications, peer comparison, and monitoring signals.
Example: Morningstar BAC report linking duration + deposit betas + macro to a unique NII/ROE trajectory.

Decision Rules (apply in order)

If 0 synthesis/original theses ⇒ Poor.

If ≤1 weak synthesis and generic framing ⇒ Fair.

If ≥2 syntheses or 1 strong original thesis (quantified, decision-relevant) ⇒ Good.

If ≥3 syntheses or ≥2 original theses with clear mechanisms, numbers, and catalysts ⇒ Excellent.
</instruction>

<output_format>
Required Output (JSON only; follow the schema exactly)
{
  "grade": "Poor | Fair | Good | Excellent",
  "summary_reasoning": "<150–250 words covering: number/nature of syntheses vs generic restatements, originality vs copying, specificity (units, catalysts), and decision relevance.>",
  "content_checks": {
    "insights_classified": [
      {"text": "<short excerpt>", "classification": "Copied | Restated | Synthesis | Original Thesis", "decision_relevant": true|false}
    ],
    "red_flags": ["Generic DCF with no novel drivers", "Boilerplate 'scale & diversification' phrasing"]
  },
  "checks": {
    "synthesis_present": true|false,
    "original_thesis_present": true|false,
    "decision_relevant_insights_count": <int>,
    "copied_or_generic_count": <int>
  },
  "flags": {
    "boilerplate_detected": true|false,
    "valuation_novel_driver_present": true|false,
    "peer_specificity_detected": true|false
  }
}
</output_format>

<calibration_hints>
Calibration Hints

Reports with generic phrases, stress-test recaps, and boilerplate DCF ⇒ Fair.

Reports with 2+ company-specific syntheses (e.g., deposit betas, efficiency ratios, fee normalization) ⇒ Good.

Reports with multiple unique theses, quantified pathways, and catalysts/peer contrasts ⇒ Excellent.
</calibration_hints>
"""

# Map a short key -> (prompt_str, output_suffix)
EVALS: Dict[str, Tuple[str, str]] = {
    "assumptions": (ASSUMPTIONS_PROMPT, "oai_assumptions_eval.json"),
    "coherence": (COHERENCE_PROMPT, "oai_coherence_eval.json"),
    "comprehensiveness": (COMPREHENSIVENESS_PROMPT, "oai_comprehensiveness_eval.json"),
    "depth": (DEPTH_PROMPT, "oai_depth_eval.json"),
    "originality": (ORIGINALITY_PROMPT, "oai_originality_eval.json"),
}

# =========================
# ---- CONFIG & CLIENT ----
# =========================
MODEL = "gpt-5"
API_KEY = os.getenv("OPENAI_API_KEY")
TIMEOUT_S = 3600
MAX_WORKERS = min(8, (os.cpu_count() or 4) * 2)  # I/O-bound

# Folders
INPUT_DIR = Path(
    "/Users/antonypapadimitriou/PycharmProjects/deepfin-benchmarking/deepfin_benchmarking/output/human_reports/2025_q2"
)
OUTPUT_DIR = Path(
    "/Users/antonypapadimitriou/PycharmProjects/deepfin-benchmarking/deepfin_benchmarking/output/evaluation/evaluation_v2/morningstar"
)

# Optional: delete uploaded files from OpenAI after use to avoid clutter
DELETE_FILES_AFTER = False

client = OpenAI(api_key=API_KEY, timeout=TIMEOUT_S)


# =========================
# ------- UTILITIES -------
# =========================
def ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def extract_text(resp: Any) -> str:
    """
    Be liberal in parsing text from the Responses API return value.
    Tries resp.output_text first, then walks output/content to collect text parts.
    """
    text = getattr(resp, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text

    output = getattr(resp, "output", None) or []
    collected = []
    for msg in output:
        content = getattr(msg, "content", None) or []
        for item in content:
            t = getattr(item, "type", None) or (isinstance(item, dict) and item.get("type"))
            if t in {"output_text", "text"}:
                val = getattr(item, "text", None) or (isinstance(item, dict) and item.get("text"))
                if val:
                    collected.append(val)
    return "\n".join(collected).strip()


def try_json(text: str) -> tuple[bool, Any]:
    try:
        return True, json.loads(text)
    except json.JSONDecodeError:
        return False, text


def backoff_sleep(attempt: int) -> None:
    # exponential backoff with light jitter
    time.sleep(min(2 ** attempt, 30) + 0.1 * attempt)


def upload_pdf(path: Path, retries: int = 3) -> str:
    """
    Uploads a PDF and returns file_id. Uses purpose='user_data' as per PDF guide.
    """
    last_err = None
    for attempt in range(retries + 1):
        try:
            with path.open("rb") as f:
                up = client.files.create(file=f, purpose="user_data")
            return up.id
        except Exception as e:
            last_err = e
            if attempt < retries:
                backoff_sleep(attempt + 1)
            else:
                raise last_err


def call_eval_with_file(prompt: str, file_id: str, retries: int = 3) -> str:
    """
    Calls the model with a text prompt + attached PDF file_id using the Responses API.
    """
    # Per docs, pass content as parts: input_text + input_file (PDF) to the user message.
    # https://platform.openai.com/docs/guides/pdf-files (Python, chat mode) and
    # https://platform.openai.com/docs/api-reference/responses
    last_err = None
    prompt += "\n The report is in the attached PDF file."
    for attempt in range(retries + 1):
        try:
            resp = client.responses.create(
                model=MODEL,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {"type": "input_file", "file_id": file_id},
                        ],
                    }
                ],
            )
            return extract_text(resp)
        except Exception as e:
            last_err = e
            if attempt < retries:
                backoff_sleep(attempt + 1)
            else:
                raise last_err


@dataclass
class EvalTask:
    eval_key: str
    prompt: str
    ticker: str
    file_id: str
    out_path: Path


def run_task(task: EvalTask) -> tuple[str, Path, bool, str]:
    """
    Executes one evaluation; writes JSON (or raw) to disk.
    Returns: (eval_key, out_path, is_json, error_message_if_any)
    """
    ensure_dir(task.out_path)
    try:
        raw = call_eval_with_file(task.prompt, task.file_id)
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


def build_tasks(pdf_path: Path, file_id: str) -> list[EvalTask]:
    ticker = pdf_path.stem  # assumes <ticker>.pdf
    tasks: list[EvalTask] = []
    for key, (prompt, suffix) in EVALS.items():
        out_path = OUTPUT_DIR / f"{ticker}_{suffix}"
        tasks.append(EvalTask(eval_key=key, prompt=prompt, ticker=ticker, file_id=file_id, out_path=out_path))
    return tasks


def main() -> None:
    pdfs = sorted([p for p in INPUT_DIR.glob("*Morningstar.pdf") if p.is_file()])
    if not pdfs:
        print(f"No PDFs found under: {INPUT_DIR}")
        return

    # 1) Upload each PDF once
    file_ids: dict[Path, str] = {}
    for p in tqdm(pdfs, desc="Uploading PDFs"):
        try:
            file_ids[p] = upload_pdf(p)
        except Exception as e:
            print(f"Failed to upload {p.name}: {e}")

    # 2) Build all eval tasks
    all_tasks: list[EvalTask] = []
    for p, fid in file_ids.items():
        all_tasks.extend(build_tasks(p, fid))

    # 3) Run evaluations in parallel
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(run_task, t) for t in all_tasks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
            results.append(fut.result())

    ok_count = sum(1 for _, _, is_json, _ in results if is_json)
    total = len(results)
    print(f"\nDone. Parsed valid JSON for {ok_count}/{total} evaluations.")

    # 4) Optional cleanup of uploaded files
    if DELETE_FILES_AFTER:
        for fid in set(fid for fid in file_ids.values()):
            try:
                client.files.delete(fid)
            except Exception as e:
                print(f"Failed to delete file {fid}: {e}")


if __name__ == "__main__":
    main()
