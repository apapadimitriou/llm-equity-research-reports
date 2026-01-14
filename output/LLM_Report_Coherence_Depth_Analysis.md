# Analysis: Why LLM Reports Underperform on Coherence and Depth

**An examination of systematic failure patterns in LLM-generated equity research reports**

---

## Executive Summary

Based on examination of evaluation files and actual report excerpts across analyst reports (Morningstar) and LLM providers (XAI, Gemini, OpenAI, Perplexity), this analysis identifies systematic patterns explaining the performance gap in coherence and depth metrics.

| Metric | Analyst (Morningstar) | LLM Reports |
|--------|----------------------|-------------|
| **Coherence** | Consistently "Good" | Overwhelmingly "Poor" |
| **Depth** | Consistently "Good" | "Poor" to "Fair" |

The gap is **not marginal** — it's categorical. Every LLM-generated report examined for AAPL received "Poor" coherence ratings, while the analyst report received "Good".

---

## Part 1: Coherence Failures

### 1.1 Temporal Impossibilities (Most Critical)

LLM reports suffer from severe **temporal confusion** — citing future events as historical facts. This is the single most damaging coherence failure.

#### XAI NVDA Report (dated July 15, 2025)

```
Sources:
[5] TipRanks.com – "Nvidia (NVDA) Earnings Dates", tipranks.com, 2025-08-08
[6] Public.com – "NVIDIA Stock Forecast", public.com, 2025-08-08
```

**Evaluator Assessment:**
> *"Report dated July 15, 2025 discusses April 27, 2025 earnings as recent while citing sources from August 8, 2025 (temporal impossibility)"*

#### Gemini NVDA Report (dated July 15, 2025)

The report cites:
> *"NVIDIA Announces Financial Results for Second Quarter Fiscal 2026, August 27, 2025"*

**Evaluator Assessment:**
> *"Report dated July 15, 2025 cites 'NVIDIA Announces Financial Results for Second Quarter Fiscal 2026, August 27, 2025' - an impossible future event"*

#### Root Cause
LLMs lack genuine temporal reasoning. They retrieve and assemble information without verifying the logical consistency of dates. Professional analysts operate within a single temporal frame and rigorously date their sources.

---

### 1.2 The "Insufficient Data" Epidemic

LLM reports are plagued by pervasive data gaps, creating hollow scaffolding where analysis should exist.

#### XAI NVDA Report - Cover Block

```
Last close insufficient data | Fair-Value Estimate $227.95
Price/FVE insufficient data | Market Cap insufficient data
Economic Moat insufficient data | Uncertainty rating insufficient data
Capital Allocation rating insufficient data
```

#### XAI NVDA Financial Snapshot Table

| Metric | FY 2022 | FY 2023 | FY 2024 | FY 2025 | FY 2026 | FY 2027 | FY 2028 | FY 2029 |
|--------|---------|---------|---------|---------|---------|---------|---------|---------|
| Revenue (B) | Insufficient data | Insufficient data | Insufficient data | 91.1 (partial) | Insufficient data | Insufficient data | Insufficient data | Insufficient data |
| Op-Margin % | Insufficient data | Insufficient data | Insufficient data | Insufficient data | Insufficient data | Insufficient data | Insufficient data | Insufficient data |
| EPS | Insufficient data | Insufficient data | Insufficient data | Insufficient data | Insufficient data | Insufficient data | Insufficient data | Insufficient data |
| FCF (B) | Insufficient data | Insufficient data | Insufficient data | Insufficient data | Insufficient data | Insufficient data | Insufficient data | Insufficient data |
| ROIC % | Insufficient data | Insufficient data | Insufficient data | Insufficient data | Insufficient data | Insufficient data | Insufficient data | Insufficient data |

**Evaluator Assessment:**
> *"Analyst Note cites specific EPS of $0.81 and margins of 61% while Financials Snapshot shows 'insufficient data' for the same metrics"*

#### Contrast: Morningstar NVDA Report

Complete financial projections with explicit FY 2022–2029 data:

| Metric | FY 2022A | FY 2023A | FY 2024A | FY 2025A | FY 2026E | FY 2027E | FY 2028E | FY 2029E |
|--------|----------|----------|----------|----------|----------|----------|----------|----------|
| Revenue | 26.9 | 27.0 | 60.9 | 130.5 | 201.0 | 251.2 | 288.9 | 303.3 |
| Operating Margin % | 37.3 | 15.7 | 54.1 | 62.4 | 63.0 | 64.0 | 63.0 | 62.0 |
| Free Cash Flow | 8.1 | 3.8 | 27.0 | 60.9 | 85.4 | 108.0 | 125.7 | 133.4 |
| ROIC (%) | 30.2 | 13.1 | 72.9 | 101.8 | 115.5 | 125.1 | 120.3 | 118.7 |

---

### 1.3 Internal Contradictions

LLM reports contain **unreconciled Bulls/Bears arguments** that contradict the base-case thesis.

#### Perplexity AAPL Report

**Evaluator Assessment:**
> *"Claims iPhone revenue is declining while financial table shows consistent revenue growth from $405.2B to $482.0B (2025-2029)"*
>
> *"Bulls say Services growth of 14% supports valuation while Bears say Services growth may decelerate, with no reconciliation in base case"*

#### XAI AAPL Report

```markdown
**Bulls Say:**
1. Services revenue growth of 12% in Q2 2025 demonstrates the durability of Apple's ecosystem

**Bears Say:**
1. Greater China revenue drop of 2.3% highlights vulnerability to geopolitical risks
```

**Evaluator Assessment:**
> *"Bulls/Bears arguments reference the same data points (wearables decline, China weakness) without reconciliation against the positive base-case narrative"*

#### Contrast: Morningstar NVDA Report (Good coherence)

**Evaluator Assessment:**
> *"Bulls/Bears section properly reconciled with base case, acknowledging both 'industry-leading GPUs and Cuda software platform' strengths and 'powerful chipmakers focused on in-house chip development' competitive risks"*

---

### 1.4 List-Like Sections Without Transitions

LLM reports read like **disconnected bullet dumps** rather than coherent narratives.

#### OpenAI AAPL Report

**Evaluator Assessment:**
> *"Risk section abruptly jumps from tariff costs to competitive risks to regulatory issues with no connecting transitions between distinct topics"*

#### XAI NVDA Report - Risk Section

```
Macro risks include economic slowdowns affecting data center investments.
Regulatory risks involve U.S.-China trade tensions impacting exports [insufficient data for details].
ESG risks relate to energy consumption of AI hardware [insufficient data].
Operational risks include supply chain disruptions for advanced chips [insufficient data].
These could swing valuation by 20-30% [insufficient data for precise estimate].
```

Each sentence is atomized with no connective tissue explaining how these risks interact or compound.

---

## Part 2: Depth Failures

### 2.1 Descriptive Rather Than Analytical

LLM reports **describe facts without explaining mechanisms**.

#### Perplexity NVDA Report

**Evaluator Assessment:**
> *"This report is fundamentally descriptive with minimal analytical depth...The few causal statements like 'driven primarily by Data Center segment performance' and 'creates significant switching costs' are shallow and unsupported."*

#### XAI NVDA Report Excerpt

> *"Revenue of $44.1 billion for the fiscal quarter...represents a 69% increase from the year-ago period."*

This states **what** happened but not **why** or **how** it connects to the investment thesis.

#### Contrast: Morningstar NVDA Report (Good depth)

> *"DC revenue of $39.1 billion was up 73% year over year...massive DC growth has been gross margin-accretive."*

The analyst connects the revenue metric to margin implications through a causal chain.

---

### 2.2 Unbenchmarked Assumptions

LLM reports present assumptions **without justification or industry comparison**.

#### Gemini NVDA Report

```
WACC: 9.2%
Terminal Growth Rate: 4.0%
```

**Evaluator Flags:**
- "terminal growth 4.0% with no industry benchmark"
- "WACC 9.2% components not stress-tested"

#### XAI AAPL Report

```
WACC: 8.5%
Terminal Growth: 3.0%
```

**Evaluator Flag:**
> *"terminal growth rate of 3% with no industry benchmark"*

#### Contrast: Morningstar NVDA Report

> *"WACC ~9%, detailed margin projections of '71% in fiscal 2026' declining to 'low-70% range a decade from now.'"*

Morningstar provides context for why margins will decline (competitive dynamics, product mix) rather than just stating numbers.

---

### 2.3 Absent Sensitivity Analysis

No LLM report provided quantified scenario analysis.

#### Perplexity NVDA Depth Checks

| Check | Result |
|-------|--------|
| causal_explanation_present | ❌ false |
| assumptions_explicit | ✓ true |
| assumptions_benchmarked | ❌ false |
| sensitivity_or_scenarios | ❌ **false** |
| actionable_implications_present | ❌ false |

#### OpenAI AAPL Report

**Evaluator Assessment:**
> *"Even when specific numbers are cited like 'tariff costs of $800 million,' there's no analysis of how this affects margins or valuation."*

#### Morningstar NVDA (partial sensitivity)

> *"25% gross downside risk to earnings from tariff exposure"*

While Morningstar also lacks comprehensive scenario modeling, it at least acknowledges risk magnitudes.

---

### 2.4 Missing Causal Mechanisms

LLM reports fail to explain the **"why" behind projections**.

#### OpenAI AAPL Report - Missing Mechanisms

```
- "Services growth → margin impact not explained"
- "AI delays → sales impact mechanism missing"
- "Ecosystem lock-in → retention rates not quantified"
```

#### Perplexity AAPL Report - Missing Mechanisms

```
- "Services growth → margin expansion linkage unexplained"
- "AI delays → competitive impact not quantified"
- "China risks → revenue correlation not modeled"
```

#### Contrast: Morningstar AAPL (Good depth)

> *"iPhone revenue declining 1% year over year tied to specific drivers...we project 7% compound annual revenue growth with detailed segment breakdowns."*

---

## Part 3: Interesting Observations

### 3.1 Provider Variance in Quality

**Gemini** produces the most coherent LLM reports, yet still fails on fundamentals:

| Stock | XAI | Gemini | Perplexity | OpenAI |
|-------|-----|--------|------------|--------|
| NVDA Coherence | Poor | Poor | — | — |
| AAPL Coherence | Poor | Poor | Poor | Poor |
| NVDA Depth | Poor | Fair | Poor | — |
| AAPL Depth | Fair | Fair | Poor | Poor |

Gemini's NVDA report at least has a complete DCF model with explicit assumptions — it just lacks benchmarking and sensitivity analysis.

---

### 3.2 The "Fabricated Sources" Problem

**Gemini NVDA Report** cites **357 sources** — many with implausible or future-dated URLs.

Sample from sources:
```
[274] NVIDIA Corporation – "NVIDIA Announces Financial Results for Second Quarter
      Fiscal 2026", NVIDIA Newsroom, 2025-08-27

[340] NVIDIA Corporation – "NVIDIA Announces Financial Results for Second Quarter
      Fiscal 2026", NVIDIA Newsroom, 2025-08-27
```

This appears to be **citation hallucination** — fabricating authoritative-looking sources to match retrieved claims.

---

### 3.3 The "X Posts" Credibility Collapse

**XAI NVDA** relies heavily on Twitter/X as a data source:

```
"Adjusted EPS came in at $0.81, against estimates of $0.93...
[from X posts, used for sentiment]"

"gross margin at 61% versus estimates of 71%
[from X posts, used for sentiment]"
```

Using social media sentiment as primary financial data undermines analytical credibility entirely.

---

### 3.4 Structural Similarity Despite Failure

All LLM reports follow the **same template structure**:

1. Cover Block
2. Analyst Note
3. Business Description
4. Business Strategy & Outlook
5. Bulls Say / Bears Say
6. Economic Moat
7. Fair Value and Profit Drivers
8. Risk & Uncertainty
9. Capital Allocation
10. Financials Snapshot
11. ESG Risk
12. Appendix
13. Sources

Yet they fail on execution within that structure. This suggests the **prompt-based template approach** may be a design flaw, creating form without substance.

---

### 3.5 Evaluator Consistency Across Judges

The coherence and depth failures are **consistently identified across different LLM evaluators** (Claude, Gemini, GPT-5), suggesting these are objective structural problems rather than evaluator bias.

---

## Root Cause Analysis

| Failure Mode | Underlying Cause |
|--------------|------------------|
| **Temporal confusion** | LLMs lack genuine temporal reasoning; they pattern-match dates without logical verification |
| **"Insufficient data"** | LLMs cannot reliably retrieve precise financial metrics from knowledge cutoffs |
| **Unreconciled contradictions** | LLMs generate locally coherent text without global consistency checks |
| **Missing causal mechanisms** | LLMs describe patterns in training data but lack causal world models |
| **Unbenchmarked assumptions** | LLMs output plausible-sounding numbers without validation against industry norms |
| **No sensitivity analysis** | LLMs aren't prompted/trained to generate quantified scenario ranges |

---

## Detailed Comparison: NVDA Reports

### Coherence Evaluation Summary

| Provider | Grade | Key Failures |
|----------|-------|--------------|
| **Morningstar** | Good | None flagged |
| **XAI** | Poor | Temporal impossibility, contradictions, "insufficient data" throughout |
| **Gemini** | Poor | Future-dated sources, unreconciled Bulls/Bears, data inconsistencies |
| **Perplexity** | Poor | Descriptive content, missing mechanisms, no sensitivity |

### Depth Evaluation Summary

| Provider | Grade | Causal | Explicit | Benchmarked | Sensitivity | Actionable |
|----------|-------|--------|----------|-------------|-------------|------------|
| **Morningstar** | Good | ✓ | ✓ | ✓ | Partial | ✓ |
| **XAI** | Poor | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Gemini** | Fair | ✓ | ✓ | ❌ | ❌ | ✓ |
| **Perplexity** | Poor | ❌ | ✓ | ❌ | ❌ | ❌ |

---

## Key Excerpts: Good vs. Poor Analysis

### On Fair Value Methodology

**Morningstar NVDA (Good):**
> *"The fair value estimate for Nvidia is raised to $195 per share from $170 following the U.S. government's decision on July 15, 2025, to reverse its export ban on the company's H20 artificial intelligence accelerators to China. This policy shift materially improves the near-term financial outlook, removing a significant headwind previously estimated at $8.0 billion for the second quarter of fiscal 2026. The restoration of access to the high-demand Chinese market prompts the reincorporation of approximately $30 billion in cumulative revenue through fiscal 2027 into the forecast model."*

**XAI NVDA (Poor):**
> *"Our fair value estimate of $227.95 is based on analyst predictions for 2025. Due to insufficient data for a detailed model, we omit specific revenue CAGR, margins, WACC, EPS bridge, and implied multiples."*

### On Risk Analysis

**Morningstar NVDA (Good):**
> *"Despite the positive revision, a Very High Uncertainty rating is maintained. The stock's valuation, at over 30 times forward earnings estimates, reflects expectations of near-flawless execution and offers a limited margin of safety."*

**XAI NVDA (Poor):**
> *"These could swing valuation by 20-30% [insufficient data for precise estimate]."*

---

## Conclusions

The coherence and depth gaps between LLM and analyst reports stem from fundamental architectural limitations:

### Coherence Gap

**Coherence** requires:
- **Temporal grounding** — maintaining a consistent timeline
- **Global consistency** — ensuring all sections align
- **Data integrity** — not mixing real and fabricated information

LLMs fail on all three dimensions because they generate text token-by-token without maintaining a global state or verifying logical consistency.

### Depth Gap

**Depth** requires:
- **Causal reasoning** — explaining *why* things happen
- **Domain-specific judgment** — knowing what matters in equity analysis
- **Scenario thinking** — modeling ranges, not point estimates

LLMs substitute pattern-matching for genuine understanding. They can reproduce the *form* of analytical writing but not the *substance*.

### The Fundamental Problem

The analyst advantage isn't just expertise — it's the ability to:

1. **Maintain a coherent mental model** of a company across all report sections
2. **Reconcile contradictory data** into a unified thesis
3. **Explain *why*** something matters for valuation
4. **Quantify uncertainty** through scenarios and sensitivity analysis

LLMs produce locally fluent text that collapses when examined for logical structure. They can describe; they cannot analyze.

---

## Recommendations for Improvement

1. **Implement temporal verification** — Pre-process sources to filter impossible date combinations
2. **Add global consistency checks** — Cross-reference Bulls/Bears with base case assumptions
3. **Require assumption benchmarking** — Compare WACC, terminal growth to industry medians
4. **Mandate scenario analysis** — Generate bull/base/bear cases with explicit triggers
5. **Source quality filtering** — Exclude social media as primary financial data sources
6. **Fill data gaps proactively** — Either retrieve accurate data or explicitly state limitations rather than building on "insufficient data"

---

*Analysis generated from examination of 2,877 evaluation files and 197 reports across 5 providers (Morningstar, XAI, Gemini, OpenAI, Perplexity) covering 50 stocks.*
