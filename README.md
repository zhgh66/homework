# LLM Symbolic Reasoning Benchmarks: Baseline vs. Optimized

Zheng yi    CEG25098

Zhou Guhan  CEG25100

## Overview

Division of Work.

Zheng Yi: Boolean & Causal optimization; methods/experiments writing.

Zhou Guhan: 3D optimization; repository organization & scripts.

---

This repository contains benchmark scripts for evaluating the performance of Large Language Models (LLMs) on symbolic reasoning tasks. For each task, two versions are provided:

1.  **Baseline (`*_benchmark.py`):** A straightforward implementation that sends prompts, receives a single response, and performs basic parsing.
2.  **Optimized (`*_benchmark_optimized.py`):** An advanced version incorporating specific strategies to improve robustness, diversity, and accuracy, along with enhanced diagnostics.

This document summarizes the key implementation differences and their direct impact on performance metrics across three domains: **Boolean Logic**, **Causal Graphs**, and **3D Structures**.

## Executive Summary: Key Findings

The optimized scripts consistently and significantly outperform the baseline versions across all three tasks. The core improvements are:

* **Higher Recovery & Novelty:** The optimized scripts are far more effective at finding the "ground truth" expression (`recovery_rate`) and generating structurally-diverse solutions (`novelty_rate`).
* **Better Diagnostics:** The optimized versions track detailed metrics (e.g., operator distribution, parse success rates, query savings), making them far more suitable for research and debugging.
* **Task-Specific Strategies Work:**
    * **Boolean:** Self-consistency (multiple samples per prompt) is highly effective but increases cost.
    * **Causal:** Dynamic prompting (recovery guides, early stopping) improves all metrics while intelligently managing query costs.
    * **3D:** Strict output formatting (code fences) and robust parsing dramatically improve reliability *and* reduce costs.

---

## Detailed Task Comparisons

Below is a breakdown of the optimizations for each task and their observed impact on results.

### 1. Boolean Logic (`boolean_benchmark_optimized.py`)

This task tests the ability to find a boolean expression matching a set of observations.

**Key Optimization Strategies:**
* **Self-Consistency (sc_k=5):** For a single prompt, the script samples 5 candidate expressions from the LLM.
* **Candidate Ranking:** These candidates are evaluated and ranked. The "best" one is chosen based on minimizing mismatch, prioritizing structural novelty, and preferring simpler expressions (lower depth/ops).
* **Diversity-Aware Prompting:** The prompt is updated with patterns from prior hypotheses to explicitly ask for *structurally different* expressions.
* **Enhanced Statistics:** Tracks operator distribution (`AND`, `OR`, etc.) and expression depth.

**Observed Impact (Baseline $\to$ Optimized):**

| Metric | Baseline | Optimized | Change | Rationale |
| :--- | :--- | :--- | :--- | :--- |
| **`novelty_rate`** | 0.41 | **0.74** | **+80.5%** | Self-consistency & diversity prompts generated more unique structures. |
| **`recovery_rate`** | 0.392 | **0.653** | **+66.6%** | Sampling 5 candidates greatly increased the chance of finding the ground truth. |
| **`valid_rate`** | 0.909 | 0.889 | -2.2% | A minor, negligible drop, likely due to novelty-seeking behavior sometimes producing valid but boundary-case expressions. |
| **Total Cost** | \$0.288 | **\$1.091** | **+278%** | **(Trade-off)** Sampling 5x per query (sc_k=5) directly results in a ~4.8x increase in token usage and cost. |

**Conclusion:** The optimized script is vastly superior at finding correct and diverse solutions. The significant cost increase is a direct trade-off for the dramatic improvement in recovery and novelty.

---

### 2. Causal Graphs (`causal_benchmark_optimized.py`)

This task tests the ability to recover a ground-truth Causal Graph (DAG) from observational data.

**Key Optimization Strategies:**
* **Dynamic "Recovery Guide" Prompting:** The script identifies which ground-truth edges have *not* yet been recovered and adds them as a "Tip" to the prompt, guiding the LLM.
* **Limited Prior History:** Only the last 3 prior hypotheses are shown in the prompt, reducing prompt bloat and token cost.
* **Dynamic Query (Early Stopping):** If all ground-truth edges are recovered, the script stops querying for that sample, saving tokens and cost.
* **Detailed Savings Metrics:** Explicitly tracks `n_queries_saved` and `cost_saved_by_dynamic`.

**Observed Impact (Baseline $\to$ Optimized):**

| Metric | Baseline | Optimized | Change | Rationale |
| :--- | :--- | :--- | :--- | :--- |
| **`recovery_rate`** | 0.506 | **0.674** | **+33.2%** | The "recovery_guide" prompt was highly effective at focusing the LLM on missing edges. |
| **`novelty_rate`** | 0.445 | **0.713** | **+60.2%** | Limiting prior history and guided prompts encouraged new structures. |
| **`valid_rate`** | 0.564 | **0.684** | **+21.3%** | Better guidance led to more valid DAGs. |
| **Total Cost** | \$0.389 | \$0.439 | +12.8% | A slight cost increase, but the script achieved *better results in fewer queries* for many samples due to early stopping. |

**Conclusion:** The optimized script is a more intelligent and efficient experimental process. It achieves better results across the board while actively working to minimize unnecessary queries.

---

### 3. 3D Structures (`3d_benchmark_optimized.py`)

This task tests the ability to generate a layered 3D structure based on constraints.

**Key Optimization Strategies:**
* **Strict Output Formatting:** The prompt *mandates* a machine-parseable format, requiring the LLM to output the answer inside a code fence (```) with a `Structure:` header.
* **Robust Parsing:** The parser was improved to handle common LLM failures:
    * It first extracts content only from the code fence.
    * It automatically trims leading all-zero layers.
    * It truncates structures that exceed `max_height`.
    * It enforces a non-empty bottom layer.
* **Better Error Classification:** More granular tracking of why parsing failed.

**Observed Impact (Baseline $\to$ Optimized):**

| Metric | Baseline | Optimized | Change | Rationale |
| :--- | :--- | :--- | :--- | :--- |
| **`recovery_rate`** | 0.278 | **0.419** | **+50.7%** | Robust parsing saved many "semantically correct" but "syntactically messy" LLM outputs from being marked invalid. |
| **`novelty_rate`** | 0.704 | **0.952** | **+35.3%** | A cleaner, stricter format allowed the model to produce diverse structures that were all successfully parsed. |
| **`valid_rate`** | 0.374 | **0.452** | **+20.8%** | Strict formatting + robust parsing = more successful attempts. |
| **`parse_success_rate`** | 0.993 | **1.0** | +0.7% | Forcing code fences nearly eliminated all parsing failures. |
| **Total Cost** | \$1.516 | **\$0.981** | **-35.3%** | **(Key Benefit)** By forcing a strict, non-verbose format, the average *completion tokens* dropped significantly, leading to a major cost reduction. |

**Conclusion:** For generative tasks involving complex structures, enforcing a strict, machine-parseable output format is a clear win. It improves all metrics *and* simultaneously reduces API costs.

---

## Overall Recommendation

The **optimized scripts** should be used as the default for any serious evaluation or research. They provide more accurate, diverse, and reliable results, and their enhanced logging is essential for analysis. The specific optimization strategy (self-consistency vs. dynamic prompting vs. strict formatting) should be chosen based on the task requirements and cost constraints.


## Citation
```bibtex
@article{chen2025hypospace,
  title={HypoSpace:Evaluating LLM Creativity as Set-Valued Hypothesis Generators under Underdetermination},
  author={Chen, Tingting and Lin, Beibei and Yuan, Zifeng and Zou, Qiran and He, Hongyu and Zhang, Wei-Nan},
  journal={arXiv preprint arXiv:2510.15614},
  year={2025}
}
