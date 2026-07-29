# PyMC 5 → PyMC 6 Migration Baseline Comparison

This is the permanent attachment structure for #1048. It is not evidence by itself: the coordinator must generate a populated Markdown report with `scripts/migration_baseline/harness.py compare` and attach that generated report together with its four input artifacts.

**Result:** `PASS` or `FAIL` from the generated comparison.

## Scope and attribution

- Reference checkout: `79c0a87072fd4653bfaed1eb085f965594c7f03a` in the PyMC 5 environment.
- Candidate checkout: `18a524a1a8512aaa21c46e0ccddbc54501c9eb1a` in the PyMC 6 migration environment.
- Both sampled checkouts had an empty `git status --porcelain` before CausalPy import. The harness implementation checkout is not a sampled candidate checkout after it is committed.
- Attribution statement: this evidence covers only the fixed migration delta. Any behavior on a later source commit is a separate feature comparison and must not be described as migration drift.

## Provenance and deterministic protocol

Record the four artifact paths and SHA-256 fields, the exact CausalPy import paths, Python/PyMC/PyTensor/ArviZ versions, platform, harness SHA, fixed fixture hashes, and command log.

Record that each capture used PyMC NUTS with four serialized chains (`cores=1`), 1,000 tuning iterations, 1,000 retained draws, master seed `1048`, target acceptance `0.95`, and a maximum tree depth of `12`. State that `cores=1` was mandatory on local macOS and intentionally retained on all platforms to serialize the chain schedule.

## Evidence validity gate

Report that both independent captures within each stack produced exact same-stack raw-draw digests for every captured metric. Do not compare raw draw digests across stacks.

Report the per-case divergent-draw count, tree-depth result when exposed, finite-value result, maximum rank R-hat, minimum bulk ESS, and minimum tail ESS. A failure in any of these checks invalidates the numerical comparison rather than being treated as a migration measurement.

## Hard migration gates

Apply each hard gate to every unrounded posterior scalar from the representative Difference-in-Differences and Synthetic Control scenarios, including each draw-wise R² and counterfactual coordinate:

1. `abs(candidate_mean - reference_mean) <= max(4 * hypot(reference_mcse, candidate_mcse), 1e-6 + 1e-4 * abs(reference_mean))`.
2. `abs(candidate_mean - reference_mean) / pooled_posterior_sd <= 0.1`, with the absolute rule used when the pooled posterior SD is degenerate.
3. Effect-summary table schema, requested `0.94` HDI metadata and labels, metric selectors, dimensions, shapes, and coordinate values are semantically identical.

Mutual mean-in-94%-HDI containment is diagnostic only. It must appear in the metric table but never change the overall decision.

## Representative outputs

Include the generated metric table for the following contracts:

| Scenario | Contract | Required output |
|---|---|---|
| Difference-in-Differences | Public effect summary | Semantic table equality with unrounded treatment-effect mean and explicit 0.94 HDI |
| Difference-in-Differences | In-sample fit | Draw-wise R² derived from conditional expected `mu` |
| Difference-in-Differences | Counterfactual | Treated-post `mu` summaries with exact coordinate equality |
| Synthetic Control | Public effect summary | Semantic table equality with unrounded average and cumulative effect means and explicit 0.94 HDIs |
| Synthetic Control | In-sample fit | Pre-treatment draw-wise R² derived from conditional expected `mu` |
| Synthetic Control | Counterfactual | Post-treatment `mu` summaries with exact coordinate equality |

## Conclusion and follow-on use

State the overall gate result and identify every failed hard gate, if any. A diagnostic HDI-containment failure without a failed hard gate is reported but does not change the conclusion.

For #157, reuse the fixed inputs, summary extraction, and semantic contracts as correctness-test infrastructure, but evaluate known simulation ground truth with independently registered posterior-SD-unit bounds. Do not make PyMC 5 output the correctness oracle.
