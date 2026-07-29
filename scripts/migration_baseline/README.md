# PyMC Migration Baseline Harness

This permanent harness produces reproducible baseline evidence for the PyMC 5 → PyMC 6 migration at the only two revisions that may be attributed to that migration: PyMC 5 reference `79c0a87072fd4653bfaed1eb085f965594c7f03a` and PyMC 6 migration candidate `18a524a1a8512aaa21c46e0ccddbc54501c9eb1a`. It rejects every other checkout revision so that later features are investigated as separate changes rather than mislabeled migration drift.

The tracked implementation is `scripts/migration_baseline/harness.py`; generated JSON and Markdown evidence deliberately belongs outside every Git checkout. The harness rejects output paths inside its own checkout or either sampled checkout, so an attachment cannot accidentally become a stale tracked result. Each sampled checkout must be clean and have the exact pinned `HEAD`; the implementation checkout only hosts the harness and is never a sampled PyMC 6 source tree after this change is committed.

## What is captured

The fixed suite has two representative Bayesian experiments using input rows serialized directly in the harness rather than regenerated through package-version-dependent simulation code:

- `DifferenceInDifferences` with `LinearRegression`, including its public effect-summary table, treatment-effect posterior, fitted draw-wise R², and treated-post conditional-mean counterfactual.
- `SyntheticControl` with `WeightedSumFitter`, including its public average/cumulative effect-summary rows, post-impact posteriors, fitted draw-wise R², and post-treatment conditional-mean counterfactual.

All counterfactuals are adapter-returned `mu` conditional expected values. The harness rejects noisy `y_hat` predictions, missing canonical dimensions, or changed coordinate values.

## Registered protocol

- Explicit HDI probability: `0.94`, passed to every `effect_summary()` as `alpha=0.06`.
- Sampler: PyMC NUTS, four chains with `cores=1`, 1,000 tuning iterations, 1,000 retained draws, master seed `1048`, target acceptance `0.95`, and maximum tree depth `12`.
- `cores=1` is intentional on every platform. In particular it is mandatory on local macOS to avoid Accelerate/numba fork failures and to make the chain schedule deterministic.
- Capture fails before comparison for divergent chains, tree-depth saturation when that statistic is exposed, missing sample statistics, non-finite values, rank R-hat above `1.01`, bulk ESS below `400`, or tail ESS below `400`.
- The draw-wise R² formula is the CausalPy formula evaluated independently for every `(chain, draw, treated_unit)`: `var_obs(mu) / (var_obs(mu) + var_obs(y - mu))` with `ddof=0`. The harness keeps its posterior draws long enough to calculate MCSE and convergence diagnostics before it serializes only summaries.

The JSON artifact records the exact CausalPy import path, Git SHA, harness SHA, package versions, platform, sampler configuration, fixture rows and hash, table semantics, coordinate semantics, posterior summaries, diagnostics, and raw-draw digests. It is strict JSON with deterministic key ordering and atomic writes.

## Required coordinator run

The following commands are intentionally **pending coordinator execution**. They require the separate editable-install prefixes prescribed for the two sampled source worktrees; do not run both stacks through one shared environment.

The coordinator must provision `PYMC6_ROOT` as a separate clean detached worktree at `18a524a1a8512aaa21c46e0ccddbc54501c9eb1a` and install it into its own editable-install prefix. Do not use the committed `migration/1048-baseline-harness` checkout as `PYMC6_ROOT`: its source `HEAD` intentionally differs from the migration candidate.

```bash
MAMBA=/opt/anaconda3/condabin/mamba
MIGRATION_ROOT=/Users/carlostrujillo/Documents/GitHub/_worktrees/CausalPy-1048-baselines
PYMC5_ROOT=/Users/carlostrujillo/Documents/GitHub/_worktrees/CausalPy-1048-pymc5
PYMC5_PREFIX=/Users/carlostrujillo/Documents/GitHub/_worktrees/.mamba/CausalPy-1048-pymc5
PYMC6_ROOT=/Users/carlostrujillo/Documents/GitHub/_worktrees/CausalPy-1048-pymc6
PYMC6_PREFIX=/Users/carlostrujillo/Documents/GitHub/_worktrees/.mamba/CausalPy-1048-pymc6
EVIDENCE_ROOT=/Users/carlostrujillo/Documents/GitHub/_worktrees/migration-baseline-evidence
HARNESS="$MIGRATION_ROOT/scripts/migration_baseline/harness.py"

mkdir -p "$EVIDENCE_ROOT"

"$MAMBA" run -p "$PYMC5_PREFIX" python "$HARNESS" capture --stack pymc5 --repo-root "$PYMC5_ROOT" --output "$EVIDENCE_ROOT/pymc5-run-1.json"
"$MAMBA" run -p "$PYMC5_PREFIX" python "$HARNESS" capture --stack pymc5 --repo-root "$PYMC5_ROOT" --output "$EVIDENCE_ROOT/pymc5-run-2.json"
"$MAMBA" run -p "$PYMC6_PREFIX" python "$HARNESS" capture --stack pymc6 --repo-root "$PYMC6_ROOT" --output "$EVIDENCE_ROOT/pymc6-run-1.json"
"$MAMBA" run -p "$PYMC6_PREFIX" python "$HARNESS" capture --stack pymc6 --repo-root "$PYMC6_ROOT" --output "$EVIDENCE_ROOT/pymc6-run-2.json"
"$MAMBA" run -p "$PYMC6_PREFIX" python "$HARNESS" compare \
  --reference-first "$EVIDENCE_ROOT/pymc5-run-1.json" \
  --reference-second "$EVIDENCE_ROOT/pymc5-run-2.json" \
  --candidate-first "$EVIDENCE_ROOT/pymc6-run-1.json" \
  --candidate-second "$EVIDENCE_ROOT/pymc6-run-2.json" \
  --output "$EVIDENCE_ROOT/1048-baseline-comparison.json" \
  --report "$EVIDENCE_ROOT/1048-baseline-report.md"
```

Run the two captures for a stack as independent process invocations. The `capture` command rejects a dirty sampled checkout before importing CausalPy. The `compare` command requires four distinct artifact paths and first verifies exact raw-draw digests only within each pair. A mismatch makes the entire comparison fail as non-deterministic evidence. It never compares a PyMC 5 digest or raw draw with a PyMC 6 digest or raw draw.

A failed comparison still writes its JSON decision and Markdown report, then exits with status `1`; malformed or invalid evidence exits with status `2`. The generated report records all four artifact paths and content SHA-256 values, capture provenance, and actual validity diagnostics. Attach it and its four input artifacts to #1048 with the command log. The static attachment outline is in [REPORT_TEMPLATE.md](REPORT_TEMPLATE.md).

## Registered migration decision gates

The comparator applies these migration hard gates to every independent posterior scalar, including effect quantities, each draw-wise R² series, and each counterfactual coordinate:

1. `abs(candidate_mean - reference_mean) <= max(4 * hypot(reference_mcse, candidate_mcse), 1e-6 + 1e-4 * abs(reference_mean))`.
2. `abs(candidate_mean - reference_mean) / pooled_posterior_sd <= 0.1`, where `pooled_posterior_sd = sqrt((reference_sd² + candidate_sd²) / 2)`. A degenerate posterior uses the absolute gate because standardized drift has no meaningful denominator.
3. Effect-table schema, explicit 0.94 HDI column labels, metric selectors, dimension order, shape, and coordinate values must match exactly.

Mutual mean-in-94%-HDI containment is a diagnostic in the rendered table only. It does not pass or fail a migration gate.

## Reuse for #157

The harness is reusable infrastructure, not a claim that a PyMC 5 posterior is ground truth. #157 correctness tests can reuse its fixed fixture serialization, explicit HDI extraction, draw-wise R² calculation, semantic schema checks, and reporting layout, but their primary assertions must compare simulated-data estimates against known data-generating parameters with separately registered posterior-SD-unit tolerances. Single-realization 94% HDI coverage remains diagnostic in that later correctness work.
