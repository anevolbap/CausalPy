"""Capture and compare the pinned PyMC 5 and PyMC 6 migration baselines.

The capture command deliberately imports CausalPy from ``--repo-root`` after
checking its exact Git revision. Run it once in each dedicated environment,
twice per stack, then pass all four JSON artifacts to ``compare``. Comparison
uses posterior summaries and semantic schemas, never cross-stack draw equality.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

SUITE_NAME = "causalpy-pymc-migration-baseline"
ARTIFACT_SCHEMA_VERSION = 1
COMPARISON_SCHEMA_VERSION = 1
SCENARIO_VERSION = 1

PYMC5_COMMIT = "79c0a87072fd4653bfaed1eb085f965594c7f03a"
PYMC6_COMMIT = "18a524a1a8512aaa21c46e0ccddbc54501c9eb1a"
STACK_COMMITS = {"pymc5": PYMC5_COMMIT, "pymc6": PYMC6_COMMIT}

HDI_PROB = 0.94
EFFECT_SUMMARY_ALPHA = 0.06
CHAINS = 4
DRAWS = 1_000
TUNE = 1_000
CORES = 1
MASTER_SEED = 1048
TARGET_ACCEPT = 0.95
MAX_TREEDEPTH = 12
MAX_RHAT = 1.01
MIN_ESS_BULK = 400.0
MIN_ESS_TAIL = 400.0

ABSOLUTE_TOLERANCE_FLOOR = 1e-6
RELATIVE_TOLERANCE_FLOOR = 1e-4
MCSE_MULTIPLIER = 4.0
MAX_STANDARDIZED_DRIFT = 0.1
DEGENERATE_POSTERIOR_SD = 1e-12

# DifferenceInDifferences requires stable unit labels even when the formula
# omits them.
DID_RECORDS: tuple[dict[str, Any], ...] = (
    {"unit": "control", "t": 0, "group": 0, "post_treatment": False, "y": 10.00},
    {"unit": "control", "t": 1, "group": 0, "post_treatment": False, "y": 10.65},
    {"unit": "control", "t": 2, "group": 0, "post_treatment": False, "y": 11.10},
    {"unit": "control", "t": 3, "group": 0, "post_treatment": False, "y": 11.82},
    {"unit": "control", "t": 4, "group": 0, "post_treatment": False, "y": 12.34},
    {"unit": "control", "t": 5, "group": 0, "post_treatment": False, "y": 12.81},
    {"unit": "control", "t": 6, "group": 0, "post_treatment": True, "y": 13.15},
    {"unit": "control", "t": 7, "group": 0, "post_treatment": True, "y": 13.72},
    {"unit": "control", "t": 8, "group": 0, "post_treatment": True, "y": 14.10},
    {"unit": "control", "t": 9, "group": 0, "post_treatment": True, "y": 14.70},
    {"unit": "control", "t": 10, "group": 0, "post_treatment": True, "y": 15.10},
    {"unit": "control", "t": 11, "group": 0, "post_treatment": True, "y": 15.60},
    {"unit": "treated", "t": 0, "group": 1, "post_treatment": False, "y": 11.30},
    {"unit": "treated", "t": 1, "group": 1, "post_treatment": False, "y": 11.75},
    {"unit": "treated", "t": 2, "group": 1, "post_treatment": False, "y": 12.40},
    {"unit": "treated", "t": 3, "group": 1, "post_treatment": False, "y": 13.00},
    {"unit": "treated", "t": 4, "group": 1, "post_treatment": False, "y": 13.42},
    {"unit": "treated", "t": 5, "group": 1, "post_treatment": False, "y": 13.98},
    {"unit": "treated", "t": 6, "group": 1, "post_treatment": True, "y": 15.30},
    {"unit": "treated", "t": 7, "group": 1, "post_treatment": True, "y": 15.90},
    {"unit": "treated", "t": 8, "group": 1, "post_treatment": True, "y": 16.40},
    {"unit": "treated", "t": 9, "group": 1, "post_treatment": True, "y": 17.00},
    {"unit": "treated", "t": 10, "group": 1, "post_treatment": True, "y": 17.50},
    {"unit": "treated", "t": 11, "group": 1, "post_treatment": True, "y": 18.20},
)

SYNTHETIC_CONTROL_RECORDS: tuple[dict[str, Any], ...] = (
    {"t": 0, "a": 10.00, "b": 9.80, "c": 10.40, "actual": 10.02},
    {"t": 1, "a": 10.30, "b": 10.05, "c": 10.20, "actual": 10.17},
    {"t": 2, "a": 10.55, "b": 10.30, "c": 10.65, "actual": 10.49},
    {"t": 3, "a": 10.25, "b": 10.50, "c": 10.70, "actual": 10.45},
    {"t": 4, "a": 10.85, "b": 10.75, "c": 10.90, "actual": 10.82},
    {"t": 5, "a": 11.10, "b": 10.95, "c": 11.25, "actual": 11.07},
    {"t": 6, "a": 11.35, "b": 11.20, "c": 11.10, "actual": 11.24},
    {"t": 7, "a": 11.15, "b": 11.35, "c": 11.50, "actual": 11.28},
    {"t": 8, "a": 11.65, "b": 11.55, "c": 11.80, "actual": 11.57},
    {"t": 9, "a": 11.95, "b": 11.75, "c": 11.60, "actual": 11.81},
    {"t": 10, "a": 12.20, "b": 12.05, "c": 12.25, "actual": 12.14},
    {"t": 11, "a": 12.05, "b": 12.30, "c": 12.50, "actual": 12.20},
    {"t": 12, "a": 12.55, "b": 12.45, "c": 12.70, "actual": 14.34},
    {"t": 13, "a": 12.85, "b": 12.70, "c": 12.90, "actual": 14.58},
    {"t": 14, "a": 13.10, "b": 12.90, "c": 13.20, "actual": 14.89},
    {"t": 15, "a": 12.90, "b": 13.10, "c": 13.35, "actual": 14.91},
    {"t": 16, "a": 13.40, "b": 13.30, "c": 13.55, "actual": 15.33},
    {"t": 17, "a": 13.70, "b": 13.55, "c": 13.80, "actual": 15.62},
    {"t": 18, "a": 13.95, "b": 13.75, "c": 14.10, "actual": 15.92},
    {"t": 19, "a": 14.25, "b": 14.00, "c": 14.30, "actual": 16.23},
)


class HarnessError(RuntimeError):
    """Raised when a baseline artifact is not valid migration evidence."""


def _is_within(path: Path, root: Path) -> bool:
    """Return whether ``path`` resolves below ``root``."""
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _script_repository_root() -> Path:
    """Return the tracked repository containing this permanent harness."""
    return Path(__file__).resolve().parents[2]


def _require_external_output(path: Path, *tracked_roots: Path) -> Path:
    """Reject generated evidence paths inside a tracked checkout."""
    resolved = path.expanduser().resolve()
    for root in tracked_roots:
        if _is_within(resolved, root):
            raise HarnessError(
                f"Generated evidence must be outside tracked checkouts, got {resolved} "
                f"inside {root.resolve()}."
            )
    return resolved


def _canonical_value(value: Any) -> Any:
    """Convert scalar metadata to deterministic JSON-compatible values."""
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return _canonical_value(value.item())
        except ValueError:
            pass
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, float):
        if not math.isfinite(value):
            raise HarnessError(f"Non-finite value in artifact metadata: {value!r}")
        return float(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, tuple):
        return [_canonical_value(item) for item in value]
    return str(value)


def _canonical_json(value: Any) -> str:
    """Return canonical JSON for deterministic hashes and artifact serialization."""
    return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)


def _sha256_text(text: str) -> str:
    """Return the SHA-256 digest of UTF-8 text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _records_payload(records: tuple[dict[str, Any], ...]) -> list[dict[str, Any]]:
    """Copy fixed fixture records into JSON-safe artifact data."""
    return [
        {key: _canonical_value(value) for key, value in row.items()} for row in records
    ]


def _fixture_payload(name: str, records: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    """Return fixed, serialized input data and its content hash."""
    payload = _records_payload(records)
    return {
        "name": name,
        "records": payload,
        "sha256": _sha256_text(_canonical_json(payload)),
    }


def _atomic_write_text(path: Path, text: str) -> None:
    """Write text atomically so incomplete evidence is never mistaken for valid."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write sorted, strict JSON atomically."""
    _assert_finite_json(payload)
    serialized = json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    _atomic_write_text(path, serialized)


def _assert_finite_json(value: Any, path: str = "artifact") -> None:
    """Reject NaN and infinity anywhere in a JSON-compatible object."""
    if isinstance(value, float):
        if not math.isfinite(value):
            raise HarnessError(f"{path} contains non-finite float {value!r}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _assert_finite_json(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_finite_json(item, f"{path}.{key}")


def _run_git(repo_root: Path, *args: str) -> str:
    """Run a read-only Git query against an explicitly selected repository."""
    result = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip()
        raise HarnessError(f"Git query failed in {repo_root}: {message}")
    return result.stdout.strip()


def _activate_repository(repo_root: Path) -> Path:
    """Put exactly one checkout first on ``sys.path`` and return its source path."""
    resolved = repo_root.expanduser().resolve()
    source = resolved / "causalpy"
    if not source.is_dir():
        raise HarnessError(f"{resolved} does not contain a causalpy source directory")
    sys.path.insert(0, str(resolved))
    return resolved


def _import_capture_dependencies(repo_root: Path) -> dict[str, Any]:
    """Import sampling dependencies only after pinning the requested checkout."""
    resolved = _activate_repository(repo_root)
    import arviz as az
    import numpy as np
    import pandas as pd
    import pymc as pm
    import xarray as xr

    import causalpy as cp

    module_path = Path(cp.__file__).resolve()
    if not _is_within(module_path, resolved):
        raise HarnessError(
            "CausalPy was not imported from --repo-root: "
            f"expected below {resolved}, imported {module_path}."
        )
    return {"az": az, "cp": cp, "np": np, "pd": pd, "pm": pm, "xr": xr}


def _distribution_version(name: str) -> str:
    """Return an installed distribution version without failing on optional packages."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _sample_kwargs(pm: Any) -> tuple[dict[str, Any], bool]:
    """Build the registered serialized-chain sampling configuration."""
    kwargs: dict[str, Any] = {
        "chains": CHAINS,
        "cores": CORES,
        "draws": DRAWS,
        "tune": TUNE,
        "random_seed": MASTER_SEED,
        "progressbar": False,
        "target_accept": TARGET_ACCEPT,
        "max_treedepth": MAX_TREEDEPTH,
    }
    try:
        supports_nuts_sampler = (
            "nuts_sampler" in inspect.signature(pm.sample).parameters
        )
    except (TypeError, ValueError):
        supports_nuts_sampler = False
    if supports_nuts_sampler:
        kwargs["nuts_sampler"] = "pymc"
    return kwargs, supports_nuts_sampler


def _protocol(supports_nuts_sampler: bool) -> dict[str, Any]:
    """Return the preregistered protocol recorded in every capture artifact."""
    return {
        "scenario_version": SCENARIO_VERSION,
        "hdi_prob": HDI_PROB,
        "effect_summary_alpha": EFFECT_SUMMARY_ALPHA,
        "draw_wise_r2": {
            "definition": (
                "var_obs(mu) / (var_obs(mu) + var_obs(y - mu)) with ddof=0 "
                "for every chain, draw, and treated unit"
            ),
            "prediction_kind": "conditional_expected_mu",
        },
        "sampling": {
            "sampler": "pymc",
            "chains": CHAINS,
            "cores": CORES,
            "draws": DRAWS,
            "tune": TUNE,
            "master_seed": MASTER_SEED,
            "target_accept": TARGET_ACCEPT,
            "max_treedepth": MAX_TREEDEPTH,
            "nuts_sampler_argument_used": supports_nuts_sampler,
        },
        "evidence_validity": {
            "max_rhat": MAX_RHAT,
            "min_ess_bulk": MIN_ESS_BULK,
            "min_ess_tail": MIN_ESS_TAIL,
            "divergences": 0,
            "reject_tree_depth_saturation_when_exposed": True,
        },
        "migration_hard_gates": {
            "absolute_mean_delta": (
                "abs(candidate_mean - reference_mean) <= "
                "max(4 * hypot(reference_mcse, candidate_mcse), "
                "1e-6 + 1e-4 * abs(reference_mean))"
            ),
            "standardized_mean_drift": (
                "abs(candidate_mean - reference_mean) / pooled_posterior_sd <= 0.1"
            ),
            "semantic_equality": "effect-table schema and prediction coordinates",
            "hdi_containment": "diagnostic only",
        },
    }


def _group(idata: Any, name: str) -> Any:
    """Read an InferenceData or DataTree group through the common public shape."""
    group = getattr(idata, name, None)
    if group is None:
        try:
            group = idata[name]
        except (KeyError, TypeError):
            group = None
    if group is None:
        raise HarnessError(f"Inference output is missing the required {name!r} group")
    dataset = getattr(group, "ds", None)
    return dataset if dataset is not None else group


def _sampling_quality(idata: Any, np: Any) -> dict[str, Any]:
    """Reject divergent or tree-depth-saturated chains before comparing outputs."""
    sample_stats = _group(idata, "sample_stats")
    if "diverging" not in sample_stats:
        raise HarnessError("Inference output is missing sample_stats.diverging")
    divergences = int(np.asarray(sample_stats["diverging"].values, dtype=int).sum())
    if divergences:
        raise HarnessError(
            f"Divergent chains invalidate baseline evidence ({divergences} draws)"
        )

    tree_depth_status = "not-exposed"
    tree_depth_events = 0
    max_observed_tree_depth: int | None = None
    if "reached_max_treedepth" in sample_stats:
        tree_depth_events = int(
            np.asarray(sample_stats["reached_max_treedepth"].values, dtype=int).sum()
        )
        tree_depth_status = "reached_max_treedepth"
    elif "tree_depth" in sample_stats:
        tree_depth = np.asarray(sample_stats["tree_depth"].values, dtype=int)
        max_observed_tree_depth = int(tree_depth.max())
        tree_depth_events = int((tree_depth >= MAX_TREEDEPTH).sum())
        tree_depth_status = "tree_depth"
    if tree_depth_events:
        raise HarnessError(
            "Tree-depth saturation invalidates baseline evidence "
            f"({tree_depth_events} draws; source={tree_depth_status})."
        )

    return {
        "divergences": divergences,
        "tree_depth_source": tree_depth_status,
        "tree_depth_events": tree_depth_events,
        "max_observed_tree_depth": max_observed_tree_depth,
    }


def _finite_scalar(value: Any, label: str, np: Any) -> float:
    """Convert one scalar diagnostic to a finite Python float."""
    array = np.asarray(value)
    if array.size != 1:
        raise HarnessError(f"{label} must be scalar, got shape {array.shape}")
    result = float(array.reshape(()))
    if not math.isfinite(result):
        raise HarnessError(f"{label} is non-finite: {result!r}")
    return result


def _draw_digest(draws: Any, np: Any) -> str:
    """Hash a raw draw matrix only for same-stack repeatability checks."""
    values = np.ascontiguousarray(np.asarray(draws, dtype="<f8"))
    header = _canonical_json({"dtype": "float64-le", "shape": list(values.shape)})
    digest = hashlib.sha256(header.encode("utf-8"))
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _hdi_interval(flattened_draws: Any, az: Any, np: Any) -> Any:
    """Call the installed ArviZ HDI API with an explicitly requested probability."""
    try:
        parameter_names = inspect.signature(az.hdi).parameters
    except (TypeError, ValueError):
        parameter_names = {}
    keyword = "prob" if "prob" in parameter_names else "hdi_prob"
    return np.asarray(az.hdi(flattened_draws, **{keyword: HDI_PROB}), dtype=float)


def _summary_from_draws(draws: Any, az: Any, np: Any, label: str) -> dict[str, float]:
    """Summarize one chain-by-draw scalar with validity diagnostics."""
    values = np.asarray(draws, dtype=float)
    if values.shape != (CHAINS, DRAWS):
        raise HarnessError(
            f"{label} must have shape ({CHAINS}, {DRAWS}), got {values.shape}"
        )
    if not np.isfinite(values).all():
        raise HarnessError(f"{label} contains non-finite posterior draws")

    flattened = values.reshape(-1)
    interval = _hdi_interval(flattened, az, np)
    if interval.shape != (2,):
        raise HarnessError(f"{label} returned invalid HDI shape {interval.shape}")

    summary = {
        "mean": _finite_scalar(values.mean(), f"{label} mean", np),
        "posterior_sd": _finite_scalar(values.std(ddof=1), f"{label} posterior SD", np),
        "mcse_mean": _finite_scalar(
            az.mcse(values, method="mean"), f"{label} MCSE", np
        ),
        "rhat": _finite_scalar(az.rhat(values, method="rank"), f"{label} R-hat", np),
        "ess_bulk": _finite_scalar(
            az.ess(values, method="bulk"), f"{label} bulk ESS", np
        ),
        "ess_tail": _finite_scalar(
            az.ess(values, method="tail"), f"{label} tail ESS", np
        ),
        "hdi_lower": _finite_scalar(interval[0], f"{label} HDI lower", np),
        "hdi_upper": _finite_scalar(interval[1], f"{label} HDI upper", np),
    }
    if summary["hdi_lower"] > summary["hdi_upper"]:
        raise HarnessError(f"{label} has inverted {HDI_PROB:.2f} HDI bounds")
    if summary["rhat"] > MAX_RHAT:
        raise HarnessError(
            f"{label} has R-hat {summary['rhat']:.6g}, above {MAX_RHAT:.6g}"
        )
    if summary["ess_bulk"] < MIN_ESS_BULK:
        raise HarnessError(
            f"{label} has bulk ESS {summary['ess_bulk']:.6g}, below {MIN_ESS_BULK:.6g}"
        )
    if summary["ess_tail"] < MIN_ESS_TAIL:
        raise HarnessError(
            f"{label} has tail ESS {summary['ess_tail']:.6g}, below {MIN_ESS_TAIL:.6g}"
        )
    return summary


def _coordinate_values(data: Any, dimension: str) -> list[Any]:
    """Return one dimension coordinate in semantic JSON form."""
    if dimension not in data.coords:
        raise HarnessError(f"Prediction output is missing coordinate {dimension!r}")
    coordinate = data.coords[dimension]
    if coordinate.ndim != 1 or coordinate.dims != (dimension,):
        raise HarnessError(
            f"Coordinate {dimension!r} must be one-dimensional over itself, got "
            f"dims={coordinate.dims!r}"
        )
    return [_canonical_value(value) for value in coordinate.values.tolist()]


def _array_semantics(data: Any) -> dict[str, Any]:
    """Capture names, order, shapes, and values of all meaningful dimensions."""
    dimensions = list(data.dims)
    if dimensions[:2] != ["chain", "draw"]:
        raise HarnessError(
            "Posterior output must begin with canonical ('chain', 'draw') dimensions, "
            f"got {tuple(data.dims)!r}"
        )
    return {
        "dims": dimensions,
        "shape": [int(size) for size in data.shape],
        "coords": {
            dimension: _coordinate_values(data, dimension) for dimension in dimensions
        },
    }


def _metric_id(series_name: str, selector: dict[str, Any]) -> str:
    """Build a stable identifier for one scalar extracted from a draw series."""
    if not selector:
        return series_name
    rendered = ", ".join(f"{key}={value}" for key, value in selector.items())
    return f"{series_name}[{rendered}]"


def _capture_series(
    name: str,
    data: Any,
    az: Any,
    np: Any,
    *,
    expected_dims: tuple[str, ...] | None = None,
    expected_name: str | None = None,
) -> dict[str, Any]:
    """Capture scalar summaries and same-stack draw digests from a posterior array."""
    if expected_dims is not None and tuple(data.dims) != expected_dims:
        raise HarnessError(
            f"{name} dimensions changed: expected {expected_dims!r}, "
            f"got {tuple(data.dims)!r}"
        )
    if expected_name is not None and data.name != expected_name:
        raise HarnessError(
            f"{name} must be extracted from {expected_name!r}, got {data.name!r}"
        )
    value_dimensions = [
        dimension for dimension in data.dims if dimension not in {"chain", "draw"}
    ]
    ordered = data.transpose("chain", "draw", *value_dimensions)
    semantics = _array_semantics(ordered)
    values = np.asarray(ordered.values, dtype=float)
    value_shape = values.shape[2:]
    indexes = [()] if not value_shape else list(np.ndindex(value_shape))
    metrics: list[dict[str, Any]] = []

    for index in indexes:
        selector = {
            dimension: semantics["coords"][dimension][offset]
            for dimension, offset in zip(value_dimensions, index, strict=True)
        }
        draws = values[(slice(None), slice(None), *index)]
        metric_name = _metric_id(name, selector)
        metrics.append(
            {
                "id": metric_name,
                "selector": selector,
                "draw_digest": _draw_digest(draws, np),
                "summary": _summary_from_draws(draws, az, np, metric_name),
            }
        )
    return {"name": name, "semantics": semantics, "metrics": metrics}


def _coordinates_match(left: Any, right: Any, dimension: str, np: Any) -> bool:
    """Return whether two xarray coordinates have the same values and order."""
    return np.array_equal(left.coords[dimension].values, right.coords[dimension].values)


def _draw_wise_r2(observed: Any, expected_mu: Any, xr: Any, np: Any) -> Any:
    """Recompute CausalPy's Bayesian R-squared without collapsing posterior draws."""
    expected_dims = ("chain", "draw", "obs_ind", "treated_units")
    if tuple(expected_mu.dims) != expected_dims:
        raise HarnessError(
            "Expected-value predictions must have canonical dimensions "
            f"{expected_dims!r}, got {tuple(expected_mu.dims)!r}"
        )
    if expected_mu.name != "mu":
        raise HarnessError(
            "Draw-wise R-squared requires conditional expected 'mu', "
            f"got {expected_mu.name!r}"
        )
    if tuple(observed.dims) != ("obs_ind", "treated_units"):
        raise HarnessError(
            "Observed outcomes must have ('obs_ind', 'treated_units') dimensions, "
            f"got {tuple(observed.dims)!r}"
        )
    for dimension in ("obs_ind", "treated_units"):
        if not _coordinates_match(observed, expected_mu, dimension, np):
            raise HarnessError(
                f"Observed and expected-value {dimension!r} coordinates differ"
            )

    prediction = np.asarray(expected_mu.values, dtype=float)
    outcome = np.asarray(observed.values, dtype=float)
    if not np.isfinite(prediction).all() or not np.isfinite(outcome).all():
        raise HarnessError(
            "Draw-wise R-squared requires finite outcomes and expected values"
        )
    variance_of_prediction = np.var(prediction, axis=2, ddof=0)
    variance_of_residual = np.var(
        outcome[None, None, :, :] - prediction,
        axis=2,
        ddof=0,
    )
    denominator = variance_of_prediction + variance_of_residual
    if not np.isfinite(denominator).all() or np.any(denominator <= 0):
        raise HarnessError(
            "Draw-wise R-squared has a non-positive or non-finite denominator"
        )
    r2 = variance_of_prediction / denominator
    if not np.isfinite(r2).all():
        raise HarnessError("Draw-wise R-squared contains non-finite values")
    return xr.DataArray(
        r2,
        dims=("chain", "draw", "treated_units"),
        coords={
            "chain": expected_mu.coords["chain"],
            "draw": expected_mu.coords["draw"],
            "treated_units": expected_mu.coords["treated_units"],
        },
        name="draw_wise_r2",
    )


def _table_semantics(table: Any) -> dict[str, Any]:
    """Capture the non-numeric semantic contract of an effect-summary table."""
    required = {"mean", "hdi_lower", "hdi_upper"}
    columns = [str(column) for column in table.columns.tolist()]
    missing = required.difference(columns)
    if missing:
        raise HarnessError(
            "Effect-summary table is missing required columns: "
            f"{', '.join(sorted(missing))}"
        )
    return {
        "shape": [int(table.shape[0]), int(table.shape[1])],
        "columns": columns,
        "index": [_canonical_value(value) for value in table.index.tolist()],
        "index_name": _canonical_value(table.index.name),
        "hdi": {
            "probability": HDI_PROB,
            "lower_column": "hdi_lower",
            "upper_column": "hdi_upper",
        },
    }


def _series_metric(series: dict[str, Any], selector: dict[str, Any]) -> dict[str, Any]:
    """Return one captured metric selected by all non-sample coordinates."""
    for metric in series["metrics"]:
        if metric["selector"] == selector:
            return metric
    raise HarnessError(
        f"Missing metric in series {series['name']!r} for selector {selector!r}"
    )


def _verify_effect_table(
    table: Any,
    bindings: list[dict[str, Any]],
    series_by_name: dict[str, dict[str, Any]],
    np: Any,
) -> None:
    """Verify that the public table reports unrounded 94% HDI summary statistics."""
    for binding in bindings:
        row = binding["table_row"]
        if row not in table.index:
            raise HarnessError(f"Effect-summary table is missing expected row {row!r}")
        metric = _series_metric(series_by_name[binding["series"]], binding["selector"])
        expected = metric["summary"]
        for column, statistic in (
            ("mean", "mean"),
            ("hdi_lower", "hdi_lower"),
            ("hdi_upper", "hdi_upper"),
        ):
            reported = float(table.loc[row, column])
            if not math.isfinite(reported):
                raise HarnessError(
                    f"Effect-summary table {row!r}/{column!r} is non-finite"
                )
            if not np.isclose(reported, expected[statistic], rtol=1e-10, atol=1e-10):
                raise HarnessError(
                    f"Effect-summary table {row!r}/{column!r} does not match the "
                    f"unrounded {HDI_PROB:.2f} posterior summary."
                )


def _capture_difference_in_differences(
    dependencies: dict[str, Any], sample_kwargs: dict[str, Any]
) -> dict[str, Any]:
    """Capture a representative coefficient-based counterfactual analysis."""
    cp = dependencies["cp"]
    np = dependencies["np"]
    pd = dependencies["pd"]
    xr = dependencies["xr"]
    az = dependencies["az"]

    data = pd.DataFrame(_records_payload(DID_RECORDS))
    result = cp.DifferenceInDifferences(
        data,
        formula="y ~ 1 + group * post_treatment",
        time_variable_name="t",
        group_variable_name="group",
        model=cp.pymc_models.LinearRegression(sample_kwargs=dict(sample_kwargs)),
    )
    quality = _sampling_quality(result.idata, np)
    effect_summary = result.effect_summary(alpha=EFFECT_SUMMARY_ALPHA)
    fitted_mu = result._model_backend.predict(result.design["X"])
    draw_wise_r2 = _draw_wise_r2(result.design["y"], fitted_mu, xr, np)

    series = [
        _capture_series("did.causal_impact", result.causal_impact, az, np),
        _capture_series("did.draw_wise_r2", draw_wise_r2, az, np),
        _capture_series(
            "did.counterfactual_mu",
            result.y_pred_counterfactual,
            az,
            np,
            expected_dims=("chain", "draw", "obs_ind", "treated_units"),
            expected_name="mu",
        ),
    ]
    series_by_name = {item["name"]: item for item in series}
    bindings = [
        {
            "table_row": "treatment_effect",
            "series": "did.causal_impact",
            "selector": {},
        }
    ]
    _verify_effect_table(effect_summary.table, bindings, series_by_name, np)

    return {
        "name": "difference_in_differences",
        "fixture": _fixture_payload("fixed_difference_in_differences", DID_RECORDS),
        "sampling_quality": quality,
        "effect_summary": {
            "alpha": EFFECT_SUMMARY_ALPHA,
            "hdi_prob": HDI_PROB,
            "table": _table_semantics(effect_summary.table),
            "metric_bindings": bindings,
        },
        "counterfactual": {
            "series": "did.counterfactual_mu",
            "prediction_kind": "conditional_expected_mu",
        },
        "series": series,
    }


def _capture_synthetic_control(
    dependencies: dict[str, Any], sample_kwargs: dict[str, Any]
) -> dict[str, Any]:
    """Capture a representative simplex-weighted counterfactual analysis."""
    cp = dependencies["cp"]
    np = dependencies["np"]
    pd = dependencies["pd"]
    xr = dependencies["xr"]
    az = dependencies["az"]

    data = pd.DataFrame(_records_payload(SYNTHETIC_CONTROL_RECORDS)).set_index("t")
    result = cp.SyntheticControl(
        data,
        treatment_time=12,
        control_units=["a", "b", "c"],
        treated_units=["actual"],
        model=cp.pymc_models.WeightedSumFitter(sample_kwargs=dict(sample_kwargs)),
    )
    quality = _sampling_quality(result.idata, np)
    effect_summary = result.effect_summary(
        alpha=EFFECT_SUMMARY_ALPHA,
        cumulative=True,
        relative=False,
    )
    post_average_impact = result.post_impact.mean(dim="obs_ind")
    post_cumulative_impact = result.post_impact_cumulative.isel(obs_ind=-1)
    draw_wise_r2 = _draw_wise_r2(result.pre_design["treated"], result.pre_pred, xr, np)

    series = [
        _capture_series("sc.post_average_impact", post_average_impact, az, np),
        _capture_series("sc.post_cumulative_impact", post_cumulative_impact, az, np),
        _capture_series("sc.draw_wise_r2", draw_wise_r2, az, np),
        _capture_series(
            "sc.counterfactual_mu",
            result.post_pred,
            az,
            np,
            expected_dims=("chain", "draw", "obs_ind", "treated_units"),
            expected_name="mu",
        ),
    ]
    series_by_name = {item["name"]: item for item in series}
    bindings = [
        {
            "table_row": "average",
            "series": "sc.post_average_impact",
            "selector": {"treated_units": "actual"},
        },
        {
            "table_row": "cumulative",
            "series": "sc.post_cumulative_impact",
            "selector": {"treated_units": "actual"},
        },
    ]
    _verify_effect_table(effect_summary.table, bindings, series_by_name, np)

    return {
        "name": "synthetic_control",
        "fixture": _fixture_payload(
            "fixed_synthetic_control", SYNTHETIC_CONTROL_RECORDS
        ),
        "sampling_quality": quality,
        "effect_summary": {
            "alpha": EFFECT_SUMMARY_ALPHA,
            "hdi_prob": HDI_PROB,
            "table": _table_semantics(effect_summary.table),
            "metric_bindings": bindings,
        },
        "counterfactual": {
            "series": "sc.counterfactual_mu",
            "prediction_kind": "conditional_expected_mu",
        },
        "series": series,
    }


def _capture_artifact(stack: str, repo_root: Path) -> dict[str, Any]:
    """Run the fixed scenarios and return a strict, self-describing artifact."""
    expected_commit = STACK_COMMITS[stack]
    resolved_root = repo_root.expanduser().resolve()
    actual_commit = _run_git(resolved_root, "rev-parse", "HEAD")
    if actual_commit != expected_commit:
        raise HarnessError(
            f"{stack} capture requires {expected_commit}, found {actual_commit}. "
            "Do not attribute later changes to this migration baseline."
        )
    dirty_paths = _run_git(resolved_root, "status", "--porcelain")
    if dirty_paths:
        raise HarnessError(
            f"{stack} capture checkout must be clean; found uncommitted paths:\n"
            f"{dirty_paths}"
        )

    dependencies = _import_capture_dependencies(resolved_root)
    cp = dependencies["cp"]
    pm = dependencies["pm"]
    sample_kwargs, supports_nuts_sampler = _sample_kwargs(pm)
    cases = [
        _capture_difference_in_differences(dependencies, sample_kwargs),
        _capture_synthetic_control(dependencies, sample_kwargs),
    ]
    script_path = Path(__file__).resolve()

    artifact = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "suite": SUITE_NAME,
        "provenance": {
            "stack": stack,
            "expected_commit": expected_commit,
            "actual_commit": actual_commit,
            "repo_root": str(resolved_root),
            "causalpy_path": str(Path(cp.__file__).resolve()),
            "harness_path": str(script_path),
            "harness_sha256": hashlib.sha256(script_path.read_bytes()).hexdigest(),
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "dependencies": {
                "arviz": _distribution_version("arviz"),
                "causalpy": _distribution_version("causalpy"),
                "numpy": _distribution_version("numpy"),
                "pandas": _distribution_version("pandas"),
                "pymc": _distribution_version("pymc"),
                "pytensor": _distribution_version("pytensor"),
                "xarray": _distribution_version("xarray"),
            },
        },
        "protocol": _protocol(supports_nuts_sampler),
        "cases": cases,
    }
    _validate_artifact(artifact)
    return artifact


def _read_json(path: Path) -> dict[str, Any]:
    """Read one JSON artifact with a helpful error for malformed evidence."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HarnessError(f"Could not read artifact {path}: {error}") from error
    if not isinstance(payload, dict):
        raise HarnessError(f"Artifact {path} must contain a JSON object")
    _assert_finite_json(payload, str(path))
    _validate_artifact(payload)
    return payload


def _case_map(artifact: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Index artifact cases while rejecting duplicated names."""
    cases = artifact.get("cases")
    if not isinstance(cases, list):
        raise HarnessError("Artifact cases must be a list")
    result: dict[str, dict[str, Any]] = {}
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("name"), str):
            raise HarnessError("Each artifact case must have a string name")
        name = case["name"]
        if name in result:
            raise HarnessError(f"Artifact has duplicate case {name!r}")
        result[name] = case
    return result


def _series_map(case: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Index case series while rejecting duplicate names."""
    series = case.get("series")
    if not isinstance(series, list):
        raise HarnessError(f"Case {case.get('name')!r} must contain a series list")
    result: dict[str, dict[str, Any]] = {}
    for item in series:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise HarnessError(f"Case {case.get('name')!r} has malformed series")
        name = item["name"]
        if name in result:
            raise HarnessError(
                f"Case {case.get('name')!r} has duplicate series {name!r}"
            )
        result[name] = item
    return result


def _metric_map(series: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Index scalar metrics while rejecting duplicate identifiers."""
    metrics = series.get("metrics")
    if not isinstance(metrics, list) or not metrics:
        raise HarnessError(f"Series {series.get('name')!r} has no captured metrics")
    result: dict[str, dict[str, Any]] = {}
    for metric in metrics:
        if not isinstance(metric, dict) or not isinstance(metric.get("id"), str):
            raise HarnessError(f"Series {series.get('name')!r} has malformed metric")
        metric_id = metric["id"]
        if metric_id in result:
            raise HarnessError(
                f"Series {series.get('name')!r} has duplicate metric {metric_id!r}"
            )
        result[metric_id] = metric
    return result


def _validate_metric(metric: dict[str, Any]) -> None:
    """Validate one serialized scalar posterior metric."""
    digest = metric.get("draw_digest")
    if not isinstance(digest, str) or len(digest) != 64:
        raise HarnessError(f"Metric {metric.get('id')!r} has an invalid draw digest")
    summary = metric.get("summary")
    if not isinstance(summary, dict):
        raise HarnessError(f"Metric {metric.get('id')!r} has no summary")
    required = {
        "mean",
        "posterior_sd",
        "mcse_mean",
        "rhat",
        "ess_bulk",
        "ess_tail",
        "hdi_lower",
        "hdi_upper",
    }
    missing = required.difference(summary)
    if missing:
        raise HarnessError(
            f"Metric {metric.get('id')!r} is missing summary fields: "
            f"{', '.join(sorted(missing))}"
        )
    for key in required:
        value = summary[key]
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise HarnessError(f"Metric {metric.get('id')!r} has invalid {key!r}")
    if summary["hdi_lower"] > summary["hdi_upper"]:
        raise HarnessError(f"Metric {metric.get('id')!r} has inverted HDI bounds")
    if summary["rhat"] > MAX_RHAT:
        raise HarnessError(f"Metric {metric.get('id')!r} exceeds the registered R-hat")
    if summary["ess_bulk"] < MIN_ESS_BULK:
        raise HarnessError(f"Metric {metric.get('id')!r} fails the registered bulk ESS")
    if summary["ess_tail"] < MIN_ESS_TAIL:
        raise HarnessError(f"Metric {metric.get('id')!r} fails the registered tail ESS")


def _validate_case(case: dict[str, Any]) -> None:
    """Validate fixed input, explicit HDI, semantic table, and posterior series."""
    fixture = case.get("fixture")
    if not isinstance(fixture, dict) or not isinstance(fixture.get("records"), list):
        raise HarnessError(f"Case {case.get('name')!r} has no serialized fixture")
    fixture_hash = fixture.get("sha256")
    if fixture_hash != _sha256_text(_canonical_json(fixture["records"])):
        raise HarnessError(
            f"Case {case.get('name')!r} fixture hash does not match records"
        )

    quality = case.get("sampling_quality")
    if not isinstance(quality, dict) or quality.get("divergences") != 0:
        raise HarnessError(f"Case {case.get('name')!r} is not divergence-free")
    if quality.get("tree_depth_events") != 0:
        raise HarnessError(f"Case {case.get('name')!r} has tree-depth saturation")

    effect_summary = case.get("effect_summary")
    if not isinstance(effect_summary, dict):
        raise HarnessError(
            f"Case {case.get('name')!r} is missing effect summary metadata"
        )
    if not math.isclose(effect_summary.get("hdi_prob", -1), HDI_PROB):
        raise HarnessError(
            f"Case {case.get('name')!r} did not request a {HDI_PROB:.2f} HDI"
        )
    if not math.isclose(effect_summary.get("alpha", -1), EFFECT_SUMMARY_ALPHA):
        raise HarnessError(
            f"Case {case.get('name')!r} did not request alpha={EFFECT_SUMMARY_ALPHA}"
        )
    table = effect_summary.get("table")
    if not isinstance(table, dict):
        raise HarnessError(f"Case {case.get('name')!r} has no table semantics")
    hdi = table.get("hdi")
    if not isinstance(hdi, dict) or not math.isclose(
        hdi.get("probability", -1), HDI_PROB
    ):
        raise HarnessError(
            f"Case {case.get('name')!r} table lacks explicit HDI metadata"
        )
    if hdi.get("lower_column") != "hdi_lower" or hdi.get("upper_column") != "hdi_upper":
        raise HarnessError(f"Case {case.get('name')!r} table HDI columns changed")

    counterfactual = case.get("counterfactual")
    if not isinstance(counterfactual, dict):
        raise HarnessError(f"Case {case.get('name')!r} lacks counterfactual metadata")
    if counterfactual.get("prediction_kind") != "conditional_expected_mu":
        raise HarnessError(
            f"Case {case.get('name')!r} captured noisy predictions instead of mu"
        )
    series_by_name = _series_map(case)
    if counterfactual.get("series") not in series_by_name:
        raise HarnessError(
            f"Case {case.get('name')!r} counterfactual series is missing"
        )

    for series in series_by_name.values():
        semantics = series.get("semantics")
        if not isinstance(semantics, dict):
            raise HarnessError(f"Series {series['name']!r} has no semantic schema")
        if semantics.get("dims", [])[:2] != ["chain", "draw"]:
            raise HarnessError(
                f"Series {series['name']!r} lacks canonical sample dimensions"
            )
        if not isinstance(semantics.get("coords"), dict):
            raise HarnessError(f"Series {series['name']!r} lacks coordinates")
        for metric in _metric_map(series).values():
            _validate_metric(metric)


def _validate_artifact(artifact: dict[str, Any]) -> None:
    """Validate schema, exact pins, protocol, and every captured case."""
    if artifact.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise HarnessError("Unsupported artifact schema version")
    if artifact.get("suite") != SUITE_NAME:
        raise HarnessError("Artifact comes from a different comparison suite")
    provenance = artifact.get("provenance")
    if not isinstance(provenance, dict):
        raise HarnessError("Artifact is missing provenance")
    stack = provenance.get("stack")
    if stack not in STACK_COMMITS:
        raise HarnessError(f"Artifact has unknown stack {stack!r}")
    expected_commit = STACK_COMMITS[stack]
    if provenance.get("expected_commit") != expected_commit:
        raise HarnessError(f"Artifact has an unexpected registered commit for {stack}")
    if provenance.get("actual_commit") != expected_commit:
        raise HarnessError(
            f"Artifact actual commit differs from the pinned {stack} revision"
        )
    for key in ("repo_root", "causalpy_path", "harness_path", "harness_sha256"):
        if not isinstance(provenance.get(key), str) or not provenance[key]:
            raise HarnessError(f"Artifact is missing provenance field {key!r}")
    if not _is_within(Path(provenance["causalpy_path"]), Path(provenance["repo_root"])):
        raise HarnessError(
            "Artifact CausalPy import path is outside its recorded checkout"
        )

    protocol = artifact.get("protocol")
    if not isinstance(protocol, dict):
        raise HarnessError("Artifact is missing the preregistered protocol")
    if protocol.get("scenario_version") != SCENARIO_VERSION:
        raise HarnessError("Artifact scenario version differs")
    if not math.isclose(protocol.get("hdi_prob", -1), HDI_PROB):
        raise HarnessError("Artifact HDI probability differs from the registered 0.94")
    sampling = protocol.get("sampling")
    if not isinstance(sampling, dict):
        raise HarnessError("Artifact is missing sampling protocol")
    expected_sampling = {
        "sampler": "pymc",
        "chains": CHAINS,
        "cores": CORES,
        "draws": DRAWS,
        "tune": TUNE,
        "master_seed": MASTER_SEED,
        "target_accept": TARGET_ACCEPT,
        "max_treedepth": MAX_TREEDEPTH,
    }
    for key, value in expected_sampling.items():
        if sampling.get(key) != value:
            raise HarnessError(f"Artifact sampling protocol differs at {key!r}")

    cases = _case_map(artifact)
    expected_cases = {"difference_in_differences", "synthetic_control"}
    if set(cases) != expected_cases:
        raise HarnessError(
            f"Artifact cases must be {sorted(expected_cases)!r}, got {sorted(cases)!r}"
        )
    for case in cases.values():
        _validate_case(case)


def _comparable_protocol(protocol: dict[str, Any]) -> dict[str, Any]:
    """Drop runtime implementation details while retaining every statistical setting."""
    comparable = json.loads(_canonical_json(protocol))
    comparable["sampling"].pop("nuts_sampler_argument_used", None)
    return comparable


def _repeatability(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    """Verify exact raw-draw reproducibility within one stack only."""
    first_provenance = first["provenance"]
    second_provenance = second["provenance"]
    if first_provenance["stack"] != second_provenance["stack"]:
        raise HarnessError("Repeatability artifacts must come from the same stack")
    for key in ("actual_commit", "harness_sha256"):
        if first_provenance[key] != second_provenance[key]:
            raise HarnessError(f"Repeatability artifacts differ at provenance {key!r}")
    if _comparable_protocol(first["protocol"]) != _comparable_protocol(
        second["protocol"]
    ):
        raise HarnessError("Repeatability artifacts use different protocols")

    first_cases = _case_map(first)
    second_cases = _case_map(second)
    if set(first_cases) != set(second_cases):
        raise HarnessError("Repeatability artifacts have different scenario sets")

    mismatches: list[str] = []
    metric_count = 0
    for case_name in sorted(first_cases):
        first_case = first_cases[case_name]
        second_case = second_cases[case_name]
        if first_case["fixture"] != second_case["fixture"]:
            raise HarnessError(f"Repeatability fixture changed for {case_name!r}")
        first_series = _series_map(first_case)
        second_series = _series_map(second_case)
        if set(first_series) != set(second_series):
            raise HarnessError(f"Repeatability series changed for {case_name!r}")
        for series_name in sorted(first_series):
            first_item = first_series[series_name]
            second_item = second_series[series_name]
            if first_item["semantics"] != second_item["semantics"]:
                raise HarnessError(
                    f"Repeatability coordinate semantics changed for {series_name!r}"
                )
            first_metrics = _metric_map(first_item)
            second_metrics = _metric_map(second_item)
            if set(first_metrics) != set(second_metrics):
                raise HarnessError(f"Repeatability metrics changed for {series_name!r}")
            for metric_id in sorted(first_metrics):
                metric_count += 1
                if (
                    first_metrics[metric_id]["draw_digest"]
                    != second_metrics[metric_id]["draw_digest"]
                ):
                    mismatches.append(metric_id)
    return {
        "stack": first_provenance["stack"],
        "metric_count": metric_count,
        "passed": not mismatches,
        "mismatched_metric_ids": mismatches,
    }


def _semantic_check(
    reference_case: dict[str, Any], candidate_case: dict[str, Any]
) -> dict[str, Any]:
    """Evaluate the hard schema and coordinate equality gate for one scenario."""
    failures: list[str] = []
    if reference_case["fixture"] != candidate_case["fixture"]:
        failures.append("serialized fixed fixture differs")
    if reference_case["effect_summary"] != candidate_case["effect_summary"]:
        failures.append("effect-summary table schema or 0.94 HDI metadata differs")
    if reference_case["counterfactual"] != candidate_case["counterfactual"]:
        failures.append("counterfactual prediction contract differs")

    reference_series = _series_map(reference_case)
    candidate_series = _series_map(candidate_case)
    if set(reference_series) != set(candidate_series):
        failures.append("captured series names differ")
    for series_name in sorted(set(reference_series).intersection(candidate_series)):
        reference_item = reference_series[series_name]
        candidate_item = candidate_series[series_name]
        if reference_item["semantics"] != candidate_item["semantics"]:
            failures.append(f"coordinate semantics differ for {series_name}")
        reference_metrics = _metric_map(reference_item)
        candidate_metrics = _metric_map(candidate_item)
        if set(reference_metrics) != set(candidate_metrics):
            failures.append(f"metric selectors differ for {series_name}")
    return {"passed": not failures, "failures": failures}


def _compare_scalar_summaries(
    reference: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    """Apply only the preregistered numerical migration gates to one scalar."""
    reference_mean = float(reference["mean"])
    candidate_mean = float(candidate["mean"])
    difference = abs(candidate_mean - reference_mean)
    combined_mcse = math.hypot(
        float(reference["mcse_mean"]), float(candidate["mcse_mean"])
    )
    absolute_tolerance = max(
        MCSE_MULTIPLIER * combined_mcse,
        ABSOLUTE_TOLERANCE_FLOOR + RELATIVE_TOLERANCE_FLOOR * abs(reference_mean),
    )
    absolute_passed = difference <= absolute_tolerance

    pooled_posterior_sd = math.sqrt(
        (float(reference["posterior_sd"]) ** 2 + float(candidate["posterior_sd"]) ** 2)
        / 2.0
    )
    if pooled_posterior_sd <= DEGENERATE_POSTERIOR_SD:
        standardized_drift: float | None = None
        standardized_passed = absolute_passed
        standardized_rule = "degenerate posterior: use absolute mean gate"
    else:
        standardized_drift = difference / pooled_posterior_sd
        standardized_passed = standardized_drift <= MAX_STANDARDIZED_DRIFT
        standardized_rule = "abs(delta) / pooled posterior SD <= 0.1"

    reference_contains_candidate_mean = (
        float(reference["hdi_lower"]) <= candidate_mean <= float(reference["hdi_upper"])
    )
    candidate_contains_reference_mean = (
        float(candidate["hdi_lower"]) <= reference_mean <= float(candidate["hdi_upper"])
    )
    return {
        "reference_mean": reference_mean,
        "candidate_mean": candidate_mean,
        "absolute_mean_delta": difference,
        "combined_mcse": combined_mcse,
        "absolute_tolerance": absolute_tolerance,
        "absolute_mean_gate_passed": absolute_passed,
        "pooled_posterior_sd": pooled_posterior_sd,
        "standardized_mean_drift": standardized_drift,
        "standardized_mean_drift_rule": standardized_rule,
        "standardized_mean_drift_gate_passed": standardized_passed,
        "reference_hdi": [
            float(reference["hdi_lower"]),
            float(reference["hdi_upper"]),
        ],
        "candidate_hdi": [
            float(candidate["hdi_lower"]),
            float(candidate["hdi_upper"]),
        ],
        "reference_hdi_contains_candidate_mean": reference_contains_candidate_mean,
        "candidate_hdi_contains_reference_mean": candidate_contains_reference_mean,
        "mutual_hdi_containment_diagnostic": (
            reference_contains_candidate_mean and candidate_contains_reference_mean
        ),
    }


def _compare_case(
    reference_case: dict[str, Any], candidate_case: dict[str, Any]
) -> dict[str, Any]:
    """Compare independent posterior summaries for one fixed scenario."""
    semantic = _semantic_check(reference_case, candidate_case)
    reference_series = _series_map(reference_case)
    candidate_series = _series_map(candidate_case)
    metric_results: list[dict[str, Any]] = []

    for series_name in sorted(set(reference_series).intersection(candidate_series)):
        reference_metrics = _metric_map(reference_series[series_name])
        candidate_metrics = _metric_map(candidate_series[series_name])
        for metric_id in sorted(set(reference_metrics).intersection(candidate_metrics)):
            comparison = _compare_scalar_summaries(
                reference_metrics[metric_id]["summary"],
                candidate_metrics[metric_id]["summary"],
            )
            comparison["id"] = metric_id
            comparison["series"] = series_name
            comparison["passed"] = (
                comparison["absolute_mean_gate_passed"]
                and comparison["standardized_mean_drift_gate_passed"]
            )
            metric_results.append(comparison)

    return {
        "name": reference_case["name"],
        "semantic_equality": semantic,
        "metrics": metric_results,
        "passed": semantic["passed"] and all(item["passed"] for item in metric_results),
    }


def _capture_evidence(artifact: dict[str, Any]) -> dict[str, Any]:
    """Return the provenance and observed validity diagnostics for one capture."""
    cases: list[dict[str, Any]] = []
    for case_name, case in sorted(_case_map(artifact).items()):
        metrics = [
            metric
            for series in _series_map(case).values()
            for metric in _metric_map(series).values()
        ]
        summaries = [metric["summary"] for metric in metrics]
        cases.append(
            {
                "name": case_name,
                "fixture_sha256": case["fixture"]["sha256"],
                "sampling_quality": case["sampling_quality"],
                "metric_count": len(metrics),
                "max_rhat": max(summary["rhat"] for summary in summaries),
                "min_ess_bulk": min(summary["ess_bulk"] for summary in summaries),
                "min_ess_tail": min(summary["ess_tail"] for summary in summaries),
            }
        )
    return {"provenance": artifact["provenance"], "cases": cases}


def compare_artifacts(
    reference_first: dict[str, Any],
    reference_second: dict[str, Any],
    candidate_first: dict[str, Any],
    candidate_second: dict[str, Any],
) -> dict[str, Any]:
    """Verify repeatability, then compare pinned PyMC 5 and PyMC 6 summaries."""
    for artifact in (
        reference_first,
        reference_second,
        candidate_first,
        candidate_second,
    ):
        _validate_artifact(artifact)
    if reference_first["provenance"]["stack"] != "pymc5":
        raise HarnessError(
            "Reference artifacts must be captured from the pinned PyMC 5 stack"
        )
    if candidate_first["provenance"]["stack"] != "pymc6":
        raise HarnessError(
            "Candidate artifacts must be captured from the pinned PyMC 6 stack"
        )

    reference_repeatability = _repeatability(reference_first, reference_second)
    candidate_repeatability = _repeatability(candidate_first, candidate_second)
    reference_cases = _case_map(reference_first)
    candidate_cases = _case_map(candidate_first)
    if set(reference_cases) != set(candidate_cases):
        raise HarnessError(
            "Reference and candidate artifacts have different scenario sets"
        )

    protocol_equal = _comparable_protocol(
        reference_first["protocol"]
    ) == _comparable_protocol(candidate_first["protocol"])
    harness_equal = (
        reference_first["provenance"]["harness_sha256"]
        == candidate_first["provenance"]["harness_sha256"]
    )
    cases = [
        _compare_case(reference_cases[name], candidate_cases[name])
        for name in sorted(reference_cases)
    ]
    metric_results = [metric for case in cases for metric in case["metrics"]]
    semantic_passed = all(case["semantic_equality"]["passed"] for case in cases)
    absolute_passed = all(
        metric["absolute_mean_gate_passed"] for metric in metric_results
    )
    standardized_passed = all(
        metric["standardized_mean_drift_gate_passed"] for metric in metric_results
    )
    repeatability_passed = (
        reference_repeatability["passed"] and candidate_repeatability["passed"]
    )
    passed = (
        protocol_equal
        and harness_equal
        and repeatability_passed
        and semantic_passed
        and absolute_passed
        and standardized_passed
    )

    return {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "suite": SUITE_NAME,
        "reference": {
            "stack": "pymc5",
            "commit": reference_first["provenance"]["actual_commit"],
        },
        "candidate": {
            "stack": "pymc6",
            "commit": candidate_first["provenance"]["actual_commit"],
        },
        "protocol_equal": protocol_equal,
        "harness_equal": harness_equal,
        "artifact_inputs": [],
        "capture_evidence": {
            "reference_first": _capture_evidence(reference_first),
            "reference_second": _capture_evidence(reference_second),
            "candidate_first": _capture_evidence(candidate_first),
            "candidate_second": _capture_evidence(candidate_second),
        },
        "within_stack_repeatability": {
            "reference": reference_repeatability,
            "candidate": candidate_repeatability,
        },
        "hard_gates": {
            "protocol_equal": protocol_equal,
            "same_harness_version": harness_equal,
            "within_stack_repeatability": repeatability_passed,
            "semantic_table_and_coordinate_equality": semantic_passed,
            "absolute_mean_delta": absolute_passed,
            "standardized_mean_drift": standardized_passed,
        },
        "cases": cases,
        "passed": passed,
    }


def _format_number(value: float | None) -> str:
    """Format optional report values without introducing non-finite JSON values."""
    return "n/a" if value is None else f"{value:.8g}"


def render_report(comparison: dict[str, Any]) -> str:
    """Render an issue-attachment-ready Markdown report from a comparison result."""
    status = "PASS" if comparison["passed"] else "FAIL"
    reference = comparison["reference"]
    candidate = comparison["candidate"]
    reference_repeat = comparison["within_stack_repeatability"]["reference"]
    candidate_repeat = comparison["within_stack_repeatability"]["candidate"]
    artifact_inputs = {item["role"]: item for item in comparison["artifact_inputs"]}
    capture_evidence = comparison["capture_evidence"]
    capture_labels = {
        "reference_first": "PyMC 5 / capture 1",
        "reference_second": "PyMC 5 / capture 2",
        "candidate_first": "PyMC 6 / capture 1",
        "candidate_second": "PyMC 6 / capture 2",
    }
    lines = [
        "# PyMC 5 → PyMC 6 migration baseline comparison",
        "",
        f"**Result:** {status}",
        "",
        "## Scope and attribution",
        "",
        f"- Reference: PyMC 5 CausalPy commit `{reference['commit']}`.",
        f"- Candidate: PyMC 6 migration commit `{candidate['commit']}`.",
        "- This report measures only the pinned migration delta. It must not be used to attribute behavior introduced after the PyMC 6 migration commit to the migration itself.",
        "- The artifacts contain independent posterior summaries; no raw PyMC 5 draw is compared to a PyMC 6 draw.",
        "",
        "## Pre-registered protocol",
        "",
        f"- Explicit HDI probability: `{HDI_PROB}` (`alpha={EFFECT_SUMMARY_ALPHA}`).",
        f"- Sampling: PyMC NUTS, `{CHAINS}` serialized chains (`cores={CORES}`), `{TUNE}` tune iterations, `{DRAWS}` retained draws, master seed `{MASTER_SEED}`, target acceptance `{TARGET_ACCEPT}`.",
        "- The harness captures a coefficient-based Difference-in-Differences scenario and a simplex-weighted Synthetic Control scenario from fixed serialized input records embedded in each artifact.",
        "- Every captured scalar must be finite, divergence-free, non-tree-depth-saturated when that statistic is exposed, have rank R-hat at most 1.01, and have bulk and tail ESS at least 400 before it is evidence.",
        "",
        "## Within-stack deterministic repeatability",
        "",
        "| Stack | Metrics checked | Exact raw-draw digest result |",
        "|---|---:|---|",
        f"| PyMC 5 | {reference_repeat['metric_count']} | {'pass' if reference_repeat['passed'] else 'fail'} |",
        f"| PyMC 6 | {candidate_repeat['metric_count']} | {'pass' if candidate_repeat['passed'] else 'fail'} |",
        "",
        "Exact raw-draw digests are used only within a stack to establish repeatability. They are not used for any cross-stack decision.",
        "",
        "## Hard migration gates",
        "",
        "1. `abs(candidate_mean - reference_mean) <= max(4 * hypot(reference_mcse, candidate_mcse), 1e-6 + 1e-4 * abs(reference_mean))`.",
        "2. `abs(candidate_mean - reference_mean) / pooled_posterior_sd <= 0.1`, with a degenerate posterior using the absolute gate because a standardized denominator is undefined.",
        "3. Effect-summary table schemas, requested 0.94 HDI labels, metric selectors, prediction dimensions, and coordinate values must be semantically equal.",
        "",
        "Mutual mean-in-94%-HDI containment is diagnostic only. It is reported below but never changes the pass/fail decision.",
        "",
        "## Gate summary",
        "",
        "| Gate | Result |",
        "|---|---|",
    ]
    for gate, passed in comparison["hard_gates"].items():
        lines.append(f"| {gate.replace('_', ' ')} | {'pass' if passed else 'fail'} |")
    lines.extend(
        [
            "",
            "## Capture provenance",
            "",
            "| Capture | Artifact path | Artifact SHA-256 | CausalPy import | Harness SHA-256 | Package versions |",
            "|---|---|---|---|---|---|",
        ]
    )
    for role, label in capture_labels.items():
        evidence = capture_evidence[role]
        provenance = evidence["provenance"]
        dependencies = provenance["dependencies"]
        dependency_versions = ", ".join(
            f"{name} {version}" for name, version in sorted(dependencies.items())
        )
        artifact_input = artifact_inputs.get(role, {})
        lines.append(
            "| "
            f"{label} | `{artifact_input.get('path', 'not supplied')}` | "
            f"`{artifact_input.get('sha256', 'not supplied')}` | "
            f"`{provenance['causalpy_path']}` | "
            f"`{provenance['harness_sha256']}` | "
            f"{dependency_versions} |"
        )
    lines.extend(["", "Capture platforms:"])
    for role, label in capture_labels.items():
        provenance = capture_evidence[role]["provenance"]
        lines.append(
            f"- {label}: Python {provenance['python']} "
            f"({provenance['python_implementation']}) on "
            f"{provenance['platform']} ({provenance['machine']})."
        )
    lines.extend(
        [
            "",
            "## Capture validity evidence",
            "",
            "| Capture | Scenario | Fixture SHA-256 | Divergences | Tree-depth source / events / max | Max R-hat | Min bulk ESS | Min tail ESS |",
            "|---|---|---|---:|---|---:|---:|---:|",
        ]
    )
    for role, label in capture_labels.items():
        for case in capture_evidence[role]["cases"]:
            quality = case["sampling_quality"]
            tree_depth = (
                f"{quality['tree_depth_source']} / "
                f"{quality['tree_depth_events']} / "
                f"{_format_number(quality['max_observed_tree_depth'])}"
            )
            lines.append(
                "| "
                f"{label} | {case['name']} | `{case['fixture_sha256']}` | "
                f"{quality['divergences']} | "
                f"{tree_depth} | {_format_number(case['max_rhat'])} | "
                f"{_format_number(case['min_ess_bulk'])} | "
                f"{_format_number(case['min_ess_tail'])} |"
            )
    lines.extend(
        [
            "",
            "## Metric comparison",
            "",
            "| Scenario | Metric | Reference mean | Candidate mean | Absolute delta | Tolerance | Standardized drift | Mutual HDI diagnostic | Hard gates |",
            "|---|---|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for case in comparison["cases"]:
        for metric in case["metrics"]:
            diagnostic = "yes" if metric["mutual_hdi_containment_diagnostic"] else "no"
            lines.append(
                "| "
                f"{case['name']} | `{metric['id']}` | "
                f"{_format_number(metric['reference_mean'])} | "
                f"{_format_number(metric['candidate_mean'])} | "
                f"{_format_number(metric['absolute_mean_delta'])} | "
                f"{_format_number(metric['absolute_tolerance'])} | "
                f"{_format_number(metric['standardized_mean_drift'])} | "
                f"{diagnostic} | {'pass' if metric['passed'] else 'fail'} |"
            )
    lines.extend(
        [
            "",
            "## Semantic contracts",
            "",
            "| Scenario | Effect table and coordinate equality | Details |",
            "|---|---|---|",
        ]
    )
    for case in comparison["cases"]:
        semantic = case["semantic_equality"]
        details = "; ".join(semantic["failures"]) if semantic["failures"] else "matched"
        lines.append(
            f"| {case['name']} | {'pass' if semantic['passed'] else 'fail'} | {details} |"
        )
    lines.extend(
        [
            "",
            "## Reuse for #157 correctness work",
            "",
            "The fixed fixtures, unrounded posterior summaries, explicit 0.94 HDI extraction, draw-wise R-squared computation, and schema checks are reusable test inputs. They are migration evidence only: #157 correctness tests must assert simulated-data ground truth with separately registered posterior-SD-unit bounds rather than treating the PyMC 5 posterior as truth.",
            "",
            "## Attachment checklist",
            "",
            "- Attach this rendered report and its four JSON artifacts to #1048 after coordinator execution.",
            "- Preserve the artifact SHA-256 metadata and command log with the attachment.",
            "- If a later commit is under investigation, create a separately named feature comparison; this harness intentionally rejects a source revision other than the two pinned commits above.",
        ]
    )
    return "\n".join(lines) + "\n"


def _load_distinct_artifacts(paths: list[Path]) -> list[dict[str, Any]]:
    """Reject accidentally reusing one file as multiple independent captures."""
    resolved_paths = [path.expanduser().resolve() for path in paths]
    if len(set(resolved_paths)) != len(resolved_paths):
        raise HarnessError(
            "All four artifact paths must be distinct independent captures"
        )
    return [_read_json(path) for path in resolved_paths]


def _artifact_input_metadata(paths: list[Path]) -> list[dict[str, Any]]:
    """Record content-addressed source artifacts for an attachment-ready report."""
    roles = (
        "reference_first",
        "reference_second",
        "candidate_first",
        "candidate_second",
    )
    return [
        {
            "role": role,
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for role, path in zip(roles, paths, strict=True)
    ]


def _capture_command(args: argparse.Namespace) -> int:
    """Execute a pinned capture and write one external JSON artifact."""
    output = _require_external_output(
        args.output,
        _script_repository_root(),
        args.repo_root.expanduser().resolve(),
    )
    artifact = _capture_artifact(args.stack, args.repo_root)
    _atomic_write_json(output, artifact)
    print(output)
    return 0


def _compare_command(args: argparse.Namespace) -> int:
    """Compare four external artifacts and always write a decision report."""
    paths = [
        args.reference_first,
        args.reference_second,
        args.candidate_first,
        args.candidate_second,
    ]
    resolved_paths = [path.expanduser().resolve() for path in paths]
    artifacts = _load_distinct_artifacts(resolved_paths)
    tracked_roots = [_script_repository_root()]
    for artifact in artifacts:
        tracked_roots.append(Path(artifact["provenance"]["repo_root"]))
    output = _require_external_output(args.output, *tracked_roots)
    report_path = _require_external_output(args.report, *tracked_roots)
    if output == report_path:
        raise HarnessError(
            "Comparison JSON output and Markdown report must use different paths"
        )
    if output in resolved_paths or report_path in resolved_paths:
        raise HarnessError(
            "Comparison outputs must not overwrite an input evidence artifact"
        )

    comparison = compare_artifacts(*artifacts)
    comparison["artifact_inputs"] = _artifact_input_metadata(resolved_paths)
    _atomic_write_json(output, comparison)
    _atomic_write_text(report_path, render_report(comparison))
    print(output)
    print(report_path)
    return 0 if comparison["passed"] else 1


def _build_parser() -> argparse.ArgumentParser:
    """Build the two-command CLI without exposing unregistered run settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture = subparsers.add_parser(
        "capture",
        help="Sample one pinned stack and write a strict baseline JSON artifact.",
    )
    capture.add_argument("--stack", choices=sorted(STACK_COMMITS), required=True)
    capture.add_argument(
        "--repo-root",
        type=Path,
        required=True,
        help="Pinned CausalPy checkout that must supply the imported package.",
    )
    capture.add_argument(
        "--output",
        type=Path,
        required=True,
        help="External JSON destination; tracked checkouts are rejected.",
    )
    capture.set_defaults(handler=_capture_command)

    compare = subparsers.add_parser(
        "compare",
        help="Verify within-stack repeats and compare independent stack summaries.",
    )
    compare.add_argument("--reference-first", type=Path, required=True)
    compare.add_argument("--reference-second", type=Path, required=True)
    compare.add_argument("--candidate-first", type=Path, required=True)
    compare.add_argument("--candidate-second", type=Path, required=True)
    compare.add_argument(
        "--output",
        type=Path,
        required=True,
        help="External comparison JSON destination; tracked checkouts are rejected.",
    )
    compare.add_argument(
        "--report",
        type=Path,
        required=True,
        help="External Markdown attachment destination; tracked checkouts are rejected.",
    )
    compare.set_defaults(handler=_compare_command)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments and return an evidence-oriented process status."""
    args = _build_parser().parse_args(argv)
    try:
        return args.handler(args)
    except HarnessError as error:
        print(f"migration baseline harness: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
