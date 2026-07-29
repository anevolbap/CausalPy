#   Copyright 2026 - 2026 The PyMC Labs Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
"""Tests for the permanent PyMC migration baseline comparison harness."""

from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "migration_baseline" / "harness.py"


def _load_harness_module():
    spec = importlib.util.spec_from_file_location(
        "migration_baseline_harness", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _summary(
    *,
    mean: float,
    posterior_sd: float,
    mcse_mean: float,
    hdi_lower: float,
    hdi_upper: float,
) -> dict[str, float]:
    return {
        "mean": mean,
        "posterior_sd": posterior_sd,
        "mcse_mean": mcse_mean,
        "hdi_lower": hdi_lower,
        "hdi_upper": hdi_upper,
    }


def test_registered_mean_gates_accept_small_independent_drift() -> None:
    """The comparator combines independent MCSE values and pooled posterior SDs."""
    harness = _load_harness_module()
    reference = _summary(
        mean=5.0,
        posterior_sd=2.0,
        mcse_mean=0.1,
        hdi_lower=1.0,
        hdi_upper=9.0,
    )
    candidate = _summary(
        mean=5.1,
        posterior_sd=2.0,
        mcse_mean=0.1,
        hdi_lower=1.1,
        hdi_upper=9.1,
    )

    result = harness._compare_scalar_summaries(reference, candidate)

    assert result["absolute_mean_gate_passed"]
    assert result["standardized_mean_drift_gate_passed"]
    assert math.isclose(result["combined_mcse"], math.hypot(0.1, 0.1))
    assert math.isclose(result["standardized_mean_drift"], 0.05)


def test_standardized_gate_rejects_drift_hidden_by_large_mcse() -> None:
    """Large MCSE must not make a substantively large posterior shift acceptable."""
    harness = _load_harness_module()
    reference = _summary(
        mean=5.0,
        posterior_sd=2.0,
        mcse_mean=0.1,
        hdi_lower=1.0,
        hdi_upper=9.0,
    )
    candidate = _summary(
        mean=5.5,
        posterior_sd=2.0,
        mcse_mean=0.1,
        hdi_lower=1.5,
        hdi_upper=9.5,
    )

    result = harness._compare_scalar_summaries(reference, candidate)

    assert result["absolute_mean_gate_passed"]
    assert not result["standardized_mean_drift_gate_passed"]
    assert math.isclose(result["standardized_mean_drift"], 0.25)


def test_degenerate_posterior_uses_absolute_gate_without_nan() -> None:
    """A zero pooled SD uses the registered absolute rule instead of a NaN ratio."""
    harness = _load_harness_module()
    reference = _summary(
        mean=0.0,
        posterior_sd=0.0,
        mcse_mean=0.0,
        hdi_lower=0.0,
        hdi_upper=0.0,
    )
    candidate = _summary(
        mean=2e-6,
        posterior_sd=0.0,
        mcse_mean=0.0,
        hdi_lower=2e-6,
        hdi_upper=2e-6,
    )

    result = harness._compare_scalar_summaries(reference, candidate)

    assert result["standardized_mean_drift"] is None
    assert not result["absolute_mean_gate_passed"]
    assert not result["standardized_mean_drift_gate_passed"]


def test_hdi_containment_remains_diagnostic_only() -> None:
    """The returned containment flag is separate from the two numerical hard gates."""
    harness = _load_harness_module()
    reference = _summary(
        mean=0.0,
        posterior_sd=10.0,
        mcse_mean=1.0,
        hdi_lower=-0.01,
        hdi_upper=0.01,
    )
    candidate = _summary(
        mean=0.1,
        posterior_sd=10.0,
        mcse_mean=1.0,
        hdi_lower=0.09,
        hdi_upper=0.11,
    )

    result = harness._compare_scalar_summaries(reference, candidate)

    assert result["absolute_mean_gate_passed"]
    assert result["standardized_mean_drift_gate_passed"]
    assert not result["mutual_hdi_containment_diagnostic"]


def test_pymc5_singleton_effect_is_canonicalized_before_table_binding() -> None:
    """A PyMC 5-shaped singleton effect must yield the scalar table selector."""
    harness = _load_harness_module()
    pymc5_effect = xr.DataArray(
        np.arange(harness.CHAINS * harness.DRAWS, dtype=float).reshape(
            harness.CHAINS, harness.DRAWS, 1
        ),
        dims=("chain", "draw", "treated_units"),
        coords={
            "chain": np.arange(harness.CHAINS),
            "draw": np.arange(harness.DRAWS),
            "treated_units": ["unit_0"],
        },
        name="beta",
    )

    class StableArviZ:
        @staticmethod
        def hdi(values, *, prob):
            assert prob == harness.HDI_PROB
            return np.array([np.min(values), np.max(values)])

        @staticmethod
        def mcse(_values, *, method):
            assert method == "mean"
            return 0.01

        @staticmethod
        def rhat(_values, *, method):
            assert method == "rank"
            return 1.0

        @staticmethod
        def ess(_values, *, method, prob=None):
            if method == "tail":
                assert prob == harness.TAIL_ESS_PROB
            else:
                assert method == "bulk"
                assert prob is None
            return 800.0

    captured = harness._capture_series(
        "did.causal_impact",
        harness._canonical_scalar_effect(pymc5_effect),
        StableArviZ,
        np,
    )

    assert captured["semantics"]["dims"] == ["chain", "draw"]
    assert captured["metrics"][0]["selector"] == {}
    assert harness._series_metric(captured, {})["id"] == "did.causal_impact"

    multi_unit_effect = xr.concat(
        [pymc5_effect, pymc5_effect], dim="treated_units"
    ).assign_coords(treated_units=["unit_0", "unit_1"])
    with pytest.raises(harness.HarnessError, match="unexpected non-sample"):
        harness._canonical_scalar_effect(multi_unit_effect)


def test_tail_ess_probability_is_explicit_across_arviz_api_variants() -> None:
    """Tail ESS must not inherit a version-dependent default probability."""
    harness = _load_harness_module()
    assert harness._protocol(False)["evidence_validity"]["tail_ess_prob"] == [
        0.05,
        0.95,
    ]
    calls = []

    class ArviZOne:
        @staticmethod
        def ess(_draws, *, method, prob):
            calls.append(("arviz-one", method, prob))
            return 801.0

    class LegacyArviZ:
        @staticmethod
        def ess(_draws, *, method, prob=None):
            calls.append(("legacy", method, prob))
            return 802.0

    assert harness._tail_ess(object(), ArviZOne) == 801.0
    assert harness._tail_ess(object(), LegacyArviZ) == 802.0
    assert calls == [
        ("arviz-one", "tail", (0.05, 0.95)),
        ("legacy", "tail", (0.05, 0.95)),
    ]


def test_capture_rejects_dirty_pinned_checkout_before_import(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The pinned revision alone cannot make uncommitted source attributable."""
    harness = _load_harness_module()

    def fake_run_git(_repo_root: Path, *arguments: str) -> str:
        if arguments == ("rev-parse", "HEAD"):
            return harness.STACK_COMMITS["pymc6"]
        assert arguments == ("status", "--porcelain")
        return " M causalpy/experiments/diff_in_diff.py"

    def unexpected_import(_repo_root: Path) -> None:
        pytest.fail("dirty source must be rejected before importing CausalPy")

    monkeypatch.setattr(harness, "_run_git", fake_run_git)
    monkeypatch.setattr(harness, "_import_capture_dependencies", unexpected_import)

    with pytest.raises(harness.HarnessError, match="checkout must be clean"):
        harness._capture_artifact("pymc6", tmp_path)


def test_did_capture_fixture_passes_constructor_validation_without_mcmc() -> None:
    """The pinned DiD fixture reaches real constructor validation with OLS only."""
    harness = _load_harness_module()
    records_json = json.dumps(harness._records_payload(harness.DID_RECORDS))
    script = "\n".join(
        [
            "import json",
            "import pandas as pd",
            "import causalpy as cp",
            "from sklearn.linear_model import LinearRegression",
            f"data = pd.DataFrame(json.loads({records_json!r}))",
            'assert set(data.loc[data["group"] == 0, "unit"]) == {"control"}',
            'assert set(data.loc[data["group"] == 1, "unit"]) == {"treated"}',
            "result = cp.DifferenceInDifferences(",
            "    data,",
            '    formula="y ~ 1 + group * post_treatment",',
            '    time_variable_name="t",',
            '    group_variable_name="group",',
            "    model=LinearRegression(),",
            ")",
            "assert result.causal_impact is not None",
        ]
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        cwd=REPO_ROOT,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def _report_capture_evidence(
    *,
    stack: str,
    fixture_sha256: str,
) -> dict[str, object]:
    """Build one concise capture-evidence record for report rendering."""
    return {
        "provenance": {
            "causalpy_path": f"/checkouts/{stack}/causalpy/__init__.py",
            "harness_sha256": "harness-sha256",
            "python": "3.12.0",
            "python_implementation": "CPython",
            "platform": "Darwin-25.0.0-arm64",
            "machine": "arm64",
            "dependencies": {
                "arviz": "0.22.0",
                "pymc": "6.2.0",
                "pytensor": "2.37.0",
            },
        },
        "cases": [
            {
                "name": "difference_in_differences",
                "fixture_sha256": fixture_sha256,
                "sampling_quality": {
                    "divergences": 0,
                    "tree_depth_source": "tree_depth",
                    "tree_depth_events": 0,
                    "max_observed_tree_depth": 7,
                },
                "metric_count": 3,
                "max_rhat": 1.001,
                "min_ess_bulk": 800.0,
                "min_ess_tail": 750.0,
            }
        ],
    }


def test_report_records_artifact_provenance_and_observed_validity() -> None:
    """Generated attachment reports must retain the evidence behind their gates."""
    harness = _load_harness_module()
    evidence = {
        "reference_first": _report_capture_evidence(
            stack="pymc5", fixture_sha256="fixture-reference-first"
        ),
        "reference_second": _report_capture_evidence(
            stack="pymc5", fixture_sha256="fixture-reference-second"
        ),
        "candidate_first": _report_capture_evidence(
            stack="pymc6", fixture_sha256="fixture-candidate-first"
        ),
        "candidate_second": _report_capture_evidence(
            stack="pymc6", fixture_sha256="fixture-candidate-second"
        ),
    }
    comparison = {
        "passed": True,
        "reference": {"stack": "pymc5", "commit": "reference-commit"},
        "candidate": {"stack": "pymc6", "commit": "candidate-commit"},
        "within_stack_repeatability": {
            "reference": {"metric_count": 3, "passed": True},
            "candidate": {"metric_count": 3, "passed": True},
        },
        "hard_gates": {
            "protocol_equal": True,
            "same_harness_version": True,
            "within_stack_repeatability": True,
            "semantic_table_and_coordinate_equality": True,
            "absolute_mean_delta": True,
            "standardized_mean_drift": True,
        },
        "artifact_inputs": [
            {
                "role": "reference_first",
                "path": "/evidence/pymc5-run-1.json",
                "sha256": "artifact-reference-first",
            },
            {
                "role": "reference_second",
                "path": "/evidence/pymc5-run-2.json",
                "sha256": "artifact-reference-second",
            },
            {
                "role": "candidate_first",
                "path": "/evidence/pymc6-run-1.json",
                "sha256": "artifact-candidate-first",
            },
            {
                "role": "candidate_second",
                "path": "/evidence/pymc6-run-2.json",
                "sha256": "artifact-candidate-second",
            },
        ],
        "capture_evidence": evidence,
        "cases": [
            {
                "name": "difference_in_differences",
                "semantic_equality": {"passed": True, "failures": []},
                "metrics": [
                    {
                        "id": "did.causal_impact",
                        "reference_mean": 1.0,
                        "candidate_mean": 1.001,
                        "absolute_mean_delta": 0.001,
                        "absolute_tolerance": 0.01,
                        "standardized_mean_drift": 0.001,
                        "mutual_hdi_containment_diagnostic": True,
                        "passed": True,
                    }
                ],
            }
        ],
    }

    report = harness.render_report(comparison)

    assert "## Capture provenance" in report
    assert "`/evidence/pymc5-run-1.json`" in report
    assert "`artifact-reference-first`" in report
    assert "## Capture validity evidence" in report
    assert "`fixture-candidate-second`" in report
    assert "tree_depth / 0 / 7" in report
    assert "1.001" in report
    assert "750" in report
