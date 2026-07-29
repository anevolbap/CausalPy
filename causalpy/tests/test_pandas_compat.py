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
"""Behavioral regressions for pandas 2.3 and 3 compatibility."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

import causalpy as cp
from causalpy.custom_exceptions import BadIndexException, DataException
from causalpy.data.simulate_data import generate_time_series_data_simple
from causalpy.formula_utils import _normalize_patsy_data, build_formula_matrices


@pytest.mark.parametrize(
    "unit",
    [
        pd.Series(["north", "north", "south", "south"]),
        pd.Series(["north", "north", "south", "south"], dtype="string"),
    ],
    ids=["inferred-strings", "StringDtype"],
)
def test_formula_matrices_accept_string_columns_without_mutating_input(unit):
    """Patsy formulas accept pandas 3-style strings without changing the frame."""
    data = pd.DataFrame({"unit": unit, "y": [1.0, 2.0, 3.0, 4.0]})
    original = data.copy(deep=True)

    y, X = build_formula_matrices("y ~ 1 + C(unit)", data)

    assert y.shape == (4, 1)
    assert X.shape == (4, 2)
    pd.testing.assert_frame_equal(data, original)


def test_patsy_normalization_leaves_non_string_extension_dtypes_unchanged():
    """Only extension strings are normalized before Patsy consumes mixed frames."""
    data = pd.DataFrame(
        {
            "unit": pd.Series(["north", "south"], dtype="string"),
            "category": pd.Series(pd.Categorical(["a", "b"])),
            "number": pd.Series([1, 2], dtype="Int64"),
            "y": [1.0, 2.0],
        }
    )
    original = data.copy(deep=True)

    normalized = _normalize_patsy_data(data)

    assert normalized is not data
    assert normalized["unit"].dtype == object
    assert isinstance(normalized["category"].dtype, pd.CategoricalDtype)
    assert normalized["number"].dtype == data["number"].dtype

    y, X = build_formula_matrices("y ~ 1 + C(unit) + C(category)", data)

    assert y.shape == (2, 1)
    assert X.shape == (2, 3)
    pd.testing.assert_frame_equal(data, original)


def test_formula_matrices_preserve_extension_string_missingness():
    """Extension-string missing values retain Patsy's normal row-dropping behavior."""
    data = pd.DataFrame(
        {
            "unit": pd.Series(["north", "south", pd.NA, "north"], dtype="string"),
            "y": [1.0, 2.0, 3.0, 4.0],
        }
    )
    original = data.copy(deep=True)

    y, X = build_formula_matrices("y ~ 1 + C(unit)", data)

    assert y.shape == (3, 1)
    assert X.shape == (3, 2)
    assert pd.isna(data.loc[2, "unit"])
    pd.testing.assert_frame_equal(data, original)


def test_regression_discontinuity_normalizes_nullable_integers_without_mutation():
    """Nullable integer indicators become boolean only on the experiment-owned frame."""
    x = np.linspace(0.0, 1.0, 20)
    data = pd.DataFrame(
        {
            "x": x,
            "treated": pd.Series((x >= 0.5).astype(int), dtype="Int64"),
            "y": 1.0 + 2.0 * x + (x >= 0.5),
        }
    )
    original = data.copy(deep=True)

    result = cp.RegressionDiscontinuity(
        data,
        formula="y ~ 1 + x + treated + x:treated",
        treatment_threshold=0.5,
        model=LinearRegression(),
    )

    assert pd.api.types.is_bool_dtype(result.data["treated"])
    assert pd.api.types.is_integer_dtype(data["treated"])
    pd.testing.assert_frame_equal(data, original)


@pytest.mark.parametrize(
    "unit",
    [
        pd.Series(["north", "north", "south", "south"]),
        pd.Series(["north", "north", "south", "south"], dtype="string"),
    ],
    ids=["inferred-strings", "StringDtype"],
)
def test_panel_regression_uses_observed_categorical_groups(unit):
    """Demeaning ignores unused category levels and accepts string fixed effects."""
    data = pd.DataFrame(
        {
            "unit": unit,
            "time": pd.Categorical([0, 1, 0, 1], categories=[0, 1, 2]),
            "treatment": pd.Series([False, True, False, True], dtype="boolean"),
            "y": [1.0, 3.0, 2.0, 4.0],
        }
    )
    original = data.copy(deep=True)

    result = cp.PanelRegression(
        data,
        formula="y ~ treatment",
        unit_fe_variable="unit",
        time_fe_variable="time",
        fe_method="demeaned",
        model=LinearRegression(),
    )

    assert list(result._group_means["unit"].index) == ["north", "south"]
    assert list(result._group_means["time"].index) == [0, 1]
    pd.testing.assert_frame_equal(data, original)


def test_panel_regression_rejects_missing_string_fixed_effects():
    """Missing fixed-effect identifiers are rejected before grouping can drop them."""
    data = pd.DataFrame(
        {
            "unit": pd.Series(["north", pd.NA, "south", "south"], dtype="string"),
            "time": [0, 1, 0, 1],
            "treatment": [False, True, False, True],
            "y": [1.0, 3.0, 2.0, 4.0],
        }
    )
    original = data.copy(deep=True)

    with pytest.raises(
        DataException,
        match="Fixed-effect variable 'unit' must not contain missing values",
    ):
        cp.PanelRegression(
            data,
            formula="y ~ treatment",
            unit_fe_variable="unit",
            time_fe_variable="time",
            fe_method="demeaned",
            model=LinearRegression(),
        )

    pd.testing.assert_frame_equal(data, original)


def test_staggered_did_creates_float_event_times_without_mutating_input():
    """Never-treated rows receive missing float event times without mutating callers."""
    data = pd.DataFrame(
        {
            "unit": [0, 0, 1, 1, 2, 2],
            "time": [0, 1, 0, 1, 0, 1],
            "treated": [0, 1, 0, 1, 0, 0],
            "y": [1.0, 3.0, 2.0, 4.0, 1.5, 1.7],
        }
    )
    original = data.copy(deep=True)

    result = cp.StaggeredDifferenceInDifferences(
        data,
        formula="y ~ 1 + C(unit) + C(time)",
        unit_variable_name="unit",
        time_variable_name="time",
        model=LinearRegression(),
    )

    assert pd.api.types.is_float_dtype(result.data_["event_time"])
    assert result.data_.loc[result.data_["unit"] == 2, "event_time"].isna().all()
    pd.testing.assert_frame_equal(data, original)


def test_time_series_generator_has_explicit_datetime_treatment_boundary():
    """Generated series retains a datetime index and exclusive treatment boundary."""
    treatment_time = pd.Timestamp("2015-01-31")

    data = generate_time_series_data_simple(treatment_time=treatment_time, seed=42)

    expected_index = pd.date_range(
        start="2010-01-01", end="2020-01-01", freq="ME", name="date"
    )
    pd.testing.assert_index_equal(data.index, expected_index, exact=True)
    assert data.loc[:treatment_time, "causal effect"].eq(0).all()
    assert data.loc[data.index > treatment_time, "causal effect"].eq(2).all()


def test_interrupted_time_series_predicts_with_string_extension_covariates():
    """Fit and prediction matrices handle ``StringDtype`` at the Patsy boundary."""
    dates = pd.date_range("2020-01-01", periods=12, freq="MS")
    data = pd.DataFrame(
        {
            "y": np.arange(len(dates), dtype=float),
            "segment": pd.array(
                ["north", "south", "north", "south"] * 3, dtype="string"
            ),
        },
        index=dates,
    )
    original = data.copy(deep=True)

    result = cp.InterruptedTimeSeries(
        data,
        treatment_time=dates[6],
        formula="y ~ 1 + C(segment)",
        model=LinearRegression(),
    )

    assert result.pre_design["X"].sizes["obs_ind"] == 6
    assert result.post_design["X"].sizes["obs_ind"] == 6
    pd.testing.assert_frame_equal(data, original)


def test_interrupted_time_series_preserves_timezone_aware_boundary_and_input():
    """Timezone-aware treatment dates split periods correctly and reject ``NaT``."""
    dates = pd.date_range("2020-01-01", periods=12, freq="MS", tz="UTC")
    data = pd.DataFrame(
        {"y": np.arange(len(dates), dtype=float)},
        index=dates,
    )
    original = data.copy(deep=True)
    treatment_time = dates[6]

    result = cp.InterruptedTimeSeries(
        data,
        treatment_time=treatment_time,
        formula="y ~ 1",
        model=LinearRegression(),
    )

    assert result.datapre.index.max() == dates[5]
    assert result.datapost.index.min() == treatment_time
    pd.testing.assert_frame_equal(data, original)

    with pytest.raises(BadIndexException, match="treatment_time must not be missing"):
        cp.InterruptedTimeSeries(
            data,
            treatment_time=pd.NaT,
            formula="y ~ 1",
            model=LinearRegression(),
        )
