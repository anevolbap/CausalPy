"""Injected code to mock pm.sample for faster notebook execution."""

import numpy as np
import pymc as pm
import xarray as xr

# Minimum draws needed to satisfy notebook code that iterates over posterior samples
MIN_DRAWS = 100
FALLBACK_COMPILE_MODE = "FAST_COMPILE"


def mock_sample(*args, **kwargs):
    """Mock pm.sample using prior predictive sampling for speed."""
    random_seed = kwargs.get("random_seed")
    model = kwargs.get("model")
    idata_kwargs = kwargs.get("idata_kwargs") or {}

    # If no model is provided via kwargs, try to infer it from positional args
    if model is None and args:
        first_arg = args[0]
        if isinstance(first_arg, pm.Model):
            model = first_arg

    requested_draws = kwargs.get("draws")
    if requested_draws is None and len(args) > 1 and isinstance(args[1], int):
        requested_draws = args[1]

    # Ensure enough draws for notebook code while keeping execution fast.
    n_draws = max(MIN_DRAWS, requested_draws or MIN_DRAWS)

    try:
        idata = pm.sample_prior_predictive(
            model=model,
            random_seed=random_seed,
            draws=n_draws,
        )
    except ZeroDivisionError:
        idata = pm.sample_prior_predictive(
            model=model,
            random_seed=random_seed,
            draws=n_draws,
            compile_kwargs={"mode": FALLBACK_COMPILE_MODE},
        )
    idata["posterior"] = idata["prior"].to_dataset().copy()

    log_likelihood = idata_kwargs.get("log_likelihood")
    if log_likelihood:
        var_names = None if log_likelihood is True else log_likelihood
        idata = pm.compute_log_likelihood(
            idata,
            model=model,
            var_names=var_names,
            extend_inferencedata=True,
            progressbar=False,
        )

    # Create mock sample stats with diverging data
    if "sample_stats" not in idata:
        n_chains = 1
        sample_stats = xr.Dataset(
            {
                "diverging": xr.DataArray(
                    np.zeros((n_chains, n_draws), dtype=int),
                    dims=("chain", "draw"),
                )
            }
        )
        idata["sample_stats"] = sample_stats

    del idata["prior"]
    if "prior_predictive" in idata:
        del idata["prior_predictive"]

    return idata


pm.sample = mock_sample
pm.HalfFlat = pm.HalfNormal
pm.Flat = pm.Normal
