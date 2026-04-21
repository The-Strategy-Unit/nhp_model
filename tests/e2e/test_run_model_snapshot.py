"""Test single model runs for the NHP model."""

import pandas as pd
import pytest

from nhp.model import (
    AaEModel,
    HealthStatusAdjustmentInterpolated,
    InpatientsModel,
    Model,
    ModelIteration,
    OutpatientsModel,
    load_sample_params,
)
from nhp.model.data import Local
from nhp.model.results import combine_results
from nhp.model.run import _run_model


@pytest.fixture(scope="session")
def model_results(data_dir):
    # this is reimplementing the logic in nhp.model.run.run_all, but that method writes files to
    # disk. it's much easier for us to capture the model results in memory here so we can snapshot
    # test them, but
    # TODO: refactor run_all to separate the logic of running the model and saving results
    params = load_sample_params(model_runs=32)
    nhp_data = Local.create(data_dir)

    model_types = [InpatientsModel, OutpatientsModel, AaEModel]
    run_params = Model.generate_run_params(params)

    # set the data path in the HealthStatusAdjustment class
    hsa = HealthStatusAdjustmentInterpolated(
        nhp_data(params["start_year"], params["dataset"]), params["start_year"]
    )

    return combine_results(
        [
            _run_model(
                m,
                params,
                nhp_data,
                hsa,
                run_params,
                lambda _: None,
                False,
            )
            for m in model_types
        ]
    )


def test_model_results_returns_expected_keys(model_results, data_regression):
    data_regression.check(set(model_results.keys()))


@pytest.mark.parametrize("result_key", ["default", "step_counts"])
def test_all_model_runs(model_results, result_key, dataframe_regression):
    actual = model_results[result_key]

    actual_summarised = (
        actual.query("model_run > 0")
        .groupby([i for i in actual if i not in ["model_run", "value"]])
        .agg({"value": "mean"})
        .sort_index()
        .reset_index()
    )

    actual_summarised["value"] = actual_summarised["value"].round(1)

    dataframe_regression.check(actual_summarised)
