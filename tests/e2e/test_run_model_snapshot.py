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

# we need to capture the expected model result keys in a list so we can
#   a. test that all expected keys are present in the results
#   b. parameterize the snapshot test to check each result type separately
# if result types are added/removed from the model results, this list should be
# updated and the snapshot test re-run to update the snapshots.
expected_model_result_keys = [
    "acuity",
    "age",
    "attendance_category",
    "avoided_activity",
    "default",
    "delivery_episode_in_spell",
    "sex+age_group",
    "sex+tretspef_grouped",
    "tretspef",
    "tretspef+los_group",
    "step_counts",
]


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

    results, step_counts = combine_results(
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

    results["step_counts"] = step_counts

    return results


def test_model_results_returns_expected_keys(model_results):
    assert set(model_results.keys()) == set(expected_model_result_keys)


@pytest.mark.parametrize("result_key", expected_model_result_keys)
def test_all_model_runs(model_results, result_key, dataframe_regression):
    actual = model_results[result_key]
    actual = actual.sort_values(by=actual.columns.tolist())

    actual_summarised = (
        actual.query("model_run > 0")
        .groupby(list(set(actual.columns) - {"model_run", "value"}))
        .agg({"value": "mean"})
        .reset_index()
    )

    dataframe_regression.check(actual_summarised)
