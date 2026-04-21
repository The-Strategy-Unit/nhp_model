"""Test single model runs for the NHP model."""

import pytest

from nhp.model import (
    AaEModel,
    InpatientsModel,
    ModelIteration,
    OutpatientsModel,
    load_sample_params,
)
from nhp.model.data import Local
from nhp.model.run import run_all


@pytest.fixture(scope="session")
def model_single_run_results(data_dir):
    params = load_sample_params()
    data = Local.create(data_dir)

    results = {}
    for k, model_class in [("ip", InpatientsModel), ("op", OutpatientsModel), ("aae", AaEModel)]:
        model = model_class(params, data)
        m_run = ModelIteration(model, 1)
        model_results, step_counts = m_run.get_aggregate_results()
        results[k] = model_results
        results[k]["step_counts"] = step_counts
    return results


def test_model_returns_expected_aggregations(model_single_run_results, data_regression):
    actual_aggregations = {k: list(v.keys()) for k, v in model_single_run_results.items()}
    data_regression.check(actual_aggregations)


@pytest.mark.parametrize("activity_type", ["ip", "op", "aae"])
def test_model_default_results(activity_type, model_single_run_results, dataframe_regression):
    df = model_single_run_results[activity_type]["default"]
    dataframe_regression.check(df.to_frame())


@pytest.mark.parametrize("activity_type", ["ip", "op", "aae"])
def test_model_step_counts(activity_type, model_single_run_results, dataframe_regression):
    df = model_single_run_results[activity_type]["step_counts"]
    dataframe_regression.check(df.to_frame())
