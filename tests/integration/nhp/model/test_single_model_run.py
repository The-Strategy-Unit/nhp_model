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


@pytest.fixture(scope="session")
def model_single_run_results(data_dir):
    params = load_sample_params()
    data = Local.create(data_dir)

    results = {}
    for k, model_class in [("ip", InpatientsModel), ("op", OutpatientsModel), ("aae", AaEModel)]:
        model = model_class(params, data)
        m_run = ModelIteration(model, 1)
        model_results = m_run.get_aggregate_results()
        results[k] = model_results
    return results


@pytest.mark.integration
def test_model_returns_expected_aggregations(model_single_run_results, data_regression):
    actual_aggregations = {k: list(v.keys()) for k, v in model_single_run_results.items()}
    data_regression.check(actual_aggregations)


@pytest.mark.integration
@pytest.mark.parametrize("activity_type", ["ip", "op", "aae"])
def test_model_default_results(activity_type, model_single_run_results, dataframe_regression):
    res = model_single_run_results[activity_type]["default"]
    dataframe_regression.check(res.to_frame())


@pytest.mark.integration
@pytest.mark.parametrize("activity_type", ["ip", "op", "aae"])
def test_model_step_counts(activity_type, model_single_run_results, dataframe_regression):
    res = (
        model_single_run_results[activity_type]["step_counts"]
        # Store stable decimal text in snapshots to avoid float serialization artefacts
        # such as 3607.4000000000001 from dataframe_regression's %.17g formatting.
        .map(lambda value: f"{value:.1f}")
    )
    dataframe_regression.check(res.to_frame())
