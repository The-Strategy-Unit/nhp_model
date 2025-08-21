"""Test single model runs for the NHP model."""

import pandas as pd
import pytest

from nhp.model import AaEModel, InpatientsModel, ModelIteration, OutpatientsModel, load_params
from nhp.model.data import Local
from nhp.model.run import run_all


@pytest.mark.parametrize(
    "model_class, expected_aggregations",
    [
        (
            InpatientsModel,
            {
                "sex+tretspef_grouped",
                "tretspef",
                "tretspef+los_group",
            },
        ),
        (
            OutpatientsModel,
            {
                "sex+tretspef_grouped",
                "tretspef",
            },
        ),
        (
            AaEModel,
            {
                "acuity",
                "attendance_category",
            },
        ),
    ],
)
def test_single_model_run(model_class, expected_aggregations, params_path, data_dir):
    # arrange
    params = load_params(params_path)
    data = Local.create(data_dir)
    model = model_class(params, data)
    expected_aggregations |= {
        "default",
        "sex+age_group",
        "age",
        "avoided_activity",
    }

    # act
    # rather than using the run_single_model_run function, we directly instantiate ModelIteration
    # this is so we can work with the results. run_single_model_run is used to print some output to
    # the console.
    m_run = ModelIteration(model, 1)
    model_results, step_counts = m_run.get_aggregate_results()

    # assert
    assert {isinstance(v, pd.Series) for v in model_results.values()} == {True}
    assert set(model_results.keys()) == expected_aggregations
    assert isinstance(step_counts, pd.Series)


def test_all_model_runs(params_path, data_dir):
    # arrange
    params = load_params(params_path)
    params["model_runs"] = 4

    nhp_data = Local.create(data_dir)
    # act
    actual = run_all(params, nhp_data)

    # assert
    res_path = "results/synthetic/test/20220101_000000"
    assert actual == (
        [
            f"{res_path}/{i}.parquet"
            for i in [
                "acuity",
                "age",
                "attendance_category",
                "avoided_activity",
                "default",
                "sex+age_group",
                "sex+tretspef_grouped",
                "tretspef",
                "tretspef+los_group",
                "step_counts",
            ]
        ]
        + [
            f"{res_path}/params.json",
        ],
        "synthetic/test-20220101_000000",
    )
