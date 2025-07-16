"""Methods to work with results of the model.

This module allows you to work with the results of the model. Namely, combining the monte-carlo runs
into a single panda's dataframe, and helping with saving the results files.
"""

import json
import logging
import os
from typing import Dict, List

# janitor complains of being unused, but it is used (.complete())
import janitor  # pylint: disable=unused-import
import pandas as pd

from nhp.model.model_iteration import ModelRunResult


def _complete_model_runs(
    res: List[pd.DataFrame], model_runs: int, include_baseline: bool = True
) -> pd.DataFrame:
    """Complete the data frame for all model runs.

    if any aggregation returns rows for only some of the model runs, we need to add a "0" row for
    that run

    :param res: list of model results
    :type res: List[pd.DataFrame]
    :param model_runs: the number of model runs
    :type model_runs: int
    :param include_baseline: whether to include model run 0 (the baseline) or not, optional
        (defaults to True)
    :type include_baseline: bool
    :return: combined and completed data frame
    :rtype: pd.DataFrame
    """
    results = pd.concat(res)
    results: pd.DataFrame = results.groupby(  # type: ignore
        [i for i in results.columns if i != "value"], as_index=False
    )["value"].sum()

    return janitor.complete(
        results,
        [i for i in results.columns if i != "model_run" if i != "value"],
        {"model_run": range(0 if include_baseline else 1, model_runs + 1)},
        fill_value={"value": 0},
    )


def _combine_model_results(
    results: list[list[ModelRunResult]],
) -> dict[str, pd.DataFrame]:
    """Combine the results of the monte carlo runs.

    Takes as input a list of lists, where the outer list contains an item for inpatients,
    outpatients and a&e runs, and the inner list contains the results of the monte carlo runs.

    :param results: a list containing the model results
    :type results: list
    :return: DataFrame containing the model results
    :rtype: pd.DataFrame
    """
    aggregations = sorted(list({k for r in results for v, _ in r for k in v.keys()}))

    model_runs = len(results[0]) - 1

    return {
        k: _complete_model_runs(
            [
                v[k].reset_index().assign(model_run=i)
                for r in results
                for (i, (v, _)) in enumerate(r)
                if k in v
            ],
            model_runs,
        )
        for k in aggregations
    }


def _combine_step_counts(results: list):
    """Combine the step counts of the monte carlo runs.

    Takes as input a list of lists, where the outer list contains an item for inpatients,
    outpatients and a&e runs, and the inner list contains the results of the monte carlo runs.

    :param results: a list containing the model results
    :type results: list
    :return: DataFrame containing the model step counts
    :rtype: pd.DataFrame
    """
    model_runs = len(results[0]) - 1
    return _complete_model_runs(
        [
            v
            # TODO: handle the case of daycase conversion, it's duplicating values
            # need to figure out exactly why, but this masks the issue for now
            .groupby(v.index.names).sum().reset_index().assign(model_run=i)
            for r in results
            for i, (_, v) in enumerate(r)
            if i > 0
        ],
        model_runs,
        include_baseline=False,
    )


def generate_results_json(
    combined_results: dict[str, pd.DataFrame],
    combined_step_counts: pd.DataFrame,
    params: dict,
    run_params: dict,
) -> str:
    """Generate the results in the json format and save."""

    def agg_to_dict(res):
        results_df = res.set_index("model_run")
        return (
            pd.concat(
                [
                    results_df.loc[0]
                    .set_index([i for i in results_df.columns if i != "value"])
                    .rename(columns={"value": "baseline"}),
                    results_df.loc[results_df.index != 0]
                    .groupby([i for i in results_df.columns if i != "value"])
                    .agg(list)
                    .rename(columns={"value": "model_runs"}),
                ],
                axis=1,
            )
            .reset_index()
            .to_dict(orient="records")
        )

    dict_results = {k: agg_to_dict(v) for k, v in combined_results.items()}

    dict_results["step_counts"] = (
        combined_step_counts.groupby(
            [
                "pod",
                "change_factor",
                "strategy",
                "sitetret",
                "activity_type",
                "measure",
            ]
        )[["value"]]
        .agg(list)
        .reset_index()
        .to_dict("records")
    )

    for i in dict_results["step_counts"]:
        i["model_runs"] = i.pop("value")
        if i["change_factor"] == "baseline":
            i["model_runs"] = i["model_runs"][0:1]
        if i["strategy"] == "-":
            i.pop("strategy")

    filename = f"{params['dataset']}/{params['scenario']}-{params['create_datetime']}"
    os.makedirs(f"results/{params['dataset']}", exist_ok=True)
    with open(f"results/{filename}.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "params": params,
                "population_variants": run_params["variant"],
                "results": dict_results,
            },
            file,
        )
    return filename


def save_results_files(results: dict, params: dict) -> list:
    """Saves aggregated and combined results as parquet, and params as JSON.

    :param dict_results: the results of running the models, processed into one dictionary
    :type dict_results: dict
    :param params: the parameters used for the model run
    :type params: dict
    :return: filepaths to saved files
    :rtype: list
    """
    path = (
        f"results/{params['dataset']}/{params['scenario']}/{params['create_datetime']}"
    )
    os.makedirs(path, exist_ok=True)

    return [
        *[_save_parquet_file(path, k, v, params) for k, v in results.items()],
        _save_params_file(path, params),
    ]


def _add_metadata_to_dataframe(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Add metadata as columns to the dataframe.

    Add metadata as columns to the dataframe, so that the saved parquet files have useful
    information regarding their provenance.

    :param df: The dataframe that we want to add the metadata to
    :type df: pd.DataFrame
    :param params: The parameters for the model run, which include metadata
    :type params: dict
    :return: The dataframe, with additional columns "dataset", "scenario" and "create_datetime"
    :rtype: pd.DataFrame
    """
    metadata_to_save = ["dataset", "scenario", "app_version", "create_datetime"]
    for m in metadata_to_save:
        df[m] = params[m]
    return df


def _save_parquet_file(
    path: str, results_name: str, results_df: pd.DataFrame, params: dict
) -> str:
    """Save a results dataframe as parquet.

    :param path: the folder where we want to save the results to
    :type path: str
    :param results_name: the name of this aggregation
    :type results_name: str
    :param results_df: the results dataframe
    :type results_df: pd.DataFrame
    :return: the filename of the saved file
    :rtype: str
    """
    results_df = _add_metadata_to_dataframe(results_df, params)
    results_df.to_parquet(filename := f"{path}/{results_name}.parquet")
    return filename


def _save_params_file(path: str, params: dict) -> str:
    """Save the model runs parameters as json.

    :param path: the folder where we want to save the results to
    :type path: str
    :param params: the parameters the model was run with
    :type params: dict
    :return: the filename of the saved file
    :rtype: str
    """
    with open(filename := f"{path}/params.json", "w", encoding="utf-8") as file:
        json.dump(params, file)
    return filename


def _patch_converted_sdec_activity(
    results: Dict[str, pd.DataFrame], column: str, col_value: str
) -> None:
    """Patch the converted SDEC activity in the dataframe."""
    results_df = results[column]
    agg_cols = ["pod", "sitetret", "measure", "model_run"]

    default_sdec = (
        results["default"]
        .query("pod == 'aae_type-05'")
        .set_index(agg_cols)["value"]
        .rename("b")
    )

    missing_sdec_activity = (
        pd.concat(
            [
                default_sdec,
                (
                    results_df.query("pod == 'aae_type-05'")
                    .groupby(agg_cols)["value"]
                    .sum()
                    .rename("a")
                ),
            ],
            axis=1,
        )
        .fillna(0)
        .reset_index()
        .assign(value=lambda x: x["b"] - x["a"])
        .drop(columns=["b", "a"])
    )
    missing_sdec_activity[column] = col_value

    df_fixed = (
        pd.concat([results_df, missing_sdec_activity], axis=0)
        .groupby(
            ["pod", "sitetret", "measure", column, "model_run"],
            as_index=False,
        )
        .sum()
    )

    df_fixed["value"] = df_fixed["value"].astype("int64")

    results[column] = df_fixed


def combine_results(
    results: list[list[ModelRunResult]],
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Combine the results into a single dictionary.

    When we run the models we have an array containing 3 items [inpatients, outpatient, a&e].
    Each of which contains one item for each model run, which is a dictionary.

    :param results: the results of running the models
    :type results: list[list[ModelRunResult]]
    :return: combined model results
    :rtype: dict
    """
    logging.info(" * starting to combine results")

    combined_results = _combine_model_results(results)
    combined_step_counts = _combine_step_counts(results)

    # TODO: this is a bit of a hack, but we need to patch the converted SDEC activity
    # because inpatients activity is aggregated differently to a&e, the a&e aggregations will be
    # missing the converted SDEC activity, so we need to add it back in
    _patch_converted_sdec_activity(combined_results, "acuity", "standard")
    _patch_converted_sdec_activity(combined_results, "attendance_category", "1")

    logging.info(" * finished combining results")
    return combined_results, combined_step_counts
