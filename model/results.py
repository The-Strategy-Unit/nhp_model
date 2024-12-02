"""
Methods to work with results of the model

This module allows you to work with the results of the model. Namely, combining the monte-carlo runs
into a single panda's dataframe, and helping with saving the results files.
"""

import logging

# janitor complains of being unused, but it is used (.complete())
import janitor  # pylint: disable=unused-import
import pandas as pd


def _combine_model_results(results: list) -> pd.DataFrame:
    """Combine the results of the monte carlo runs

    Takes as input a list of lists, where the outer list contains an item for inpatients,
    outpatients and a&e runs, and the inner list contains the results of the monte carlo runs.

    :param results: a list containing the model results
    :type results: list
    :return: DataFrame containing the model results
    :rtype: pd.DataFrame
    """
    combined_results = pd.concat(
        [v.assign(model_run=i) for r in results for i, (v, _) in enumerate(r)],
        ignore_index=True,
    )

    return combined_results.complete(
        [i for i in combined_results.columns if i != "model_run" if i != "value"],
        "model_run",
        fill_value={"value": 0},
    )


def _combine_step_counts(results: list):
    """Combine the step counts of the monte carlo runs

    Takes as input a list of lists, where the outer list contains an item for inpatients,
    outpatients and a&e runs, and the inner list contains the results of the monte carlo runs.

    :param results: a list containing the model results
    :type results: list
    :return: DataFrame containing the model step counts
    :rtype: pd.DataFrame
    """
    combined_step_counts = pd.concat(
        [
            v
            # TODO: handle the case of daycase conversion, it's duplicating values
            # need to figure out exactly why, but this masks the issue for now
            .groupby(v.index.names).sum().reset_index().assign(model_run=i)
            for r in results
            for i, (_, v) in enumerate(r)
            if i > 0
        ],
        ignore_index=True,
    )

    return combined_step_counts.complete(
        [i for i in combined_step_counts.columns if i != "model_run" if i != "value"],
        "model_run",
        fill_value={"value": 0},
    )


def _generate_results_json(
    combined_results: pd.DataFrame, combined_step_counts: pd.DataFrame
) -> dict:
    """Generate the results in the json format"""

    def get_agg(*args):
        df = combined_results.groupby(
            ["model_run", "pod", "sitetret", *args, "measure"]
        )["value"].sum()
        return (
            pd.concat(
                [
                    df.loc[0].rename("baseline"),
                    df.loc[1:]
                    .groupby(level=df.index.names[1:])
                    .agg(list)
                    .rename("model_runs"),
                ],
                axis=1,
            )
            .reset_index()
            .to_dict(orient="records")
        )

    dict_results = {
        "default" if not v else "+".join(v): get_agg(*v)
        for v in [
            [],
            ["sex", "age_group"],
            ["age"],
            # aae specific
            ["acuity"],
            ["attendance_category"],
            # ip/op
            ["sex", "tretspef"],
            ["tretspef_raw"],
            # ip specific
            ["tretspef_raw", "los_group"],
        ]
    }

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

    return dict_results


def combine_results(results: list) -> dict:
    """Combine the results into a single dictionary

    When we run the models we have an array containing 3 items [inpatients, outpatient, a&e].
    Each of which contains one item for each model run, which is a dictionary.

    :param results: the results of running the models
    :type results: list
    :return: combined model results
    :rtype: dict
    """

    logging.info(" * starting to combine results")

    combined_results = _combine_model_results(results)
    combined_step_counts = _combine_step_counts(results)

    dict_results = _generate_results_json(combined_results, combined_step_counts)

    logging.info(" * finished combining results")
    return dict_results
