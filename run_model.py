"""
Functions to run the model

This module allows you to run the various models. It allows you to run a single model run of one of
the different types of models for debugging purposes, or it allows you to run all of the models in
parallel saving the results to disk.

There are existing launch profiles for vscode that use this file, or you can use it directly in the
console, e.g.

    python run_model.py data/[DATASET]/results/[SCENARIO]/[RUN_TIME] 0 1 --debug -t=ip

will run a single run of the inpatients model, returning the results to display.
"""

import argparse
import json
import logging
import os
import time
from collections import defaultdict
from multiprocessing import Pool
from typing import Any, Callable

import numpy as np
import pandas as pd
from tqdm.auto import tqdm as base_tqdm

from model.aae import AaEModel
from model.health_status_adjustment import (
    HealthStatusAdjustment,
    HealthStatusAdjustmentInterpolated,
)
from model.helpers import load_params
from model.inpatients import InpatientsModel
from model.model import Model
from model.model_run import ModelRun
from model.outpatients import OutpatientsModel


def timeit(func: Callable, *args) -> Any:
    """
    Time how long it takes to evaluate function `f` with arguments `*args`.
    """
    start = time.time()
    results = func(*args)
    print(f"elapsed: {time.time() - start:.3f}s")
    return results


def _combine_results(results: list) -> dict:
    """Combine the results into a single dictionary

    When we run the models we have an array containing 3 items [inpatients, outpatient, a&e].
    Each of which contains one item for each model run, which is a dictionary.

    :param results: the results of running the models
    :type results: list
    :return: combined model results
    :rtype: dict
    """
    # results are a list containing each model runs aggregations
    # the aggregations are dicts of dicts
    combined_results = defaultdict(lambda: defaultdict(lambda: [0] * len(results[0])))
    # first, we need to invert the [{{}}] to {{[]}}
    for model_result in results:
        for i, res in enumerate(model_result):
            for agg_type, agg_values in res.items():
                for key, value in agg_values.items():
                    if agg_type == "step_counts":
                        combined_results[agg_type][key][i] = value
                    else:
                        combined_results[agg_type][key][i] += value
    # now we can convert this to the results we want
    combined_results = {
        k0: [
            {**dict(k1), "baseline": v1[0], "principal": v1[1], "model_runs": v1[2:]}
            for k1, v1 in v0.items()
        ]
        for k0, v0 in combined_results.items()
    }
    # finally we split the model runs out, this function modifies values in place
    for agg_type, values in combined_results.items():
        _split_model_runs_out(agg_type, values)

    return combined_results


def _split_model_runs_out(agg_type: str, results: dict) -> None:
    """updates a single result so the baseline and principal runs are split out
    and summary statistics are generated

    :param agg_type: which aggregate type are we using
    :type key: string
    :param results: the results for this aggregation
    :type results: dict
    """
    for result in results:
        if agg_type == "step_counts":
            if result["strategy"] == "-":
                result.pop("strategy")
            result.pop("baseline")
            result["value"] = result.pop("principal")
            if result["change_factor"] == "baseline":
                result.pop("model_runs")
            continue

        [lwr, median, upr] = np.quantile(result["model_runs"], [0.05, 0.5, 0.95])
        result["lwr_ci"] = lwr
        result["median"] = median
        result["upr_ci"] = upr

        if not agg_type in ["default", "bed_occupancy", "theatres_available"]:
            result.pop("model_runs")
        else:
            result["model_runs"] = [int(i) for i in result["model_runs"]]

        result["baseline"] = int(result["baseline"])
        result["principal"] = int(result["principal"])


def _run_model(
    model_type: Model,
    params: dict,
    path: str,
    hsa: Any,
    run_params: dict,
    progress_callback,
) -> dict:
    """Run the model iterations

    Runs the model for all of the model iterations, returning the aggregated results

    :param model_type: the type of model that we want to run
    :type model_type: Model
    :param params: the parameters to run the model with
    :type params: dict
    :param path: where the data is stored
    :type path: str
    :param hsa: an instance of the HealthStatusAdjustment class
    :type hsa: HealthStatusAdjustment
    :param run_params: the generated run parameters for the model run
    :type run_params: dict
    :return: a dictionary containing the aggregated results
    :rtype: dict
    """
    model_class = model_type.__name__[:-5]  # pylint: disable=protected-access
    logging.info("%s", model_class)
    logging.info(" * instantiating")
    model = model_type(params, path, hsa, run_params)
    logging.info(" * running")

    model_runs = list(range(-1, params["model_runs"] + 1))

    cpus = os.cpu_count()
    batch_size = int(os.getenv("BATCH_SIZE", "16"))

    class tqdm(base_tqdm):
        def update(self, n=1):
            super().update(n)
            progress_callback(self.n)

    with Pool(cpus) as pool:
        results = list(
            tqdm(
                pool.imap(
                    model.go,
                    model_runs,
                    chunksize=batch_size,
                ),
                f"Running {model.__class__.__name__[:-5].rjust(11)} model",  # pylint: disable=protected-access
                total=len(model_runs),
            )
        )
    logging.info(" * finished")

    return results


def run_all(params: dict, data_path: str, progress_callback) -> dict:
    """Run the model

    runs all 3 model types, aggregates and combines the results

    :param params: the parameters to use for this model run
    :type params: dict
    :param data_path: where the data is stored
    :type data_path: str
    :return: the filename of the saved results
    :rtype: dict
    """
    model_types = [InpatientsModel, OutpatientsModel, AaEModel]
    run_params = Model.generate_run_params(params)

    # set the data path in the HealthStatusAdjustment class
    HealthStatusAdjustment.data_path = data_path
    hsa = HealthStatusAdjustmentInterpolated(
        f"{data_path}/{params['start_year']}/{params['dataset']}",
        params["start_year"],
        params["end_year"],
    )

    pcallback = progress_callback(params["id"])

    results = _combine_results(
        [
            _run_model(
                m,
                params,
                data_path,
                hsa,
                run_params,
                lambda n: pcallback(m.__name__[:-5], n),
            )
            for m in model_types
        ]
    )

    filename = f"{params['dataset']}/{params['scenario']}-{params['create_datetime']}"
    os.makedirs(f"results/{params['dataset']}", exist_ok=True)

    with open(f"results/{filename}.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "params": params,
                "population_variants": run_params["variant"],
                "results": results,
            },
            file,
        )

    return filename


def run_single_model_run(
    params: dict, data_path: str, model_type: Model, model_run: int
) -> None:
    """
    Runs a single model iteration for easier debugging in vscode
    """
    run_params = Model.generate_run_params(params)
    # set the data path in the HealthStatusAdjustment class
    HealthStatusAdjustment.data_path = data_path
    hsa = HealthStatusAdjustmentInterpolated(
        f"{data_path}/{params['start_year']}/{params['dataset']}",
        params["start_year"],
        params["end_year"],
    )

    print("initialising model...  ", end="")
    model = timeit(model_type, params, data_path, hsa, run_params)
    print("running model...       ", end="")
    m_run = timeit(ModelRun, model, model_run)
    print("aggregating results... ", end="")
    agg_results = timeit(m_run.get_aggregate_results)
    #
    print()
    print("change factors:")
    step_counts = pd.DataFrame(
        [{**dict(k), "value": v} for k, v, in agg_results["step_counts"].items()]
    ).drop(columns=["strategy", "activity_type"])
    # pylint: disable=unsubscriptable-object
    cf_values = step_counts["change_factor"].unique()
    step_counts = (
        step_counts.groupby(["change_factor", "measure"], as_index=False)
        .sum()
        .pivot(index="change_factor", columns="measure")
        .loc[cf_values]
    )
    step_counts.loc["total"] = step_counts.sum()
    print(step_counts)
    #
    print()
    print("aggregated (default) results:")
    default_results = (
        pd.DataFrame(
            [{**dict(k), "value": v} for k, v, in agg_results["default"].items()]
        )
        .groupby(["pod", "measure"], as_index=False)
        .agg({"value": sum})
        .pivot(index=["pod"], columns="measure")
        .fillna(0)
    )
    default_results.loc["total"] = default_results.sum()
    print(default_results)


def _run_model_argparser() -> argparse.Namespace:  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "params_file",
        nargs="?",
        default="queue/sample_params.json",
        help="Path to the params.json file",
    )
    parser.add_argument("-d", "--data-path", help="Path to the data", default="data")
    parser.add_argument(
        "-r", "--model-run", help="Which model iteration to run", default=0, type=int
    )
    parser.add_argument(
        "-t",
        "--type",
        default="all",
        choices=["all", "aae", "ip", "op"],
        help="Model type, either: all, ip, op, aae",
        type=str,
    )
    return parser.parse_args()


def main() -> None:
    """
    Main method

    Runs when __name__ == "__main__"
    """

    # Grab the Arguments
    args = _run_model_argparser()
    #
    params = load_params(args.params_file)
    # define the model to run
    match args.type:
        case "all":
            logging.basicConfig(
                format="%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s",
                level=logging.INFO,
                datefmt="%Y-%m-%d %H:%M:%S",
            )

            run_all(params, args.data_path)
            return
        case "aae":
            model_type = AaEModel
        case "ip":
            model_type = InpatientsModel
        case "op":
            model_type = OutpatientsModel
    run_single_model_run(params, args.data_path, model_type, args.model_run)


def init():
    """method for calling main"""
    if __name__ == "__main__":
        main()


init()
