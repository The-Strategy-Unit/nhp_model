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
import time
from typing import Any, Callable

import pandas as pd

from model.aae import AaEModel
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


def debug_run(model_type: Model, params: dict, path: str, model_run: int) -> None:
    """
    Runs a single model iteration for easier debugging in vscode
    """
    print("initialising model...  ", end="")
    model = timeit(model_type, params, path)
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
    step_counts = (
        step_counts.groupby(["change_factor", "measure"], as_index=False)
        .sum()
        .pivot(index="change_factor", columns="measure")
        .loc[step_counts["change_factor"].unique()]
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
    parser.add_argument("params_file", help="Path to the params.json file")
    parser.add_argument("-d", "--data-path", help="Path to the data", default="data")
    parser.add_argument(
        "-r", "--model-run", help="Which model iteration to run", default=0, type=int
    )
    parser.add_argument(
        "-t",
        "--type",
        default="ip",
        choices=["aae", "ip", "op"],
        help="Model type, either: ip, op, aae",
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
    # define the models to run
    models = {"aae": AaEModel, "ip": InpatientsModel, "op": OutpatientsModel}
    if args.type != "all":
        models = {args.type: models[args.type]}
    #
    params = load_params(args.params_file)

    debug_run(models[args.type], params, args.data_path, args.model_run)


def init():
    """method for calling main"""
    if __name__ == "__main__":
        main()


init()
