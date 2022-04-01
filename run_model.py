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
import os
import time

from model.aae import AaEModel
from model.inpatients import InpatientsModel
from model.outpatients import OutpatientsModel


def timeit(func, *args):
    """
    Time how long it takes to evaluate function `f` with arguments `*args`.
    """
    start = time.time()
    results = func(*args)
    print(f"elapsed: {time.time() - start:.3f}")
    return results


def run_model(model_type, results_path, run_start, model_runs, cpus, batch_size):
    """
    Run the model

    * model_type: which model to run? one of AaEModel, InpatientsModel, OutpatientsModel
    * results_path: where the model run's data is stored, see notes below.
    * run_start: the model run to start at
    * model_runs: how many runs to perform
    * cpus: how many cpu cores should we use
    * batch_size: how many runs should we perform each iteration

    The results_path should be of the form `data/[DATASET]/results/[SCENARIO]/[RUN_TIME]`.
    """
    try:
        model = model_type(results_path)
    except FileNotFoundError as exc:
        # handle the dataset not existing: we simply skip
        if str(exc).endswith(".parquet"):
            print(f"file {str(exc)} not found: skipping")
        # if it's not the data file that missing, re-raise the error
        else:
            raise exc
    print(f"Running: {model.__class__.__name__}")
    model.multi_model_runs(run_start, model_runs, cpus, batch_size)


def main():
    """
    Main method

    Runs when __name__ == "__main__"
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("results_path", nargs=1, help="Path to the results")
    parser.add_argument(
        "run_start", nargs=1, help="Where to start model run from", type=int
    )
    parser.add_argument(
        "model_runs", nargs=1, help="How many model runs to perform", type=int
    )
    parser.add_argument(
        "-t",
        "--type",
        default="all",
        help="Model type, either ip, op, aae, or all",
        type=str,
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=4,
        help="Size of the batches to run the model in",
        type=int,
    )
    parser.add_argument(
        "-c",
        "--cpus",
        default=os.cpu_count(),
        help="Number of CPU cores to use",
        type=int,
    )
    parser.add_argument("-d", "--debug", action="store_true")
    # Grab the Arguments
    args = parser.parse_args()
    # define the models to run
    models = {"aae": AaEModel, "ip": InpatientsModel, "op": OutpatientsModel}
    if args.type != "all":
        models = {args.type: models[args.type]}
    #
    if args.debug:
        assert (
            args.type != "all"
        ), "can only debug a single model at a time: make sure to set the --type argument"
        model = models[args.type](args.results_path[0])
        print(
            "running model:",
        )
        change_factors, results = timeit(model.run, args.run_start[0])
        print(
            "aggregating results:",
        )
        agg_results = timeit(model.aggregate, results)
        print(json.dumps(change_factors, indent=2))
        print()
        print(agg_results)
    else:
        for i in models.values():
            run_model(
                i,
                args.results_path[0],
                args.run_start[0],
                args.model_runs[0],
                args.cpus,
                args.batch_size,
            )


if __name__ == "__main__":
    main()
