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
from multiprocessing import Pool

import pandas as pd
from tqdm.auto import tqdm

from model.aae import AaEModel
from model.inpatients import InpatientsModel
from model.model_save import CosmosDBSave, LocalSave
from model.outpatients import OutpatientsModel


def timeit(func, *args):
    """
    Time how long it takes to evaluate function `f` with arguments `*args`.
    """
    start = time.time()
    results = func(*args)
    print(f"elapsed: {time.time() - start:.3f}s")
    return results


def debug_run(model, model_run):
    """
    Runs a single model iteration for easier debugging in vscode
    """
    print("running model... ", end="")
    change_factors, results = timeit(model.run, model_run)
    print("aggregating results... ", end="")
    agg_results = timeit(model.aggregate, results)
    #
    print()
    print("change factors:")
    print(change_factors)
    #
    print()
    print("aggregated (default) results:")
    print(
        pd.DataFrame.from_dict(agg_results["default"])
        .pivot(index="pod", columns="measure")
        .fillna(0)
    )


def multi_model_runs(save_model, run_start, model_runs, n_cpus=1, batch_size=16):
    """
    Run multiple model runs in parallel
    """

    run_end = run_start + model_runs

    with Pool(n_cpus) as pool:
        results = list(
            tqdm(
                pool.imap(
                    save_model.run_model,
                    range(run_start, run_end),
                    chunksize=batch_size,
                ),
                "Running model",
                total=model_runs,
            )
        )

    save_model.post_runs(results)

    assert len(results) == model_runs
    # # make sure to get the results - if we don't then no errors that occurred will be raised
    # print(f"Model runs completed: {sum(len(r) for r in results.get())} / {model_runs}")
    # assert sum(len(r.get()) for r in results) == model_runs


def run_model(
    params,
    data_path,
    save_model_class,
    save_model_path,
    run_start,
    model_runs,
    cpus,
    batch_size,
):  # pylint: disable=too-many-arguments
    """
    Run the model

    * model_type: which model to run? one of AaEModel, InpatientsModel, OutpatientsModel
    * params_file: the params file to use for this model run
    * data_path: where the model data is stored
    * results_path: where the model run's data is stored, see notes below.
    * run_start: the model run to start at
    * model_runs: how many runs to perform
    * cpus: how many cpu cores should we use
    * batch_size: how many runs should we perform each iteration

    The results_path should be of the form `data/[DATASET]/results/[SCENARIO]/[RUN_TIME]`.
    """

    def run_model_fn(model_type):
        try:
            model = model_type(params, data_path)
        except FileNotFoundError as exc:
            # handle the dataset not existing: we simply skip
            if str(exc).endswith(".parquet"):
                print(f"file {str(exc)} not found: skipping")
            # if it's not the data file that missing, re-raise the error
            else:
                raise exc
        print(f"Running: {model.__class__.__name__}")
        save_model = save_model_class(model, save_model_path)
        multi_model_runs(save_model, run_start, model_runs, cpus, batch_size)

    return run_model_fn


def _run_model_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("params_file", help="Path to the params.json file")
    parser.add_argument("--data-path", help="Path to the data", default="data")
    parser.add_argument(
        "--results-path", help="Path to the results", default="run_results"
    )
    parser.add_argument(
        "--run-start", help="Where to start model run from", type=int, default=0
    )
    parser.add_argument(
        "--model-runs",
        help="How many model runs to perform",
        type=int,
        default=1,
    )
    parser.add_argument(
        "-t",
        "--type",
        default="all",
        choices=["all", "aae", "ip", "op"],
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
    parser.add_argument(
        "--save-type",
        default="local",
        choices=["local", "cosmos"],
        help="What type of save method to use",
        type=str,
    )
    parser.add_argument("-d", "--debug", action="store_true")
    return parser


def main():
    """
    Main method

    Runs when __name__ == "__main__"
    """

    # Grab the Arguments
    args = _run_model_argparser().parse_args()
    # define the models to run
    models = {"aae": AaEModel, "ip": InpatientsModel, "op": OutpatientsModel}
    if args.type != "all":
        models = {args.type: models[args.type]}
    #
    with open(args.params_file, "r", encoding="UTF-8") as prf:
        params = json.load(prf)
    if args.debug:
        assert (
            args.type != "all"
        ), "can only debug a single model at a time: make sure to set the --type argument"
        model = models[args.type](params, args.data_path)
        debug_run(model, args.run_start)
    else:
        if args.save_type == "local":
            save_model_class = LocalSave
        elif args.save_type == "cosmos":
            save_model_class = CosmosDBSave
        runner = run_model(
            params,
            args.data_path,
            save_model_class,
            args.results_path,
            args.run_start,
            args.model_runs,
            args.cpus,
            args.batch_size,
        )
        list(map(runner, models.values()))


if __name__ == "__main__":
    main()
