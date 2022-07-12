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
import os
import time
from multiprocessing import Pool
from typing import Any, Callable

import pandas as pd
from tqdm.auto import tqdm

from model.aae import AaEModel
from model.helpers import load_params
from model.inpatients import InpatientsModel
from model.model import Model
from model.model_save import CosmosDBSave, LocalSave, ModelSave
from model.outpatients import OutpatientsModel


def timeit(func: Callable, *args) -> Any:
    """
    Time how long it takes to evaluate function `f` with arguments `*args`.
    """
    start = time.time()
    results = func(*args)
    print(f"elapsed: {time.time() - start:.3f}s")
    return results


def debug_run(model: Model, model_run: int) -> None:
    """
    Runs a single model iteration for easier debugging in vscode
    """
    print("running model... ", end="")
    change_factors, results = timeit(model.run, model_run)
    print("aggregating results... ", end="")
    agg_results = timeit(model.aggregate, results, model_run)
    #
    print()
    print("change factors:")
    print(change_factors)
    #
    print()
    print("aggregated (default) results:")
    print(
        pd.DataFrame.from_dict(
            [{**k._asdict(), "value": v} for k, v in agg_results["default"].items()]
        )
        .pivot(index="pod", columns="measure")
        .fillna(0)
    )


def run_model(
    save_model: ModelSave,
    run_start: int,
    model_runs: int,
    cpus: int = os.cpu_count(),
    batch_size: int = 8,
) -> Callable[[str], None]:
    """
    Run the model

    * save_model: an instance of ModelSave class
    * run_start: the model run to start at
    * model_runs: how many runs to perform
    * cpus: how many cpu cores should we use
    * batch_size: how many runs should we perform each iteration

    returns: a function which accepts a model instance
    """

    def run_model_fn(model):
        try:
            save_model.set_model(model)
            if cpus == 1:
                results = list(
                    tqdm(
                        map(
                            save_model.run_model,
                            range(run_start, run_start + model_runs),
                        ),
                        f"Running {save_model._model.__class__.__name__[:-5].rjust(11)} model",  # pylint: disable=protected-access
                        total=model_runs,
                    )
                )
            else:
                with Pool(cpus) as pool:
                    results = list(
                        tqdm(
                            pool.imap(
                                save_model.run_model,
                                range(run_start, run_start + model_runs),
                                chunksize=batch_size,
                            ),
                            f"Running {save_model._model.__class__.__name__[:-5].rjust(11)} model",  # pylint: disable=protected-access
                            total=model_runs,
                        )
                    )
            assert len(results) == model_runs
        except FileNotFoundError as exc:
            # handle the dataset not existing: we simply skip
            if str(exc).endswith(".parquet"):
                print(f"file {str(exc)} not found: skipping")
            # if it's not the data file that missing, re-raise the error
            else:
                raise exc

    return run_model_fn


def _run_model_argparser() -> argparse.Namespace:  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument("params_file", help="Path to the params.json file")
    parser.add_argument("--data-path", help="Path to the data", default="data")
    parser.add_argument("--results-path", help="Path to the results", default="results")
    parser.add_argument(
        "--temp-results-path",
        help="Path to the temporary results path",
        default=None,
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
        default=8,
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
    parser.add_argument(
        "--run-postruns", help="Run the ModelSave post_run method", action="store_true"
    )
    parser.add_argument(
        "--save-results", help="Save the full model results", action="store_true"
    )
    parser.add_argument("-d", "--debug", action="store_true")
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

        save_model = save_model_class(
            params, args.results_path, args.temp_results_path, args.save_results
        )

        runner = run_model(
            save_model,
            args.run_start,
            args.model_runs,
            args.cpus,
            args.batch_size,
        )

        list(map(runner, [m(params, args.data_path) for m in models.values()]))

        if args.run_postruns:
            print("Running         post-runs")
            save_model.post_runs()


if __name__ == "__main__":
    main()
