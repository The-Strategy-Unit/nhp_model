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
import logging
import os
import time
from multiprocessing import Pool
from typing import Any, Callable

from tqdm.auto import tqdm as base_tqdm

from model.aae import AaEModel
from model.data import Data, Local
from model.health_status_adjustment import HealthStatusAdjustmentInterpolated
from model.helpers import load_params
from model.inpatients import InpatientsModel
from model.model import Model
from model.model_iteration import ModelIteration
from model.outpatients import OutpatientsModel
from model.results import combine_results, generate_results_json, save_results_files


class tqdm(base_tqdm):  # pylint: disable=inconsistent-mro, invalid-name
    """Custom tqdm class that provides a callback function on update"""

    # ideally this would be set in the contstructor, but as this is a pretty
    # simple use case just implemented as a static variable. this does mean that
    # you need to update the value before using the class (each time)
    progress_callback = None

    def update(self, n=1):
        """overide the default tqdm update function to run the callback method"""
        super().update(n)
        if tqdm.progress_callback:
            tqdm.progress_callback(self.n)  # pylint: disable=not-callable


def timeit(func: Callable, *args) -> Any:
    """
    Time how long it takes to evaluate function `f` with arguments `*args`.
    """
    start = time.time()
    results = func(*args)
    print(f"elapsed: {time.time() - start:.3f}s")
    return results


def _run_model(
    model_type: Model,
    params: dict,
    data: Data,
    hsa: Any,
    run_params: dict,
    progress_callback,
    save_full_model_results: bool,
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
    model = model_type(params, data, hsa, run_params, save_full_model_results)
    logging.info(" * running")

    # set the progress callback for this run
    tqdm.progress_callback = progress_callback

    # model run 0 is the baseline
    # model run 1:n are the monte carlo sims
    model_runs = list(range(params["model_runs"] + 1))

    cpus = os.cpu_count()
    batch_size = int(os.getenv("BATCH_SIZE", "1"))

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


def run_all(
    params: dict, data_path: str, progress_callback, save_full_model_results: bool
) -> str:
    """Run the model

    runs all 3 model types, aggregates and combines the results

    :param params: the parameters to use for this model run
    :type params: dict
    :param data_path: where the data is stored
    :type data_path: str
    :return: the filename of the saved results
    :rtype: str
    """
    model_types = [InpatientsModel, OutpatientsModel, AaEModel]
    run_params = Model.generate_run_params(params)

    nhp_data = Local.create(data_path)

    # set the data path in the HealthStatusAdjustment class
    hsa = HealthStatusAdjustmentInterpolated(
        nhp_data(params["start_year"], params["dataset"]), params["start_year"]
    )

    pcallback = progress_callback()

    results, step_counts = combine_results(
        [
            _run_model(
                m,
                params,
                nhp_data,
                hsa,
                run_params,
                pcallback(m.__name__[:-5]),
                save_full_model_results,
            )
            for m in model_types
        ]
    )

    json_filename = generate_results_json(results, step_counts, params, run_params)

    # TODO: once generate_results_json is deperecated this step should be moved into combine_results
    results["step_counts"] = step_counts
    # TODO: this should be what the model returns once generate_results_json is deprecated
    saved_files = save_results_files(results, params)  # pylint: disable=unused-variable

    return saved_files, json_filename


def run_single_model_run(
    params: dict, data_path: str, model_type: Model, model_run: int
) -> None:
    """
    Runs a single model iteration for easier debugging in vscode
    """
    data = Local.create(data_path)

    print("initialising model...  ", end="")
    model = timeit(model_type, params, data)
    print("running model...       ", end="")
    m_run = timeit(ModelIteration, model, model_run)
    print("aggregating results... ", end="")
    model_results, step_counts = timeit(m_run.get_aggregate_results)
    #
    print()
    print("change factors:")
    step_counts = (
        step_counts.reset_index()
        .groupby(["change_factor", "measure"], as_index=False)["value"]
        .sum()
        .pivot(index="change_factor", columns="measure")
    )
    step_counts.loc["total"] = step_counts.sum()
    print(step_counts.fillna(0).astype(int))
    #
    print()
    print("aggregated (default) results:")

    default_results = (
        model_results["default"]
        .reset_index()
        .groupby(["pod", "measure"], as_index=False)
        .agg({"value": "sum"})
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
        "-r", "--model-run", help="Which model iteration to run", default=1, type=int
    )
    parser.add_argument(
        "-t",
        "--type",
        default="all",
        choices=["all", "aae", "ip", "op"],
        help="Model type, either: all, ip, op, aae",
        type=str,
    )
    parser.add_argument("--save-full-model-results", action="store_true")
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

            run_all(
                params,
                args.data_path,
                lambda: lambda _: None,
                args.save_full_model_results,
            )
            return
        case "aae":
            model_type = AaEModel
        case "ip":
            model_type = InpatientsModel
        case "op":
            model_type = OutpatientsModel
        case _:
            raise ValueError(f"Unknown model type: {args.type}")

    run_single_model_run(params, args.data_path, model_type, args.model_run)


def init():
    """method for calling main"""
    if __name__ == "__main__":
        main()


init()
