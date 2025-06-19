"""
Run the model


"""

import logging
import os
import time
from multiprocessing import Pool
from typing import Any, Callable, Tuple, Type

from tqdm.auto import tqdm as base_tqdm

from nhp.model.aae import AaEModel
from nhp.model.data import Data, Local
from nhp.model.health_status_adjustment import HealthStatusAdjustmentInterpolated
from nhp.model.inpatients import InpatientsModel
from nhp.model.model import Model
from nhp.model.model_iteration import ModelIteration
from nhp.model.outpatients import OutpatientsModel
from nhp.model.results import combine_results, generate_results_json, save_results_files


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
    model_type: Type[Model],
    params: dict,
    data: Callable[[int, str], Data],
    hsa: Any,
    run_params: dict,
    progress_callback,
    save_full_model_results: bool,
) -> list:
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
) -> Tuple[list, str]:
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
    params: dict, data_path: str, model_type: Type[Model], model_run: int
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
