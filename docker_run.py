"""Run the model inside of the docker container"""

import os
from collections import defaultdict
from multiprocessing import Pool

from tqdm.auto import tqdm

from model.aae import AaEModel
from model.helpers import load_params
from model.inpatients import InpatientsModel
from model.model import Model
from model.outpatients import OutpatientsModel


def run(model_type: Model, params: dict, path: str) -> dict:
    """Run the model iterations

    Runs the model for all of the model iterations, returning the aggregated results

    :param model_type: the type of model that we want to run
    :type model_type: Model
    :param params: the parameters to run the model with
    :type params: dict
    :param path: where the data is stored
    :type path: str
    :return: a dictionary containing the aggregated results
    :rtype: dict
    """
    model = model_type(params, path)

    model_runs = list(range(-1, model.params["model_runs"] + 1))

    cpus = os.cpu_count()
    batch_size = 16

    with Pool(cpus) as pool:
        results = list(
            tqdm(
                pool.imap(model.go, model_runs, chunksize=batch_size),
                f"Running {model.__class__.__name__[:-5].rjust(11)} model",  # pylint: disable=protected-access
                total=len(model_runs),
            )
        )

    return results


def combine_results(results: list) -> dict:
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
    return {
        k0: [
            {**dict(k1), "baseline": v1[0], "principal": v1[1], "model_runs": v1[2:]}
            for k1, v1 in v0.items()
        ]
        for k0, v0 in combined_results.items()
    }


def main():
    """the main method"""
    params = load_params("sample_params.json")

    model_types = [InpatientsModel, OutpatientsModel, AaEModel]

    results = combine_results([run(m, params, "data") for m in model_types])

    print(results.keys())


if __name__ == "__main__":
    main()
