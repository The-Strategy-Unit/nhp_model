"""Run the model inside of the docker container"""

import argparse
import json
import os
import uuid
from collections import defaultdict
from multiprocessing import Pool
from typing import Any

import numpy as np
from azure.cosmos import CosmosClient
from azure.identity import ManagedIdentityCredential
from azure.storage.blob import BlobServiceClient
from azure.storage.filedatalake import DataLakeServiceClient
from tqdm.auto import tqdm

from model.aae import AaEModel
from model.inpatients import InpatientsModel
from model.model import Model
from model.outpatients import OutpatientsModel


def load_params(storage_account: str, filename: str, credential: Any) -> dict:
    """_summary_

    :param storage_account: _description_
    :type storage_account: str
    :param filename: _description_
    :type filename: str
    :param credential: _description_
    :type credential: Any
    :return: _description_
    :rtype: dict
    """
    container = BlobServiceClient(
        account_url=f"https://{storage_account}.blob.core.windows.net",
        credential=credential,
    ).get_container_client("queue")

    params_content = container.download_blob(filename).readall()
    return json.loads(params_content)


def get_data(storage_account: str, version: str, dataset: str, credential: Any) -> None:
    """_summary_

    :param storage_account: _description_
    :type storage_account: str
    :param version: _description_
    :type version: str
    :param dataset: _description_
    :type dataset: str
    :param credential: _description_
    :type credential: Any
    """
    if os.path.exists(f"/app/data/{dataset}"):
        return

    fs_client = DataLakeServiceClient(
        account_url=f"https://{storage_account}.dfs.core.windows.net",
        credential=credential,
    ).get_file_system_client("data")

    directory_path = f"{version}/{dataset}"

    paths = [p.name for p in fs_client.get_paths(directory_path)]

    os.makedirs(f"/app/data/tmp/{dataset}", exist_ok=True)

    for filename in paths:
        local_name = "/app/data/tmp" + filename.removeprefix(version)
        with open(local_name, "wb") as local_file:
            file_client = fs_client.get_file_client(filename)
            local_file.write(file_client.download_file().readall())


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


def upload_to_cosmos(params: dict, results: dict) -> None:
    """_summary_

    :param params: _description_
    :type params: dict
    :param results: _description_
    :type results: dict
    """
    cosmos_client = CosmosClient(
        os.environ["COSMOS_ENDPOINT"], os.environ["COSMOS_KEY"]
    )
    cosmos_db = cosmos_client.get_database_client(os.environ["COSMOS_DB"])

    model_run_container = cosmos_db.get_container_client("model_runs")
    results_container = cosmos_db.get_container_client("model_results")

    params["id"] = str(uuid.uuid4())
    params["app_version"] = "dev"  # TODO: build into the container build process?

    for key, value in results.items():
        if key != "step_counts":
            for v in value:
                [lwr, median, upr] = np.quantile(v["model_runs"], [0.05, 0.5, 0.95])
                v["lwr_ci"] = lwr
                v["median"] = median
                v["upr_ci"] = upr

                if not key in ["default", "bed_occupancy", "theatres_available"]:
                    v.pop("model_runs")
                else:
                    v["model_runs"] = [int(i) for i in v["model_runs"]]

                v["baseline"] = int(v["baseline"])
                v["principal"] = int(v["principal"])

        item = {
            "id": str(uuid.uuid4()),
            "model_run_id": params["id"],
            "aggregation": key,
            "values": value,
        }
        results_container.create_item(item)

    model_run_container.create_item(params)


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


def update_step_counts(results: dict) -> None:
    """updates the step counts item in results

    Modifies the results object in place. The step counts needs to be restructured to remove the
    "baseline" item from non-baseline steps, and remove the model run items from the "baseline"
    step.
    """
    step_counts = results.pop("step_counts")
    for i in step_counts:
        if i["strategy"] == "-":
            i.pop("strategy")
        if i["change_factor"] != "baseline":
            i.pop("baseline")
        else:
            i["baseline"] = i.pop("principal")
            i.pop("model_runs")
    results["step_counts"] = step_counts


def main(param_filename):
    """the main method"""

    storage_account = os.environ["STORAGE_ACCOUNT"]

    data_version = "v0.2.0"  # update as required, but bake into container

    # we can either pass the storage account key in via an enivornment variable
    # or we could use a managed identity (when running in Azure)
    if not (credential := os.environ.get("STORAGE_KEY", None)):
        credential = ManagedIdentityCredential()

    params = load_params(storage_account, param_filename, credential)
    get_data(storage_account, data_version, params["dataset"], credential)

    model_types = [InpatientsModel, OutpatientsModel, AaEModel]

    results = combine_results([run(m, params, "data") for m in model_types])

    upload_to_cosmos(params, results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "params_file", help="Name of the parameters file stored in Azure"
    )

    args = parser.parse_args()
    main(args.params_file)
