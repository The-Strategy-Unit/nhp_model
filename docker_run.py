#!/opt/conda/bin/python
"""Run the model inside of the docker container"""

import argparse
import hashlib as hl
import json
import logging
import os
import uuid
from collections import defaultdict
from multiprocessing import Pool
from typing import Any

import numpy as np
from azure.cosmos import CosmosClient
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from azure.storage.blob import BlobServiceClient
from azure.storage.filedatalake import DataLakeServiceClient

import config
from model.aae import AaEModel
from model.health_status_adjustment import HealthStatusAdjustmentInterpolated
from model.inpatients import InpatientsModel
from model.model import Model
from model.outpatients import OutpatientsModel


def load_params(filename: str) -> dict:
    """Load model parameters

    if the file exists in the local folder `queue/`, then we will load the file from there.
    otherwise, the file will be downloaded from blob storage

    :param filename: the name of the parameter file to load
    :type filename: str
    :return: the parameters to use for a model run
    :rtype: dict
    """
    if os.path.exists(f"queue/{filename}"):
        with open(f"queue/{filename}", "rb") as params_file:
            params_content = params_file.read()
    else:
        logging.info("downloading params")

        container = BlobServiceClient(
            account_url=f"https://{config.STORAGE_ACCOUNT}.blob.core.windows.net",
            credential=DefaultAzureCredential(),
        ).get_container_client("queue")

        params_content = container.download_blob(filename).readall()

    app_version = config.APP_VERSION

    # generate an id using sha256 hash of the params combined with the app version
    run_id = hl.sha256(params_content + app_version.encode()).hexdigest()

    # convert to a dictionary, insert the id and app version
    params = json.loads(params_content)
    params["id"] = run_id
    params["app_version"] = app_version

    return params


def get_data(dataset: str) -> None:
    """Get the data to run the model

    if the data exists locally in the folder `data/`, then this function does nothing.
    otherwise, it will download the data from Azure Data Lake storage

    :param dataset: the name of the dataset that we are loading
    :type dataset: str
    """
    if os.path.exists(f"data/{dataset}"):
        return

    logging.info("downloading data")
    fs_client = DataLakeServiceClient(
        account_url=f"https://{config.STORAGE_ACCOUNT}.dfs.core.windows.net",
        credential=DefaultAzureCredential(),
    ).get_file_system_client("data")

    version = config.DATA_VERSION
    directory_path = f"{version}/{dataset}"

    paths = [p.name for p in fs_client.get_paths(directory_path)]

    os.makedirs(f"data/{dataset}", exist_ok=True)

    for filename in paths:
        logging.info(" * %s", filename)
        local_name = "data" + filename.removeprefix(version)
        with open(local_name, "wb") as local_file:
            file_client = fs_client.get_file_client(filename)
            local_file.write(file_client.download_file().readall())


def _upload_to_cosmos(params: dict, results: dict) -> None:
    """upload the model results to cosmos db

    :param params: the parameters used for this model run
    :type params: dict
    :param results: the results of the model run
    :type results: dict
    """
    logging.info("uploading results to cosmos")

    kyv_client = SecretClient(config.KEYVAULT_ENDPOINT, DefaultAzureCredential())
    cosmos_key = kyv_client.get_secret("cosmos-key").value

    cosmos_client = CosmosClient(config.COSMOS_ENDPOINT, cosmos_key)
    cosmos_db = cosmos_client.get_database_client(config.COSMOS_DB)

    model_run_container = cosmos_db.get_container_client("model_runs")
    results_container = cosmos_db.get_container_client("model_results")

    for agg_type, values in results.items():
        results_container.create_item(
            {
                "id": str(uuid.uuid4()),
                "model_run_id": params["id"],
                "aggregation": agg_type,
                "values": values,
            }
        )

    model_run_container.create_item(params)


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
            if result["change_factor"] != "baseline":
                result.pop("baseline")
            else:
                result["baseline"] = result.pop("principal")
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
    model_type: Model, params: dict, path: str, hsa: Any, run_params: dict
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
    batch_size = 16

    with Pool(cpus) as pool:
        results = list(pool.imap(model.go, model_runs, chunksize=batch_size))
    logging.info(" * finished")

    return results


def run(params: dict, data_path: str) -> dict:
    """Run the model

    runs all 3 model types, aggregates and combines the results

    :param params: the parameters to use for this model run
    :type params: dict
    :param data_path: where the data is stored
    :type data_path: str
    :return: the results of running the model
    :rtype: dict
    """
    model_types = [InpatientsModel, OutpatientsModel, AaEModel]
    run_params = Model.generate_run_params(params)

    hsa = HealthStatusAdjustmentInterpolated(
        f"{data_path}/{params['dataset']}", params["life_expectancy"]
    )

    return _combine_results(
        [_run_model(m, params, data_path, hsa, run_params) for m in model_types]
    )


def main():
    """the main method"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "params_file",
        nargs="?",
        default="sample_params.json",
        help="Name of the parameters file stored in Azure",
    )
    parser.add_argument("--skip-upload-to-cosmos", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("azure.storage.common.storageclient").setLevel(logging.WARNING)
    logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(
        logging.WARNING
    )

    params = load_params(args.params_file)
    get_data(params["dataset"])

    results = run(params, "data")

    if not args.skip_upload_to_cosmos:
        _upload_to_cosmos(params, results)

    logging.info("complete")


def init():
    """method for calling main"""
    if __name__ == "__main__":
        main()


init()
