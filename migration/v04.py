"""Data Migration

Scripts to migrate from a previous version of the model to a new version
"""

import gzip
import json
import os

from tqdm.auto import tqdm

import run_model as rm
from migration.azure_storage import upload_results
from model.helpers import load_params


def v03_to_v04(path, data_path="data"):
    """Convert v0.3 to v0.4 results

    :param path: the path to the old results
    :type path: string
    :param data_path: the path to where the data lives, defaults to "data"
    :type data_path: str, optional
    """
    time_profile_mappings = load_params("queue/sample_params.json")[
        "time_profile_mappings"
    ]

    files = [
        f"{path}/{d}/{i}" for d in os.listdir(path) for i in os.listdir(f"{path}/{d}")
    ]

    for i in tqdm(files):
        with gzip.open(i) as gzf:
            params = json.load(gzf)["params"]

        params.pop("health_status_adjustment")
        params.pop("life_expectancy")
        params["time_profile_mappings"] = time_profile_mappings

        results_file = rm.run_all(params, data_path, lambda _: lambda _: None)

        metadata = {
            k: str(v)
            for k, v in params.items()
            if not isinstance(v, dict) and not isinstance(v, list)
        }

        upload_results(results_file, "v0.4", metadata)

        os.remove(i)
        os.remove(f"results/{results_file}.json")
