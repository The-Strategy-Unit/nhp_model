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


def convert_all_files_in_folder(path: str) -> None:
    """Convert all files in a folder from v03 to v04

    :param path: the folder which contains the old results
    :type path: str
    """
    files = [
        f"{path}/{d}/{i}" for d in os.listdir(path) for i in os.listdir(f"{path}/{d}")
    ]
    for i in tqdm(files):
        params = v03_to_v04(i)

        results_file = rm.run_all(params, "data", lambda: lambda _: None)

        metadata = {
            k: str(v)
            for k, v in params.items()
            if not isinstance(v, dict) and not isinstance(v, list)
        }

        upload_results(results_file, "v0.4", metadata)

        os.remove(i)
        os.remove(f"results/{results_file}.json")


def v03_to_v04(filename: str) -> dict:
    """Convert v0.3 to v0.4 results

    :param filename: the path to the old results file which contains the parameters
    :type filename: str
    :return: the new parameters
    :rtype: dict
    """
    time_profile_mappings = load_params("queue/sample_params.json")[
        "time_profile_mappings"
    ]

    with gzip.open(filename) as gzf:
        params = json.load(gzf)["params"]

    if "health_status_adkustment" in params:
        params.pop("health_status_adjustment")
        params.pop("life_expectancy")

    params["time_profile_mappings"] = time_profile_mappings

    efficiencies = params["efficiencies"]
    e_ip = efficiencies["ip"]
    for j in [
        "bads_daycase_occasional",
        "bads_daycase",
        "bads_outpatients",
        "bads_outpatients_or_daycase",
    ]:
        if not j in e_ip:
            continue

        e_ip[j].pop("op_dc_split")
        k = e_ip[j].pop("baseline_target_rate")
        e_ip[j]["interval"] = [
            1 - e_ip[j]["interval"][1] + k,
            1 - e_ip[j]["interval"][0] + k,
        ]
    e_op = efficiencies["op"]
    for j in [
        "convert_to_tele_adult_surgical",
        "convert_to_tele_adult_non-surgical",
        "convert_to_tele_child_surgical",
        "convert_to_tele_child_non-surgical",
    ]:
        if not j in e_op:
            continue
        e_op[j]["interval"] = [
            1 - e_op[j]["interval"][1],
            1 - e_op[j]["interval"][0],
        ]

    return params
