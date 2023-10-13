"""Data Migration

Scripts to migrate from a previous version of the model to a new version
"""

import gzip
import json

from migration.convert_all_files_in_folder import _convert_all_files_in_folder
from model.helpers import load_params


def convert_all_files_in_folder(path: str) -> None:
    """Convert all files in a folder from v03 to v04

    :param path: the folder which contains the old results
    :type path: str
    """
    _convert_all_files_in_folder(path, "v0.4", v03_to_v04)


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
