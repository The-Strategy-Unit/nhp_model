"""Data Migration

Scripts to migrate from a previous version of the model to a new version
"""

import gzip
import json

from migration.convert_all_files_in_folder import _convert_all_files_in_folder


def convert_all_files_in_folder(path: str) -> None:
    """Convert all files in a folder from v03 to v04

    :param path: the folder which contains the old results
    :type path: str
    """
    _convert_all_files_in_folder(path, "v0.6", v05_to_v06)


def v05_to_v06(filename: str) -> dict:
    """Convert v0.5 to v0.6 results

    :param filename: the path to the old results file which contains the parameters
    :type filename: str
    :return: the new parameters
    :rtype: dict
    """

    with gzip.open(filename) as gzf:
        params = json.load(gzf)["params"]

    # drop theatres
    params.drop("theatres")
    # drop a key that was sometimes incorrectly added
    params.drop("non_demographic_adjustment")
    # waiting list data needs to be dropped, this was recalculated in the inputs app
    params["waiting_list_adjustment"] = {"ip": {}, "op": {}}

    return params
