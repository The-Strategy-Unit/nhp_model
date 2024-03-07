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
    _convert_all_files_in_folder(path, "v1.0", v06_to_v10)


def v06_to_v10(filename: str) -> dict:
    """Convert v0.6 to v1.0 results

    :param filename: the path to the old results file which contains the parameters
    :type filename: str
    :return: the new parameters
    :rtype: dict
    """

    with gzip.open(filename) as gzf:
        params = json.load(gzf)["params"]

    params["app_version"] = "v1.0"
    params["time_profile_mappings"]["non-demographic_adjustment"] = "none"

    return params
