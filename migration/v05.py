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
    _convert_all_files_in_folder(path, "v0.5", v04_to_v05)



def v04_to_v05(filename: str) -> dict:
    """Convert v0.4 to v0.5 results

    :param filename: the path to the old results file which contains the parameters
    :type filename: str
    :return: the new parameters
    :rtype: dict
    """

    with gzip.open(filename) as gzf:
        params = json.load(gzf)["params"]

    if "non-demographic_adjustment" in params:
        params.pop("non-demographic_adjustment")

    return params
