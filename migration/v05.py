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


def _get_mean_nda_param(values):
    mults = {
        " 0- 4": 5,
        " 5-14": 10,
        "15-34": 20,
        "35-49": 15,
        "50-64": 15,
        "65-84": 20,
        "85+": 15,
    }
    return [
        sum(values.get(k, [1, 1])[i] * m for k, m in mults.items())
        / sum(mults.values())
        for i in [0, 1]
    ]


def v04_to_v05(filename: str) -> dict:
    """Convert v0.4 to v0.5 results

    :param filename: the path to the old results file which contains the parameters
    :type filename: str
    :return: the new parameters
    :rtype: dict
    """

    with gzip.open(filename) as gzf:
        params = json.load(gzf)["params"]

    params["app_version"] = "v0.5"

    if "non-demographic_adjustment" in params:
        p = params.pop("non-demographic_adjustment")
        params["non-demographic_adjustment"] = {
            "ip": {
                k: _get_mean_nda_param(p[k])
                for k in ["non-elective", "elective", "maternity"]
                if p[k] != {}
            },
            "op": {},
            "aae": {},
        }

    return params
