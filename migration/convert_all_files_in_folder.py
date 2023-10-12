"""Data Migration

Scripts to migrate from a previous version of the model to a new version
"""

import os
from typing import Callable

from tqdm.auto import tqdm

import run_model as rm
from migration.azure_storage import upload_results


def convert_all_files_in_folder(path: str, migration: Callable[[str], dict]) -> None:
    """Convert all files in a folder from v03 to v04

    :param path: the folder which contains the old results
    :type path: str
    :param migration: the migration function to call
    :type migration: Callable[[str], dict]
    """
    files = [
        f"{path}/{d}/{i}" for d in os.listdir(path) for i in os.listdir(f"{path}/{d}")
    ]
    for i in tqdm(files, position=0):
        params = migration(i)

        results_file = rm.run_all(params, "data", lambda: lambda _: None)

        metadata = {
            k: str(v)
            for k, v in params.items()
            if not isinstance(v, dict) and not isinstance(v, list)
        }

        upload_results(results_file, "v0.4", metadata)

        os.remove(i)
        os.remove(f"results/{results_file}.json")
