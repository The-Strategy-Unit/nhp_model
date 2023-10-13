"""Data Migration

Scripts to migrate from a previous version of the model to a new version
"""

import os
from typing import Callable

from tqdm.auto import tqdm

import run_model as rm
from migration.azure_storage import upload_results


def _convert_all_files_in_folder(
    path: str, dest: str, migration: Callable[[str], dict]
) -> None:
    """Convert all files in a folder from one version to another

    :param path: the folder which contains the old results
    :type path: str
    :param dest: the destination to save new results to in azure
    :type dest: str
    :param migration: the migration function to call
    :type migration: Callable[[str], dict]
    """
    files = [
        f"{path}/{d}/{i}" for d in os.listdir(path) for i in os.listdir(f"{path}/{d}")
    ]
    for i in (pbar := tqdm(files, position=0)):
        pbar.set_description(i)
        params = migration(i)

        results_file = rm.run_all(params, "data", lambda: lambda _: None)

        metadata = {
            k: str(v)
            for k, v in params.items()
            if not isinstance(v, dict) and not isinstance(v, list)
        }

        upload_results(results_file, dest, metadata)

        os.remove(i)
        os.remove(f"results/{results_file}.json")
