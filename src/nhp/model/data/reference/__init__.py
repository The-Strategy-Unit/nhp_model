"""Reference Data.

Any reference data needed for the model should be stored in this folder.

Helper methods for loading the reference data should be created here.
"""

import json
import pathlib

import pandas as pd


def _ref_path(filename):
    path = pathlib.Path(__file__).parent.resolve()
    return path.joinpath(filename)


def variant_lookup() -> dict:
    """Variant Lookup (Health Status Adjustment).

    Returns:
        A dictionary of the variant lookups.
    """
    with _ref_path("variant_lookup.json").open("r", encoding="UTF-8") as vlup_file:
        return json.load(vlup_file)


def life_expectancy() -> pd.DataFrame:
    """Life Expectancy (Health Status Adjustment).

    Returns:
        A pandas DataFrame containing life expectancy data.
    """
    return pd.read_csv(_ref_path("life_expectancy.csv"))


def split_normal_params() -> pd.DataFrame:
    """Split Normal Parameters (Health Status Adjustment).

    Returns:
        A pandas DataFrame containing split normal parameters.
    """
    return pd.read_csv(_ref_path("hsa_split_normal_params.csv"))
