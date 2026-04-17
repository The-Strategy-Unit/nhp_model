"""Reference Data.

Any reference data needed for the model should be stored in this folder.

Helper methods for loading the reference data should be created here.
"""

import json
import pathlib

import pandas as pd
from metalog_jax.base import MetalogParameters, MetalogRandomVariableParameters
from metalog_jax.metalog import Metalog
from metalog_jax.utils import JaxUniformDistributionParameters


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


def life_expectancy(base_year: int, target_year: int) -> pd.Series:
    """Life Expectancy (Health Status Adjustment).

    Args:
        base_year: The baseline year for the model.
        target_year: The year the model is running for.

    Returns:
        A pandas DataFrame containing life expectancy data.
    """
    return (
        pd.read_csv(_ref_path("life_expectancy.csv"))
        .set_index(["year", "sex", "age"])
        .loc[([base_year, target_year]), "ex"]
        # ensure index is sorted to remove PerformanceWarning from pandas when indexing with .loc
        .sort_index()
    )


def hsa_metalog_parameters(target_year: int) -> dict[int, Metalog]:
    """Health Status Adjustment Metalog Parameters.

    Args:
        target_year: The target year for the metalog parameters.

    Returns:
        A dictionary containing the health status adjustment metalog parameters.
    """

    def sex_to_int(sex: str) -> int:
        match sex:
            case "m":
                return 1
            case "f":
                return 2
            case _:
                raise ValueError(f"Invalid sex: {sex}")

    with _ref_path("hsa_metalog_parameters.json").open("r", encoding="UTF-8") as f:
        return {
            sex_to_int(k): Metalog(
                MetalogParameters(**v["metalog_params"]),
                v["a"],
            )
            for k, v in json.load(f)[str(target_year)].items()
        }
