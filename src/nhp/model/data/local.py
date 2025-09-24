"""NHP Data Loaders.

Classes for loading data for the NHP model. Each class supports loading data from different sources,
such as from local storage or directly from DataBricks.
"""

import pickle
from typing import Any, Callable, Dict

import pandas as pd

from nhp.model.data import Data


class Local(Data):
    """Load NHP data from local storage."""

    def __init__(self, data_path: str, year: int, dataset: str):
        """Initialise Local data loader class."""
        self._data_path = data_path
        self._year = str(year)
        self._dataset = dataset

    def _file_path(self, file):
        return "/".join([self._data_path, file, f"fyear={self._year}", f"dataset={self._dataset}"])

    @staticmethod
    def create(data_path: str) -> Callable[[int, str], Any]:
        """Create Local Data object.

        :param data_path: the path to where the data is stored locally
        :type data_path: str
        :return: a function to initialise the object
        :rtype: Callable[[str, str], Local]
        """
        return lambda year, dataset: Local(data_path, year, dataset)

    def get_ip(self) -> pd.DataFrame:
        """Get the inpatients dataframe.

        :return: the inpatients dataframe
        :rtype: pd.DataFrame
        """
        return self._get_parquet("ip")

    def get_ip_strategies(self) -> dict[str, pd.DataFrame]:
        """Get the inpatients strategies dataframe.

        :return: the inpatients strategies dataframes
        :rtype: dict[str, pd.DataFrame]
        """
        return {
            i: self._get_parquet(f"ip_{i}_strategies")
            for i in ["activity_avoidance", "efficiencies"]
        }

    def get_op(self) -> pd.DataFrame:
        """Get the outpatients dataframe.

        :return: the outpatients dataframe
        :rtype: pd.DataFrame
        """
        return self._get_parquet("op").rename(columns={"index": "rn"})

    def get_aae(self) -> pd.DataFrame:
        """Get the A&E dataframe.

        :return: the A&E dataframe
        :rtype: pd.DataFrame
        """
        return self._get_parquet("aae").rename(columns={"index": "rn"})

    def get_birth_factors(self) -> pd.DataFrame:
        """Get the birth factors dataframe.

        :return: the birth factors dataframe
        :rtype: pd.DataFrame
        """
        return self._get_parquet("birth_factors")

    def get_demographic_factors(self) -> pd.DataFrame:
        """Get the demographic factors dataframe.

        :return: the demographic factors dataframe
        :rtype: pd.DataFrame
        """
        return self._get_parquet("demographic_factors")

    def get_hsa_activity_table(self) -> pd.DataFrame:
        """Get the demographic factors dataframe.

        :return: the demographic factors dataframe
        :rtype: pd.DataFrame
        """
        return self._get_parquet("hsa_activity_tables")

    def get_hsa_gams(self):
        """Get the health status adjustment gams.

        :return: the health status adjustment gams
        :rtype: _
        """
        with open(f"{self._data_path}/hsa_gams.pkl", "rb") as hsa_pkl:
            return pickle.load(hsa_pkl)

    def get_inequalities(self):
        """Get the inequalities dataframe.

        :return: the inequalities dataframe
        :rtype: pd.DataFrame
        """
        return self._get_parquet("inequalities")

    def _get_parquet(self, file) -> pd.DataFrame:
        """Load specific parquet file using Pandas.

        :param file: Specific parquet filename to open
        :type file: str
        :return: DataFrame containing the data
        :rtype: pd.DataFrame
        """
        inequalities_df = pd.read_parquet(self._file_path(file))
        inequalities_df["provider"] = self._dataset  # need provider for PBM
        return inequalities_df
