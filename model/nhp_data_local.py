"""NHP Data Loaders

Classes for loading data for the NHP model. Each class supports loading data from different sources,
such as from local storage or directly from DataBricks.
"""

import pickle
from typing import Any, Callable, Dict

import pandas as pd

from model.nhp_data import NHPData


class NHPDataLocal(NHPData):
    """Load NHP data from local storage"""

    def __init__(self, data_path: str, year: int, dataset: str):
        self._data_path = "/".join([data_path, str(year), dataset])

    @staticmethod
    def create(data_path: str) -> Callable[[int, str], Any]:
        """Create NHPDataLocal object

        :param data_path: the path to where the data is stored locally
        :type data_path: str
        :return: a function to initialise the object
        :rtype: Callable[[str, str], NHPDataLocal]
        """
        return lambda year, dataset: NHPDataLocal(data_path, year, dataset)

    def get_ip(self) -> pd.DataFrame:
        """Get the inpatients dataframe

        :return: the inpatients dataframe
        :rtype: pd.DataFrame
        """
        return self._get_parquet("ip.parquet")

    def get_ip_strategies(self) -> Dict[str, pd.DataFrame]:
        """Get the inpatients strategies dataframe

        :return: the inpatients strategies dataframe
        :rtype: pd.DataFrame
        """
        return {
            i: self._get_parquet(f"ip_{i}_strategies.parquet")
            for i in ["activity_avoidance", "efficiencies"]
        }

    def get_op(self) -> pd.DataFrame:
        """Get the outpatients dataframe

        :return: the outpatients dataframe
        :rtype: pd.DataFrame
        """
        return self._get_parquet("op.parquet")

    def get_aae(self) -> pd.DataFrame:
        """Get the A&E dataframe

        :return: the A&E dataframe
        :rtype: pd.DataFrame
        """
        return self._get_parquet("aae.parquet")

    def get_birth_factors(self) -> pd.DataFrame:
        """Get the birth factors dataframe

        :return: the birth factors dataframe
        :rtype: pd.DataFrame
        """
        return self._get_csv("birth_factors.csv")

    def get_demographic_factors(self) -> pd.DataFrame:
        """Get the demographic factors dataframe

        :return: the demographic factors dataframe
        :rtype: pd.DataFrame
        """
        return self._get_csv("demographic_factors.csv")

    def get_hsa_activity_table(self) -> pd.DataFrame:
        """Get the demographic factors dataframe

        :return: the demographic factors dataframe
        :rtype: pd.DataFrame
        """
        return self._get_csv("hsa_activity_table.csv")

    def get_hsa_gams(self):
        """Get the health status adjustment gams

        :return: the health status adjustment gams
        :rtype: _
        """
        with open(f"{self._data_path}/hsa_gams.pkl", "rb") as hsa_pkl:
            return pickle.load(hsa_pkl)

    def _get_parquet(self, file) -> pd.DataFrame:
        """_summary_

        :param file: _description_
        :type file: _type_
        :return: _description_
        :rtype: pd.DataFrame
        """
        return pd.read_parquet(f"{self._data_path}/{file}")

    def _get_csv(self, file) -> pd.DataFrame:
        """_summary_

        :param file: _description_
        :type file: _type_
        :return: _description_
        :rtype: pd.DataFrame
        """
        return pd.read_csv(f"{self._data_path}/{file}")
