"""NHP Data Loaders

Classes for loading data for the NHP model. Each class supports loading data from different sources,
such as from local storage or directly from DataBricks.
"""

from typing import Any, Callable

import pandas as pd
import pyspark.sql.functions as F
from pyspark import SparkContext

from model.data import Data


class Databricks(Data):
    """Load NHP data from databricks"""

    def __init__(self, spark: SparkContext, fyear: int, dataset: str):
        self._spark = spark
        self._fyear = fyear
        self._dataset = dataset

    @staticmethod
    def create(spark: SparkContext) -> Callable[[int, str], Any]:
        """Create Databricks object

        :param spark: a SparkContext for selecting data
        :type spark: SparkContext
        :return: a function to initialise the object
        :rtype: Callable[[str, str], Databricks]
        """
        return lambda fyear, dataset: Databricks(spark, fyear, dataset)

    @property
    def _apc(self):
        return (
            self._spark.read.table("apc")
            .filter(F.col("provider") == self._dataset)
            .filter(F.col("fyear") == self._fyear)
            .withColumnRenamed("epikey", "rn")
        )

    def get_ip(self) -> pd.DataFrame:
        """Get the inpatients dataframe

        :return: the inpatients dataframe
        :rtype: pd.DataFrame
        """
        return self._apc.withColumn("tretspef_raw", F.col("tretspef")).toPandas()

    def get_ip_strategies(self) -> pd.DataFrame:
        """Get the inpatients strategies dataframe

        :return: the inpatients strategies dataframe
        :rtype: pd.DataFrame
        """
        mitigators = (
            self._spark.read.table("apc_mitigators")
            .withColumnRenamed("epikey", "rn")
            .join(self._apc, "rn", "semi")
        )

        return {
            k: mitigators.filter(F.col("type") == v).drop("type").toPandas()
            for k, v in [
                ("activity_avoidance", "activity_avoidance"),
                ("efficiencies", "efficiency"),
            ]
        }

    def get_op(self) -> pd.DataFrame:
        """Get the outpatients dataframe

        :return: the outpatients dataframe
        :rtype: pd.DataFrame
        """
        op = (
            self._spark.read.table("opa")
            .filter(F.col("provider") == self._dataset)
            .filter(F.col("fyear") == self._fyear)
            .toPandas()
        )
        return (
            op.sort_values(list(op.columns))
            .reset_index(drop=True)
            .reset_index()
            .rename(columns={"index": "rn"})
        )

    def get_aae(self) -> pd.DataFrame:
        """Get the A&E dataframe

        :return: the A&E dataframe
        :rtype: pd.DataFrame
        """
        aae = (
            self._spark.read.table("ecds")
            .filter(F.col("provider") == self._dataset)
            .filter(F.col("fyear") == self._fyear)
            .toPandas()
        )
        return (
            aae.sort_values(list(aae.columns))
            .reset_index(drop=True)
            .reset_index()
            .rename(columns={"index": "rn"})
        )

    def get_birth_factors(self) -> pd.DataFrame:
        """Get the birth factors dataframe

        :return: the birth factors dataframe
        :rtype: pd.DataFrame
        """

        return (
            self._spark.read.table("birth_factors")
            .filter(F.col("provider") == self._dataset)
            .drop("provider")
            .toPandas()
        )

    def get_demographic_factors(self) -> pd.DataFrame:
        """Get the demographic factors dataframe

        :return: the demographic factors dataframe
        :rtype: pd.DataFrame
        """

        return (
            self._spark.read.table("demographic_factors")
            .filter(F.col("provider") == self._dataset)
            .drop("provider")
            .toPandas()
        )

    def get_hsa_activity_table(self) -> pd.DataFrame:
        """Get the demographic factors dataframe

        :return: the demographic factors dataframe
        :rtype: pd.DataFrame
        """
        return (
            self._spark.read.table("hsa_activity_tables")
            .filter(F.col("provider") == self._dataset)
            .filter(F.col("fyear") == self._fyear)
            .drop("provider", "fyear")
            .toPandas()
        )

    def get_hsa_gams(self):
        """Get the health status adjustment gams"""
        # this is not supported in our data bricks environment currently
        raise NotImplementedError
