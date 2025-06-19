"""NHP Data Loaders

Classes for loading data for the NHP model. Each class supports loading data from different sources,
such as from local storage or directly from DataBricks.
"""

from typing import Any, Callable

import pandas as pd
import pyspark.sql.functions as F
from pyspark import SparkContext

from nhp.model.data import Data


class Databricks(Data):
    """Load NHP data from databricks"""

    def __init__(self, spark: SparkContext, data_path: str, year: int, dataset: str):
        self._spark = spark
        self._data_path = data_path
        self._year = year
        self._dataset = dataset

    @staticmethod
    def create(spark: SparkContext, data_path: str) -> Callable[[int, str], Any]:
        """Create Databricks object

        :param spark: a SparkContext for selecting data
        :type spark: SparkContext
        :param data_path: the path to where the parquet files are stored
        :type data_path: str
        :return: a function to initialise the object
        :rtype: Callable[[str, str], Databricks]
        """
        return lambda fyear, dataset: Databricks(spark, data_path, fyear, dataset)

    @property
    def _apc(self):
        return (
            self._spark.read.parquet(f"{self._data_path}/ip")
            .filter(F.col("dataset") == self._dataset)
            .filter(F.col("fyear") == self._year)
            .persist()
        )

    def get_ip(self) -> pd.DataFrame:
        """Get the inpatients dataframe

        :return: the inpatients dataframe
        :rtype: pd.DataFrame
        """
        return self._apc.toPandas()

    def get_ip_strategies(self) -> pd.DataFrame:
        """Get the inpatients strategies dataframe

        :return: the inpatients strategies dataframe
        :rtype: pd.DataFrame
        """
        return {
            k: self._spark.read.parquet(f"{self._data_path}/ip_{k}_strategies")
            .filter(F.col("dataset") == self._dataset)
            .filter(F.col("fyear") == self._year)
            .join(self._apc, "rn", "semi")
            .toPandas()
            for k in ["activity_avoidance", "efficiencies"]
        }

    def get_op(self) -> pd.DataFrame:
        """Get the outpatients dataframe

        :return: the outpatients dataframe
        :rtype: pd.DataFrame
        """
        return (
            self._spark.read.parquet(f"{self._data_path}/op")
            .filter(F.col("dataset") == self._dataset)
            .filter(F.col("fyear") == self._year)
            .withColumnRenamed("index", "rn")
            .toPandas()
        )

    def get_aae(self) -> pd.DataFrame:
        """Get the A&E dataframe

        :return: the A&E dataframe
        :rtype: pd.DataFrame
        """
        return (
            self._spark.read.parquet(f"{self._data_path}/aae")
            .filter(F.col("dataset") == self._dataset)
            .filter(F.col("fyear") == self._year)
            .withColumnRenamed("index", "rn")
            .toPandas()
        )

    def get_birth_factors(self) -> pd.DataFrame:
        """Get the birth factors dataframe

        :return: the birth factors dataframe
        :rtype: pd.DataFrame
        """

        return (
            self._spark.read.parquet(f"{self._data_path}/birth_factors")
            .filter(F.col("dataset") == self._dataset)
            .filter(F.col("fyear") == self._year)
            .drop("dataset")
            .toPandas()
        )

    def get_demographic_factors(self) -> pd.DataFrame:
        """Get the demographic factors dataframe

        :return: the demographic factors dataframe
        :rtype: pd.DataFrame
        """

        return (
            self._spark.read.parquet(f"{self._data_path}/demographic_factors")
            .filter(F.col("dataset") == self._dataset)
            .filter(F.col("fyear") == self._year)
            .drop("dataset")
            .toPandas()
        )

    def get_hsa_activity_table(self) -> pd.DataFrame:
        """Get the demographic factors dataframe

        :return: the demographic factors dataframe
        :rtype: pd.DataFrame
        """
        return (
            self._spark.read.parquet(f"{self._data_path}/hsa_activity_tables")
            .filter(F.col("dataset") == self._dataset)
            .filter(F.col("fyear") == self._year)
            .drop("dataset", "fyear")
            .toPandas()
        )

    def get_hsa_gams(self):
        """Get the health status adjustment gams"""
        # this is not supported in our data bricks environment currently
        raise NotImplementedError


class DatabricksNational(Data):
    """Load NHP data from databricks"""

    def __init__(
        self,
        spark: SparkContext,
        data_path: str,
        year: int,
        sample_rate: float,
        seed: int,
    ):
        self._spark = spark
        self._data_path = data_path
        self._year = year
        self._sample_rate = sample_rate
        self._seed = seed

        self._apc = (
            self._spark.read.parquet(f"{self._data_path}/ip")
            .filter(F.col("fyear") == self._year)
            .withColumn("dataset", F.lit("NATIONAL"))
            .withColumn("sitetret", F.lit("NATIONAL"))
            .sample(fraction=self._sample_rate, seed=self._seed)
            .persist()
        )

    @staticmethod
    def create(
        spark: SparkContext, data_path: str, sample_rate: float, seed: int
    ) -> Callable[[int, str], Any]:
        """Create Databricks object

        :param spark: a SparkContext for selecting data
        :type spark: SparkContext
        :param data_path: the path to where the parquet files are stored
        :type data_path: str
        :param sample_rate: the rate to sample inpatient data at
        :type sample_rate: float
        :return: a function to initialise the object
        :rtype: Callable[[str, str], Databricks]
        """
        return lambda fyear, _: DatabricksNational(
            spark, data_path, fyear, sample_rate, seed
        )

    def get_ip(self) -> pd.DataFrame:
        """Get the inpatients dataframe

        :return: the inpatients dataframe
        :rtype: pd.DataFrame
        """
        return self._apc.toPandas()

    def get_ip_strategies(self) -> pd.DataFrame:
        """Get the inpatients strategies dataframe

        :return: the inpatients strategies dataframe
        :rtype: pd.DataFrame
        """
        return {
            k: self._spark.read.parquet(f"{self._data_path}/ip_{k}_strategies")
            .filter(F.col("fyear") == self._year)
            .join(self._apc, "rn", "semi")
            .toPandas()
            for k in ["activity_avoidance", "efficiencies"]
        }

    def get_op(self) -> pd.DataFrame:
        """Get the outpatients dataframe

        :return: the outpatients dataframe
        :rtype: pd.DataFrame
        """
        op = self._spark.read.parquet(f"{self._data_path}/op")

        return (
            self._spark.read.parquet(f"{self._data_path}/op")
            .filter(F.col("fyear") == self._year)
            .withColumn("dataset", F.lit("NATIONAL"))
            .withColumn("sitetret", F.lit("NATIONAL"))
            # TODO: temporary fix, see #353
            .withColumn("sushrg_trimmed", F.lit("HRG"))
            .withColumn("imd_quintile", F.lit(0))
            .groupBy(op.drop("index", "fyear", "attendances", "tele_attendances").columns)
            .agg(
                (F.sum("attendances") * self._sample_rate).alias("attendances"),
                (F.sum("tele_attendances") * self._sample_rate).alias("tele_attendances"),
            )
            # TODO: how do we make this stable? at the moment we can't use full model results with
            # national
            .withColumn("rn", F.expr("uuid()"))
            .toPandas()
        )

    def get_aae(self) -> pd.DataFrame:
        """Get the A&E dataframe

        :return: the A&E dataframe
        :rtype: pd.DataFrame
        """
        aae = self._spark.read.parquet(f"{self._data_path}/aae")

        return (
            self._spark.read.parquet(f"{self._data_path}/aae")
            .filter(F.col("fyear") == self._year)
            .withColumn("dataset", F.lit("NATIONAL"))
            .withColumn("sitetret", F.lit("NATIONAL"))
            .groupBy(aae.drop("index", "fyear", "arrivals").columns)
            .agg((F.sum("arrivals") * self._sample_rate).alias("arrivals"))
            # TODO: how do we make this stable? at the moment we can't use full model results with
            # national
            .withColumn("rn", F.expr("uuid()"))
            .toPandas()
        )

    def get_birth_factors(self) -> pd.DataFrame:
        """Get the birth factors dataframe

        :return: the birth factors dataframe
        :rtype: pd.DataFrame
        """

        return (
            self._spark.read.table("nhp.population_projections.births")
            .filter(F.col("area_code").rlike("^E0[6-9]"))
            .withColumn("sex", F.lit(2))
            .groupBy("projection", "age", "sex")
            .pivot("year")
            .agg(F.sum("value").alias("value"))
            .withColumnRenamed("projection", "variant")
            .toPandas()
        )

    def get_demographic_factors(self) -> pd.DataFrame:
        """Get the demographic factors dataframe

        :return: the demographic factors dataframe
        :rtype: pd.DataFrame
        """

        return (
            self._spark.read.table("nhp.population_projections.demographics")
            .filter(F.col("area_code").rlike("^E0[6-9]"))
            .groupBy("projection", "age", "sex")
            .pivot("year")
            .agg(F.sum("value").alias("value"))
            .withColumnRenamed("projection", "variant")
            .toPandas()
        )

    def get_hsa_activity_table(self) -> pd.DataFrame:
        """Get the demographic factors dataframe

        :return: the demographic factors dataframe
        :rtype: pd.DataFrame
        """
        return (
            self._spark.read.table("nhp.default.hsa_activity_tables_national")
            .filter(F.col("fyear") == self._year * 100 + (self._year + 1) % 100)
            .groupBy("hsagrp", "sex", "age")
            .agg(F.mean("activity").alias("activity"))
            .toPandas()
        )

    def get_hsa_gams(self):
        """Get the health status adjustment gams"""
        # this is not supported in our data bricks environment currently
        raise NotImplementedError
