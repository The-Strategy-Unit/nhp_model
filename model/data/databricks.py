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
        self._fyear = fyear * 100 + (fyear + 1) % 100
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
            .withColumn("sex", F.col("sex").cast("int"))
            .persist()
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
            .select("rn", "type", "strategy", "sample_rate")
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
        return (
            self._spark.read.table("opa")
            .filter(F.col("provider") == self._dataset)
            .filter(F.col("fyear") == self._fyear)
            .withColumn("tretspef_raw", F.col("tretspef"))
            .withColumn("is_wla", F.lit(True))
            .toPandas()
            .rename(columns={"index": "rn"})
        )

    def get_aae(self) -> pd.DataFrame:
        """Get the A&E dataframe

        :return: the A&E dataframe
        :rtype: pd.DataFrame
        """
        return (
            self._spark.read.table("ecds")
            .filter(F.col("provider") == self._dataset)
            .filter(F.col("fyear") == self._fyear)
            .toPandas()
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


class DatabricksNational(Data):
    """Load NHP data from databricks"""

    def __init__(self, spark: SparkContext, fyear: int, sample_rate: float, seed: int):
        self._spark = spark
        self._fyear = fyear * 100 + (fyear + 1) % 100
        self._sample_rate = sample_rate
        self._seed = seed

        self._apc = (
            self._spark.read.table("apc")
            .filter(F.col("fyear") == self._fyear)
            .withColumnRenamed("epikey", "rn")
            .withColumn("sex", F.col("sex").cast("int"))
            .withColumn("provider", F.lit("NATIONAL"))
            .withColumn("sitetret", F.lit("NATIONAL"))
            .drop("fyear")
            .sample(fraction=self._sample_rate, seed=self._seed)
            .persist()
        )

    @staticmethod
    def create(
        spark: SparkContext, sample_rate: float, seed: int
    ) -> Callable[[int, str], Any]:
        """Create Databricks object

        :param spark: a SparkContext for selecting data
        :type spark: SparkContext
        :param sample_rate: the rate to sample inpatient data at
        :type sample_rate: float
        :return: a function to initialise the object
        :rtype: Callable[[str, str], Databricks]
        """
        return lambda fyear, _: DatabricksNational(spark, fyear, sample_rate, seed)

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
            .select("rn", "type", "strategy", "sample_rate")
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
        op = self._spark.read.table("opa")

        return (
            op.filter(F.col("fyear") == self._fyear)
            .withColumn("provider", F.lit("NATIONAL"))
            .withColumn("sitetret", F.lit("NATIONAL"))
            .withColumn("sex", F.col("sex").cast("int"))
            .groupBy(
                op.drop("index", "fyear", "attendances", "tele_attendances").columns
            )
            .agg(
                (F.sum("attendances") * self._sample_rate).alias("attendances"),
                (F.sum("tele_attendances") * self._sample_rate).alias(
                    "tele_attendances"
                ),
            )
            .withColumn("tretspef_raw", F.col("tretspef"))
            .withColumn("is_wla", F.lit(True))
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
        ecds = self._spark.read.table("ecds")

        return (
            ecds.filter(F.col("fyear") == self._fyear)
            .withColumn("provider", F.lit("NATIONAL"))
            .withColumn("sitetret", F.lit("NATIONAL"))
            .withColumn("sex", F.col("sex").cast("int"))
            .groupBy(ecds.drop("index", "fyear", "arrivals").columns)
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
            self._spark.read.parquet(
                "/Volumes/su_data/nhp/population-projections/birth_data"
            )
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
            self._spark.read.parquet(
                "/Volumes/su_data/nhp/population-projections/demographic_data"
            )
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
            self._spark.read.table("hsa_activity_tables")
            .filter(F.col("fyear") == self._fyear)
            .groupBy("hsagrp", "sex", "age")
            .agg(F.mean("activity").alias("activity"))
            .toPandas()
        )

    def get_hsa_gams(self):
        """Get the health status adjustment gams"""
        # this is not supported in our data bricks environment currently
        raise NotImplementedError
