"""NHP Data Loaders.

Classes for loading data for the NHP model. Each class supports loading data from different sources,
such as from local storage or directly from DataBricks.
"""

from typing import Any, Callable

import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql import SparkSession

from nhp.model.data import Data


class Databricks(Data):
    """Load NHP data from databricks."""

    def __init__(self, spark: SparkSession, data_path: str, year: int, dataset: str):
        """Initialise Databricks data loader class."""
        self._spark = spark
        self._data_path = data_path
        self._year = year
        self._dataset = dataset

    @staticmethod
    def create(spark: SparkSession, data_path: str) -> Callable[[int, str], Any]:
        """Create Databricks object.

        Args:
            spark: A SparkSession for selecting data.
            data_path: The path to where the parquet files are stored.

        Returns:
            A function to initialise the object.
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
        """Get the inpatients dataframe.

        Returns:
            The inpatients dataframe.
        """
        return self._apc.toPandas()

    def get_ip_strategies(self) -> dict[str, pd.DataFrame]:
        """Get the inpatients strategies dataframe.

        Returns:
            The inpatients strategies dataframes.
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
        """Get the outpatients dataframe.

        Returns:
            The outpatients dataframe.
        """
        return (
            self._spark.read.parquet(f"{self._data_path}/op")
            .filter(F.col("dataset") == self._dataset)
            .filter(F.col("fyear") == self._year)
            .withColumnRenamed("index", "rn")
            .toPandas()
        )

    def get_aae(self) -> pd.DataFrame:
        """Get the A&E dataframe.

        Returns:
            The A&E dataframe.
        """
        return (
            self._spark.read.parquet(f"{self._data_path}/aae")
            .filter(F.col("dataset") == self._dataset)
            .filter(F.col("fyear") == self._year)
            .withColumnRenamed("index", "rn")
            .toPandas()
        )

    def get_birth_factors(self) -> pd.DataFrame:
        """Get the birth factors dataframe.

        Returns:
            The birth factors dataframe.
        """
        return (
            self._spark.read.parquet(f"{self._data_path}/birth_factors")
            .filter(F.col("dataset") == self._dataset)
            .filter(F.col("fyear") == self._year)
            .drop("dataset")
            .toPandas()
        )

    def get_demographic_factors(self) -> pd.DataFrame:
        """Get the demographic factors dataframe.

        Returns:
            The demographic factors dataframe.
        """
        return (
            self._spark.read.parquet(f"{self._data_path}/demographic_factors")
            .filter(F.col("dataset") == self._dataset)
            .filter(F.col("fyear") == self._year)
            .drop("dataset")
            .toPandas()
        )

    def get_hsa_activity_table(self) -> pd.DataFrame:
        """Get the demographic factors dataframe.

        Returns:
            The demographic factors dataframe.
        """
        return (
            self._spark.read.parquet(f"{self._data_path}/hsa_activity_tables")
            .filter(F.col("dataset") == self._dataset)
            .filter(F.col("fyear") == self._year)
            .drop("dataset", "fyear")
            .toPandas()
        )

    def get_hsa_gams(self):
        """Get the health status adjustment gams.

        Raises:
            NotImplementedError: This is not supported in our databricks environment currently.
        """
        # this is not supported in our data bricks environment currently
        raise NotImplementedError

    def get_inequalities(self) -> pd.DataFrame:
        """Get the inequalities dataframe.

        Returns:
            The inequalities dataframe.
        """
        return (
            self._spark.read.table("nhp.default.inequalities")
            .filter(F.col("fyear") == self._year * 100 + (self._year + 1) % 100)
            .toPandas()
        )


class DatabricksNational(Data):
    """Load NHP data from databricks."""

    def __init__(
        self,
        spark: SparkSession,
        data_path: str,
        year: int,
        sample_rate: float,
        seed: int,
    ):
        """Initialise DatabricksNational data loader class."""
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
        spark: SparkSession, data_path: str, sample_rate: float, seed: int
    ) -> Callable[[int, str], Any]:
        """Create Databricks object.

        Args:
            spark: A SparkSession for selecting data.
            data_path: The path to where the parquet files are stored.
            sample_rate: The rate to sample inpatient data at.
            seed: The random seed for sampling.

        Returns:
            A function to initialise the object.
        """
        return lambda fyear, _: DatabricksNational(spark, data_path, fyear, sample_rate, seed)

    def get_ip(self) -> pd.DataFrame:
        """Get the inpatients dataframe.

        Returns:
            The inpatients dataframe.
        """
        return self._apc.toPandas()

    def get_ip_strategies(self) -> dict[str, pd.DataFrame]:
        """Get the inpatients strategies dataframe.

        Returns:
            The inpatients strategies dataframes.
        """
        return {
            k: self._spark.read.parquet(f"{self._data_path}/ip_{k}_strategies")
            .filter(F.col("fyear") == self._year)
            .join(self._apc, "rn", "semi")
            .toPandas()
            for k in ["activity_avoidance", "efficiencies"]
        }

    def get_op(self) -> pd.DataFrame:
        """Get the outpatients dataframe.

        Returns:
            The outpatients dataframe.
        """
        op = self._spark.read.parquet(f"{self._data_path}/op")

        return (
            self._spark.read.parquet(f"{self._data_path}/op")
            .filter(F.col("fyear") == self._year)
            .withColumn("dataset", F.lit("NATIONAL"))
            .withColumn("sitetret", F.lit("NATIONAL"))
            .withColumn("icb", F.lit("NATIONAL"))
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
        """Get the A&E dataframe.

        Returns:
            The A&E dataframe.
        """
        aae = self._spark.read.parquet(f"{self._data_path}/aae")

        return (
            self._spark.read.parquet(f"{self._data_path}/aae")
            .filter(F.col("fyear") == self._year)
            .withColumn("dataset", F.lit("NATIONAL"))
            .withColumn("sitetret", F.lit("NATIONAL"))
            .withColumn("icb", F.lit("NATIONAL"))
            .groupBy(aae.drop("index", "fyear", "arrivals").columns)
            .agg((F.sum("arrivals") * self._sample_rate).alias("arrivals"))
            # TODO: how do we make this stable? at the moment we can't use full model results with
            # national
            .withColumn("rn", F.expr("uuid()"))
            .toPandas()
        )

    def get_birth_factors(self) -> pd.DataFrame:
        """Get the birth factors dataframe.

        Returns:
            The birth factors dataframe.
        """
        births_df = (
            self._spark.read.parquet(f"{self._data_path}/birth_factors/")
            .filter(F.col("fyear") == self._year)
            .filter(~F.col("variant").startswith("custom"))
        )

        years = [i for i in births_df.columns if i.startswith("20")]
        years_str = ", ".join([f"'{i}', `{i}`" for i in years])
        expr = f"stack({len(years)}, {years_str}) as (year, value)"

        return (
            births_df.selectExpr("variant", "age", "sex", expr)  # noqa: PD010
            .groupBy("variant", "age", "sex")
            .pivot("year")
            .agg(F.sum("value"))
            .orderBy("variant", "age", "sex")
            .withColumnRenamed("projection", "variant")
            .toPandas()
        )

    def get_demographic_factors(self) -> pd.DataFrame:
        """Get the demographic factors dataframe.

        Returns:
            The demographic factors dataframe.
        """
        demog_df = (
            self._spark.read.parquet(f"{self._data_path}/demographic_factors/")
            .filter(F.col("fyear") == self._year)
            .filter(~F.col("variant").startswith("custom"))
        )

        years = [i for i in demog_df.columns if i.startswith("20")]
        years_str = ", ".join([f"'{i}', `{i}`" for i in years])
        expr = f"stack({len(years)}, {years_str}) as (year, value)"

        return (
            demog_df.selectExpr("variant", "age", "sex", expr)  # noqa: PD010
            .groupBy("variant", "age", "sex")
            .pivot("year")
            .agg(F.sum("value"))
            .orderBy("variant", "age", "sex")
            .withColumnRenamed("projection", "variant")
            .toPandas()
        )

    def get_hsa_activity_table(self) -> pd.DataFrame:
        """Get the demographic factors dataframe.

        Returns:
            The demographic factors dataframe.
        """
        return (
            self._spark.read.table("nhp.default.hsa_activity_tables_national")
            .filter(F.col("fyear") == self._year * 100 + (self._year + 1) % 100)
            .groupBy("hsagrp", "sex", "age")
            .agg(F.mean("activity").alias("activity"))
            .toPandas()
        )

    def get_hsa_gams(self):
        """Get the health status adjustment gams.

        Raises:
            NotImplementedError: This is not supported in our databricks environment currently.
        """
        # this is not supported in our data bricks environment currently
        raise NotImplementedError

    def get_inequalities(self) -> pd.DataFrame:
        """Get the inequalities dataframe.

        Returns:
            The inequalities dataframe.
        """
        return (
            self._spark.read.table("nhp.default.inequalities")
            .filter(F.col("fyear") == self._year * 100 + (self._year + 1) % 100)
            .toPandas()
        )
