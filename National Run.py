# Databricks notebook source
import os
import json
os.makedirs("/tmp/data/2019/national", exist_ok=True)

os.environ["BATCH_SIZE"] = "8"

# COMMAND ----------

import model as mdl
import pandas as pd
import pyspark.sql.functions as F

from model.health_status_adjustment import HealthStatusAdjustmentInterpolated

from run_model import _combine_results, _run_model
from model.data.local import Local

# COMMAND ----------

spark.catalog.setCurrentCatalog("su_data")
spark.catalog.setCurrentDatabase("nhp")

# COMMAND ----------

"""NHP Data Loaders

Classes for loading data for the NHP model. Each class supports loading data from different sources,
such as from local storage or directly from DataBricks.
"""

from typing import Any, Callable

import pandas as pd
import pyspark.sql.functions as F
from pyspark import SparkContext

from model.data import Data

class DatabricksNational(Data):
    """Load NHP data from databricks"""

    def __init__(self, spark: SparkContext, fyear: int, sample_rate: float):
        self._spark = spark
        self._fyear = fyear * 100 + (fyear + 1) % 100
        self._sample_rate = sample_rate
        # TODO: currently the demographic datasets are only created at provider levels, need to load a specific provider in
        self._dataset = "R0A"

    @staticmethod
    def create(spark: SparkContext, sample_rate: float) -> Callable[[int, str], Any]:
        """Create Databricks object

        :param spark: a SparkContext for selecting data
        :type spark: SparkContext
        :param sample_rate: the rate to sample inpatient data at
        :type sample_rate: float
        :return: a function to initialise the object
        :rtype: Callable[[str, str], Databricks]
        """
        return lambda fyear, _: DatabricksNational(spark, fyear, sample_rate)

    @property
    def _apc(self):
        return (
            self._spark.read.table("apc")
            .filter(F.col("fyear") == self._fyear)
            .withColumnRenamed("epikey", "rn")
            .withColumn("sex", F.col("sex").cast("int"))
            .withColumn("provider", F.lit("NATIONAL"))
            .withColumn("sitetret", F.lit("NATIONAL"))
            .drop("fyear")
            .sample(fraction=self._sample_rate)
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
        op = (
            self._spark.read.table("opa")
            .filter(F.col("fyear") == self._fyear)
            .withColumn("provider", F.lit("NATIONAL"))
            .withColumn("sitetret", F.lit("NATIONAL"))
            .withColumn("sex", F.col("sex").cast("int"))
            .groupBy(
                "provider",
                "sitetret",
                "age",
                "sex",
                "tretspef",
                "has_procedures",
                "is_main_icb",
                "is_surgical_specialty",
                "is_adult",
                "is_gp_ref",
                "is_cons_cons_ref",
                "is_first",
                "type",
                "group",
                "hsagrp"
            )
            .agg(
                (F.sum("attendances") * self._sample_rate).alias("attendances"),
                (F.sum("tele_attendances") * self._sample_rate).alias("tele_attendances")
            )
            .withColumn("tretspef_raw", F.col("tretspef"))
            .withColumn("is_wla", F.lit(True))
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
            .filter(F.col("fyear") == self._fyear)
            .withColumn("provider", F.lit("NATIONAL"))
            .withColumn("sitetret", F.lit("NATIONAL"))
            .withColumn("sex", F.col("sex").cast("int"))
            .groupBy(
                "provider",
                "age",
                "sex",
                "sitetret",
                "aedepttype",
                "attendance_category",
                "is_main_icb",
                "is_ambulance",
                "is_frequent_attender",
                "is_low_cost_referred_or_discharged",
                "is_left_before_treatment",
                "is_discharged_no_treatment",
                "group",
                "hsagrp",
                "tretspef"
            )
            .agg(
                (F.sum("arrivals") * self._sample_rate).alias("arrivals")
            )
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


# COMMAND ----------

# MAGIC %md
# MAGIC # Get Data
# MAGIC
# MAGIC For now, load the data locally to `/tmp` because running with databricks itself causes weird issues...

# COMMAND ----------

data = DatabricksNational(spark, 2019, 0.01)

data.get_ip().to_parquet("/tmp/data/2019/national/ip.parquet")

strats = data.get_ip_strategies()

strats["activity_avoidance"].to_parquet("/tmp/data/2019/national/ip_activity_avoidance_strategies.parquet")
strats["efficiencies"].to_parquet("/tmp/data/2019/national/ip_efficiencies_strategies.parquet")

data.get_op().to_parquet("/tmp/data/2019/national/op.parquet")
data.get_aae().to_parquet("/tmp/data/2019/national/aae.parquet")

data.get_demographic_factors().to_csv("/tmp/data/2019/national/demographic_factors.csv")
data.get_birth_factors().to_csv("/tmp/data/2019/national/birth_factors.csv")
data.get_hsa_activity_table().to_csv("/tmp/data/2019/national/hsa_activity_table.csv")

# COMMAND ----------

params = mdl.load_params("queue/sample_params.json")
params["health_status_adjustment"] = False
params["dataset"] = "national"
run_params = mdl.Model.generate_run_params(params)

nhp_data = Local.create("/tmp/data")

# set the data path in the HealthStatusAdjustment class
hsa = mdl.HealthStatusAdjustmentInterpolated(
    nhp_data(params["start_year"], params["dataset"]), params["start_year"]
)

results_dict = {}
pcallback = lambda _: None

# COMMAND ----------

# MAGIC %md
# MAGIC # Run Model

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run Inpatients Model
# MAGIC
# MAGIC There is some issue with day procedures, remove these from params for now

# COMMAND ----------

params["efficiencies"]["ip"] = {
    k: v
    for k, v in params["efficiencies"]["ip"].items()
    if not k.startswith("day_procedures")
}

# COMMAND ----------

results_dict["inpatients"] = _run_model(
    mdl.InpatientsModel,
    params,
    nhp_data,
    hsa,
    run_params,
    pcallback,
    False,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run Outpatients Model

# COMMAND ----------

results_dict["outpatients"] = _run_model(
    mdl.OutpatientsModel,
    params,
    nhp_data,
    hsa,
    run_params,
    pcallback,
    False,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run A&E Model

# COMMAND ----------

results_dict["aae"] = _run_model(
    mdl.AaEModel,
    params,
    nhp_data,
    hsa,
    run_params,
    pcallback,
    False,
)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Combine Results

# COMMAND ----------

results = _combine_results(list(results_dict.values()), params["model_runs"])

# COMMAND ----------

# MAGIC %md
# MAGIC # Check results

# COMMAND ----------

df = (
    pd.DataFrame(results["default"])
    .drop(columns = "time_profiles")
    .rename(columns = {"model_runs": "value"})
    .assign(model_run = lambda x: x["value"].apply(lambda y: list(range(len(y)))))
    .explode(["model_run", "value"])
    .reset_index(drop=True)
)

df.groupby(["pod", "measure", "baseline"]).agg(value=("value", "mean"))

# COMMAND ----------

# MAGIC %md
# MAGIC # Save the results
# MAGIC
# MAGIC Push the results to a storage container we have configured. This should be replaced by a proper upload, compress to gzip, add metadata

# COMMAND ----------

# filename = f"{params['dataset']}-{params['scenario']}-{params['create_datetime']}"
# os.makedirs(f"results/{params['dataset']}", exist_ok=True)

# with open(f"/Volumes/su_data/nhp/reference_data/{filename}.json", "w", encoding="utf-8") as file:
#     json.dump(
#         {
#             "params": params,
#             "population_variants": run_params["variant"],
#             "results": results,
#         },
#         file,
#     )

