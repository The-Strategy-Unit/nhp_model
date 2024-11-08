# Databricks notebook source
import json
import os

os.makedirs("/tmp/data/2019/national", exist_ok=True)

os.environ["BATCH_SIZE"] = "8"

# COMMAND ----------

import pandas as pd
import pyspark.sql.functions as F

import model as mdl
from model.data.local import Local
from model.health_status_adjustment import HealthStatusAdjustmentInterpolated
from run_model import _combine_results, _run_model

# COMMAND ----------

params = mdl.load_params("queue/sample_params.json")

params["dataset"] = "national"
params["demographic_factors"]["variant_probabilities"] = {"principal_proj": 1.0}

nhp_data = Local.create("/tmp/data")

# COMMAND ----------

spark.catalog.setCurrentCatalog("su_data")
spark.catalog.setCurrentDatabase("nhp")

# COMMAND ----------

# MAGIC %md
# MAGIC # Get Data
# MAGIC
# MAGIC For now, load the data locally to `/tmp` because running with databricks itself causes weird issues...

# COMMAND ----------

data = DatabricksNational(spark, 2019, 0.01)

# COMMAND ----------

data.get_ip().to_parquet("/tmp/data/2019/national/ip.parquet")

# COMMAND ----------

strats = data.get_ip_strategies()
strats["activity_avoidance"].to_parquet(
    "/tmp/data/2019/national/ip_activity_avoidance_strategies.parquet"
)
strats["efficiencies"].to_parquet(
    "/tmp/data/2019/national/ip_efficiencies_strategies.parquet"
)

# COMMAND ----------

data.get_op().to_parquet("/tmp/data/2019/national/op.parquet")

# COMMAND ----------

data.get_aae().to_parquet("/tmp/data/2019/national/aae.parquet")

# COMMAND ----------

data.get_demographic_factors().to_csv(
    "/tmp/data/2019/national/demographic_factors.csv", index=False
)

# COMMAND ----------

data.get_birth_factors().to_csv(
    "/tmp/data/2019/national/birth_factors.csv", index=False
)

# COMMAND ----------

data.get_hsa_activity_table().to_csv(
    "/tmp/data/2019/national/hsa_activity_table.csv", index=False
)

# COMMAND ----------

run_params = mdl.Model.generate_run_params(params)

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
    .drop(columns="time_profiles")
    .rename(columns={"model_runs": "value"})
    .assign(model_run=lambda x: x["value"].apply(lambda y: list(range(len(y)))))
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
