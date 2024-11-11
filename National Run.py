# Databricks notebook source
dbutils.widgets.text("params_file", "queue/sample_params.json", "Params File")

# COMMAND ----------

import json
import os
import gzip
from datetime import datetime
from azure.storage.blob import ContainerClient

import pandas as pd
import pyspark.sql.functions as F

import model as mdl
from model.data.databricks import DatabricksNational
from model.health_status_adjustment import HealthStatusAdjustmentInterpolated
from run_model import _combine_results, _run_model

os.environ["BATCH_SIZE"] = "8"

# COMMAND ----------

# Upload JSON params file to queue folder and provide filepath in the params_file widget above
params = mdl.load_params(dbutils.widgets.get("params_file"))

params["dataset"] = "national"
params["demographic_factors"]["variant_probabilities"] = {"principal_proj": 1.0}

# COMMAND ----------

spark.catalog.setCurrentCatalog("su_data")
spark.catalog.setCurrentDatabase("nhp")

# COMMAND ----------

# MAGIC %md
# MAGIC # Run Model

# COMMAND ----------

nhp_data = DatabricksNational.create(spark, 0.01, params["seed"])
runtime = datetime.now().strftime(format="%Y%m%d-%H%M%S")

# COMMAND ----------

hsa = mdl.HealthStatusAdjustmentInterpolated(
    nhp_data(params["start_year"], None), params["start_year"]
)

run_params = mdl.Model.generate_run_params(params)

results_dict = {}
pcallback = lambda _: None

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run Inpatients Model

# COMMAND ----------

mdl.

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
# MAGIC Push the results to storage

# COMMAND ----------

filename = f"{params['dataset']}-{params['scenario']}-{runtime}"

zipped_results = gzip.compress(json.dumps({
            "params": params,
            "population_variants": run_params["variant"],
            "results": results,
        }).encode("utf-8"))

metadata = {
    k: str(v)
    for k, v in params.items()
    if not isinstance(v, dict) and not isinstance(v, list)
}

# Metadata "dataset" needs to be SYNTHETIC otherwise it will not be viewable in outputs
metadata['dataset'] = 'synthetic'

# COMMAND ----------

url = dbutils.secrets.get("nhpsa-results", "url")
sas = dbutils.secrets.get("nhpsa-results", "sas-token")
cont = ContainerClient.from_container_url(f"{url}?{sas}")
cont.upload_blob(f'prod/dev/synthetic/{filename}.json.gz', zipped_results, metadata=metadata)
