# Databricks notebook source
# MAGIC %md
# MAGIC # National run - population based model
# MAGIC
# MAGIC 1. Run the first cell below. Once this has been done, a small widget will appear at the top of the notebook, allowing us to specify different params files, and sample rates for the scaling of the inpatients results. Upload your JSON params file to queue folder and provide the filename in the params_file widget. Defaults to sample_params.json.
# MAGIC
# MAGIC 2. To save model results, you will need to ensure that Databricks has a valid SAS token for Azure storage. The code below generates a key that is valid for one day, so this only needs to be run once per day.
# MAGIC
# MAGIC First, set up databricks CLI. [Instructions here](https://learn.microsoft.com/en-us/azure/databricks/dev-tools/auth/azure-cli); this only needs to be done once.
# MAGIC
# MAGIC Once authenticated, run the commands below in your terminal. Generates a SAS token that expires on the same day and saves it as a secret.
# MAGIC
# MAGIC ```
# MAGIC $date = (Get-Date).AddDays(1).ToString("yyyy-MM-dd")
# MAGIC
# MAGIC $sas_token=az storage container generate-sas `
# MAGIC     --account-name nhpsa `
# MAGIC     --name results `
# MAGIC     --permissions aclrw `
# MAGIC     --expiry $date `
# MAGIC     --auth-mode login `
# MAGIC     --as-user `
# MAGIC     --output tsv
# MAGIC     
# MAGIC
# MAGIC echo $sas_token | databricks secrets put-secret nhpsa-results sas-token
# MAGIC ```

# COMMAND ----------

dbutils.widgets.text("params_file", "sample_params.json", "Params File")
dbutils.widgets.text("sample_rate", "0.01", "Sample Rate")

# COMMAND ----------

import sys
sys.path.append(spark.conf.get("bundle.sourcePath", "."))

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
from run_model import _run_model
from model.results import combine_results

os.environ["BATCH_SIZE"] = "8"

# COMMAND ----------

# Upload JSON params file to queue folder and provide filepath in the params_file widget above
params = mdl.load_params(f"../queue/{dbutils.widgets.get('params_file')}")

params["dataset"] = "national"
params["demographic_factors"]["variant_probabilities"] = {"principal_proj": 1.0}

# COMMAND ----------

spark.catalog.setCurrentCatalog("su_data")
spark.catalog.setCurrentDatabase("nhp")

# COMMAND ----------

# MAGIC %md
# MAGIC # Run Model

# COMMAND ----------

SAMPLE_RATE = float(dbutils.widgets.get("sample_rate"))

if not 0 < SAMPLE_RATE <= 1:
  raise ValueError("Sample rate must be between 0 and 1")

nhp_data = DatabricksNational.create(spark, SAMPLE_RATE, params["seed"])
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

results = combine_results(list(results_dict.values()))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Scale results back up

# COMMAND ----------

def multiply_results_by_sample_rate(results, key, sample_rate):
  for res in results[key]:
    if key == "step_counts":
      fn = lambda x: x / sample_rate
    else:
      fn = lambda x: round(x / sample_rate)

    if "baseline" in res:
      res["baseline"] = fn(res["baseline"])
    res["model_runs"] = [fn(r) for r in res["model_runs"]]
    if "time_profiles" in res:
      res["time_profiles"] = [fn(r) for r in res["time_profiles"]]

for k in results.keys():
  multiply_results_by_sample_rate(results, k, SAMPLE_RATE)

# COMMAND ----------

# MAGIC %md
# MAGIC # Check results

# COMMAND ----------

(
  pd.DataFrame(results["step_counts"])
  .query("change_factor == 'activity_avoidance'")
  .query("measure == 'admissions'")
  .explode("model_runs")
  .groupby("strategy")
  ["model_runs"].mean()
)

# COMMAND ----------

df = (
    pd.DataFrame(results["default"])
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
