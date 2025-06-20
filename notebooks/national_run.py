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

# MAGIC %md
# MAGIC # Compute Setup
# MAGIC
# MAGIC You need to alter the compute used to install the nhp-model library
# MAGIC
# MAGIC 1. go to the compute tab, and select the compute you are going to use to run the model
# MAGIC 2. click onto the libraries tab, then click the install new button
# MAGIC 3. select file path/adls, and choose the "Python package" option
# MAGIC 4. paste in /Volumes/nhp/default/app/nhp_model-dev-py3-none-any.whl, changing the version as needed

# COMMAND ----------

dbutils.widgets.text("data_path", "/Volumes/nhp/model_data/files", "Data Path")
dbutils.widgets.text("data_version", "dev", "Data Version")
dbutils.widgets.text("params_file", "params-sample.json", "Params File")
dbutils.widgets.text("sample_rate", "0.01", "Sample Rate")

# COMMAND ----------


import gzip
import json
import os
from datetime import datetime

import pandas as pd
import pyspark.sql.functions as F
from azure.storage.blob import ContainerClient

from nhp import model as mdl
from nhp.model.data.databricks import DatabricksNational
from nhp.model.health_status_adjustment import HealthStatusAdjustmentInterpolated
from nhp.model.results import combine_results, generate_results_json, save_results_files
from nhp.model.run import _run_model

os.environ["BATCH_SIZE"] = "8"

# COMMAND ----------

# Upload JSON params file to queue folder and provide filepath in the params_file widget above
params = mdl.load_params(f"../queue/{dbutils.widgets.get('params_file')}")

params["dataset"] = "national"
params["demographic_factors"]["variant_probabilities"] = {"principal_proj": 1.0}

# COMMAND ----------

spark.catalog.setCurrentCatalog("nhp")
spark.catalog.setCurrentDatabase("default")

# COMMAND ----------

# MAGIC %md
# MAGIC # Run Model

# COMMAND ----------

SAMPLE_RATE = float(dbutils.widgets.get("sample_rate"))

if not 0 < SAMPLE_RATE <= 1:
    raise ValueError("Sample rate must be between 0 and 1")

DATA_PATH = f"{dbutils.widgets.get('data_path')}/{dbutils.widgets.get('data_version')}/"

nhp_data = DatabricksNational.create(spark, DATA_PATH, SAMPLE_RATE, params["seed"])
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

# save_full_model_results set to True
# This creates folders with the results for each of the 256 Monte Carlo simulations in notebooks/results/national/SCENARIONAME/CREATE_DATETIME

results_dict["inpatients"] = _run_model(
    mdl.InpatientsModel,
    params,
    nhp_data,
    hsa,
    run_params,
    pcallback,
    True,
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
    True,
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
    True,
)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Combine Results

# COMMAND ----------

results, step_counts = combine_results(list(results_dict.values()))

# COMMAND ----------

# MAGIC %md
# MAGIC scale results back

# COMMAND ----------

for k in results.keys():
    results[k]["value"] /= SAMPLE_RATE

step_counts["value"] /= SAMPLE_RATE

# COMMAND ----------

# MAGIC %md
# MAGIC create the json file before adding step counts into results

# COMMAND ----------

json_filename = generate_results_json(results, step_counts, params, run_params)

# COMMAND ----------

# MAGIC %md
# MAGIC # Check results

# COMMAND ----------

(
    step_counts.groupby(["pod", "change_factor", "measure", "model_run"])["value"]
    .sum()
    .groupby(["pod", "change_factor", "measure"])
    .mean()
)

# COMMAND ----------


def get_principal(df):
    cols = [i for i in df.columns if i != "value" if i != "model_run"]
    df = df.set_index(cols)

    baseline = df.query("model_run == 0")["value"].rename("baseline")
    principal = df.groupby(level=cols)["value"].mean().rename("principal")

    return pd.concat([baseline, principal], axis=1)


get_principal(results["default"])

# COMMAND ----------

# MAGIC %md
# MAGIC # Save the results
# MAGIC
# MAGIC Push the results to storage

# COMMAND ----------

outputs_version = dbutils.widgets.get("data_version").rsplit(".", 1)[0]
outputs_version

# COMMAND ----------

# JSON file

with open(f"results/{json_filename}.json", "rb") as f:
    zipped_results = gzip.compress(f.read())

metadata = {
    k: str(v)
    for k, v in params.items()
    if not isinstance(v, dict) and not isinstance(v, list)
}

url = dbutils.secrets.get("nhpsa-results", "url")
sas = dbutils.secrets.get("nhpsa-results", "sas-token")
cont = ContainerClient.from_container_url(f"{url}?{sas}")
cont.upload_blob(
    f"prod/{outputs_version}/national/{json_filename}.json.gz",
    zipped_results,
    metadata=metadata,
    overwrite=True,
)

# COMMAND ----------

# Save aggregated parquets as well (new format of model results, for future proofing)

saved_files = save_results_files(results, params)
for file in saved_files:
    filename = file[8:]
    with open(file, "rb") as f:
        cont.upload_blob(
            f"aggregated-model-results/{outputs_version}/{filename}",
            f.read(),
            overwrite=True,
        )


# COMMAND ----------

from pathlib import Path

# Save the IP full model results to storage
# From docker_run._upload_full_model_results
dataset = params["dataset"]
scenario = params["scenario"]
create_datetime = params["create_datetime"]

path = Path(f"results/{dataset}/{scenario}/{create_datetime}")
for file in path.glob("**/*.parquet"):
    filename = file.as_posix()[8:]
    with open(file, "rb") as f:
        cont.upload_blob(
            f"full-model-results/{outputs_version}/{filename}",
            f.read(),
            overwrite=True,
        )

# COMMAND ----------

print(
    f"Full results saved here; use this path for avoided_activity: \n full-model-results/{outputs_version}/{dataset}/{scenario}/{create_datetime}"
)
