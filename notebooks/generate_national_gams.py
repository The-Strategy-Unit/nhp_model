# Databricks notebook source
# MAGIC %pip install pygam
# MAGIC

# COMMAND ----------

dbutils.widgets.text("version", "dev")

# COMMAND ----------


import sys

sys.path.append(spark.conf.get("bundle.sourcePath", "."))

import os
import pickle as pkl
from functools import reduce

import numpy as np
import pandas as pd
import pyspark.sql.functions as F
from pygam import GAM
from pyspark.sql import DataFrame
from tqdm.auto import tqdm

# COMMAND ----------

spark.catalog.setCurrentCatalog("su_data")
spark.catalog.setCurrentDatabase("nhp")

# COMMAND ----------

save_path = f"/Volumes/su_data/nhp/old_nhp_data/{dbutils.widgets.get('version')}"

# COMMAND ----------

# MAGIC %md
# MAGIC # Load data

# COMMAND ----------


dfr = (
    reduce(
        DataFrame.unionByName,
        [
            (
                spark.read.parquet(f"{save_path}/{ds}")
                .groupBy("fyear", "age", "sex", "hsagrp")
                .count()
            )
            for ds in ["ip", "op", "aae"]
        ],
    )
    .filter(~F.col("hsagrp").isin(["birth", "maternity", "paeds"]))
    .filter(F.col("fyear").isin([2019, 2022, 2023]))
)

dfr.display()

# COMMAND ----------

# MAGIC %md
# MAGIC load the demographics data, then cross join to the distinct HSA groups

# COMMAND ----------

demog = (
    spark.read.parquet("/Volumes/su_data/nhp/population-projections/demographic_data")
    .filter(F.col("area_code").rlike("^E0[6-9]"))
    .filter(F.col("projection") == "principal_proj")
    .filter(F.col("age") >= 18)
    .filter(F.col("year") == 2019)
    .groupBy("age", "sex")
    .agg(F.sum("value").alias("pop"))
    .crossJoin(dfr.select("hsagrp").distinct())
)
demog.display()

# COMMAND ----------

# MAGIC %md
# MAGIC generate the data. we right join to the demographics and fill the missing rows with 0's, before calculating the activity rate as the amount of activity (count) divided by the population.

# COMMAND ----------

dfr = (
    dfr.join(demog, ["age", "sex", "hsagrp"], "right")
    .fillna(0)
    .filter(F.col("fyear") > 0)
    .withColumn("activity_rate", F.col("count") / F.col("pop"))
    .drop("count", "pop")
    .toPandas()
)
dfr

# COMMAND ----------

# MAGIC %md
# MAGIC # Generate GAMs
# MAGIC
# MAGIC generate the GAMs as a nested dictionary by dataset/year/(HSA group, sex).
# MAGIC
# MAGIC This may be amenable to some parallelisation? or other speed tricks possible with pygam?

# COMMAND ----------

all_gams = {"NATIONAL": {}}
for fyear, v2 in list(dfr.groupby("fyear")):
    g = {
        k: GAM().gridsearch(
            v[["age"]].to_numpy(), v["activity_rate"].to_numpy(), progress=False
        )
        for k, v in list(v2.groupby(["hsagrp", "sex"]))
    }
    all_gams["NATIONAL"][fyear] = g


# COMMAND ----------

all_gams[2019]

# COMMAND ----------

all_ages = np.arange(0, 101)


def to_fyear(year):
    return year * 100 + (year + 1) % 100


def from_fyear(fyear):
    return fyear // 100


hsa_activity_tables = spark.createDataFrame(
    pd.concat(
        {
            dataset: pd.concat(
                {
                    to_fyear(year): pd.concat(
                        {
                            k: pd.Series(
                                g.predict(all_ages), index=all_ages, name="activity"
                            )
                            for k, g in v2.items()
                        }
                    )
                    for year, v2 in v1.items()
                }
            )
            for dataset, v1 in tqdm(all_gams.items())
        }
    )
    .rename_axis(["dataset", "fyear", "hsagrp", "sex", "age"])
    .reset_index()
)

for i in ["fyear", "sex", "age"]:
    hsa_activity_tables = hsa_activity_tables.withColumn(i, F.col(i).cast("int"))

hsa_activity_tables.display()

# COMMAND ----------

hsa_activity_tables.write.mode("overwrite").saveAsTable("hsa_activity_tables_NATIONAL")
