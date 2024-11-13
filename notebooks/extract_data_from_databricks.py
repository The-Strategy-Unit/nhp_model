# Databricks notebook source
dbutils.widgets.text("version", "dev")

# COMMAND ----------

import sys

sys.path.append(spark.conf.get("bundle.sourcePath", "."))
import pyspark.sql.functions as F

# COMMAND ----------

spark.catalog.setCurrentCatalog("su_data")
spark.catalog.setCurrentDatabase("nhp")

# COMMAND ----------

# MAGIC %md
# MAGIC # Save files

# COMMAND ----------

save_path = f"/Volumes/su_data/nhp/old_nhp_data/{dbutils.widgets.get('version')}"

# COMMAND ----------

apc = (
    spark.read.table("apc")
    .filter(F.col("fyear").isin([201920, 202223]))
    .withColumnRenamed("epikey", "rn")
    .withColumnRenamed("provider", "dataset")
    .withColumn("tretspef_raw", F.col("tretspef"))
    .withColumn("fyear", F.floor(F.col("fyear") / 100))
    .withColumn("sex", F.col("sex").cast("int"))
)

# COMMAND ----------

(
    apc.repartition(1)
    .write.mode("overwrite")
    .partitionBy(["fyear", "dataset"])
    .parquet(f"{save_path}/ip")
)


# COMMAND ----------

for k, v in [
    ("activity_avoidance", "activity_avoidance"),
    ("efficiencies", "efficiency"),
]:

    (
        spark.read.table("apc_mitigators")
        .filter(F.col("type") == v)
        .drop("type")
        .withColumnRenamed("epikey", "rn")
        .join(apc, "rn", "inner")
        .select("dataset", "fyear", "rn", "strategy", "sample_rate")
        .repartition(1)
        .write.mode("overwrite")
        .partitionBy(["fyear", "dataset"])
        .parquet(f"{save_path}/ip_{k}_strategies")
    )


# COMMAND ----------

(
    spark.read.table("opa")
    .filter(F.col("fyear").isin([201920, 202223]))
    .withColumnRenamed("provider", "dataset")
    .withColumn("fyear", F.floor(F.col("fyear") / 100))
    .withColumn("tretspef_raw", F.col("tretspef"))
    .withColumn("is_wla", F.lit(True))
    .repartition(1)
    .write.mode("overwrite")
    .partitionBy(["fyear", "dataset"])
    .parquet(f"{save_path}/op")
)


# COMMAND ----------

(
    spark.read.table("ecds")
    .filter(F.col("fyear").isin([201920, 202223]))
    .withColumnRenamed("provider", "dataset")
    .withColumn("fyear", F.floor(F.col("fyear") / 100))
    .repartition(1)
    .write.mode("overwrite")
    .partitionBy(["fyear", "dataset"])
    .parquet(f"{save_path}/aae")
)

# COMMAND ----------

df = (
    spark.read.table("birth_factors")
    .withColumnRenamed("provider", "dataset")
    .repartition(1)
    .write.mode("overwrite")
    .partitionBy("dataset")
)

df.parquet(f"{save_path}/birth_factors/fyear=2019")
df.parquet(f"{save_path}/birth_factors/fyear=2022")

# COMMAND ----------

df = (
    spark.read.table("demographic_factors")
    .withColumnRenamed("provider", "dataset")
    .repartition(1)
    .write.mode("overwrite")
    .partitionBy("dataset")
)

df.parquet(f"{save_path}/demographic_factors/fyear=2019")
df.parquet(f"{save_path}/demographic_factors/fyear=2022")

# COMMAND ----------

(
    spark.read.table("hsa_activity_tables")
    .filter(F.col("fyear").isin([201920, 202223]))
    .withColumnRenamed("provider", "dataset")
    .withColumn("fyear", F.floor(F.col("fyear") / 100))
    .repartition(1)
    .write.mode("overwrite")
    .partitionBy("fyear", "dataset")
    .parquet(f"{save_path}/hsa_activity_tables")
)
