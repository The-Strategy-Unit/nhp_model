# Databricks notebook source
dbutils.widgets.text("version", "dev")
dbutils.widgets.text("fyear", "201920")

# COMMAND ----------

import sys

sys.path.append(spark.conf.get("bundle.sourcePath", "."))
import pyspark.sql.functions as F

# COMMAND ----------

spark.catalog.setCurrentCatalog("su_data")
spark.catalog.setCurrentDatabase("nhp")

spark.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")

# COMMAND ----------

# MAGIC %md
# MAGIC # Save files

# COMMAND ----------

save_path = f"/Volumes/su_data/nhp/old_nhp_data/{dbutils.widgets.get('version')}"
fyear = int(dbutils.widgets.get("fyear"))

# COMMAND ----------

apc = (
    spark.read.table("apc")
    .filter(F.col("fyear") == fyear)
    .withColumnRenamed("epikey", "rn")
    .withColumnRenamed("provider", "dataset")
    .withColumn("tretspef_raw", F.col("tretspef"))
    .withColumn("fyear", F.floor(F.col("fyear") / 100))
    .withColumn("sex", F.col("sex").cast("int"))
    .withColumn("sushrg_trimmed", F.expr("substring(sushrg, 1, 4)"))
)

# COMMAND ----------

# add tretspef categories

specialties = [
    "100",
    "101",
    "110",
    "120",
    "130",
    "140",
    "150",
    "160",
    "170",
    "300",
    "301",
    "320",
    "330",
    "340",
    "400",
    "410",
    "430",
    "520",
]

tretspef_column = (
    F.when(F.col("tretspef_raw").isin(specialties), F.col("tretspef_raw"))
    .when(F.expr("tretspef_raw RLIKE '^1(?!80|9[02])'"), F.lit("Other (Surgical)"))
    .when(
        F.expr("tretspef_raw RLIKE '^(1(80|9[02])|[2346]|5(?!60)|83[134])'"),
        F.lit("Other (Medical)"),
    )
    .otherwise(F.lit("Other"))
)

apc = apc.withColumn("tretspef", tretspef_column)

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
    .filter(F.col("fyear") == fyear)
    .withColumnRenamed("provider", "dataset")
    .withColumn("fyear", F.floor(F.col("fyear") / 100))
    .withColumn("tretspef_raw", F.col("tretspef"))
    .withColumn("tretspef", tretspef_column)
    .withColumn("is_wla", F.lit(True))
    .repartition(1)
    .write.mode("overwrite")
    .partitionBy(["fyear", "dataset"])
    .parquet(f"{save_path}/op")
)


# COMMAND ----------

(
    spark.read.table("ecds")
    .filter(F.col("fyear") == fyear)
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

df.parquet(f"{save_path}/birth_factors/fyear={fyear // 100}")

# COMMAND ----------

df = (
    spark.read.table("demographic_factors")
    .withColumnRenamed("provider", "dataset")
    .repartition(1)
    .write.mode("overwrite")
    .partitionBy("dataset")
)

df.parquet(f"{save_path}/demographic_factors/fyear={fyear // 100}")
