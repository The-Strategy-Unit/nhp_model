import numpy as np
import os
import sys
import pandas as pd
import pickle
import pyarrow.parquet as pq
from janitor import complete
from pygam import GAM

def create_gams(p, pop, file, ignored_hsagrps = []):
  df = pq.read_pandas(p(f"{file}.parquet")).to_pandas()
  df = df[~df["hsagrp"].isin(ignored_hsagrps)]
  df = df[df["age"] >= 18]
  df = df[df["age"] <= 90]
  #
  df = (df
    .groupby(["age", "sex", "hsagrp"])
    .size()
    .reset_index(name = "n")
    .complete({ "age": range(18, 91) }, "sex", "hsagrp")
    .fillna(0)
    .sort_values(["age", "sex", "hsagrp"])
    .merge(pop, on = ["age", "sex"])
  )
  df["activity_rate"] = df["n"] / df["base_year"]
  #
  return (df, {
    k: GAM().gridsearch(v[["age"]].to_numpy(), v["activity_rate"].to_numpy())
    for k, v in tuple(df.groupby(["hsagrp", "sex"]))
  })

if __name__ == "__main__":
  if len(sys.argv) != 3:
    raise "Must provide exactly 2 argument: the path to the data and the base year"
  path, base_year = sys.argv[1:]
  # create a helper function for creating paths
  p = lambda f: os.path.join(path, f)
  # load the population data
  pop = pd.read_csv(p("demographic_factors.csv"))
  pop["age"] = pop["age"].clip(upper = 90)
  pop = (pop[pop["variant"] == "principal"][["sex", "age", str(base_year)]]
    .rename(columns = {str(base_year): "base_year", "age": "age"})
    .groupby(["sex", "age"])
    .agg("sum")
    .reset_index()
  )
  # create the gams
  _, ip_gams = create_gams(p, pop, "ip", ["birth", "maternity", "paeds"])
  _, op_gams = create_gams(p, pop, "op")
  # combine the gams
  gams = { **ip_gams, **op_gams }
  # save the gams to disk
  with open(p("hsa_gams.pkl"), "wb") as f:
    pickle.dump(gams, f)