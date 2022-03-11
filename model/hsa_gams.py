import numpy as np
import os
import sys
import pandas as pd
import pickle
import pyarrow.parquet as pq
from janitor import complete
from pygam import GAM

def create_gams(path, base_year, final_year):
  p = lambda f: os.path.join(path, f)
  df = pq.read_pandas(p("ip.parquet")).to_pandas()
  df = df[~df["admigrp"].isin(["birth", "maternity", "paeds"])]
  df = df[df["admiage"] >= 18]
  df = df[df["admiage"] <= 90]
  #
  df = (df
    .groupby(["admiage", "sex", "admigrp"])
    .size()
    .reset_index(name = "n")
    .complete({ "admiage": range(18, 91) }, "sex", "admigrp")
    .fillna(0)
    .sort_values(["admiage", "sex", "admigrp"])
  )
  #
  pop = pd.read_csv(p("demographic_factors.csv"))
  pop = pop[pop["variant"] == "principal"][["sex", "age", str(base_year), str(final_year)]] \
    .rename(columns = {str(base_year): "base_year", str(final_year): "final_year", "age": "admiage"})
  pop["admiage"] = pop["admiage"].astype(float)
  pop["sex"] = pop["sex"].astype(str)
  #
  df = df.merge(pop, on = ["sex", "admiage"])
  #
  gams = {
    k: GAM().gridsearch(v[["admiage"]].to_numpy(), v["n"].to_numpy())
    for k, v in tuple(df.groupby(["admigrp", "sex"]))
  }
  #
  with open(p("hsa_gams.pkl"), "wb") as f:
    pickle.dump(gams, f)

if __name__ == "__main__":
  if len(sys.argv) != 4:
    raise "Must provide exactly 3 argument: the path to the data, the base year, and the final year"
  create_gams(*sys.argv[1:])