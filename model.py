import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import os
import json
import time

from collections import defaultdict
from random import choices

# helper functions
def rnorm (lo, hi): 
  mean = (hi + lo) / 2
  sd   = (hi - lo) / 3.289707 # magic number: 2 * qnorm(0.95)
  return np.random.normal(mean, sd, 1)[0]

# pre-process data: shouldn't be needed in practice
if not os.path.exists(strategies_path := "test/data/synthetic/ip_strategies.json"):
  strategies = pq.read_pandas("test/data/synthetic/ip_strategies.parquet").to_pandas()
  # set up of strategies... pickle this?
  sd = defaultdict(lambda: ["NULL"])
  #
  for index, row in strategies.iterrows():
    sd[row["rn"]].append(row["strategy"])
  #
  with open(strategies_path, "w") as f:
    json.dump(sd, f)

def admission_avoidance(params):
  return pd.DataFrame.from_dict({
    k: min(1, rnorm(v["avoidance_low"], v["avoidance_high"])) if "avoidance_low" in v else 1
    for k, v in params["strategy_params"].items()
  }, "index", columns = ["factor"])

def los_run(data, params):
  sp = params["strategy_params"]
  #
  for k, v in tuple(data.groupby("strategy")):
    v = v["speldur"]
    if not k in sp:
      yield v
      continue
    #
    s = sp[k]
    if not "los_reduction_low" in s:
      yield v
      continue
    #
    m = np.mean(v)
    r = rnorm(s["los_reduction_low"], s["los_reduction_high"])
    if r >= m:
      yield v
      continue
    yield np.array([np.random.binomial(x, r / m, 1)[0] for x in v])

# TODO: code should be a dictionary or code/proportion
def demog_factors(params):
  params = params["demographic_factors"]
  #
  code = params["local_authorities"][0]
  start_year = params.get("start_year", "2018")
  end_year = params.get("end_year", "2043")
  #
  demog_factors = pd.read_csv("nhpmodel/inst/data/demographic_factors.csv")
  demog_factors = demog_factors[(demog_factors["code"] == code) & (demog_factors["age"] != "all")]
  #
  demog_factors["age"] = demog_factors["age"].astype(int)
  demog_factors["sex"] = np.where(demog_factors["sex"] == "males", 1, 2)
  demog_factors["factor"] = demog_factors[end_year] / demog_factors[start_year]
  #
  demog_factors = demog_factors[["variant", "age", "sex", "factor"]].groupby("variant")
  demog_factors = { k: v.drop("variant", axis = 1) for k, v in tuple(demog_factors) }
  #
  variants = list(params["variant_probabilities"].keys())
  probabilities = list(params["variant_probabilities"].values()) 
  #
  return lambda: demog_factors[np.random.choice(variants, 1, False,  probabilities)[0]]

def model_run(model_run, data, strategies, params, demog_factors_fn):
  # select just relevant columns for modelling
  data_run = data.loc[:, ["rn", "speldur", "sex", "admiage"]]
  # select a strategy
  data_run["strategy"] = data_run["rn"].apply(lambda k: choices(strategies[k])[0])
  # choose a demographic factor
  data_run = data_run.merge(demog_factors_fn(), left_on = ["admiage", "sex"], right_on = ["age", "sex"])
  # choose an admission avoidance factor
  data_run = data_run.merge(admission_avoidance(params), how = "left", left_on = ["strategy"], right_index = True)
  # create a single factor for how many times to select that row
  data_run["n"] = (
    data_run["factor_x"] * np.nan_to_num(data_run["factor_y"], nan = 1.0)
  ).apply(lambda x: np.random.poisson(x))
  # drop columns we don't need and repeat rows n times
  data_run = data_run.drop(["admiage", "age", "sex", "factor_x", "factor_y"], axis = 1)
  data_run = data_run.loc[data_run.index.repeat(data_run["n"])].drop(["n"], axis = 1)
  # choose new los
  data_run["speldur"] = np.concatenate([x for x in los_run(data_run, params)])
  # add in the model run and return the data
  data_run["model_run"] = model_run
  #
  return data_run.reset_index()

def test():
  # load data
  data = pq.read_pandas("test/data/synthetic/ip.parquet").to_pandas()
  data["rn"] = data["rn"].astype(str)
  data["admiage"] = data["admiage"].astype(int)
  data["sex"] = data["sex"].astype(int)

  data = data[data["sex"] <= 2]

  with open(strategies_path, "r") as f:
    sd = json.load(f)

  with open("test/queue/test.json", "r") as f:
    params = json.load(f)
    
  dfn = demog_factors(params)
  # fudge params for now
  params["strategy_params"]["ambulatory_emergency_care_high"] = {
    "los_reduction_low": 0.7,
    "los_reduction_high": 0.85
  }
  params["strategy_params"]["ambulatory_emergency_care_low"] = {
    "los_reduction_low": 0.8,
    "los_reduction_high": 0.95
  }
  params["strategy_params"]["ambulatory_emergency_care_moderate"] = {
    "los_reduction_low": 0.75,
    "los_reduction_high": 0.9
  }
  params["strategy_params"]["ambulatory_emergency_care_very_high"] = {
    "los_reduction_low": 0.65,
    "los_reduction_high": 0.8
  }
  return lambda i: model_run(i, data, sd, params, dfn)

if __name__ == "__main__":
  N_RUNS = 10
  mr = test()
  s = time.time()
  runs = [mr(i) for i in range(N_RUNS)]
  e = time.time() - s
  print(f"total time: {e}, avg time: {e / N_RUNS}")