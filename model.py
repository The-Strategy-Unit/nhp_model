import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import os
import json
import time

from collections import defaultdict
from random import choices

path = "test/data/synthetic"
data_path = f"{path}/ip.parquet"
strategies_path = f"{path}/ip_strategies.json"
demog_factors_csv_path = "nhpmodel/inst/data/demographic_factors.csv"

# helper functions
def create_strategy_json():
  strategies = pq.read_pandas(f"{path}/ip_strategies.parquet").to_pandas()
  strategies.strategy.unique()
  #
  s = strategies.strategy.unique()
  s.sort()
  s = pd.DataFrame(s, columns = ["strategy"])
  s["strategy_replace"] = s.strategy
  s.loc[s.strategy.str.contains("^alcohol_partial_acute"), "strategy_replace"] = "alcohol_partial_acute"
  s.loc[s.strategy.str.contains("^alcohol_partial_chronic"), "strategy_replace"] = "alcohol_partial_chronic"
  s.loc[s.strategy.str.contains("^ambulatory_care_conditions_acute"), "strategy_replace"] = "ambulatory_care_conditions_acute"
  s.loc[s.strategy.str.contains("^ambulatory_care_conditions_chronic"), "strategy_replace"] = "ambulatory_care_conditions_chronic"
  s.loc[s.strategy.str.contains("^ambulatory_care_conditions_vaccine_preventable"), "strategy_replace"] = "ambulatory_care_conditions_vaccine_preventable"
  s.loc[s.strategy == "improved_discharge_planning_emergency"] = "improved_discharge_planning_non-elective"
  #
  strategies = strategies.merge(s, on = "strategy").drop(["strategy"], axis = "columns").rename(columns = {"strategy_replace": "strategy"}).drop_duplicates()
  #
  # set up of strategies... pickle this?
  sd = defaultdict(lambda: ["NULL"])
  #
  for index, row in strategies.iterrows():
    sd[row["rn"]].append(row["strategy"])
  #
  with open(strategies_path, "w") as f:
    json.dump(sd, f)

def rnorm (lo, hi): 
  mean = (hi + lo) / 2
  sd   = (hi - lo) / 3.289707 # magic number: 2 * qnorm(0.95)
  return np.random.normal(mean, sd, 1)[0]

def admission_avoidance(params):
  return {
    k: min(1, rnorm(*v["admission_avoidance"])) if "admission_avoidance" in v else 1
    for k, v in params["strategy_params"].items()
  }

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
    if not "los_reduction" in s:
      yield v
      continue
    #
    m = np.mean(v)
    r = rnorm(*s["los_reduction"])
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
  demog_factors = pd.read_csv(demog_factors_csv_path)
  demog_factors = demog_factors[(demog_factors["code"] == code) & (demog_factors["age"] != "all")]
  #
  demog_factors["age"] = demog_factors["age"].astype(int)
  demog_factors["sex"] = np.where(demog_factors["sex"] == "males", 1, 2)
  demog_factors["factor"] = demog_factors[end_year] / demog_factors[start_year]
  #
  demog_factors = demog_factors[["variant", "age", "sex", "factor"]].set_index(["age", "sex"]).groupby("variant")
  demog_factors = { k: v["factor"].to_dict() for k, v in tuple(demog_factors) }
  #
  variants = list(params["variant_probabilities"].keys())
  probabilities = list(params["variant_probabilities"].values()) 
  #
  return lambda: demog_factors[np.random.choice(variants, None, False,  probabilities)]

def model_run(model_run, data, data_strategies, params, demog_factors_fn):
  # select a strategy
  data["strategy"] = data["rn"].apply(lambda k: choices(data_strategies[k])[0])
  # choose a demographic factor
  dfn = demog_factors_fn()
  factor_d = np.array([dfn[k] for k in data["agesex"]])
  # choose an admission avoidance factor
  ada = admission_avoidance(params)
  factor_a = np.array([ada.get(k, 1.0) for k in data["strategy"]])
  # create a single factor for how many times to select that row
  data["n"] = [np.random.poisson(f) for f in (factor_d * factor_a)]
  # drop columns we don't need and repeat rows n times
  data = data.drop(["agesex"], axis = 1)
  data = data.loc[data.index.repeat(data["n"])].drop(["n"], axis = 1)
  # choose new los
  data["speldur"] = np.concatenate([x for x in los_run(data, params)])
  # add in the model run and return the data
  data["model_run"] = model_run
  #
  return data.reset_index()

# load data
data = pq.read_pandas(data_path, ["rn", "speldur", "admiage", "sex"]).to_pandas()
data["agesex"] = list(zip(data["admiage"].astype(int), data["sex"].astype(int)))
data = data.loc[data["sex"].isin(["1", "2"]), ["rn", "speldur", "agesex"]]

# load strategies, convert the key (rn) to an int
with open(strategies_path, "r") as f:
  data_strategies = { int(k): v for k, v in json.load(f).items() }

# load the parameters
with open("test/queue/test.json", "r") as f:
  params = json.load(f)

if __name__ == "__main__":
  mr = lambda i: model_run(i, data, data_strategies, params, demog_factors(params))
  N_RUNS = 10
  s = time.time()
  runs = [mr(i) for i in range(N_RUNS)]
  e = time.time() - s
  print(f"total time: {e}, avg time: {e / N_RUNS}")

# optimisations:
# could we pre-join the demographic factors for all variants, then just select one option?