import pyarrow.parquet as pq
import pandas as pd
import numpy as np
from pathos.multiprocessing import ProcessPool
import json
import time

from collections import defaultdict
from random import choice

N_CPUS = 10

path = "test/data/synthetic"
data_path = f"{path}/ip.parquet"
strategies_path = f"{path}/ip_strategies.json"
demog_factors_csv_path = "nhpmodel/inst/data/demographic_factors.csv"

# helper functions
def create_strategy_json(strategy_params):
  # figure out the types of strategies
  admission_avoidance = strategy_params["admission_avoidance"].keys()
  los_reduction = strategy_params["los_reduction"].keys()

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
  sd = {
    "admission_avoidance": defaultdict(lambda: ["NULL"]),
    "los_reduction": defaultdict(lambda: ["NULL"])
  }
  #
  for index, row in strategies.iterrows():
    rn, strategy = row["rn"], row["strategy"]
    if strategy in admission_avoidance:
      sd["admission_avoidance"][rn].append(strategy)
    if strategy in los_reduction:
      sd["los_reduction"][rn].append(strategy)
  #
  with open(strategies_path, "w") as f:
    json.dump(sd, f)

def rnorm (lo, hi): 
  mean = (hi + lo) / 2
  sd   = (hi - lo) / 3.289707 # magic number: 2 * qnorm(0.95)
  return np.random.normal(mean, sd, 1)[0]

def in01 (v):
  return max(0, min(1, v))

def admission_avoidance(strategy_params):
  return defaultdict(lambda: 1, {
    k: in01(rnorm(*v["interval"]))
    for k, v in strategy_params["admission_avoidance"].items()
  })

def los_run(data, strategy_params):
  #
  for k, v in tuple(data.groupby("strategy")):
    v = v["speldur"]
    if not k in strategy_params:
      yield v
      continue
    #
    s = strategy_params[k]
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
def demog_factors(data, params):
  code = params["local_authorities"][0]
  start_year = params.get("start_year", "2018")
  end_year = params.get("end_year", "2043")
  #
  demog_factors = pd.read_csv(demog_factors_csv_path)
  demog_factors = demog_factors[(demog_factors["code"] == code) & (demog_factors["age"] != "all")]
  #
  demog_factors["agesex"] = list(zip(demog_factors["age"].astype(int), np.where(demog_factors["sex"] == "males", 1, 2)))
  demog_factors["factor"] = demog_factors[end_year] / demog_factors[start_year]
  #
  variants = list(params["variant_probabilities"].keys())
  probabilities = list(params["variant_probabilities"].values())
  #
  data = {
    k: v.drop(["variant", "agesex"], axis = "columns")
    for k, v in tuple(data.merge(demog_factors[["agesex", "variant", "factor"]], on = "agesex").groupby(["variant"]))
  }
  return lambda: data[np.random.choice(variants, p = probabilities)]

def new_los(row, strategy_params):
  if row.los_reduction_strategy == "NULL" or row.speldur == 0:
    return row.speldur
  sp = strategy_params["los_reduction"][row.los_reduction_strategy]
  #
  r = in01(rnorm(*sp["interval"]))
  if sp["type"] == "1-2_to_0":
    # TODO: check that this is the correct ordering for the <
    return row.speldur if row.speldur > 2 or np.random.uniform() < r else 0
  #
  return np.random.binomial(row.speldur, r)

def model_run(model_run, data_fn, data_strategies, strategy_params):
  # choose a demographic factor
  data = data_fn()
  # select a strategy
  data["admission_avoidance_strategy"] = data["rn"].apply(lambda k: choice(data_strategies["admission_avoidance"][k]))
  data["los_reduction_strategy"] = data["rn"].apply(lambda k: choice(data_strategies["los_reduction"][k]))
  # choose an admission avoidance factor
  ada = admission_avoidance(strategy_params)
  factor_a = np.array([ada[k] for k in data["admission_avoidance_strategy"]])
  # create a single factor for how many times to select that row
  data["n"] = [np.random.poisson(f) for f in (data["factor"] * factor_a)]
  # drop columns we don't need and repeat rows n times
  data = data.drop(["factor"], axis = 1).loc[data.index.repeat(data["n"])].drop(["n"], axis = 1)
  # choose new los
  data["speldur"] = [new_los(r, params["strategy_params"]) for r in data.itertuples()]
  # add in the model run and return the data
  data["model_run"] = model_run
  #
  #return data.reset_index()
  return len(data.index)

def multi_model_runs(N_RUNS):
  pool = ProcessPool(ncpus = N_CPUS)
  results = pool.amap(mr, [i for i in range(N_RUNS)])
  pool.close()
  pool.join()
  #
  return np.mean(results.get())

# load data
data = pq.read_pandas(data_path, ["rn", "speldur", "admiage", "sex"]).to_pandas()
data["agesex"] = list(zip(data["admiage"].astype(int), data["sex"].astype(int)))
data = data.loc[data["sex"].isin(["1", "2"]), ["rn", "speldur", "agesex"]]

# load strategies, convert the key (rn) to an int
with open(strategies_path, "r") as f:
  data_strategies = { k1: defaultdict(
    lambda: ['NULL'],
    { int(k2): v2 for k2, v2 in v1.items() }
   ) for k1, v1 in json.load(f).items() }

# load the parameters
with open("test/queue/test.json", "r") as f:
  params = json.load(f)

def timeit(f, *args):
  s = time.time()
  f(*args)
  print(f"elapsed: {time.time() - s:.3f}")

dfn = demog_factors(data, params["demographic_factors"])
mr = lambda i: model_run(i, dfn, data_strategies, params["strategy_params"])

if __name__ == "__main__":
  dfn = demog_factors(data, params["demographic_factors"])
  N_RUNS = 1000
  s = time.time()
  print(f"avg n results: {multi_model_runs(N_RUNS)}")
  e = time.time() - s
  print(f"total time: {e}, avg time: {e / N_RUNS * N_CPUS}")
