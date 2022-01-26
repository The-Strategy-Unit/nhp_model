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
demog_factors_csv_path = "nhpmodel/inst/data/demographic_factors.csv"

# helper functions

def timeit(f, *args):
  """
  Time how long it takes to evaluate function `f` with arguments `*args`.
  """
  s = time.time()
  f(*args)
  print(f"elapsed: {time.time() - s:.3f}")

def prepare_strategies(data, strategy_params):
  """
  take the raw strategies file and create the prepared admission_avoidance/los_reduction files
  """
  # figure out the types of strategies
  admission_avoidance = strategy_params["admission_avoidance"].keys()
  los_reduction = strategy_params["los_reduction"].keys()
  #
  strategies = pq.read_pandas(f"{path}/ip_strategies.parquet").to_pandas()
  # semijoin back to filtered data
  strategies = strategies[strategies.rn.isin(data.index)]
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
  s.loc[s.strategy.str.contains("^falls_related_admissions_implicit"), "strategy_replace"] = "falls_related_admissions_implicit"
  #
  s["type"] = s.strategy_replace.apply(lambda s: "admission_avoidance" if s in admission_avoidance else "los_reduction" if s in los_reduction else None)
  #
  strategies_merged = tuple(strategies
      .merge(s, on = "strategy")
      .drop(["strategy"], axis = "columns")
      .rename(columns = {"strategy_replace": "strategy"})
      .drop_duplicates()
      .groupby(["type"])
    )
  #
  null_strats = pd.DataFrame(["NULL"] * len(data.index), index = data.index, columns = ["strategy"])
  null_strats.index.rename("rn", inplace = True)
  for k, v in strategies_merged:
    (pd
      .concat([null_strats, v.drop(["type"], axis = "columns").set_index(["rn"])])
      .rename({ "strategy": f"{k}_strategy" }, axis = "columns")
      .to_parquet(f"{path}/ip_{k}_strategies.parquet")
    )

def rnorm (lo, hi):
  """
  Create a single random normal value from a 90% confidence interval
  """
  mean = (hi + lo) / 2
  sd   = (hi - lo) / 3.289707 # magic number: 2 * qnorm(0.95)
  return np.random.normal(mean, sd)

def in01 (v):
  """
  Force a value to be in the interval [0, 1]
  """
  return max(0, min(1, v))

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
    k: v.drop(["variant", "agesex"], axis = "columns").set_index(["rn"])
    for k, v in tuple(data.merge(demog_factors[["agesex", "variant", "factor"]], on = "agesex").groupby(["variant"]))
  }
  # create and return a closure that selects a variant at random, returning a tuple containing the (variant, data)
  def f():
    v = np.random.choice(variants, p = probabilities)
    return (v, data[v])
  return f

def admission_avoidance(strategy_params):
  return defaultdict(lambda: 1, {
    k: in01(rnorm(*v["interval"]))
    for k, v in strategy_params["admission_avoidance"].items()
  })

def new_los(row, strategy_params):
  if row.los_reduction_strategy == "NULL" or row.speldur == 0:
    return row.speldur
  sp = strategy_params["los_reduction"][row.los_reduction_strategy]
  #
  r = in01(rnorm(*sp["interval"]))
  if sp["type"] == "1-2_to_0":
    # TODO: check that this is the correct ordering for the <
    # with r = 0.8, < will convert 80% of 1/2 to 0. with > it will convert 20%
    return row.speldur if row.speldur > 2 or np.random.uniform() < r else 0
  #
  return np.random.binomial(row.speldur, r)

def model(data_fn, strategies_admission_avoidance, strategies_los_reduction, strategy_params):
  # choose a demographic factor
  variant, data = data_fn()
  # select a strategy
  random_strat = lambda df: data.merge(df.sample(frac = 1).groupby(["rn"]).head(1), left_index = True, right_index = True)
  data = random_strat(strategies_admission_avoidance)
  data = random_strat(strategies_los_reduction)
  # choose an admission avoidance factor
  ada = admission_avoidance(strategy_params)
  factor_a = np.array([ada[k] for k in data["admission_avoidance_strategy"]])
  # create a single factor for how many times to select that row
  data["n"] = [np.random.poisson(f) for f in (data["factor"] * factor_a)]
  # drop columns we don't need and repeat rows n times
  data = data.loc[data.index.repeat(data["n"])].drop(["factor", "n"], axis = "columns")
  # choose new los
  data["speldur"] = [new_los(r, strategy_params) for r in data.itertuples()]
  # return the data
  return (variant, data.reset_index())
  
def save_model_run(model_run_n, mr):
  variant, mr_data = mr
  mr_data.to_parquet(f"test/tmp/results/mr={model_run_n}.parquet")
  with open(f"test/tmp/selected_variant/mr={model_run_n}.txt", "w") as f:
    f.write(variant)

def run_model(data_fn, strategies_admission_avoidance, strategies_los_reduction, strategy_params):
  def f(n):
    m = model(data_fn, strategies_admission_avoidance, strategies_los_reduction, strategy_params)
    save_model_run(n, m)
    return len(m.index)
  return f

def multi_model_runs(N_RUNS):
  pool = ProcessPool(ncpus = N_CPUS)
  results = pool.amap(mr, range(N_RUNS))
  pool.close()
  pool.join()
  #
  return np.mean(results.get())

# load data
data = pq.read_pandas(data_path, ["rn", "speldur", "admiage", "sex"]).to_pandas()
data["agesex"] = list(zip(data["admiage"].astype(int), data["sex"].astype(int)))
data = data.loc[data["sex"].isin(["1", "2"]), ["rn", "speldur", "agesex"]]

strategies_admission_avoidance = pq.read_pandas(f"{path}/ip_admission_avoidance_strategies.parquet").to_pandas()
strategies_los_reduction = pq.read_pandas(f"{path}/ip_los_reduction_strategies.parquet").to_pandas()

# load the parameters
with open("test/queue/test.json", "r") as f: params = json.load(f)

data_fn = demog_factors(data, params["demographic_factors"])
strategy_params = params["strategy_params"]
mr = run_model(data_fn, strategies_admission_avoidance, strategies_los_reduction, strategy_params)

if __name__ == "__main__":
  dfn = demog_factors(data, params["demographic_factors"])
  N_RUNS = 1000
  s = time.time()
  print(f"avg n results: {multi_model_runs(N_RUNS)}")
  e = time.time() - s
  print(f"total time: {e}, avg time: {e / N_RUNS * N_CPUS}")
