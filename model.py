import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import json
import time
import os
import argparse

from pathlib import Path
from pathos.multiprocessing import ProcessPool
from datetime import datetime
from collections import defaultdict

# helper functions

def timeit(f, *args):
  """
  Time how long it takes to evaluate function `f` with arguments `*args`.
  """
  s = time.time()
  r = f(*args)
  print(f"elapsed: {time.time() - s:.3f}")
  return r

# TODO: this should just become part of the data extraction process
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

def inrange(v, lo = 0, hi = 1):
  """
  Force a value to be in the interval [lo, hi]
  """
  return max(lo, min(hi, v))

def rnorm(rng, lo, hi):
  """
  Create a single random normal value from a 90% confidence interval
  """
  mean = (hi + lo) / 2
  sd   = (hi - lo) / 3.289707 # magic number: 2 * qnorm(0.95)
  return rng.normal(mean, sd)

class InpatientsModel:
  """
  Inpatients Model
  
  * params: a dictionary from a parsed json file containing the model parameters.

  Implements the model for inpatient data. In order to run the model you need to pass in a parsed json file of
  parameters. Once the object is constructed you can call either m.run() to run the model and return the data, or
  m.save_run() to run the model and save the results.
  """
  def __init__(self, results_path):
    # load the parameters file
    with open(f"{results_path}/params.json", "r") as f: params = json.load(f)
    self._params = params
    # load the required data
    self._path = str(Path(results_path).parent.parent.parent)
    self._results_path = results_path
    #
    # load the data
    # don't assign to self yet, handle in prepare demographic factors
    data = self._load_parquet("ip", ["rn", "speldur", "admiage", "sex"]) 
    data["agesex"] = list(zip(data["admiage"].astype(int), data["sex"].astype(int)))
    data = data.loc[data["sex"].isin(["1", "2"]), ["rn", "speldur", "agesex"]]
    #
    self._strategies = { x: self._load_parquet(f"ip_{x}_strategies") for x in ["admission_avoidance", "los_reduction"] }
    #
    # prepare demographic factors
    dfp = params["demographic_factors"]
    start_year = dfp.get("start_year", "2018")
    end_year = dfp.get("end_year", "2043")
    #
    demog_factors = pd.read_csv(os.path.join(self._path, dfp["file"]))
    #
    demog_factors["agesex"] = list(zip(demog_factors["age"].astype(int), np.where(demog_factors["sex"] == "males", 1, 2)))
    demog_factors["factor"] = demog_factors[end_year] / demog_factors[start_year]
    #
    self._variants = list(dfp["variant_probabilities"].keys())
    self._probabilities = list(dfp["variant_probabilities"].values())
    #
    self._data = {
      k: v.drop(["variant", "agesex"], axis = "columns").set_index(["rn"])
      for k, v in tuple(data.merge(demog_factors[["agesex", "variant", "factor"]], on = "agesex").groupby(["variant"]))
    }
  #
  def _load_parquet(self, file, *args):
    """
    Load a parquet file from the file path created by the constructor.

    You can selectively load columns by passing an array of column names to *args.
    """
    return pq.read_pandas(os.path.join(self._path, f"{file}.parquet"), *args).to_pandas()
  #
  def _select_variant(self, rng):
    """
    Randomly select a single variant to use for a model run
    """
    v = rng.choice(self._variants, p = self._probabilities)
    return (v, self._data[v])
  #
  def _admission_avoidance(self, rng):
    """
    Create a dictionary of the admission avoidance factors to use for a run
    """
    strategy_params = self._params["strategy_params"]
    return defaultdict(lambda: 1, {
      k: inrange(rnorm(rng, *v["interval"]))
      for k, v in strategy_params["admission_avoidance"].items()
    })
  #
  def _new_los(self, rng, row):
    """
    Create a new Length of Stay for a row of data
    """
    strategy_params = self._params["strategy_params"]
    if row.los_reduction_strategy == "NULL" or row.speldur == 0:
      return row.speldur
    sp = strategy_params["los_reduction"][row.los_reduction_strategy]
    #
    r = inrange(rnorm(rng, *sp["interval"]))
    if sp["type"] == "1-2_to_0":
      # TODO: check that this is the correct ordering for the <
      # with r = 0.8, > will convert 80% of 1/2 to 0. with < it will convert 20%
      return row.speldur if row.speldur > 2 or rng.uniform() < r else 0
    #
    return rng.binomial(row.speldur, r)
  #
  def _random_strategy(self, rng, data, strategy_type):
    """
    Select one strategy per record

    * data: the pandas DataFrame that we are updating
    * strategy_type: a string of which type of strategy to update, e.g. "admission_avoidance", "los_reduction"

    returns: an updated DataFrame with a new column for the selected strategy
    """
    df = (self._strategies[strategy_type]
      .sample(frac = 1, random_state = rng.bit_generator)
      .groupby(["rn"])
      .head(1)
    )
    return data.merge(df, left_index = True, right_index = True)
  #
  def run(self, model_run):
    """
    Run the model once

    returns: a tuple of the selected varient and the updated DataFrame
    """
    # create a random number generator for this run
    rng = np.random.default_rng(self._params["seed"] + model_run)
    # choose a demographic factor
    variant, data = self._select_variant(rng)
    # select a strategy
    data = self._random_strategy(rng, data, "admission_avoidance")
    data = self._random_strategy(rng, data, "los_reduction")
    # choose an admission avoidance factor
    ada = self._admission_avoidance(rng)
    factor_a = np.array([ada[k] for k in data["admission_avoidance_strategy"]])
    # create a single factor for how many times to select that row
    data["n"] = [rng.poisson(f) for f in (data["factor"] * factor_a)]
    # drop columns we don't need and repeat rows n times
    data = data.loc[data.index.repeat(data["n"])].drop(["factor", "n"], axis = "columns")
    # choose new los
    data["speldur"] = [self._new_los(rng, r) for r in data.itertuples()]
    # return the data
    return (variant, data.reset_index())
  #
  def save_run(self, model_run):
    """
    Save the model run

    * model_run: the number of this model run

    Runs the model, saving the results to the results folder
    """
    variant, mr_data = self.run(model_run)
    mr_data.to_parquet(f"{self._results_path}/results/model_run={model_run}.parquet")
    with open(f"{self._results_path}/selected_variants/model_run={model_run}.txt", "w") as f:
      f.write(variant)
  #
  def multi_model_runs(self, run_start, model_runs, N_CPUS = 1):
    #
    pool = ProcessPool(ncpus = N_CPUS)
    pool.amap(self.save_run, range(run_start, run_start + model_runs))
    pool.close()
    pool.join()

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("results_path", nargs = 1, help = "Path to the results")
  parser.add_argument("run_start", nargs = 1, help = "Where to start model run from", type = int)
  parser.add_argument("model_runs", nargs = 1, help = "How many model runs to perform", type = int)
  parser.add_argument("-c", "--cpus", default = os.cpu_count(), help = "Number of CPU cores to use", type = int)
  parser.add_argument("-f", "--force", action = "store_true")
  # Grab the Arguments
  args = parser.parse_args()
  #
  m = InpatientsModel(args.results_path[0])
  m.multi_model_runs(args.run_start[0], args.model_runs[0], args.cpus)

if __name__ == "__main__":
  main()
# TODO: debugging purposes: remove from production
else:
  m = InpatientsModel("test/data/synthetic/results/test/20220110_104353")