import pickle
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

def _los_none(row, rng):
  return row.speldur

def _los_all(row, rng, lo, hi):
  r = inrange(rnorm(rng, lo, hi))
  return rng.binomial(row.speldur, r)

def _los_bads(row, rng, lo, hi):
  # we have the previous rate
  return row.speldur if row.classpat == 1 else 0

def _los_aec(row, rng, minv, lo, hi):
  if row.speldur == 0: return 0
  r = inrange(rnorm(rng, lo, hi), minv, 1)
  u = rng.uniform()
  return 0 if u <= r else row.speldur

def _los_preop(row, rng, days, lo, hi):
  r = inrange(rnorm(rng, lo, hi))
  u = rng.uniform()
  return row.speldur - (days if u <= r else 0)

los_methods = defaultdict(lambda: _los_none, {
  "all": _los_all,
  "aec": _los_aec,
  "bads": _los_bads,
  "pre-op": _los_preop
})

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
    #
    with open(f"{self._path}/hsa_gams.pkl", "rb") as f: self._hsa_gams = pickle.load(f)
    # don't assign to self yet, handle in prepare demographic factors
    data = self._load_parquet("ip", ["rn", "speldur", "admiage", "sex", "admimeth", "classpat", "tretspef", "admigrp"]) 
    data["age"] = data["admiage"].astype(int)
    data["sex"] = data["sex"].astype(int)
    data = data.loc[data["sex"].isin([1, 2]), ["rn", "speldur", "admiage", "sex", "admimeth", "classpat", "tretspef", "admigrp"]]
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
    demog_factors["age"] = demog_factors["age"].astype(int)
    demog_factors["sex"] = demog_factors["sex"].astype(int)
    demog_factors["factor"] = demog_factors[end_year] / demog_factors[start_year]
    #
    self._variants = list(dfp["variant_probabilities"].keys())
    self._probabilities = list(dfp["variant_probabilities"].values())
    #
    self._data = {
      k: v.drop(["variant"], axis = "columns").set_index(["rn"])
      for k, v in tuple(data.merge(demog_factors[["age", "sex", "variant", "factor"]], left_on = ["admiage", "sex"], right_on = ["age", "sex"]).groupby(["variant"]))
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
    if row.los_reduction_strategy == "NULL": return row.speldur
    sp = self._params["strategy_params"]["los_reduction"][row.los_reduction_strategy]
    return los_methods[sp["type"]](row, rng, *sp["interval"])
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
      .set_index(["rn"])
    )
    return data.merge(df, left_index = True, right_index = True)
  #
  def _waiting_list_adjustment(self, data):
    pwla = self._params["waiting_list_adjustment"]["inpatients"].copy()
    dv = pwla.pop("X01")
    pwla = defaultdict(lambda: dv, pwla)
    return [pwla[row.tretspef] if row.admimeth == "11" else 1 for row in data.itertuples()]
  #
  def _health_status_adjustment(self, rng, data):
    lo, hi = self._params["health_status_adjustment"]
    y = rnorm(rng, lo, hi)
    #
    ages = np.arange(55, 91)
    hsa = pd.concat([
      pd.DataFrame({
        "admigrp": a,
        "sex": int(s),
        "admiage": ages,
        "hsa_f": g.predict(ages - y) / g.predict(ages)
      })
      for (a, s), g in self._hsa_gams.items()
    ])
    return data.merge(hsa, on = ["admigrp", "sex", "admiage"], how = "left").fillna(1)
  #
  def _bads_conversion(self, rng, row):
    if row.los_reduction_strategy == "NULL": return row.classpat
    sp = self._params["strategy_params"]["los_reduction"][row.los_reduction_strategy]
    if sp["type"] != "bads": return row.classpat
    #
    if row.classpat == (2 if sp["target_type"] == "daycase" else -1): return row.classpat
    #
    btr, ods = sp["baseline_target_rate"], sp["op_dc_split"]
    ods = sp["op_dc_split"]
    r = inrange((rnorm(rng, *sp["interval"]) - btr) / (1 - btr))
    u = [ods * r, (1 - ods) * r]
    # update the patient class to be daycase, or -1 to indicate this is now outpatients
    return rng.choice([row.classpat, 2, -1], p = [1 - sum(u)] + u)
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
    row_count = len(data.index) # for assert below
    data = self._random_strategy(rng, data, "admission_avoidance")
    data = self._random_strategy(rng, data, "los_reduction")
    # double check that joining in the strategies didn't drop any rows
    assert len(data.index) == row_count, "Row's lost when selecting strategies: has the NULL strategy not been included?"
    # choose an admission avoidance factor
    ada = self._admission_avoidance(rng)
    factor_a = np.array([ada[k] for k in data["admission_avoidance_strategy"]])
    # waiting list adjustments
    factor_w = self._waiting_list_adjustment(data)
    # hsa
    data = self._health_status_adjustment(rng, data)
    # create a single factor for how many times to select that row
    data["n"] = [rng.poisson(f) for f in (data["factor"] * data["hsa_f"] * factor_a * factor_w)]
    # drop columns we don't need and repeat rows n times
    data = data.loc[data.index.repeat(data["n"])].drop(["factor", "hsa_f", "n"], axis = "columns")
    # handle bads rows
    data["classpat"] = [self._bads_conversion(rng, r) for r in data.itertuples()]
    # choose new los
    data["speldur"] = [self._new_los(rng, r) for r in data.itertuples()]
    # return the data
    return (variant, data.reset_index().drop(["admimeth", "tretspef"], axis = "columns"))
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
  parser.add_argument("-d", "--debug", action = "store_true")
  # Grab the Arguments
  args = parser.parse_args()
  #
  m = InpatientsModel(args.results_path[0])
  if args.debug:
    _, r = timeit(m.run, 0)
    print (r)
  else:
    m.multi_model_runs(args.run_start[0], args.model_runs[0], args.cpus)

if __name__ == "__main__":
  main()
# TODO: debugging purposes: remove from production
else:
  m = InpatientsModel("test/data/synthetic/results/test/20220110_104353")
  with open("test/queue/test.json", "r") as f: params = json.load(f)
  data = m._data["principal"]