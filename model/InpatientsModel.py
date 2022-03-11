import pyarrow.parquet as pq
import pandas as pd
import numpy as np

from collections import defaultdict

from model.helpers import rnorm, inrange
from model.Model import Model

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

_los_methods = defaultdict(lambda: _los_none, {
  "all": _los_all,
  "aec": _los_aec,
  "bads": _los_bads,
  "pre-op": _los_preop
})

class InpatientsModel(Model):
  """
  Inpatients Model
  
  * params: a dictionary from a parsed json file containing the model parameters.

  Implements the model for inpatient data. In order to run the model you need to pass in a parsed json file of
  parameters. Once the object is constructed you can call either m.run() to run the model and return the data, or
  m.save_run() to run the model and save the results.
  """
  def __init__(self, results_path):
    # call the parent init function
    Model.__init__(self, results_path)
    #
    # load the data
    #
    data = self._load_parquet("ip", ["rn", "speldur", "admiage", "sex", "admimeth", "classpat", "tretspef", "admigrp"]) 
    # TODO: this data type conversion should happen at data extraction
    data["admiage"] = data["admiage"].astype(int)
    data["sex"] = data["sex"].astype(int)
    # TODO: this filtering should happen at the data extraction stage
    data = data.loc[data["sex"].isin([1, 2]), ["rn", "speldur", "admiage", "sex", "admimeth", "classpat", "tretspef", "admigrp"]]
    #
    self._strategies = { x: self._load_parquet(f"ip_{x}_strategies") for x in ["admission_avoidance", "los_reduction"] }
    #
    # prepare demographic factors
    #
    self._data = {
      k: v.drop(["variant"], axis = "columns").set_index(["rn"])
      for k, v in tuple(data.merge(self._demog_factors, left_on = ["admiage", "sex"], right_index = True).groupby(["variant"]))
    }
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
    return _los_methods[sp["type"]](row, rng, *sp["interval"])
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
    params = self._params["health_status_adjustment"]
    #
    ages = np.arange(params["min_age"], params["max_age"] + 1)
    adjusted_ages = ages - [rnorm(rng, *i) for i in params["intervals"]]
    hsa = pd.concat([
      pd.DataFrame({
        "admigrp": a,
        "sex": int(s),
        "admiage": ages,
        "hsa_f": g.predict(adjusted_ages) / g.predict(ages)
      })
      for (a, s), g in self._hsa_gams.items()
    ])
    return (data
      .reset_index()
      .merge(hsa, on = ["admigrp", "sex", "admiage"], how = "left")
      .fillna(1)
      .set_index(["rn"])
    )
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
    return (variant, data.reset_index()[["rn", "speldur", "classpat", "admission_avoidance_strategy", "los_reduction_strategy"]])
