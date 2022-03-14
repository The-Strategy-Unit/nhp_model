import numpy as np
import pandas as pd

from collections import defaultdict

from model.helpers import rnorm, inrange
from model.Model import Model

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
    # load the data
    data = (self
      ._load_parquet("ip", ["rn", "speldur", "age", "sex", "admimeth", "classpat", "tretspef", "hsagrp"])
      .merge(self._demog_factors, left_on = ["age", "sex"], right_index = True)
      .groupby(["variant"])
    )
    self._strategies = {
      x: self._load_parquet(f"ip_{x}_strategies") for x in ["admission_avoidance", "los_reduction"]
    }
    self._data = { k: v.drop(["variant"], axis = "columns").set_index(["rn"]) for k, v in tuple(data) }
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
  def _los_reduction(self, rng):
    """
    Create a dictionary of the los reduction factors to use for a run
    """
    p = self._params["strategy_params"]["los_reduction"]
    losr = pd.DataFrame.from_dict(p, orient = "index")
    losr["losr_f"] = [inrange(rnorm(rng, *i), *r) for i, r in zip(losr["interval"], losr["range"])]
    return losr
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
    w = np.ones_like(data.index).astype(float)
    i = list(data.admimeth == "11")
    w[i] = [pwla[t] for t in data[i].tretspef]
    return w
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
    # Admission Avoidance ----------------------------------------------------------------------------------------------
    # choose an admission avoidance factor
    ada = self._admission_avoidance(rng)
    factor_a = np.array([ada[k] for k in data["admission_avoidance_strategy"]])
    # waiting list adjustments
    factor_w = self._waiting_list_adjustment(data)
    # hsa
    data = self._health_status_adjustment(rng, data)
    # create a single factor for how many times to select that row
    n = rng.poisson(data["factor"] * data["hsa_f"] * factor_a * factor_w)
    # drop columns we don't need and repeat rows n times
    data = data.loc[data.index.repeat(n)].drop(["factor", "hsa_f"], axis = "columns")
    data.reset_index(inplace = True)
    # LoS Reduction ----------------------------------------------------------------------------------------------------
    # get the parameters
    losr = self._los_reduction(rng)
    # set the index for easier querying
    data.set_index(["los_reduction_strategy"], inplace = True)
    data = self._losr_all(data, losr, rng)
    data = self._losr_aec(data, losr, rng)
    data = self._losr_preop(data, losr, rng)
    data = self._losr_bads(data, losr, rng)
    # return the data
    return (variant, data.reset_index()[["rn", "speldur", "classpat", "admission_avoidance_strategy", "los_reduction_strategy"]])
  #
  def _losr_all(self, data, losr, rng):
    i = losr.type == "all"
    s = losr.index[i]
    data.loc[s, "speldur"] = rng.binomial(data.loc[s, "speldur"], losr.loc[data.loc[s].index, "losr_f"])
    return data
  def _losr_aec(self, data, losr, rng):
    i = losr.type == "aec"
    s = losr.index[i]
    n = len(data.loc[s, "speldur"])
    data.loc[s, "speldur"] *= rng.uniform(size = n) >= losr.loc[data.loc[s].index, "losr_f"]
    return data
  def _losr_preop(self, data, losr, rng): # this is exactly the same as _losr_aec
    i = losr.type == "pre-op"
    s = losr.index[i]
    n = len(data.loc[s, "speldur"])
    data.loc[s, "speldur"] *= rng.uniform(size = n) >= losr.loc[data.loc[s].index, "losr_f"]
    return data
  def _losr_bads(self, data, losr, rng):
    i = losr.type == "bads"
    bads_df = data.merge(
      losr[i][["baseline_target_rate", "op_dc_split", "losr_f"]],
      left_index = True,
      right_index = True
    )
    r = (
      (bads_df["losr_f"] - bads_df["baseline_target_rate"]) / (1 - bads_df["baseline_target_rate"])
    ).apply(inrange)
    u1 = bads_df["op_dc_split"] * r
    u2 = (1 - bads_df["op_dc_split"]) * r
    u0 = 1 - u1 - u2
    x = np.random.uniform(size = len(r))
    #
    bads_df.loc[(x >= u0) & (x < u0 + u1), "classpat"] = "2"
    bads_df.loc[(x >= u0 + u1), "classpat"] = "-1"
    #
    s = losr.index[i]
    data.loc[s, "classpat"] = bads_df["classpat"]
    data.loc[s, "speldur"] *= data.loc[s, "classpat"] == 1
    return data