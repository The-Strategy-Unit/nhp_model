import pyarrow.parquet as pq
import pandas as pd
import numpy as np

from model.helpers import rnorm, inrange
from model.Model import Model

class AaEModel(Model):
  """
  Accident and Emergency Model

  Implements the model for accident and emergency data. See `Model()` for documentation on the generic class. 
  """
  def __init__(self, results_path):
    self._MODEL_TYPE = "aae"
    # call the parent init function
    Model.__init__(self, results_path)
  #
  def _generate_run_params(self, rng):
    p = self._params["aae_factors"]
    f = lambda v: inrange(rnorm(rng, *v["interval"]), *v["range"])
    return {
      k1: {
        k2: f(v2) for k2, v2 in v1.items()
      } for k1, v1 in p.items()
    }
  # 
  def _low_cost_discharged(self, data, run_params):
    return self._factor_helper(data, run_params["low_cost_discharged"], {
      "is_low_cost_referred_or_discharged": 1
    })
  #
  def _left_before_seen(self, data, run_params):
    return self._factor_helper(data, run_params["left_before_seen"], {
      "is_left_before_treatment": 1
    })
  #
  def _frequent_attenders(self, data, run_params):
    return self._factor_helper(data, run_params["frequent_attenders"], {
      "is_frequent_attender": 1
    })
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
    # hsa
    hsa_params, hsa_f = self._health_status_adjustment(rng, data)
    #
    run_params = self._generate_run_params(rng)
    # create a single factor for how many times to select that row
    factor = (data["factor"].to_numpy()
      * hsa_f
      * self._low_cost_discharged(data, run_params)
      * self._left_before_seen(data, run_params)
      * self._frequent_attenders(data, run_params)
    )
    data["arrivals"] = rng.poisson(data["arrivals"] * factor)
    data = data[data["arrivals"] > 0]
    # return the data
    run_params["selected_variant"] = variant
    run_params["hsa"] = hsa_params
    return (run_params, data[["arrivals"]].reset_index())
