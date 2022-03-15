import pyarrow.parquet as pq
import pandas as pd
import numpy as np

from model.helpers import rnorm, inrange
from model.Model import Model

class OutpatientsModel(Model):
  """
  Outpatients Model

  Implements the model for outpatient data. See `Model()` for documentation on the generic class. 
  """
  def __init__(self, results_path):
    self._MODEL_TYPE = "op"
    # call the parent init function
    Model.__init__(self, results_path)
  #
  def _followup_reduction(self, data, rng):
    p = self._params["outpatient_factors"]["followup_reduction"]
    f = pd.DataFrame({
      "has_procedures": [0] * len(p),
      "is_first": [0] * len(p),
      "type": p.keys(),
      "fur_f": [inrange(rnorm(rng, *v)) for v in p.values()]
    })
    #
    data = data.merge(f, how = "left", on = list(f.columns[:-1]))
    data[f.columns[-1]].fillna(1, inplace = True)
    return data
  #
  def _consultant_to_consultant_reduction(self, data, rng):    
    p = self._params["outpatient_factors"]["consultant_to_consultant_reduction"]
    f = pd.DataFrame({
      "is_cons_cons_ref": [1] * len(p),
      "type": p.keys(),
      "c2c_f": [inrange(rnorm(rng, *v)) for v in p.values()]
    })
    #
    data = data.merge(f, how = "left", on = list(f.columns[:-1]))
    data[f.columns[-1]].fillna(1, inplace = True)
    return data
  #
  def _convert_to_tele(self, data, rng):
    p = {
      k: inrange(rnorm(rng, *v))
      for k, v in self._params["outpatient_factors"]["convert_to_tele"].items()
    }
    tc = np.random.binomial(data["attendances"], [p[t] for t in data["type"]])
    data["attendances"] -= tc
    data["tele_attendances"] += tc
    return data
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
    data = self._health_status_adjustment(rng, data)
    # first to follow-up adjustments
    data = self._followup_reduction(data, rng)
    # consultant to consultant referrals adjustments
    data = self._consultant_to_consultant_reduction(data, rng)
    # create a single factor for how many times to select that row
    factor = data["factor"] * data["hsa_f"] * data["fur_f"]
    data["attendances"] = rng.poisson(data["attendances"] * factor)
    data["tele_attendances"] = rng.poisson(data["tele_attendances"] * factor)
    data = data[data["attendances"] + data["tele_attendances"] > 0]
    # convert attendances to tele attendances
    data = self._convert_to_tele(data, rng)
    # return the data
    return (variant, data[["attendances", "tele_attendances"]].reset_index())
