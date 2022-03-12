import pyarrow.parquet as pq
import pandas as pd
import numpy as np

from collections import defaultdict

from model.helpers import rnorm, inrange
from model.Model import Model


class OutpatientsModel(Model):
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
      ._load_parquet("op")
      .merge(self._demog_factors, left_on = ["age", "sex"], right_index = True)
      .groupby(["variant"])
    )
    self._data = { k: v.drop(["variant"], axis = "columns").set_index(["rn"]) for k, v in tuple(data) }
  #
  def _followup_reduction(self, data, rng):
    p = self._params["outpatient_factors"]["followup_reduction"]
    f = pd.DataFrame([
      [0] * len(p),
      [0] * len(p),
      p.keys(),
      [inrange(rnorm(rng, *v)) for v in p.values()]
    ], columns = ["has_procedures", "is_first", "type", "fur_f"])
    #
    data = data.merge(f, how = "left", on = ["has_procedures", "is_first", "type"])
    data["fur_f"].fillna(1, inplace = True)
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
    # create a single factor for how many times to select that row
    factor = data["factor"] * data["hsa_f"] * data["fur_f"]
    data["attendances"] = rng.poisson(data["attendances"] * factor)
    data["tele_attendances"] = rng.poisson(data["tele_attendances"] * factor)
    data = data[data["attendances"] + data["tele_attendances"] > 0]
    # convert attendances to tele attendances
    data = self._convert_to_tele(data, rng)
    # return the data
    return (variant, data[["attendances", "tele_attendances"]].reset_index())
