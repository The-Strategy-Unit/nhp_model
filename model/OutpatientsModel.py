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
  def _followup_reduction(self, rng, row):
    if row.has_procedures: return 1
    interval = self._params["outpatient_factors"]["followup_reduction"][row.type]
    return inrange(rnorm(rng, *interval))
  #
  def _convert_to_tele(self, rng, row):
    interval = self._params["outpatient_factors"]["convert_to_tele"][row.type]
    r = inrange(rnorm(rng, *interval))
    return rng.binomial(row.attendances, r)
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
    ffu = [self._followup_reduction(rng, r) for r in data.itertuples()]
    # create a single factor for how many times to select that row
    data["attendances"] = [rng.poisson(f) for f in (data["attendances"] * data["factor"] * data["hsa_f"] * ffu)]
    data["tele_attendances"] = [rng.poisson(f) for f in (data["tele_attendances"] * data["factor"] * data["hsa_f"] * ffu)]
    data = data[data["attendances"] + data["tele_attendances"] > 0]
    # convert attendances to tele attendances
    tr = [self._convert_to_tele(rng, r) for r in data.itertuples()]
    data["attendances"] -= tr
    data["tele_attendances"] += tr
    # return the data
    return (variant, data[["attendances", "tele_attendances"]].reset_index())
