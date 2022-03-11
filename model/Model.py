import json
import pickle
import os

import numpy as np
import pyarrow.parquet as pq
import pandas as pd

from collections import defaultdict
from pathlib import Path
from pathos.multiprocessing import ProcessPool

from model.helpers import rnorm, inrange

class Model:
  def __init__(self, results_path):
    # load the parameters file
    with open(f"{results_path}/params.json", "r") as f: self._params = json.load(f)
    # store the path where the data is stored and the results are stored
    self._path = str(Path(results_path).parent.parent.parent)
    self._results_path = results_path
    #
    # load the data that's shared across different model types
    #
    with open(f"{self._path}/hsa_gams.pkl", "rb") as f: self._hsa_gams = pickle.load(f)
    self._load_demog_factors()
  #
  def _select_variant(self, rng):
    """
    Randomly select a single variant to use for a model run
    """
    v = rng.choice(self._variants, p = self._probabilities)
    return (v, self._data[v])
  #
  def _load_demog_factors(self):
    dfp = self._params["demographic_factors"]
    start_year = dfp.get("start_year", "2018")
    end_year = dfp.get("end_year", "2043")
    #
    df = pd.read_csv(os.path.join(self._path, dfp["file"]))
    df[["age", "sex"]] = df[["age", "sex"]].astype(int)
    df["factor"] = df[end_year] / df[start_year]
    self._demog_factors = df.set_index(["age", "sex"])[["variant", "factor"]]
    #
    self._variants = list(dfp["variant_probabilities"].keys())
    self._probabilities = list(dfp["variant_probabilities"].values())
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
        "hsagrp": a,
        "sex": int(s),
        "age": ages,
        "hsa_f": g.predict(adjusted_ages) / g.predict(ages)
      })
      for (a, s), g in self._hsa_gams.items()
    ])
    return (data
      .reset_index()
      .merge(hsa, on = ["hsagrp", "sex", "age"], how = "left")
      .fillna(1)
      .set_index(["rn"])
    )
  # 
  def _load_parquet(self, file, *args):
    """
    Load a parquet file from the file path created by the constructor.

    You can selectively load columns by passing an array of column names to *args.
    """
    return pq.read_pandas(os.path.join(self._path, f"{file}.parquet"), *args).to_pandas()
  #
  def save_run(self, model_run):
    """
    Save the model run

    * model_run: the number of this model run

    Runs the model, saving the results to the results folder
    """
    variant, mr_data = self.run(model_run)
    os.mkdir(f"{self._results_path}/results/model_run={model_run}")
    mr_data.to_parquet(f"{self._results_path}/results/model_run={model_run}/{model_run}.parquet")
    with open(f"{self._results_path}/selected_variants/model_run={model_run}.txt", "w") as f:
      f.write(variant)
  #
  def multi_model_runs(self, run_start, model_runs, N_CPUS = 1):
    pool = ProcessPool(ncpus = N_CPUS)
    pool.amap(self.save_run, range(run_start, run_start + model_runs))
    pool.close()
    pool.join()
  #
  def run(self, model_run):
    """
    Run the model once

    returns: a tuple of the selected varient and the updated DataFrame
    """
    pass